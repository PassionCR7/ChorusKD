"""S 阶段：能力匹配调度。

为什么黑盒比白盒更需要这个
--------------------------
白盒能实时看教师分布的熵，直接判断"这个位置教师有多确信"。黑盒看不到，
只能靠**离线统计**推断难度——具体说，用学生在该样本上的 loss 反推。

调度逻辑来自 ZPD（zone of proximal development）：
始终让学生做"略高于当前能力"的题。太低没梯度，太高学不动。

控制回路
--------
每轮结束后看学生在本轮样本上的平均 loss：

    loss > target_high  → 太难，降档
    loss < target_low   → 太简单，升档
    区间内              → 保持

三个工程细节
------------
1. **加滞回（hysteresis）**。只在单调越界时才动作，且动作后至少保持
   一轮。否则 loss 在阈值附近抖动会引起档位来回跳，每轮问的题完全不同，
   学生学不到连续的东西。

2. **升降不对称**。降档比升档更激进（`down_step > up_step`）：太难会
   直接浪费整轮预算且损伤模型，太简单只是低效。

3. **记录档位历史**。档位轨迹本身是很好的诊断信号：如果一直在最低档
   打转，说明学生根本没在进步，应该停下来查问题而不是继续烧预算。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from config import SchedulerCfg


@dataclass
class SchedulerState:
    """调度的全部可变状态，可序列化以便断点续跑。"""

    current_bin: int = 1
    num_bins: int = 5
    # 每轮的 (轮次, 档位, 平均 loss)
    history: list[tuple[int, int, float]] = field(default_factory=list)
    # 上一轮是否刚动作过（用于滞回）
    just_moved: bool = False
    move_direction: int = 0


class DifficultyScheduler:
    def __init__(self, cfg: SchedulerCfg, num_bins: int):
        self.cfg = cfg
        self.num_bins = num_bins
        start = min(max(cfg.start_bin, 0), num_bins - 1)
        self.state = SchedulerState(
            current_bin=start, num_bins=num_bins,
        )

    @property
    def current_bin(self) -> int:
        return self.state.current_bin

    def observe(self, round_idx: int, mean_loss: float) -> tuple[int, str]:
        """喂入本轮的平均 loss，返回 (新档位, 说明)。

        这是唯一改变档位的入口，保证所有决策都有记录。
        """
        cfg = self.cfg
        self.state.history.append((round_idx, self.state.current_bin, mean_loss))

        if not cfg.enabled:
            return self.state.current_bin, "调度关闭"

        # 滞回：上一轮刚动过，本轮先观察
        if self.state.just_moved:
            self.state.just_moved = False
            return self.state.current_bin, "滞回期，保持"

        new_bin = self.state.current_bin
        note = "区间内保持"

        if mean_loss > cfg.target_loss_high:
            # 太难，降档。降幅更激进。
            step = max(cfg.step, 2)
            new_bin = max(self.state.current_bin - step, 0)
            note = f"loss {mean_loss:.2f} > {cfg.target_loss_high}，降档 {step}"
        elif mean_loss < cfg.target_loss_low:
            # 太简单，升档。升幅保守。
            step = cfg.step
            new_bin = min(self.state.current_bin + step, self.num_bins - 1)
            note = f"loss {mean_loss:.2f} < {cfg.target_loss_low}，升档 {step}"

        if new_bin != self.state.current_bin:
            self.state.move_direction = 1 if new_bin > self.state.current_bin else -1
            self.state.current_bin = new_bin
            self.state.just_moved = True

        return self.state.current_bin, note

    def trend(self) -> str:
        """档位趋势诊断。

        一直在最低档 → 学生没进步，该查问题不是继续跑
        档位持续升高 → 正常进步
        忽高忽低 → 阈值设置不当或 loss 波动过大
        """
        h = self.state.history
        if len(h) < 2:
            return "样本不足"
        bins = [b for _, b, _ in h]
        losses = [l for _, _, l in h if l == l]
        if not bins:
            return "无有效记录"

        if max(bins) == 0 and len(bins) >= 2:
            return "⚠ 始终在最低档：学生未进步或阈值过严"

        if len(bins) >= 3:
            # 先剔掉 0（档位未变的轮次），否则 0 会把相邻的同向变化
            # 隔断，导致「降-停-升-停-降」这种真震荡被判成非震荡。
            diffs = [d for d in (
                bins[i + 1] - bins[i] for i in range(len(bins) - 1)
            ) if d != 0]
            # 震荡 = 方向交替**两次以上**。单次"降了又升"是正常的
            # 自适应过程（先发现太难，降档，学会后再升），不能算震荡。
            sign_changes = sum(
                1 for i in range(len(diffs) - 1)
                if diffs[i] * diffs[i + 1] < 0
            )
            if sign_changes >= 2:
                return "⚠ 档位震荡：阈值区间偏窄或 loss 方差过大"
            if diffs and all(d > 0 for d in diffs):
                return f"↑ 档位稳步上升 {bins[0]}→{bins[-1]}"
            if diffs and all(d < 0 for d in diffs):
                return f"↓ 档位持续下降 {bins[0]}→{bins[-1]}"
        if len(losses) >= 2 and losses[-1] < losses[0]:
            return f"↓ loss 下降 {losses[0]:.2f}→{losses[-1]:.2f}"
        return "平稳"

    def to_dict(self) -> dict:
        return {
            "current_bin": self.state.current_bin,
            "num_bins": self.num_bins,
            "history": self.state.history,
            "just_moved": self.state.just_moved,
            "move_direction": self.state.move_direction,
        }

    @classmethod
    def from_dict(cls, cfg: SchedulerCfg, d: dict) -> "DifficultyScheduler":
        obj = cls(cfg, d.get("num_bins", 5))
        obj.state.current_bin = d.get("current_bin", obj.state.current_bin)
        obj.state.history = [tuple(x) for x in d.get("history", [])]
        obj.state.just_moved = d.get("just_moved", False)
        obj.state.move_direction = d.get("move_direction", 0)
        return obj


def bin_description(bin_idx: int, num_bins: int) -> str:
    """档位的人类可读描述，用于日志。"""
    if num_bins <= 1:
        return "全部"
    ratio = bin_idx / (num_bins - 1)
    if ratio < 0.2:
        label = "最简单"
    elif ratio < 0.4:
        label = "偏简单"
    elif ratio < 0.6:
        label = "中等"
    elif ratio < 0.8:
        label = "偏难"
    else:
        label = "最难"
    return f"{label} ({bin_idx+1}/{num_bins})"