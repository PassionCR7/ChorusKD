"""P 阶段：提示词策展。

黑盒的每次请求都有成本，所以**问什么**比**问多少次**更重要。
随机从语料抽问题的做法，会把预算浪费在学生已经会了的内容上。

三种选题策略
------------
difficulty（推荐）
    用学生当前 loss 给候选池排序，优先选 loss 处在"可学习区间"的。
    这是课程学习在数据选择上的直接应用：太简单的没梯度，
    太难的学不动，中间的才有效。

diverse
    对候选做 embedding 聚类，每簇均匀取。适合学生能力还很不均匀的早期，
    或者需要覆盖多种任务类型的场景。

random
    纯随机，作消融基线。论文里几乎所有 selection 方法都必须和它比，
    否则无法证明策略本身有效。

变体生成
--------
同一条种子问题改写成 k 个变体，本质是**数据增强换预算**：
一次改写成本远低于一次教师请求，但能让同一知识点在不同表述下
都被教师标注，提升泛化。改写用模板而非另一套 LLM，避免引入
额外的不确定性来源。
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from config import PromptCfg


# --------------------------------------------------------------------------- #
# 种子读取
# --------------------------------------------------------------------------- #
def load_seeds(path: str, limit: int) -> list[str]:
    """从 SFT 语料里抽出 user 侧的问题文本。

    只取 user 轮，不取 assistant——我们要问的是"问题"，答案由教师重新生成。
    用教师现有答案会导致「学生学的是老答案，教师能力没被用上」。
    """
    out = []
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if len(out) >= limit:
                break
            ln = ln.strip()
            if not ln:
                continue
            try:
                obj = json.loads(ln)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversations") or obj.get("messages") or []
            for m in conv:
                if m.get("role") == "user":
                    c = (m.get("content") or "").strip()
                    if 4 <= len(c) <= 800:
                        out.append(c)
                    break
    return out


# --------------------------------------------------------------------------- #
# 变体生成
# --------------------------------------------------------------------------- #
# 改写模板：不改变问题本质，只换表述角度。
# 刻意避开"更详细地"这类会拉长答案的模板——答案长度会显著影响
# 一致性投票（长答案的字符 F1 天然偏低）。
_VARIANT_TEMPLATES = [
    "{q}",
    "请简要回答：{q}",
    "关于「{q}」，核心要点是什么？",
    "用一两句话解释：{q}",
]


def make_variants(q: str, k: int) -> list[str]:
    """给一个问题生成 k 个表述变体（含原句）。"""
    if k <= 1:
        return [q]
    out = []
    for i in range(k):
        tpl = _VARIANT_TEMPLATES[i % len(_VARIANT_TEMPLATES)]
        out.append(tpl.format(q=q))
    return out


# --------------------------------------------------------------------------- #
# 难度分档
# --------------------------------------------------------------------------- #
def assign_bins(
    losses: list[float],
    num_bins: int,
) -> list[int]:
    """按 loss 分位数把候选切成若干档，返回每条的档位。

    用分位数而非固定阈值：不同语料/不同学生阶段的 loss 尺度差别很大，
    固定阈值（如 <1 简单、>3 困难）换一份数据就失效。
    """
    n = len(losses)
    if n == 0:
        return []
    order = sorted(range(n), key=lambda i: losses[i])
    bins = [0] * n
    for rank, idx in enumerate(order):
        bins[idx] = min(rank * num_bins // n, num_bins - 1)
    return bins


@dataclass
class Candidate:
    prompt: str
    student_loss: float
    bin: int = 0
    group: str = ""      # 聚类组标签（diverse 策略用）

    def variants(self, k: int) -> list[str]:
        return make_variants(self.prompt, k)


# --------------------------------------------------------------------------- #
# 选题
# --------------------------------------------------------------------------- #
def _cluster_by_chars(prompts: list[str], k: int) -> list[int]:
    """无 embedding 时的降级聚类：按字符集 Jaccard 相似度做单遍贪心。

    真正的 diverse 应该用句向量 + kmeans。这里不引入额外模型依赖
    （黑盒场景往往也不允许再加载一个 embedding 模型），
    用字符集近似主题分布，对"技术/生活/代码"这类粗粒度区分够用。
    """
    sets = [set(p) for p in prompts]
    labels = [-1] * len(prompts)
    next_label = 0
    reps: list[set] = []
    for i, s in enumerate(sets):
        best, best_j = -1, 0.0
        for j, r in enumerate(reps):
            inter = len(s & r)
            union = len(s | r) or 1
            jac = inter / union
            if jac > best_j:
                best_j, best = jac, j
        if best >= 0 and best_j >= 0.35:
            labels[i] = best
        else:
            reps.append(s)
            labels[i] = next_label
            next_label += 1
    return labels


def curate(
    seeds: list[str],
    cfg: PromptCfg,
    student_losses: list[float] | None = None,
    target_bin: int | None = None,
    rng: random.Random | None = None,
    num_bins: int = 5,
) -> list[Candidate]:
    """按策略选出本轮要问的问题。

    参数
    ----
    student_losses : 与 seeds 等长，学生在每个问题现有答案上的 loss
    target_bin     : 只取该难度档（由 scheduler 决定）；None 表示全取
    """
    rng = rng or random.Random(0)
    n = len(seeds)
    if n == 0:
        return []

    if student_losses is None or len(student_losses) != n:
        student_losses = [float("nan")] * n
        has_loss = False
    else:
        has_loss = True

    cands = [
        Candidate(prompt=q, student_loss=student_losses[i])
        for i, q in enumerate(seeds)
    ]

    # ---- 分档 ----
    valid = [c.student_loss for c in cands if c.student_loss == c.student_loss]
    if valid and has_loss:
        bins = assign_bins([c.student_loss for c in cands], num_bins)
        for c, b in zip(cands, bins):
            c.bin = b

    # ---- 按策略选 ----
    if cfg.strategy == "random" or not has_loss:
        picked = cands[:]
        rng.shuffle(picked)

    elif cfg.strategy == "difficulty":
        if target_bin is not None:
            # 只要指定档位；若该档空了，退化为相邻档
            pool = [c for c in cands if c.bin == target_bin]
            if not pool:
                for d in (1, -1, 2, -2):
                    pool = [c for c in cands if c.bin == target_bin + d]
                    if pool:
                        break
            picked = pool or cands[:]
        else:
            # 无目标档时，按"离可学习区间中心最近"排序
            mid = (min(valid) + max(valid)) / 2 if valid else 0.0
            picked = sorted(cands, key=lambda c: abs(c.student_loss - mid))
        # 档内打乱，避免总取同一批
        rng.shuffle(picked)

    elif cfg.strategy == "diverse":
        labels = _cluster_by_chars(seeds, cfg.num_difficulty_bins)
        groups: dict[int, list[Candidate]] = {}
        for c, lb in zip(cands, labels):
            c.group = f"g{lb}"
            groups.setdefault(lb, []).append(c)
        # 每簇轮转取，直到凑够
        picked = []
        keys = list(groups.keys())
        rng.shuffle(keys)
        idx = {k: 0 for k in keys}
        while len(picked) < len(cands):
            progressed = False
            for k in keys:
                if idx[k] < len(groups[k]):
                    picked.append(groups[k][idx[k]])
                    idx[k] += 1
                    progressed = True
            if not progressed:
                break
    else:
        picked = cands[:]

    # ---- 截断到目标数量 ----
    return picked[: cfg.num_prompts]


def expand_prompts(cands: list[Candidate], cfg: PromptCfg) -> list[tuple[str, str]]:
    """展开成 (原始问题, 实际问句) 对。

    返回的第二个值才是发给教师的。保留原始问题是因为：
    不同变体的答案要能追溯到同一个知识点，方便一致性统计和难度回填。
    """
    out = []
    for c in cands:
        for v in c.variants(cfg.variants_per_seed):
            out.append((c.prompt, v))
    return out


# --------------------------------------------------------------------------- #
# 难度评估（学生侧）
# --------------------------------------------------------------------------- #
@dataclass
class LossProbe:
    """用学生模型给「问题 + 已有答案」打 loss，作为难度代理。

    注意用**学生**的 loss，不是教师的。理由见 filter.py 说明：
    要教的是"学生跳一跳够得着"的内容。
    """

    model: object
    tokenizer: object
    device: str = "cuda"
    max_len: int = 512

    @staticmethod
    def _encode(tok, text: str) -> list[int]:
        try:
            msgs = [{"role": "user", "content": text}]
            txt = tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            txt = text
        return tok(txt, add_special_tokens=False).input_ids

    def score(self, prompts: list[str], answers: list[str] | None = None) -> list[float]:
        """返回每个 (prompt, answer) 的平均 token loss。

        answers 为 None 时，只对 prompt 本身算 loss（信息量较少，
        仅用于没有参考答案时的粗略排序）。
        """
        import torch
        import torch.nn.functional as F

        out = []
        self.model.eval()
        with torch.no_grad():
            for i, p in enumerate(prompts):
                ids = self._encode(self.tokenizer, p)
                if answers is not None and i < len(answers) and answers[i]:
                    a_ids = self.tokenizer(
                        answers[i], add_special_tokens=False
                    ).input_ids
                    ids = ids + a_ids
                    a_start = len(ids) - len(a_ids)
                else:
                    a_start = max(len(ids) - 32, 1)

                ids = ids[: self.max_len]
                if len(ids) < 4 or a_start >= len(ids):
                    out.append(float("nan"))
                    continue

                t = torch.tensor([ids], dtype=torch.long, device=self.device)
                logits = self.model(input_ids=t).logits.float()
                lg = logits[0, a_start - 1: -1]
                lb = t[0, a_start:]
                if lg.shape[0] != lb.shape[0] or lg.shape[0] == 0:
                    out.append(float("nan"))
                    continue
                loss = F.cross_entropy(lg, lb, reduction="mean")
                out.append(float(loss))
        self.model.train()
        return out