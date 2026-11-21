"""R 阶段：质量与难度筛选。

黑盒蒸馏最大的质量风险不是"教师不够好"，而是**教师幻觉和黑盒无法区分
"不会"和"不想"**。

白盒蒸馏里，教师在某位置的分布自带不确定性信号：如果它不确定，
概率会摊在多个 token 上，学生学到的是"这里要谨慎"。黑盒只给文本，
答错的时候语气和答对时一模一样，没有任何可用的置信度信号。

自一致性投票是唯一可靠的替代
----------------------------
同一问题在多个温度下采 n 次：
  * n 次答案高度一致 → 教师对这个答案是确信的，可以当软标签用
  * n 次答案分裂成多个簇 → 要么问题有歧义，要么教师不会
    （黑盒无法区分这两者，所以统一保守处理）

实测这类"一致率"和教师答案正确率是**正相关但远弱于预期**的关系：
在 24 题 × 5 温度 × 5 采样的实测中（`experiments/exp2_consistency_correlation.py`），
Spearman 仅 +0.27（95% bootstrap CI [−0.17, +0.67]），Pearson +0.27。
早期版本声称的 0.6~0.75 未能复现，详见 README 的「实验验证」一节。

更准确的结论是：一致率能**把明显分裂的样本筛掉**，但不能作为"高一致 = 一定对"
的判据。实测中 agreement=1.00 但答案错误的样本确实存在（教师"一致地错"）。
因此本模块的定位是**粗筛**，阈值应偏保守，不要用一致率替代真实标注。

难度筛选的第二个过滤器
----------------------
一致性高不代表有用。如果学生已经会用 0.2 的 loss 复现这个答案，
这条样本没有信息量；如果 loss 高达 12，学生根本学不动（梯度被少数
难 token 主导）。所以要做"可学习性"带通滤波。

注意：难度用**学生 loss** 代理，不是教师 loss。这一点容易搞错——
我们要教的是"学生跳一跳够得着"的内容，不是"教师觉得难"的内容。
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from teacher_api import Sample, TeacherResponse


# --------------------------------------------------------------------------- #
# 一致性度量
# --------------------------------------------------------------------------- #
def _normalize(text: str) -> str:
    """去空白、去标点、小写。用于 exact 模式比较。"""
    t = re.sub(r"\s+", "", text)
    t = re.sub(r"[，。！？、；：\"''（）【】《》,.!?;:\"'()\[\]<>]", "", t)
    return t.lower()


def _token_f1(a: str, b: str) -> float:
    """字符级 token F1，用于中文（无空格分词）。

    用字符而非词：中文分词本身有歧义，且两个答案的"一致性"更多体现在
    用词覆盖率上，字符级更稳健。英文场景下字符级会偏高，但阈值可调。
    """
    ta = list(_normalize(a))
    tb = list(_normalize(b))
    if not ta or not tb:
        return 0.0
    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if overlap == 0:
        return 0.0
    p = overlap / len(ta)
    r = overlap / len(tb)
    return 2 * p * r / (p + r)


def _similar(a: str, b: str, mode: str, threshold: float) -> bool:
    if mode == "exact":
        return _normalize(a) == _normalize(b)
    if mode == "fuzzy":
        return _token_f1(a, b) >= threshold
    if mode == "semantic":
        # 无 embedding 模型时的降级：用更长窗口的字符重合
        # 真正的 semantic 需要句向量，这里退化为宽松 fuzzy
        return _token_f1(a, b) >= threshold * 0.8
    return _token_f1(a, b) >= threshold


# --------------------------------------------------------------------------- #
# 聚类（基于相似度的单遍贪心）
# --------------------------------------------------------------------------- #
def cluster_samples(
    samples: list[Sample],
    mode: str = "fuzzy",
    threshold: float = 0.75,
) -> list[list[Sample]]:
    """把 n 次采样按相似度聚成若干簇。

    用单遍贪心而非层次聚类：n 通常只有 3~8，O(n²) 完全可接受，
    而贪心的结果是确定的（不依赖初始化），便于复现和调试。
    """
    clusters: list[list[Sample]] = []
    for s in samples:
        placed = False
        for c in clusters:
            # 与该簇的**代表**（第一个元素）比较，而非簇内所有元素，
            # 避免链式相似导致两个不同答案被合并（chaining problem）
            if _similar(s.text, c[0].text, mode, threshold):
                c.append(s)
                placed = True
                break
        if not placed:
            clusters.append([s])
    return clusters


def pick_majority(clusters: list[list[Sample]]) -> tuple[list[Sample], float]:
    """返回最大簇及它占总采样的比例。"""
    if not clusters:
        return [], 0.0
    total = sum(len(c) for c in clusters)
    best = max(clusters, key=len)
    return best, len(best) / max(total, 1)


# --------------------------------------------------------------------------- #
# 拒答 / 幻觉模式检测
# --------------------------------------------------------------------------- #
def has_reject_pattern(text: str, patterns: tuple[str, ...]) -> bool:
    t = text.strip()
    for p in patterns:
        if p in t:
            return True
    # 开头就是"抱歉"类，通常也是拒答
    if re.match(r"^\s*(抱歉|对不起|Sorry|I apologize)", t, re.I):
        return True
    return False


def looks_degenerate(text: str) -> bool:
    """检测退化输出：复读、极短、纯符号。

    小模型教师（mock 模式）很容易复读，这类样本教给学生只会传染。
    """
    t = text.strip()
    if len(t) < 4:
        return True

    # 符号占比过高
    alnum = sum(ch.isalnum() or '\u4e00' <= ch <= '\u9fff' for ch in t)
    if alnum / max(len(t), 1) < 0.4:
        return True

    # 复读检测 1：字符多样性过低。
    # 正常中文文本的去重字符率通常 > 0.4；复读会掉到 0.1 以下。
    # 用比例而非绝对次数，这样短文本也能判出来。
    uniq_ratio = len(set(t)) / len(t)
    if len(t) >= 12 and uniq_ratio < 0.15:
        return True

    # 复读检测 2：固定步长重复。
    # 检测「某个短片段连续重复」——这是模型卡死的典型形态，
    # 但字符多样性可能仍不低（如 "abcabcabcabc" uniq=3/12=0.25）。
    for period in range(2, min(13, len(t) // 2 + 1)):
        last = t[-period:]
        reps = 0
        end = len(t)
        while end - period >= 0 and t[end - period:end] == last:
            reps += 1
            end -= period
        if reps >= 4 and reps * period >= len(t) * 0.6:
            return True

    return False


# --------------------------------------------------------------------------- #
# 单条样本判定
# --------------------------------------------------------------------------- #
@dataclass
class FilteredSample:
    """过筛后的一条训练样本。"""

    prompt: str
    answer: str
    # 一致但非主答案的其他采样的文本（用于"多正例"辅助损失）
    aux_answers: list[str] = field(default_factory=list)
    agreement: float = 0.0
    num_samples: int = 0
    # 学生在该答案上的 loss（难度代理），由外部填入
    student_loss: float = float("nan")
    reason: str = "ok"


def filter_response(
    resp: TeacherResponse,
    cfg,
    student_loss: float | None = None,
) -> FilteredSample | None:
    """把一个 TeacherResponse 变成一条训练样本，或判定为不可用。

    返回 None 表示丢弃，丢弃原因记录在返回值之外（调用方统计）。
    """
    ok = resp.ok_samples
    if not ok:
        return None

    # ---- 1. 逐条清洗 ----
    clean = []
    for s in ok:
        txt = s.text.strip()
        if has_reject_pattern(txt, cfg.reject_patterns):
            continue
        if looks_degenerate(txt):
            continue
        n_tok = len(txt)
        if not (cfg.min_answer_tokens <= n_tok <= cfg.max_answer_tokens):
            continue
        clean.append(Sample(
            text=txt, temperature=s.temperature,
            tokens_in=s.tokens_in, tokens_out=s.tokens_out,
        ))

    if len(clean) < 2:
        # 清洗后不足 2 条，无法做一致性判断。
        # 只有 1 条时仍可用，但 agreement 记 1/1（不可信，调用方应降权）
        if len(clean) == 1:
            return FilteredSample(
                prompt=resp.prompt, answer=clean[0].text,
                agreement=1.0, num_samples=1,
                student_loss=student_loss if student_loss is not None else float("nan"),
                reason="single",
            )
        return None

    # ---- 2. 一致性聚类 ----
    clusters = cluster_samples(clean, cfg.consistency_mode, cfg.cluster_threshold)
    best_cluster, ratio = pick_majority(clusters)

    if ratio < cfg.min_agreement_ratio:
        # 教师自己都分裂，不可信
        return None

    # ---- 3. 选主答案 ----
    # 簇内选**最短**的那条：最简洁的表述通常最贴近核心事实，
    # 冗长的版本更可能是模型在"补话"（幻觉高发区）。
    main = min(best_cluster, key=lambda s: len(s.text))
    aux = [s.text for s in best_cluster if s.text != main.text]

    # ---- 4. 难度带通 ----
    reason = "ok"
    if student_loss is not None:
        if student_loss < cfg.min_student_loss:
            reason = "too_easy"
        elif student_loss > cfg.max_student_loss:
            reason = "too_hard"

    return FilteredSample(
        prompt=resp.prompt,
        answer=main.text,
        aux_answers=aux,
        agreement=ratio,
        num_samples=len(clean),
        student_loss=student_loss if student_loss is not None else float("nan"),
        reason=reason,
    )


# --------------------------------------------------------------------------- #
# 批量筛选 + 统计
# --------------------------------------------------------------------------- #
def filter_batch(
    responses: list[TeacherResponse],
    cfg,
    student_losses: list[float] | None = None,
    keep_too_easy: bool = False,
    keep_too_hard: bool = False,
) -> tuple[list[FilteredSample], dict]:
    """批量筛选，返回 (可用样本, 统计信息)。

    保留 too_easy / too_hard 的开关是为了诊断：
    如果全部样本都被判 too_hard，说明学生太弱或阈值设置不对，
    这时应该降档而不是硬训。
    """
    kept: list[FilteredSample] = []
    stats = Counter()

    for i, r in enumerate(responses):
        sl = None
        if student_losses is not None and i < len(student_losses):
            sl = student_losses[i]
        fs = filter_response(r, cfg, student_loss=sl)
        if fs is None:
            stats["dropped"] += 1
            continue
        if fs.reason == "too_easy" and not keep_too_easy:
            stats["too_easy"] += 1
            continue
        if fs.reason == "too_hard" and not keep_too_hard:
            stats["too_hard"] += 1
            continue
        kept.append(fs)
        stats[fs.reason] += 1
        if fs.agreement >= 0.8:
            stats["high_agreement"] += 1

    total = max(len(responses), 1)
    stats["total"] = len(responses)
    stats["keep_rate"] = len(kept) / total
    return kept, dict(stats)


def format_stats(stats: dict) -> str:
    total = stats.get("total", 0)
    kept = stats.get("ok", 0) + stats.get("single", 0)
    parts = [f"输入 {total} 条 → 保留 {kept} 条 "
             f"({stats.get('keep_rate', 0)*100:.1f}%)"]
    if stats.get("dropped"):
        parts.append(f"不一致/低质丢弃 {stats['dropped']}")
    if stats.get("too_easy"):
        parts.append(f"太简单 {stats['too_easy']}")
    if stats.get("too_hard"):
        parts.append(f"太难 {stats['too_hard']}")
    if stats.get("single"):
        parts.append(f"仅单采样 {stats['single']}")
    if stats.get("high_agreement"):
        parts.append(f"高一致 {stats['high_agreement']}")
    return " | ".join(parts)