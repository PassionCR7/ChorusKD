"""U 阶段：学生自采样 + 教师偏好排序。

黑盒做不了 GKD 那种"教师给学生 prefix 打分布"，因为拿不到分布。
但有一件事黑盒能做而白盒反而麻烦：**排序**。

为什么用排序而不是打分
----------------------
LLM 给的绝对分数在同一请求内可比，跨请求不可比（同一个答案今天 7 分
明天 8 分）。排序只需要相对偏好，天然免疫这个问题。而且排序任务的
输出空间小（k 个编号的排列），比"打分"这种连续值更容易验证和解析。

为什么排序优于"只取教师答案做 SFT"
----------------------------------
只学教师答案，学生学到的是"某一种正确写法"。排序让学生看到自己的
多个写法中哪个更好，学到的是"什么样的写法更好"——这是判别能力，
泛化性显著强于单纯模仿。

成本
----
listwise 排序每题 1 次请求（把所有候选塞进一个 prompt），
pairwise 要 k(k-1)/2 次。所以默认 listwise，只在 k 很小或需要
更精确偏好时用 pairwise。

实现取舍
--------
这里产出的是 (chosen, rejected) 对，供 DPO 使用，而不是直接
用排序结果做 SFT。理由：SFT 只能表达"这个好"，
DPO 能表达"这个比那个好"，后者是排序结果的原生形式。
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from teacher_api import BlackBoxTeacher


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str
    # 来自哪个排序位置（chosen_rank, rejected_rank），用于加权
    # 排名差越大，信号越强
    rank_gap: int = 1


# --------------------------------------------------------------------------- #
# 学生自采样
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sample_candidates(
    model, tokenizer, prompt: str, device: str,
    k: int = 4, temperature: float = 0.95, max_new: int = 256,
    system_prompt: str | None = None,
) -> list[str]:
    """让学生对同一问题生成 k 个不同答案。

    用采样的多样性作为"学生当前的能力边界"探针：
    能生成的候选越多样，说明学生对该问题的把握越分散。
    全部候选几乎一样 → 学生已经收敛，这条样本没梯度了。
    """
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})

    try:
        head = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        head = prompt

    ids = tokenizer(head, return_tensors="pt", add_special_tokens=False).input_ids
    ids = ids.to(device)

    out_texts = []
    was_training = model.training
    model.eval()
    for _ in range(k):
        out = model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=max_new,
            do_sample=True,
            temperature=max(temperature, 1e-6),
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        gen = out[0, ids.shape[1]:]
        txt = tokenizer.decode(gen, skip_special_tokens=True).strip()
        if txt:
            out_texts.append(txt)
    if was_training:
        model.train()
    return out_texts


# --------------------------------------------------------------------------- #
# 偏好对构造
# --------------------------------------------------------------------------- #
def build_pairs(
    prompt: str,
    candidates: list[str],
    order: list[int],
    max_pairs: int = 3,
) -> list[PreferencePair]:
    """把排序结果转成偏好对。

    只取"最好 vs 其他"和少量"相邻对比"，而不是全排列：
    全排列会产生 k(k-1)/2 个对，其中大量是远距离对（第 1 名 vs 第 4 名），
    信号强度和"第 1 vs 第 2"差不多，但数量多得多，会淹没训练。

    相邻对（第 i vs 第 i+1）是信息量最高的：它们之间的偏好判断最难，
    也最能刻画出细粒度的质量差异。
    """
    if len(order) < 2:
        return []

    # 去重：排序里可能有重复下标（教师输出错误），也去掉完全相同的文本
    seen_idx, seen_txt = set(), set()
    clean = []
    for i in order:
        if i in seen_idx or i >= len(candidates):
            continue
        t = candidates[i]
        if t in seen_txt:
            continue
        seen_idx.add(i)
        seen_txt.add(t)
        clean.append(i)

    if len(clean) < 2:
        return []

    pairs = []
    best = clean[0]
    # 1) 最好 vs 最差 —— 信号最明确
    pairs.append(PreferencePair(
        prompt=prompt,
        chosen=candidates[best],
        rejected=candidates[clean[-1]],
        rank_gap=len(clean) - 1,
    ))
    # 2) 相邻对 —— 刻画细粒度差异
    for i in range(len(clean) - 1):
        if len(pairs) >= max_pairs:
            break
        pairs.append(PreferencePair(
            prompt=prompt,
            chosen=candidates[clean[i]],
            rejected=candidates[clean[i + 1]],
            rank_gap=1,
        ))
    return pairs[:max_pairs]


def collect_preferences(
    model, tokenizer, teacher: BlackBoxTeacher,
    prompts: list[str], cfg, device: str, rng: random.Random,
    system_prompt: str | None = None,
) -> list[PreferencePair]:
    """完整流程：学生采样 → 教师排序 → 构对。

    只对 cfg.sample_ratio 比例的样本做，因为这条链路成本是普通 SFT 的
    (k+1) 倍。用固定随机种子抽样保证可复现。
    """
    if not prompts:
        return []

    n_do = max(1, int(len(prompts) * cfg.sample_ratio))
    picked = rng.sample(prompts, min(n_do, len(prompts)))

    all_pairs = []
    for i, p in enumerate(picked):
        cands = sample_candidates(
            model, tokenizer, p, device,
            k=cfg.num_candidates,
            temperature=cfg.sample_temperature,
            system_prompt=system_prompt,
        )
        if len(cands) < 2:
            continue
        try:
            order = teacher.rank(p, cands, system=system_prompt)
        except Exception:
            continue
        all_pairs.extend(build_pairs(p, cands, order))
        if (i + 1) % 3 == 0 or (i + 1) == len(picked):
            print(f"  在线策略 {i+1}/{len(picked)} | 累计 {len(all_pairs)} 个偏好对",
                  flush=True)
    return all_pairs


# --------------------------------------------------------------------------- #
# DPO 损失
# --------------------------------------------------------------------------- #
def _seq_logprob(model, tokenizer, prompt: str, answer: str,
                 device: str, max_len: int = 512,
                 system_prompt: str | None = None) -> torch.Tensor | None:
    """算「给定 prompt 生成 answer」的对数概率（答案区间求和）。"""
    msgs = []
    if system_prompt:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": prompt})
    try:
        head = tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        )
    except Exception:
        head = prompt

    h_ids = tokenizer(head, add_special_tokens=False).input_ids
    a_ids = tokenizer(answer, add_special_tokens=False).input_ids
    if not a_ids:
        return None

    ids = (h_ids + a_ids)[:max_len]
    a_start = len(h_ids)
    if a_start >= len(ids) - 1 or a_start < 1:
        return None

    t = torch.tensor([ids], dtype=torch.long, device=device)
    logits = model(input_ids=t).logits.float()
    lg = logits[0, a_start - 1:-1]
    lb = t[0, a_start:]
    if lg.shape[0] != lb.shape[0] or lg.shape[0] == 0:
        return None
    lp = torch.log_softmax(lg, dim=-1)
    return lp.gather(-1, lb.unsqueeze(-1)).squeeze(-1).sum()


def dpo_loss(
    policy, ref, tokenizer, pairs: list[PreferencePair],
    device: str, beta: float = 0.1, max_len: int = 512,
    system_prompt: str | None = None,
) -> tuple[torch.Tensor, dict]:
    """标准 DPO 损失。

        L = -log σ(β·[(logπ(c)-logπ(r)) - (logπ_ref(c)-logπ_ref(r))])

    ref 是**冻结**的参考模型（通常是本轮开始前的学生）。它的作用是
    防止 policy 为了拉高 chosen 的概率而整体偏移分布。

    注意 β 的取值：黑盒场景下的偏好对噪声比人类标注大（教师排序也会错），
    所以 β 要比 RLHF 常用值（0.1~0.5）偏小，让更新更保守。
    """
    if not pairs:
        return torch.zeros((), device=device), {"dpo_pairs": 0}

    losses = []
    margins = []
    for pr in pairs:
        lp_c = _seq_logprob(policy, tokenizer, pr.prompt, pr.chosen,
                            device, max_len, system_prompt)
        lp_r = _seq_logprob(policy, tokenizer, pr.prompt, pr.rejected,
                            device, max_len, system_prompt)
        if lp_c is None or lp_r is None:
            continue
        with torch.no_grad():
            rf_c = _seq_logprob(ref, tokenizer, pr.prompt, pr.chosen,
                                device, max_len, system_prompt)
            rf_r = _seq_logprob(ref, tokenizer, pr.prompt, pr.rejected,
                                device, max_len, system_prompt)
        if rf_c is None or rf_r is None:
            continue

        diff = (lp_c - lp_r) - (rf_c - rf_r)
        losses.append(-torch.nn.functional.logsigmoid(beta * diff))
        margins.append(float(diff.detach()))

    if not losses:
        return torch.zeros((), device=device), {"dpo_pairs": 0}

    loss = torch.stack(losses).mean()
    return loss, {
        "dpo_pairs": len(losses),
        "dpo_margin": sum(margins) / len(margins),
    }