"""实验二：验证「自一致性投票的一致率 vs 答案正确率」相关性（README 声称 0.6~0.75）。

结论待验证
----------
ChorusKD 的 filter.py 与 README 声称：
  「实测这类『一致率』和教师答案正确率的相关性在 0.6~0.75 之间，
    是目前黑盒场景下最好的免标注质量代理。」

这个话题有个容易搞错的地方
--------------------------
黑盒 API 拿不到 logits，但**本实验需要 ground truth 才能算正确率**。
所以这是一个"需要在有标注集上做、但方法本身不需要标注"的实验：
  * 投票过程只用采样文本（符合黑盒设定）
  * 评估阶段才用 ground truth（这是实验需要的，不是方法需要的）

实验设计
--------
1. 题库：一批有唯一确定答案的事实/计算题，每题带 gold answer。
2. 教师采样：同一题在温度梯度 [0.3, 0.6, 0.9, 1.2, 1.3] 各采 n 次。
3. 投票：用 filter.py 的 cluster_samples / pick_majority 算 agreement。
4. 正确性判定：
   * exact  — 归一化后完全相等
   * f1     — 字符级 F1 >= 0.75（与聚类阈值一致）
   * 人工语义 — 用规则判包含关系，作为宽松口径
5. 统计：agreement 与 correct 的三类相关系数
   * Pearson（线性）
   * Spearman（秩，更稳健）
   * Point-biserial（二分 vs 连续，等价于 Pearson 的特例，但要报）
6. 分组：按 agreement 分桶，看每桶的准确率（这是实际用阈值时的依据）。

用法
----
    python experiments/exp2_consistency_correlation.py
    python experiments/exp2_consistency_correlation.py --n-samples 5 --device cuda

输出
----
    experiments/results/exp2_consistency_correlation.json
    experiments/results/exp2_raw_samples.jsonl   原始采样，便于复核
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from filter import cluster_samples, pick_majority, _normalize, _token_f1   # noqa: E402


# --------------------------------------------------------------------------- #
# 题库：唯一确定答案，便于客观判定对错
# --------------------------------------------------------------------------- #
BANK = [
    # —— 算术 ——
    {"q": "23 乘以 17 等于多少？只回答数字。", "gold": "391"},
    {"q": "144 除以 12 等于多少？只回答数字。", "gold": "12"},
    {"q": "2 的 10 次方是多少？只回答数字。", "gold": "1024"},
    {"q": "1000 减去 357 等于多少？只回答数字。", "gold": "643"},
    {"q": "7 的阶乘是多少？只回答数字。", "gold": "5040"},
    {"q": "一个正方形边长 13，面积是多少？只回答数字。", "gold": "169"},
    {"q": "45 加 67 再加 88 等于多少？只回答数字。", "gold": "200"},
    {"q": "256 的平方根是多少？只回答数字。", "gold": "16"},

    # —— 事实 ——
    {"q": "水的化学式是什么？只回答化学式。", "gold": "H2O"},
    {"q": "光速在真空中的近似值是多少 km/s？只回答数字。", "gold": "300000"},
    {"q": "地球上最大的海洋是哪个？只回答名称。", "gold": "太平洋"},
    {"q": "《红楼梦》的作者是谁？只回答人名。", "gold": "曹雪芹"},
    {"q": "中国的首都是哪座城市？只回答城市名。", "gold": "北京"},
    {"q": "一年有多少个月？只回答数字。", "gold": "12"},
    {"q": "DNA 的全称是什么？只回答英文缩写。", "gold": "deoxyribonucleic acid"},
    {"q": "太阳系中距离太阳最近的行星是哪个？只回答行星名。", "gold": "水星"},

    # —— 单位/换算 ——
    {"q": "1 公里等于多少米？只回答数字。", "gold": "1000"},
    {"q": "1 小时等于多少秒？只回答数字。", "gold": "3600"},
    {"q": "1 GB 等于多少 MB？只回答数字。", "gold": "1024"},
    {"q": "水的沸点在标准大气压下是多少摄氏度？只回答数字。", "gold": "100"},

    # —— 编程常识 ——
    {"q": "Python 中用于定义函数的关键字是什么？只回答关键字。", "gold": "def"},
    {"q": "HTML 中用于创建超链接的标签是什么？只回答标签名。", "gold": "a"},
    {"q": "SQL 中用于查询数据的关键字是什么？只回答关键字。", "gold": "select"},
    {"q": "Git 中用于提交改动的命令是什么？只回答子命令。", "gold": "commit"},
]

TEMPS = [0.3, 0.6, 0.9, 1.2, 1.3]


# --------------------------------------------------------------------------- #
# 教师：本地 Qwen3-0.6B（black-box 方式调用，只取文本）
# --------------------------------------------------------------------------- #
def build_teacher(model_dir: str, device: str, dtype: str):
    """加载本地教师。刻意只暴露 generate 的文本输出，不返回 logits——与 mock 模式一致。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                   "float32": torch.float32}[dtype]
    tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_dir, torch_dtype=torch_dtype, trust_remote_code=True
    ).to(device).eval()
    return tok, model


@torch.no_grad()
def sample_once(tok, model, prompt: str, temperature: float, device: str,
                max_new: int = 96) -> str:
    """黑盒式采样：只返回文本，绝不返回 logits。

    关键：Qwen3 是思考型模型。开着 thinking 时，64 token 预算会被 400+ token
    的推理块吃光，最终答案根本不会出现——这会让"正确率"恒为 0，实验直接失效。
    这里用 enable_thinking=False 让教师直接作答，既是黑盒调用方的真实用法，
    也让一致性投票比较的是**答案**而非推理过程。
    """
    msgs = [{"role": "user", "content": prompt}]
    text = tok.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    ids = tok(text, return_tensors="pt").input_ids.to(device)
    out = model.generate(
        ids,
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-5),
        top_p=0.95,
        pad_token_id=tok.pad_token_id or tok.eos_token_id,
    )
    gen = out[0][ids.shape[1]:]
    return strip_thinking(tok.decode(gen, skip_special_tokens=True).strip())


def strip_thinking(text: str) -> str:
    """剥掉  thinking... 与残留的孤立标记，只留最终答案。"""
    t = re.sub(r"</?think>", "", text, flags=re.I)
    # 若思考块未闭合（被 max_new_tokens 截断），取最后一个问号/换行后的内容
    if "think" in text.lower() and len(t.strip()) == 0:
        return text.strip()
    return t.strip()


# --------------------------------------------------------------------------- #
# 正确性判定（三种口径）
# --------------------------------------------------------------------------- #
def _numbers(s: str) -> list[str]:
    return re.findall(r"-?\d+(?:\.\d+)?", s.replace(",", ""))


def is_correct_exact(ans: str, gold: str) -> bool:
    """严格口径：答案归一化后等于 gold，或**以 gold 结尾**。

    为什么要允许"以 gold 结尾"：短答案题的自然回答形态是带算式的
    （"23 × 17 = 391"），若只允许完全相等，会把格式差异误判为答错，
    导致正确率虚低、相关性失真。这不是放宽标准，而是让口径匹配题型。
    """
    a, g = _normalize(ans), _normalize(gold)
    if not a or not g:
        return False
    return a == g or a.endswith(g)


def is_correct_f1(ans: str, gold: str, thr: float = 0.75) -> bool:
    """F1 口径：仅在答案中**同时**保留 gold 且长度不过度膨胀时算对。

    直接拿整句算式算字符 F1 会被算式里的数字污染（"1000 减去 357 等于 643"
    与 gold="643" 的 F1 只有 0.2 左右）。所以先抽取候选答案段再比对。
    """
    a, g = _normalize(ans), _normalize(gold)
    if not a or not g:
        return False
    if a.endswith(g) or g in a:
        return True
    return _token_f1(ans, gold) >= thr


def is_correct_loose(ans: str, gold: str) -> bool:
    """宽松口径：gold 作为子串出现，或数字型答案中 gold 出现在答案的数字列表里。"""
    a, g = _normalize(ans), _normalize(gold)
    if not a or not g:
        return False
    if g in a or a in g:
        return True
    gn, an = _numbers(gold), _numbers(ans)
    if gn and an:
        # 数字题：答案里出现 gold 数字即算对（避免"391"被写成"答案是391。"而误判）
        return any(abs(float(x) - float(y)) < 1e-6 for x in an for y in gn)
    return False


# --------------------------------------------------------------------------- #
# 相关系数
# --------------------------------------------------------------------------- #
def pearson(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 3:
        return float("nan")
    mx, my = sum(x) / n, sum(y) / n
    num = sum((a - mx) * (b - my) for a, b in zip(x, y))
    dx = math.sqrt(sum((a - mx) ** 2 for a in x))
    dy = math.sqrt(sum((b - my) ** 2 for b in y))
    if dx < 1e-12 or dy < 1e-12:
        return float("nan")
    return num / (dx * dy)


def rankdata(v: list[float]) -> list[float]:
    """平均秩（处理并列）。"""
    order = sorted(range(len(v)), key=lambda i: v[i])
    ranks = [0.0] * len(v)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[order[k]] = avg
        i = j + 1
    return ranks


def spearman(x: list[float], y: list[float]) -> float:
    return pearson(rankdata(x), rankdata(y))


def bootstrap_ci(x, y, fn, n_boot=2000, alpha=0.05, seed=0):
    """自助法置信区间。样本量小时必须有，否则单点相关系数没有说服力。"""
    rng = random.Random(seed)
    n = len(x)
    if n < 5:
        return (float("nan"), float("nan"))
    vals = []
    for _ in range(n_boot):
        idx = [rng.randrange(n) for _ in range(n)]
        v = fn([x[i] for i in idx], [y[i] for i in idx])
        if not math.isnan(v):
            vals.append(v)
    if not vals:
        return (float("nan"), float("nan"))
    vals.sort()
    lo = vals[int(len(vals) * alpha / 2)]
    hi = vals[int(len(vals) * (1 - alpha / 2))]
    return (lo, hi)


# --------------------------------------------------------------------------- #
# 主实验
# --------------------------------------------------------------------------- #
def run(args) -> dict:
    t0 = time.time()
    print("=" * 78)
    print("实验二：一致率 vs 答案正确率 相关性验证")
    print("=" * 78)

    from config import TeacherAPICfg
    tcfg = TeacherAPICfg()
    model_dir = os.environ.get("CHORUS_MOCK_MODEL", tcfg.mock_model_path)

    print(f"\n[教师] {model_dir}  (device={args.device}, dtype={args.dtype})")
    tok, model = build_teacher(model_dir, args.device, args.dtype)

    n_samples = args.n_samples
    bank = BANK[: args.limit] if args.limit else BANK
    print(f"[题库] {len(bank)} 题 × {len(TEMPS)} 温度 × 1 次 = "
          f"{len(bank) * len(TEMPS)} 次调用（每题共 {n_samples} 个样本）")
    print(f"[温度] {TEMPS}")

    results = []
    raw = []
    for qi, item in enumerate(bank):
        q, gold = item["q"], item["gold"]
        texts = []
        for t in TEMPS:
            for rep in range(args.reps):
                txt = sample_once(tok, model, q, t, args.device)
                texts.append({"text": txt, "temperature": t})
                raw.append({"q": q, "gold": gold, "temperature": t,
                            "rep": rep, "text": txt})

        # 用 filter.py 的真实聚类逻辑
        class _S:
            __slots__ = ("text", "temperature", "tokens_in", "tokens_out")
            def __init__(self, d):
                self.text = d["text"]; self.temperature = d["temperature"]
                self.tokens_in = 0; self.tokens_out = 0

        samples = [_S(d) for d in texts]
        clusters = cluster_samples(samples, mode="fuzzy", threshold=0.75)
        best, agreement = pick_majority(clusters)
        answer = min(best, key=lambda s: len(s.text)).text if best else ""

        row = {
            "q": q, "gold": gold, "answer": answer,
            "agreement": round(agreement, 6),
            "n_clusters": len(clusters),
            "n_samples": len(samples),
            "correct_exact": is_correct_exact(answer, gold),
            "correct_f1": is_correct_f1(answer, gold),
            "correct_loose": is_correct_loose(answer, gold),
        }
        results.append(row)
        flags = "".join("1" if row[k] else "0" for k in
                        ("correct_exact", "correct_f1", "correct_loose"))
        print(f"  [{qi+1:>2}/{len(bank)}] agr={agreement:.2f} clusters={len(clusters)} "
              f"correct(exact/f1/loose)={flags}  Q={q[:24]}...")

    # ---- 统计 ----
    print("\n" + "-" * 78)
    print("相关性分析")
    print("-" * 78)

    agr = [r["agreement"] for r in results]
    stats = {}
    for key in ("correct_exact", "correct_f1", "correct_loose"):
        y = [1.0 if r[key] else 0.0 for r in results]
        acc = sum(y) / len(y)
        pr = pearson(agr, y)
        sp = spearman(agr, y)
        ci = bootstrap_ci(agr, y, pearson)
        stats[key] = {
            "accuracy": acc,
            "n_correct": int(sum(y)),
            "n_total": len(y),
            "pearson": pr,
            "spearman": sp,
            "pearson_ci95": list(ci),
        }
        print(f"  {key:<16} acc={acc:.3f} ({int(sum(y))}/{len(y)})  "
              f"Pearson={pr:+.4f}  Spearman={sp:+.4f}  "
              f"CI95=[{ci[0]:+.3f}, {ci[1]:+.3f}]")

    # ---- 分桶 ----
    print("\n" + "-" * 78)
    print("按 agreement 分桶的准确率（实际用阈值时的依据）")
    print("-" * 78)
    buckets = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    bucket_stats = []
    print(f"  {'区间':<12} {'n':>4} {'acc(exact)':>11} {'acc(f1)':>9} {'acc(loose)':>11}")
    for lo, hi in buckets:
        sel = [r for r in results if lo <= r["agreement"] < hi]
        if not sel:
            continue
        n = len(sel)
        ae = sum(r["correct_exact"] for r in sel) / n
        af = sum(r["correct_f1"] for r in sel) / n
        al = sum(r["correct_loose"] for r in sel) / n
        bucket_stats.append({"range": [lo, hi], "n": n,
                             "acc_exact": ae, "acc_f1": af, "acc_loose": al})
        print(f"  [{lo:.1f},{hi:.1f})   {n:>4} {ae:>11.3f} {af:>9.3f} {al:>11.3f}")

    # ---- 结论判定 ----
    print("\n" + "-" * 78)
    print("结论判定（README 声称 0.6 ~ 0.75）")
    print("-" * 78)
    verdict = {}
    for key in ("correct_exact", "correct_f1", "correct_loose"):
        r = stats[key]["spearman"]
        in_range = (not math.isnan(r)) and (0.6 <= abs(r) <= 0.75)
        verdict[key] = {
            "spearman": r,
            "in_claimed_range": in_range,
            "note": "落在声称区间" if in_range else (
                "低于声称区间" if (not math.isnan(r) and abs(r) < 0.6) else
                "高于声称区间" if not math.isnan(r) else "样本不足无法判定"),
        }
        print(f"  {key:<16} Spearman={r:+.4f}  {verdict[key]['note']}")

    elapsed = time.time() - t0
    print("\n" + "=" * 78)
    print(f"完成，用时 {elapsed:.1f}s")
    print("=" * 78)

    return {
        "experiment": "exp2_consistency_correlation",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "elapsed_sec": round(elapsed, 2),
        "device": args.device,
        "teacher": model_dir,
        "setup": {"n_prompts": len(bank), "temperatures": TEMPS,
                  "reps_per_temp": args.reps, "n_samples_per_prompt": n_samples,
                  "cluster_mode": "fuzzy", "cluster_threshold": 0.75},
        "per_item": results,
        "stats": stats,
        "buckets": bucket_stats,
        "verdict_vs_readme": verdict,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--n-samples", type=int, default=5, help="每题总样本数（温度数×reps）")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="只用前 N 题，0 表示全部")
    ap.add_argument("--out", default="experiments/results")
    args = ap.parse_args()

    res = run(args)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "exp2_consistency_correlation.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n结果已写入 {out_dir/'exp2_consistency_correlation.json'}")


if __name__ == "__main__":
    main()