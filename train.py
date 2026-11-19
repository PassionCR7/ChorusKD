"""ChorusKD 主流程。

多轮闭环，每轮七个阶段顺序执行：

    ┌──────────────────────────────────────────────────────────┐
    │ round r:                                                 │
    │   P 策展   读种子 → 学生打分 → 按难度档位选问题            │
    │   Q 采样   每题 n 次不同温度向黑盒教师提问                 │
    │   R 筛选   一致性投票 → 剔除幻觉/退化/太易太难             │
    │   T 组装   多正例 + 一致性加权 → 训练集                    │
    │   U 在线   学生自采样 → 教师排序 → DPO 偏好对              │
    │   W 训练   CE + DPO → 更新学生                            │
    │   S 调度   看本轮 loss → 调整下一轮难度档位                │
    └──────────────────────────────────────────────────────────┘

为什么必须多轮而不是一次问完
----------------------------
学生的能力在变。第 1 轮问"中等难度"的题，训完学生变强了，第 2 轮
还问同样难度的题就是浪费预算。多轮 + 调度让问题难度始终跟着能力走。

这也是 ChorusKD 与"一次性造数据再 SFT"的本质区别：后者是静态数据集，
前者是**闭环控制器**。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time

import torch
from torch.utils.data import DataLoader

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import ChorusKDConfig                          # noqa: E402
from data import (                                          # noqa: E402
    ChorusDataset,
    ce_loss,
    collate_chorus,
    evaluate_loss,
    load_student,
    per_sample_loss,
)
from ensemble import DomainRouter, should_drop_on_disagreement   # noqa: E402
from filter import filter_batch, format_stats              # noqa: E402
from onpolicy import collect_preferences, dpo_loss          # noqa: E402
from prompts import (                                       # noqa: E402
    Candidate,
    LossProbe,
    curate,
    expand_prompts,
    load_seeds,
)
from scheduler import DifficultyScheduler, bin_description  # noqa: E402
from teacher_api import Budget, BlackBoxTeacher, sample_many  # noqa: E402


# --------------------------------------------------------------------------- #
# 工具
# --------------------------------------------------------------------------- #
def _epoch_train(
    model, dataset: ChorusDataset, tok, cfg, device: str,
    opt, pairs: list, ref_model=None,
) -> dict:
    """跑一轮 SFT + DPO。"""
    if len(dataset) == 0:
        return {"ce": float("nan"), "dpo": 0.0, "steps": 0}

    loader = DataLoader(
        dataset, batch_size=cfg.train.batch_size, shuffle=True,
        collate_fn=lambda b: collate_chorus(b, tok.pad_token_id),
        num_workers=0, drop_last=False,
    )

    model.train()
    opt.zero_grad(set_to_none=True)
    ce_acc, nb, step = 0.0, 0, 0
    t0 = time.time()

    for it, batch in enumerate(loader):
        ids = batch["ids"].to(device)
        # ce_loss 内部要索引 ans_mask 与 w，必须一并搬到同设备，
        # 否则 cross_entropy 会因 target 在 cpu 而报 device mismatch。
        batch = {
            "ids": ids,
            "mask": batch["mask"].to(device),
            "ans_mask": batch["ans_mask"].to(device),
            "w": batch["w"].to(device),
        }
        logits = model(input_ids=ids).logits
        loss, ce_v = ce_loss(logits, batch, cfg.train.aux_positive_weight)

        (loss / cfg.train.grad_accum).backward()
        ce_acc += ce_v
        nb += 1

        if (it + 1) % cfg.train.grad_accum == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
        elif (it + 1) == len(loader):
            # 尾批不足 grad_accum 时必须补一次 step。
            # 样本量小时（如 3 条样本 + grad_accum=4）不加这个分支，
            # 整个 epoch 一次参数更新都不会发生，loss 完全不动——
            # 表面看不出错，实际训练是空转。
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1

        if (it + 1) % max(cfg.train.log_interval, 1) == 0 or (it + 1) == len(loader):
            print(f"    iter {it+1}/{len(loader)} | ce {ce_acc/nb:.4f} "
                  f"| {time.time()-t0:.0f}s", flush=True)
            ce_acc, nb = 0.0, 0

    # ---- DPO ----
    dpo_v = 0.0
    dpo_info = {"dpo_pairs": 0}
    if pairs and ref_model is not None and cfg.train.dpo_weight > 0:
        print(f"    DPO: {len(pairs)} 个偏好对", flush=True)
        dloss, dpo_info = dpo_loss(
            model, ref_model, tok, pairs, device,
            beta=cfg.train.dpo_beta, max_len=cfg.train.max_seq_len,
        )
        if dpo_info["dpo_pairs"] > 0:
            opt.zero_grad(set_to_none=True)
            (cfg.train.dpo_weight * dloss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            dpo_v = float(dloss.detach())
            print(f"    DPO loss {dpo_v:.4f} | margin {dpo_info['dpo_margin']:.3f} "
                  f"| pairs {dpo_info['dpo_pairs']}", flush=True)

    return {"ce": 0.0, "dpo": dpo_v, "steps": step, **dpo_info}


# --------------------------------------------------------------------------- #
# 主流程
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="ChorusKD 黑盒蒸馏（多样采样 + 一致性筛选 + 能力调度 + 在线策略）"
    )
    ap.add_argument("--num_rounds", type=int, default=None)
    ap.add_argument("--num_prompts", type=int, default=None,
                    help="每轮问多少个问题")
    ap.add_argument("--n_samples", type=int, default=None,
                    help="每题采样次数")
    ap.add_argument("--max_requests", type=int, default=None,
                    help="总请求预算上限")
    ap.add_argument("--strategy", type=str, default=None,
                    choices=["difficulty", "diverse", "random"])
    ap.add_argument("--batch_size", type=int, default=None)
    ap.add_argument("--learning_rate", type=float, default=None)
    ap.add_argument("--max_seq_len", type=int, default=None)
    ap.add_argument("--no_on_policy", action="store_true")
    ap.add_argument("--no_scheduler", action="store_true")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    cfg = ChorusKDConfig()
    if args.num_rounds is not None:
        cfg.train.num_rounds = args.num_rounds
    if args.num_prompts is not None:
        cfg.prompt.num_prompts = args.num_prompts
        cfg.train.samples_per_round = args.num_prompts
    if args.n_samples is not None:
        cfg.teacher.n_samples = args.n_samples
    if args.max_requests is not None:
        cfg.teacher.max_requests = args.max_requests
    if args.strategy is not None:
        cfg.prompt.strategy = args.strategy
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.learning_rate is not None:
        cfg.train.learning_rate = args.learning_rate
    if args.max_seq_len is not None:
        cfg.train.max_seq_len = args.max_seq_len
    if args.no_on_policy:
        cfg.onpolicy.enabled = False
    if args.no_scheduler:
        cfg.scheduler.enabled = False
    if args.seed is not None:
        cfg.train.seed = args.seed
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    rng = random.Random(cfg.train.seed)
    random.seed(cfg.train.seed)
    torch.manual_seed(cfg.train.seed)

    print("=" * 74)
    print("ChorusKD | 黑盒蒸馏：多样采样 + 一致性筛选 + 能力调度 + 在线策略")
    print(f"  轮次 {cfg.train.num_rounds} | 每轮 {cfg.prompt.num_prompts} 题 "
          f"| 每题 {cfg.teacher.n_samples} 采样 | 策略 {cfg.prompt.strategy}")
    print(f"  预算上限 {cfg.teacher.max_requests} 请求 | "
          f"在线策略 {'开' if cfg.onpolicy.enabled else '关'} | "
          f"调度 {'开' if cfg.scheduler.enabled else '关'}")
    print("=" * 74)

    # ---- 学生与 tokenizer ----
    student = load_student(cfg, device)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(
        cfg.student.model_dir, trust_remote_code=True
    )
    probe = LossProbe(student, tok, device=device)

    # ---- 教师与预算 ----
    budget = Budget(cfg.teacher.max_requests,
                    cfg.teacher.max_requests * cfg.teacher.max_tokens_per_request)
    teacher = BlackBoxTeacher(cfg.teacher, budget)
    router = DomainRouter(cfg, budget) if cfg.ensemble.enabled else None

    # ---- 种子池 ----
    seeds = load_seeds(cfg.prompt.seed_file, cfg.prompt.num_prompts * 8)
    if not seeds:
        raise RuntimeError(f"种子池为空: {cfg.prompt.seed_file}")
    print(f"[种子] {len(seeds)} 条问句")

    # ---- 调度器 ----
    sched = DifficultyScheduler(cfg.scheduler, cfg.prompt.num_difficulty_bins)
    print(f"[调度] 起始档位 "
          f"{bin_description(sched.current_bin, cfg.prompt.num_difficulty_bins)}")

    # ---- 优化器 ----
    opt = torch.optim.AdamW(
        student.parameters(), lr=cfg.train.learning_rate,
        weight_decay=cfg.train.weight_decay,
    )

    round_log = []
    total_t0 = time.time()

    for rnd in range(cfg.train.num_rounds):
        print("\n" + "=" * 74)
        print(f"Round {rnd+1}/{cfg.train.num_rounds}")
        print("=" * 74)

        # ================================================================ #
        # P 策展
        # ================================================================ #
        # 用学生给候选问题打分：把种子问题输入学生，算 prompt 部分的 loss。
        # 这是"学生觉得这个问题有多难"的粗略代理（没有参考答案时的降级方案）。
        n_pool = min(len(seeds), cfg.train.samples_per_round * 6)
        pool = seeds[:n_pool]
        print(f"[P 策展] 学生给 {len(pool)} 条候选打分...")
        losses = probe.score(pool)

        cands = curate(
            pool, cfg.prompt,
            student_losses=losses,
            target_bin=sched.current_bin if cfg.scheduler.enabled else None,
            rng=rng,
            num_bins=cfg.prompt.num_difficulty_bins,
        )
        cands = cands[: cfg.train.samples_per_round]
        pairs_q = expand_prompts(cands, cfg.prompt)
        print(f"[P 策展] 选中 {len(cands)} 题（档位 "
              f"{bin_description(sched.current_bin, cfg.prompt.num_difficulty_bins)}）"
              f"，展开 {len(pairs_q)} 个问句")

        # ================================================================ #
        # Q 采样
        # ================================================================ #
        print(f"[Q 采样] 每题 {cfg.teacher.n_samples} 次（温度铺开）...")
        prompts_only = [v for _, v in pairs_q]

        if router is not None:
            responses = []
            n_cross = max(1, int(len(prompts_only) * cfg.ensemble.cross_check_ratio))
            for i, p in enumerate(prompts_only):
                res = router.route(
                    p, cfg.prompt.system_prompt,
                    n=cfg.teacher.n_samples,
                    do_cross_check=(i < n_cross),
                )
                if res.response is not None:
                    if should_drop_on_disagreement(res, cfg.ensemble):
                        print(f"  [交叉验证] 丢弃不一致样本: {p[:30]}")
                        continue
                    responses.append(res.response)
                if (i + 1) % 5 == 0 or (i + 1) == len(prompts_only):
                    print(f"  采样 {i+1}/{len(prompts_only)} | "
                          f"预算 {budget.summary()}", flush=True)
        else:
            try:
                responses = sample_many(
                    teacher, prompts_only, cfg.prompt.system_prompt,
                    n=cfg.teacher.n_samples,
                )
            except Exception as e:
                print(f"[Q 采样] 中断: {e}")
                responses = []

        if not responses:
            print("[Q 采样] 无有效响应，本轮跳过")
            continue

        # ================================================================ #
        # R 筛选
        # ================================================================ #
        # 难度用"学生 loss"判定，但此刻还没有教师答案的成绩，
        # 所以先不带 student_loss 过筛，训练前再回填一次。
        print(f"[R 筛选] 一致性投票...")
        kept, stats = filter_batch(responses, cfg.filter)
        print(f"[R 筛选] {format_stats(stats)}")

        if not kept:
            print("[R 筛选] 无可训练样本，本轮跳过")
            continue

        # ================================================================ #
        # T 组装数据集
        # ================================================================ #
        ds = ChorusDataset(
            kept, tok, cfg.train.max_seq_len,
            system_prompt=cfg.prompt.system_prompt,
        )
        if len(ds) == 0:
            print("[T 组装] 编码后为空，本轮跳过")
            continue

        # 回填逐样本难度，把 too_easy / too_hard 剔掉
        sl = per_sample_loss(student, ds, tok.pad_token_id, device)
        keep_idx = [
            i for i, v in enumerate(sl)
            if v == v and cfg.filter.min_student_loss <= v <= cfg.filter.max_student_loss
        ]
        if len(keep_idx) < max(2, len(ds) * 0.2):
            print(f"[T 组装] 难度过滤后仅剩 {len(keep_idx)}/{len(ds)}，"
                  f"放宽阈值保留全部")
            keep_idx = list(range(len(ds)))
        ds.items = [ds.items[i] for i in keep_idx]
        sl = [sl[i] for i in keep_idx]
        mean_loss = sum(v for v in sl if v == v) / max(len([v for v in sl if v == v]), 1)
        print(f"[T 组装] {len(ds)} 条训练样本 | 平均学生 loss {mean_loss:.3f}")

        # ================================================================ #
        # U 在线策略
        # ================================================================ #
        pairs = []
        if cfg.onpolicy.enabled:
            print("[U 在线策略] 学生自采样 + 教师排序...")
            ref_model = None
            # 参考模型：冻结一份当前学生（DPO 需要）
            import copy
            ref_model = copy.deepcopy(student).eval()
            for p in ref_model.parameters():
                p.requires_grad_(False)

            pairs = collect_preferences(
                student, tok, teacher,
                [c.prompt for c in cands], cfg.onpolicy, device, rng,
                system_prompt=cfg.prompt.system_prompt,
            )
            print(f"[U 在线策略] {len(pairs)} 个偏好对")
        else:
            ref_model = None

        # ================================================================ #
        # W 训练
        # ================================================================ #
        print(f"[W 训练] {len(ds)} 条样本 × {cfg.train.epochs} epoch")
        tr = _epoch_train(
            student, ds, tok, cfg, device, opt, pairs, ref_model
        )
        ep_loss = evaluate_loss(student, ds, tok.pad_token_id, device)
        print(f"[W 训练] 完成后 train loss {ep_loss:.3f} | "
              f"dpo {tr.get('dpo', 0):.4f} | steps {tr.get('steps', 0)}")

        # ================================================================ #
        # S 调度
        # ================================================================ #
        new_bin, note = sched.observe(rnd, ep_loss)
        print(f"[S 调度] {note} → 档位 "
              f"{bin_description(new_bin, cfg.prompt.num_difficulty_bins)}")
        print(f"[S 调度] 趋势: {sched.trend()}")

        round_log.append({
            "round": rnd + 1,
            "bin": new_bin,
            "num_prompts": len(cands),
            "num_responses": len(responses),
            "num_kept": len(kept),
            "num_train": len(ds),
            "num_pairs": len(pairs),
            "mean_student_loss": mean_loss,
            "train_loss": ep_loss,
            "dpo_loss": tr.get("dpo", 0.0),
            "budget": budget.summary(),
        })

    # ---- 保存 ----
    os.makedirs(cfg.student.save_dir, exist_ok=True)
    save_path = os.path.join(
        cfg.student.save_dir,
        f"{cfg.student.save_weight}_{cfg.student.hidden_size}.pth",
    )
    state = {k: v.detach().cpu() for k, v in student.state_dict().items()}
    torch.save(state, save_path)
    print(f"\n[保存] {save_path}")

    with open(os.path.join(cfg.student.save_dir, "round_log.json"), "w",
              encoding="utf-8") as f:
        json.dump({
            "rounds": round_log,
            "scheduler": sched.to_dict(),
            "budget": budget.summary(),
            "teacher_backend": teacher.backend,
        }, f, ensure_ascii=False, indent=2)
    print(f"[保存] round_log.json")
    print(f"[总耗时] {(time.time()-total_t0)/60:.1f} 分钟 | 预算 {budget.summary()}")


if __name__ == "__main__":
    main()