"""ChorusKD 推理。

    python infer.py --prompt "用三句话解释什么是知识蒸馏"
    python infer.py --interactive
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from config import ChorusKDConfig                  # noqa: E402
from data import fix_keys                          # noqa: E402


def load(weight: str, device: str, cfg: ChorusKDConfig):
    PROJECT_ROOT = r"D:\workspace\xllm"
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from model.XLLMModel import XLLMConfig, XLLMForCausalLM
    from transformers import AutoTokenizer

    s = cfg.student
    model = XLLMForCausalLM(XLLMConfig(
        hidden_size=s.hidden_size,
        num_hidden_layers=s.num_hidden_layers,
        use_moe=s.use_moe,
        vocab_size=s.vocab_size,
    ))

    cands = [
        os.path.join(s.save_dir, f"{weight}_{s.hidden_size}.pth"),
        os.path.join(s.weight_dir, f"{weight}_{s.hidden_size}.pth"),
    ]
    path = next((p for p in cands if os.path.exists(p)), None)
    if path is None:
        raise FileNotFoundError(f"未找到权重 {weight}，已尝试:\n  " + "\n  ".join(cands))

    sd = torch.load(path, map_location="cpu", weights_only=True)
    sd, n = fix_keys(sd)
    if n:
        print(f"[权重] 修正 {n} 个键名")
    model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()
    print(f"[权重] {path}")

    tok = AutoTokenizer.from_pretrained(s.model_dir, trust_remote_code=True)
    return model, tok


@torch.no_grad()
def gen(model, tok, prompt: str, device: str, system: str | None = None,
        max_new: int = 256, temperature: float = 0.7,
        top_p: float = 0.9, repetition_penalty: float = 1.3) -> str:
    msgs = []
    if system:
        msgs.append({"role": "system", "content": system})
    msgs.append({"role": "user", "content": prompt})
    try:
        txt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        txt = prompt

    ids = tok(txt, return_tensors="pt", add_special_tokens=False).input_ids.to(device)
    out = model.generate(
        input_ids=ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=max_new,
        do_sample=temperature > 0,
        temperature=max(temperature, 1e-6),
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        pad_token_id=tok.pad_token_id,
        eos_token_id=tok.eos_token_id,
    )
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def main():
    ap = argparse.ArgumentParser(description="ChorusKD 推理")
    ap.add_argument("--weight", type=str, default="chorus_kd")
    ap.add_argument("--prompt", type=str, default=None)
    ap.add_argument("--interactive", action="store_true")
    ap.add_argument("--max_new", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition_penalty", type=float, default=1.3)
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    cfg = ChorusKDConfig()
    model, tok = load(args.weight, device, cfg)

    if args.interactive:
        print("\n输入 quit 退出\n")
        while True:
            try:
                q = input("你: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if q.lower() in ("quit", "exit", "q"):
                break
            if not q:
                continue
            print(f"模型: {gen(model, tok, q, device, cfg.prompt.system_prompt, args.max_new, args.temperature, repetition_penalty=args.repetition_penalty)}\n")
        return

    demos = args.prompt and [args.prompt] or [
        "用三句话解释什么是知识蒸馏。",
        "写一个 Python 函数，判断一个数是不是质数。",
        "帮我把这句话改简洁：我觉得这个事情可能大概应该是这样做的。",
    ]
    for q in demos:
        a = gen(model, tok, q, device, cfg.prompt.system_prompt,
                args.max_new, args.temperature,
                repetition_penalty=args.repetition_penalty)
        print("=" * 74)
        print(f"[问] {q}")
        print(f"[答] {a}")
    print("=" * 74)


if __name__ == "__main__":
    main()