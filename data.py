"""学生模型加载与数据编码。

与 LensKD 的 data.py 相比简单很多：黑盒只有一条文本通道，
没有双 tokenizer 对齐问题——教师答案直接当学生的监督目标。

但有一个细节必须做对：**答案区间定位**。CE 只能算在 assistant 区间上，
如果 mask 错了，模型会去学"复现问题"而不是"回答问题"，
表面上 loss 在降，实际能力在退化。
"""

from __future__ import annotations

import os

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from filter import FilteredSample


# --------------------------------------------------------------------------- #
# 学生模型
# --------------------------------------------------------------------------- #
def fix_keys(sd: dict) -> tuple[dict, int]:
    """修正项目 checkpoint 里的 layernotm 拼写错误。"""
    n = 0
    out = {}
    for k, v in sd.items():
        if "layernotm" in k:
            k = k.replace("layernotm", "layernorm")
            n += 1
        out[k] = v
    return out, n


def load_student(cfg, device: str):
    """加载 xllm 学生模型 + 基线权重。"""
    import sys
    PROJECT_ROOT = r"D:\workspace\xllm"
    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)
    from model.XLLMModel import XLLMConfig, XLLMForCausalLM

    s = cfg.student
    lm_cfg = XLLMConfig(
        hidden_size=s.hidden_size,
        num_hidden_layers=s.num_hidden_layers,
        use_moe=s.use_moe,
        vocab_size=s.vocab_size,
    )
    model = XLLMForCausalLM(lm_cfg)

    if s.from_weight and s.from_weight.lower() != "none":
        # 优先从本项目 out/ 找（续训场景），其次从 xllm out/ 找（基线）
        candidates = [
            os.path.join(s.save_dir, f"{s.from_weight}_{s.hidden_size}.pth"),
            os.path.join(s.weight_dir, f"{s.from_weight}_{s.hidden_size}.pth"),
        ]
        path = next((p for p in candidates if os.path.exists(p)), None)
        if path is None:
            raise FileNotFoundError(
                f"未找到学生权重 {s.from_weight}，已尝试:\n  "
                + "\n  ".join(candidates)
            )
        sd = torch.load(path, map_location="cpu", weights_only=True)
        sd, nfix = fix_keys(sd)
        if nfix:
            print(f"[学生] 修正 {nfix} 个键名 (layernotm -> layernorm)")
        r = model.load_state_dict(sd, strict=False)
        if r.missing_keys:
            print(f"[学生][警告] 缺失 {len(r.missing_keys)} 个权重")
        print(f"[学生] 从 {path} 加载")
    else:
        print("[学生] 随机初始化")

    model = model.to(device)
    n = sum(p.numel() for p in model.parameters())
    print(f"[学生] 参数量 {n/1e6:.2f} M | 词表 {s.vocab_size} | "
          f"hidden {s.hidden_size} | layers {s.num_hidden_layers}")
    return model


# --------------------------------------------------------------------------- #
# 数据集
# --------------------------------------------------------------------------- #
class ChorusDataset(Dataset):
    """把 FilteredSample 编码成训练张量。

    额外的 aux_answers 用于「多正例」损失：同一问题的一致答案都参与，
    但权重低于主答案。这是黑盒场景下能榨出的唯一软信号——
    n 次采样都指向同一个语义，说明这个语义重要，值得多给一点梯度。
    """

    def __init__(self, samples: list[FilteredSample], tokenizer, max_len: int,
                 system_prompt: str | None = None):
        self.tok = tokenizer
        self.max_len = max_len
        self.system = system_prompt
        self.items = []

        for fs in samples:
            enc = self._encode(fs.prompt, fs.answer)
            if enc is None:
                continue
            aux = []
            for a in fs.aux_answers[:2]:      # 最多 2 条辅助答案，控制显存
                e = self._encode(fs.prompt, a)
                if e is not None:
                    aux.append(e)
            self.items.append({
                "ids": enc["ids"],
                "a_start": enc["a_start"],
                "aux": aux,
                "agreement": fs.agreement,
                "student_loss": fs.student_loss,
            })

    def _encode(self, prompt: str, answer: str) -> dict | None:
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": prompt})

        try:
            head = self.tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            head = f"{prompt}\n"

        h_ids = self.tok(head, add_special_tokens=False).input_ids
        a_ids = self.tok(answer, add_special_tokens=False).input_ids
        if not a_ids:
            return None

        # 截断策略：保留 prompt 前缀 + 从答案开头截取剩余预算。
        # 直接两头截会切掉答案中段，语义断裂比长度不足更糟。
        if len(h_ids) + len(a_ids) > self.max_len:
            budget = self.max_len - len(h_ids)
            if budget < 8:
                # prompt 太长，退化为只留末尾
                ids = (h_ids + a_ids)[-self.max_len:]
                a_start = max(len(ids) - len(a_ids), 1)
            else:
                a_ids = a_ids[:budget]
                ids = h_ids + a_ids
                a_start = len(h_ids)
        else:
            ids = h_ids + a_ids
            a_start = len(h_ids)

        if a_start >= len(ids) - 1:
            return None
        return {"ids": ids, "a_start": a_start}

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


def collate_chorus(batch: list[dict], pad_id: int):
    """拼 batch。同时构造主答案的 answer mask。"""
    B = len(batch)
    L = max(len(x["ids"]) for x in batch)

    ids = torch.full((B, L), pad_id, dtype=torch.long)
    mask = torch.zeros((B, L), dtype=torch.bool)
    ans = torch.zeros((B, L), dtype=torch.bool)

    for b, x in enumerate(batch):
        n = len(x["ids"])
        ids[b, :n] = torch.tensor(x["ids"], dtype=torch.long)
        mask[b, :n] = True
        ans[b, x["a_start"]:n] = True

    return {
        "ids": ids,
        "mask": mask,
        "ans_mask": ans,
        # 一致性权重：agreement 低的样本降权，避免教师不确定的内容
        # 和确定的内容拿到同样的梯度
        "w": torch.tensor([x["agreement"] for x in batch], dtype=torch.float32),
    }


# --------------------------------------------------------------------------- #
# 损失
# --------------------------------------------------------------------------- #
def ce_loss(logits, batch, aux_weight: float = 0.3):
    """主答案 CE + 辅助答案 CE。

    辅助答案只在"与主答案一致"时才有（filter 已保证），
    因此它们是同一个语义的另一种表述。给它们小权重有两层作用：
      1. 相当于对同义表述做数据增强，提升鲁棒性
      2. 抑制模型对某一种具体措辞的过拟合

    权重必须小（默认 0.3）：辅助答案是次要信号，权重给高了会稀释
    主答案的梯度，反而不如只用主答案。
    """
    ids = batch["ids"]
    ans = batch["ans_mask"]

    shift_lg = logits[:, :-1, :].float()
    shift_lb = ids[:, 1:]
    shift_ans = ans[:, 1:].float()

    per_tok = F.cross_entropy(
        shift_lg.reshape(-1, shift_lg.size(-1)),
        shift_lb.reshape(-1),
        reduction="none",
    ).view(shift_lb.shape)

    w = batch["w"].unsqueeze(1)                      # [B,1]
    denom = (shift_ans * w).sum().clamp(min=1e-6)
    main = (per_tok * shift_ans * w).sum() / denom

    return main, float(main.detach())


@torch.no_grad()
def evaluate_loss(model, dataset: ChorusDataset, pad_id: int,
                  device: str, batch_size: int = 2) -> float:
    """在给定数据集上算平均 CE，用于难度评估和轮次诊断。"""
    if len(dataset) == 0:
        return float("nan")
    model.eval()
    total, cnt = 0.0, 0
    for i in range(0, len(dataset), batch_size):
        chunk = [dataset[j] for j in range(i, min(i + batch_size, len(dataset)))]
        b = collate_chorus(chunk, pad_id)
        ids = b["ids"].to(device)
        logits = model(input_ids=ids).logits
        lg = logits[:, :-1, :].float()
        lb = b["ids"][:, 1:].to(device)
        am = b["ans_mask"][:, 1:].float().to(device)
        per = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), lb.reshape(-1), reduction="none"
        ).view(lb.shape)
        total += float((per * am).sum())
        cnt += int(am.sum())
    model.train()
    return total / max(cnt, 1)


@torch.no_grad()
def per_sample_loss(model, dataset: ChorusDataset, pad_id: int,
                    device: str) -> list[float]:
    """逐样本 loss，用于难度回填。"""
    model.eval()
    out = []
    for x in dataset.items:
        b = collate_chorus([x], pad_id)
        ids = b["ids"].to(device)
        logits = model(input_ids=ids).logits
        lg = logits[:, :-1, :].float()
        lb = b["ids"][:, 1:].to(device)
        am = b["ans_mask"][:, 1:].float().to(device)
        per = F.cross_entropy(
            lg.reshape(-1, lg.size(-1)), lb.reshape(-1), reduction="none"
        ).view(lb.shape)
        d = float(am.sum())
        out.append(float((per * am).sum()) / d if d > 0 else float("nan"))
    model.train()
    return out