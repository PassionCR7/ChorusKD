"""Q 阶段：教师多样采样。

黑盒蒸馏的信息上限由这里决定。核心认识：

**一次采样 = 一个样本；n 次采样 = 一个分布的经验估计。**

白盒能直接拿到 p(y|x) 这个完整分布，黑盒只能从里面抽 n 个点。
用 n 个点的**频率**去估分布，是黑盒能做的极限。所以：
  * n 越大估计越准，但成本线性增长
  * 温度必须铺开，否则 n 次采样都在同一模式附近，估不出多峰性

温度设计（本模块最重要的决策）
------------------------------
低温度（0.3）：教师最确信的模式。这是"标准答案"，但只看到一个峰。
高温度（1.3）：教师认为"也说得通"的其他模式。这是软标签信息的替代物。

只用低温度 → 训练出死板的学生，只会一种说法。
只用高温度 → 采样噪声大，一致性投票失效。
铺开温度 → 既拿到主峰，也能发现次峰（如果一致投票后仍存在多个簇，
说明这个位置教师确实多峰，这条样本该降权或丢弃）。

Mock 模式
---------
没有 API key 时，用本地 Qwen3 模拟"只能拿到文本"的受限通道：
加载模型但**只暴露 generate 的文本输出**，不返回 logits。
这样代码路径和真实黑盒完全一致，不会出现"本地能跑、线上挂掉"的情况。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Iterable

import torch


# --------------------------------------------------------------------------- #
# 数据结构
# --------------------------------------------------------------------------- #
@dataclass
class Sample:
    """一次教师采样的结果。"""

    text: str
    temperature: float
    tokens_in: int = 0
    tokens_out: int = 0
    latency: float = 0.0
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and bool(self.text.strip())


@dataclass
class TeacherResponse:
    """一个问题的全部 n 次采样。"""

    prompt: str
    samples: list[Sample] = field(default_factory=list)
    # 成本累计（token 数），用于预算控制
    total_tokens: int = 0

    @property
    def ok_samples(self) -> list[Sample]:
        return [s for s in self.samples if s.ok]


# --------------------------------------------------------------------------- #
# 预算守卫
# --------------------------------------------------------------------------- #
class BudgetExceeded(RuntimeError):
    pass


class Budget:
    """硬性请求/token 上限。

    黑盒按量计费，必须有熔断。设计成异常而非返回值，是为了让调用方
    无法"忘了检查"——超预算必须显式捕获。
    """

    def __init__(self, max_requests: int, max_tokens: int):
        self.max_requests = max_requests
        self.max_tokens = max_tokens
        self.requests = 0
        self.tokens = 0

    def charge(self, tokens: int):
        self.requests += 1
        self.tokens += tokens
        if self.requests > self.max_requests:
            raise BudgetExceeded(
                f"请求数超限: {self.requests}/{self.max_requests}"
            )
        if self.tokens > self.max_tokens:
            raise BudgetExceeded(
                f"token 超限: {self.tokens}/{self.max_tokens}"
            )

    def summary(self) -> str:
        pct_r = self.requests / max(self.max_requests, 1) * 100
        pct_t = self.tokens / max(self.max_tokens, 1) * 100
        return (f"请求 {self.requests}/{self.max_requests} ({pct_r:.1f}%) | "
                f"token {self.tokens}/{self.max_tokens} ({pct_t:.1f}%)")


# --------------------------------------------------------------------------- #
# 教师客户端
# --------------------------------------------------------------------------- #
class BlackBoxTeacher:
    """统一的黑盒教师接口。

    两种后端：
      * api  ：OpenAI 兼容接口（真实黑盒）
      * mock ：本地模型，但只暴露文本（模拟黑盒的受限通道）
    """

    def __init__(self, cfg, budget: Budget):
        self.cfg = cfg
        self.budget = budget
        self.backend = self._pick_backend()
        self._mock_model = None
        self._mock_tok = None
        if self.backend == "mock":
            self._load_mock()

    def _pick_backend(self) -> str:
        if self.cfg.api_key and self.cfg.api_key.strip():
            return "api"
        return "mock"

    def _load_mock(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print(f"[教师] mock 模式（无 API key）| 本地模型 {self.cfg.mock_model_path}")
        self._mock_tok = AutoTokenizer.from_pretrained(
            self.cfg.mock_model_path, trust_remote_code=True
        )
        self._mock_model = AutoModelForCausalLM.from_pretrained(
            self.cfg.mock_model_path,
            dtype=torch.bfloat16,
            trust_remote_code=True,
        ).to("cuda" if torch.cuda.is_available() else "cpu").eval()

    # ------------------------------------------------------------------ #
    # 采样入口
    # ------------------------------------------------------------------ #
    def sample(
        self,
        prompt: str,
        system: str | None = None,
        n: int | None = None,
        temperatures: Iterable[float] | None = None,
    ) -> TeacherResponse:
        """对一个问题做 n 次不同温度的采样。

        温度铺开是核心：n 次同温采样 ≈ 重复问同一个问题，信息冗余。
        不同温度才能覆盖教师分布的多个模式。
        """
        n = n or self.cfg.n_samples
        temps = list(temperatures or self.cfg.temperatures)
        # 温度列表长度与 n 不匹配时循环取
        if len(temps) < n:
            temps = [temps[i % len(temps)] for i in range(n)]
        else:
            temps = temps[:n]

        resp = TeacherResponse(prompt=prompt)
        for T in temps:
            s = self._one_call(prompt, system, T)
            resp.samples.append(s)
            resp.total_tokens += s.tokens_in + s.tokens_out
        return resp

    def _one_call(self, prompt: str, system: str | None, temperature: float) -> Sample:
        t0 = time.time()
        last_err = None
        for attempt in range(self.cfg.max_retries):
            try:
                if self.backend == "api":
                    return self._call_api(prompt, system, temperature, t0)
                return self._call_mock(prompt, system, temperature, t0)
            except BudgetExceeded:
                raise
            except Exception as e:
                last_err = e
                # 指数退避：API 限流时硬重试会加重限流
                time.sleep(min(2 ** attempt, 8))
        return Sample(
            text="", temperature=temperature,
            latency=time.time() - t0,
            error=str(last_err) if last_err else "unknown",
        )

    # ------------------------------------------------------------------ #
    # 后端 1：真实 API
    # ------------------------------------------------------------------ #
    def _call_api(self, prompt, system, temperature, t0) -> Sample:
        import urllib.request

        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        payload = json.dumps({
            "model": self.cfg.model,
            "messages": msgs,
            "temperature": temperature,
            "top_p": self.cfg.top_p,
            "max_tokens": self.cfg.max_tokens,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.cfg.base_url.rstrip('/')}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.cfg.api_key}",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self.cfg.request_timeout) as r:
            obj = json.loads(r.read().decode("utf-8"))

        text = obj["choices"][0]["message"]["content"]
        usage = obj.get("usage", {})
        tin = int(usage.get("prompt_tokens", 0))
        tout = int(usage.get("completion_tokens", 0))
        self.budget.charge(tin + tout)

        return Sample(
            text=text, temperature=temperature,
            tokens_in=tin, tokens_out=tout,
            latency=time.time() - t0,
        )

    # ------------------------------------------------------------------ #
    # 后端 2：mock（本地模型，只暴露文本）
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def _call_mock(self, prompt, system, temperature, t0) -> Sample:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": prompt})

        # 关键：这里只取 decode 后的文本，绝不返回 logits。
        # 这不是"限制自己"，而是**强制代码路径与真实黑盒一致**。
        # 任何依赖 logits 的逻辑在 mock 下都会暴露出来，不会等到线上才发现。
        try:
            txt = self._mock_tok.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
        except Exception:
            txt = f"{prompt}\n"

        ids = self._mock_tok(txt, return_tensors="pt", add_special_tokens=False).input_ids
        device = next(self._mock_model.parameters()).device
        ids = ids.to(device)

        out = self._mock_model.generate(
            input_ids=ids,
            attention_mask=torch.ones_like(ids),
            max_new_tokens=self.cfg.max_tokens,
            do_sample=True,
            temperature=max(temperature, 1e-6),
            top_p=self.cfg.top_p,
            pad_token_id=self._mock_tok.pad_token_id,
            eos_token_id=self._mock_tok.eos_token_id,
        )
        gen = out[0, ids.shape[1]:]
        text = self._mock_tok.decode(gen, skip_special_tokens=True)

        tin = int(ids.numel())
        tout = int(gen.numel())
        self.budget.charge(tin + tout)

        return Sample(
            text=text.strip(), temperature=temperature,
            tokens_in=tin, tokens_out=tout,
            latency=time.time() - t0,
        )

    # ------------------------------------------------------------------ #
    # 排序（在线策略用）
    # ------------------------------------------------------------------ #
    def rank(self, prompt: str, candidates: list[str], system: str | None = None) -> list[int]:
        """让教师给候选答案排序，返回从好到差的下标。

        为什么用排序而非打分：LLM 给的绝对分数在不同请求间不可比
        （同一次请求内可比）。排序只需要相对偏好，更稳健，也不容易
        出现"全都 8 分"这种无信息输出。
        """
        if len(candidates) <= 1:
            return list(range(len(candidates)))

        listing = "\n".join(
            f"[{i}] {c[:600]}" for i, c in enumerate(candidates)
        )
        q = (
            "下面是对同一个问题的多个回答。请判断哪个回答更好"
            "（准确性、完整性、表达清晰度）。\n"
            "只输出编号，从最好到最差，用逗号分隔，不要解释。\n\n"
            f"问题：{prompt}\n\n回答：\n{listing}\n\n"
            f"排序（{len(candidates)} 个编号）："
        )

        resp = self.sample(q, system=system, n=1, temperatures=[0.2])
        ok = resp.ok_samples
        if not ok:
            return list(range(len(candidates)))

        nums = re.findall(r"\d+", ok[0].text)
        order, seen = [], set()
        for x in nums:
            i = int(x)
            if 0 <= i < len(candidates) and i not in seen:
                order.append(i)
                seen.add(i)
        # 教师漏掉的补在末尾，保证返回完整排列
        order += [i for i in range(len(candidates)) if i not in seen]
        return order


# --------------------------------------------------------------------------- #
# 批量采样
# --------------------------------------------------------------------------- #
def sample_many(
    teacher: BlackBoxTeacher,
    prompts: list[str],
    system: str | None,
    n: int | None = None,
) -> list[TeacherResponse]:
    """对一批问题采样。

    真实 API 应该并发（cfg.concurrency），但并发会让预算统计和
    日志顺序混乱，且 mock 后端用同一个模型实例本来就不能并发。
    这里保持串行 + 进度打印，把并发留给调用方按需实现。
    """
    out = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        try:
            out.append(teacher.sample(p, system=system, n=n))
        except BudgetExceeded as e:
            print(f"\n[预算] 熔断: {e}")
            break
        if (i + 1) % 5 == 0 or (i + 1) == len(prompts):
            el = time.time() - t0
            eta = el / (i + 1) * (len(prompts) - i - 1)
            print(f"  采样 {i+1}/{len(prompts)} | {el:.0f}s "
                  f"| 预算 {teacher.budget.summary()} | 剩余约 {eta:.0f}s",
                  flush=True)
    return out