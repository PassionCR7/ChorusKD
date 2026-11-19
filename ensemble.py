"""V 阶段：多教师集成（域路由）。

为什么需要
----------
单个黑盒教师有系统性偏差：某家擅长代码，某家擅长中文写作，
某家推理强但话痨。白盒蒸馏里可以用"教师规模扫参"来选最优，
黑盒场景下往往只能用现成 API，选择空间小。

集成的诱惑是"全都问然后投票"，但黑盒按量计费，全问等于成本翻 N 倍。
所以在成本和收益之间必须做取舍。

域路由：用最少的请求拿到最合适的教师
------------------------------------
核心假设：一个问题的质量主要由"该域最强的教师"决定，而不是
"所有教师的平均"。所以先判断域，只问最强的那一个。

域判断用关键词规则而非另一个 LLM：
  * 零成本（不消耗黑盒预算——这是最关键的理由）
  * 可解释、可审计、可调
  * 准确率对粗粒度分类（代码/数学/写作/闲聊）已经够用

交叉验证：花少量预算买可信度
---------------------------
cfg.cross_check_ratio 比例的样本会问两个教师。这不是为了集成出更好答案，
而是为了**检测黑盒的不可靠性**：两个强教师都答不出的问题，
大概率不适合拿来教学生。这类样本直接丢弃比硬训更划算。
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from teacher_api import BlackBoxTeacher, TeacherResponse


# --------------------------------------------------------------------------- #
# 域判断
# --------------------------------------------------------------------------- #
# 关键词规则：按命中数打分，取最高。顺序无关，靠权重区分。
_DOMAIN_PATTERNS: dict[str, list[tuple[str, float]]] = {
    "code": [
        (r"```", 3.0),
        (r"\bdef\s+\w+\s*\(", 3.0),
        (r"\bclass\s+\w+", 2.5),
        (r"\bimport\s+\w+", 2.5),
        (r"\b(function|const|let|var|return|async|await)\b", 1.5),
        # 语言名：中英文混排时 \b 不可靠（中文与字母之间没有词边界），
        # 因此直接裸匹配。长名放前面，避免 "java" 抢先匹配 "javascript"。
        (r"(javascript|typescript|python|golang|rust|java|c\+\+|sql|html|css|json|regex|正则)", 2.5),
        (r"(报错|报错信息|异常|traceback|stack ?trace|bug|调试|编译|运行结果|输出结果)", 1.5),
        (r"[{};]\s*$", 1.0),
        # 中文语境的编程意图：「写一个函数/脚本/程序」「实现一个功能」
        (r"(写|实现|编写|生成|改)(一)?(个|段|下)?\s*(函数|脚本|程序|方法|类|接口|正则|查询)", 2.5),
        (r"(函数|脚本|代码|程序|接口|变量|数组|循环|递归|算法)", 1.2),
    ],
    "math": [
        (r"(计算|求出|求解|证明|推导|化简|积分|微分|方程|不等式)", 2.5),
        (r"\b(derive|prove|integral|derivative|equation|theorem)\b", 2.5),
        (r"[=＝].*[\dπ√∑∫]", 1.5),
        (r"\b\d+\s*[\+\-\*/]\s*\d+", 1.5),
        (r"(概率|期望|方差|矩阵|向量|级数|极限)", 1.5),
    ],
    "writing": [
        (r"(写一篇|写一段|润色|改写|重写|扩写|缩写|文案|标题)", 2.5),
        (r"(文章|小说|诗|故事|邮件|简历|演讲稿|宣传语|摘要|总结|大纲)", 2.0),
        (r"\b(write|rewrite|polish|draft|essay|poem|story|translate)\b", 2.0),
        (r"(翻译|译成|译为|中文版|英文版)", 2.5),
        (r"(风格|语气|口吻|文采|修辞)", 1.5),
    ],
    "chat": [
        (r"^(你好|hi|hello|嗨|在吗)", 2.0),
        (r"(你是谁|介绍一下你自己|你能做什么)", 2.0),
        (r"(谢谢|多谢|thanks)", 1.0),
    ],
}

_COMPILED = {
    d: [(re.compile(p, re.I), w) for p, w in pats]
    for d, pats in _DOMAIN_PATTERNS.items()
}


def detect_domain(prompt: str) -> str:
    """判断问题属于哪个域。默认归到 chat。

    用对数加成而非线性：命中多个弱特征的问题（如"用 Python 写个排序
    并解释原理"），域判断应该由最强信号决定，而不是被数量淹没。
    """
    scores = {d: 0.0 for d in _COMPILED}
    for d, pats in _COMPILED.items():
        for rx, w in pats:
            if rx.search(prompt):
                scores[d] += w
    best = max(scores.items(), key=lambda kv: kv[1])
    if best[1] <= 0:
        return "chat"
    return best[0]


# --------------------------------------------------------------------------- #
# 路由器
# --------------------------------------------------------------------------- #
@dataclass
class RouteResult:
    domain: str
    teacher_name: str
    response: TeacherResponse | None = None
    # 交叉验证结果
    cross_teacher: str | None = None
    cross_response: TeacherResponse | None = None
    agreement: float = float("nan")


class DomainRouter:
    """按域把问题分派给对应教师。

    教师实例按需创建并缓存——多个教师共用一个 Budget，
    保证总成本不超限（这是最容易出错的地方：按教师分别计预算
    会让实际花费是设定值的 N 倍）。
    """

    def __init__(self, cfg, budget, teacher_factory=None):
        self.cfg = cfg
        self.budget = budget
        self._cache: dict[str, BlackBoxTeacher] = {}
        self._factory = teacher_factory or self._default_factory

    def _default_factory(self, name: str) -> BlackBoxTeacher:
        """默认所有教师共用同一份 API 配置，仅 model 名不同。"""
        from copy import deepcopy
        tcfg = deepcopy(self.cfg.teacher)
        tcfg.model = name
        return BlackBoxTeacher(tcfg, self.budget)

    def get(self, name: str) -> BlackBoxTeacher:
        if name not in self._cache:
            self._cache[name] = self._factory(name)
        return self._cache[name]

    def pick(self, prompt: str) -> tuple[str, str]:
        """返回 (域, 教师名)。"""
        domain = detect_domain(prompt)
        names = self.cfg.routing.get(domain) or self.cfg.routing.get("chat") or ["default"]
        return domain, names[0]

    def route(
        self, prompt: str, system: str | None, n: int | None = None,
        do_cross_check: bool = False,
    ) -> RouteResult:
        domain, name = self.pick(prompt)
        res = RouteResult(domain=domain, teacher_name=name)

        try:
            res.response = self.get(name).sample(prompt, system=system, n=n)
        except Exception as e:
            print(f"[路由] {name} 失败: {e}")
            return res

        if do_cross_check:
            names = self.cfg.routing.get(domain) or ["default"]
            # 选一个不同的教师做交叉验证
            other = next((x for x in names if x != name), None)
            if other is None:
                # 该域只有一个教师，退化到默认教师
                all_names = {x for v in self.cfg.routing.values() for x in v}
                other = next((x for x in all_names if x != name), None)
            if other:
                res.cross_teacher = other
                try:
                    res.cross_response = self.get(other).sample(
                        prompt, system=system, n=n
                    )
                    res.agreement = _response_agreement(
                        res.response, res.cross_response
                    )
                except Exception:
                    pass
        return res


def _response_agreement(a: TeacherResponse, b: TeacherResponse) -> float:
    """两个教师答案集的一致性（互相比对取最大相似度）。"""
    from filter import _token_f1

    ta = [s.text for s in a.ok_samples]
    tb = [s.text for s in b.ok_samples]
    if not ta or not tb:
        return float("nan")
    scores = [max(_token_f1(x, y) for y in tb) for x in ta]
    return sum(scores) / len(scores)


def should_drop_on_disagreement(
    res: RouteResult, cfg, threshold: float = 0.5,
) -> bool:
    """交叉验证不一致时该不该丢。

    语义判断：两个不同来源的强教师对同一问题给出差异很大的答案，
    说明这题要么有歧义、要么超出双方能力。两种情况下拿来教学生
    都是有害的（学生学到的是混乱）。
    """
    if cfg.on_disagreement != "drop":
        return False
    if res.cross_response is None:
        return False
    if res.agreement != res.agreement:      # nan
        return False
    return res.agreement < threshold