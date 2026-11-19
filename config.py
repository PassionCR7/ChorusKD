"""ChorusKD：黑盒（闭源 API）蒸馏的理论最优形态。

与 LensKD 的根本差别
--------------------
LensKD 能拿到教师的 hidden、logits、每层激活——因为教师是本地白盒。
黑盒只有一条文本通道：发 prompt，收字符串。

于是所有"从分布里榨信息"的手段全部失效：
  ✗ 软标签 KL（拿不到 logits）
  ✗ 表征对齐（拿不到 hidden）
  ✗ 温度缩放（拿不到分布）

剩下唯一能榨的是**采样**：同一个问题问多次，从答案的多样性反推教师的行为
倾向。这条路的信息上限远低于白盒，所以工程重点必须转移到：
  1. 有限预算内怎么问到信息量最大的问题
  2. 怎么把"一条文本"的事实性错误率压到最低
  3. 怎么处理教师答不好的样本（黑盒无法区分"不会"和"不想"）

七个阶段
--------
P 提示词策展   → prompts.py     按学生当前能力缺口选题
Q 教师多样采样 → teacher_api.py  每题 n 采样 + 多温度
R 质量与难度   → filter.py      一致性投票 + 可学习性筛选
S 能力匹配调度 → scheduler.py   动态调整难度分布
T 全量 logprob → （黑盒不可用，用 n-gram 覆盖近似，见 snapshot.py）
U 学生自采样   → onpolicy.py    自采样 + 教师偏好打分
V 多教师集成   → ensemble.py    按能力域路由

本文件只放配置。
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


PROJECT_ROOT = r"D:\workspace\xllm"
CHORUS_ROOT = r"D:\workspace\ChorusKD"


# --------------------------------------------------------------------------- #
# 教师（黑盒 API）
# --------------------------------------------------------------------------- #
@dataclass
class TeacherAPICfg:
    """黑盒教师配置。

    支持任何 OpenAI 兼容接口。没有 API key 时自动走 mock 模式，
    用本地 Qwen3 模拟"只能拿到文本"的受限通道——这样代码可以完整跑通，
    行为与真实黑盒一致（不暴露 logits）。
    """

    provider: str = "openai_compatible"
    base_url: str = os.environ.get("CHORUS_BASE_URL", "https://api.openai.com/v1")
    api_key: str = os.environ.get("CHORUS_API_KEY", "")
    model: str = os.environ.get("CHORUS_MODEL", "gpt-4o-mini")

    # mock 模式用的本地模型（无 API key 时启用）
    mock_model_path: str = os.environ.get(
        "XLLM_TEACHER_PATH",
        r"D:\models\models\Qwen--Qwen3-0.6B\snapshots\master",
    )

    # ---- 采样参数：多样性的来源 ----
    # 同一问题采 n 次，温度从低到高铺开。低温度拿到"教师最确信的答案"，
    # 高温度暴露"教师认为合理的其他答案"——后者是软标签信息的文本替代物。
    n_samples: int = 5
    temperatures: tuple[float, ...] = (0.3, 0.7, 0.9, 1.1, 1.3)
    top_p: float = 0.95
    max_tokens: int = 512

    # ---- 预算控制 ----
    # 黑盒按 token 计费，必须有硬上限，否则一轮跑飞就超支
    max_requests: int = 2000
    max_tokens_per_request: int = 700
    request_timeout: float = 60.0
    max_retries: int = 3
    # 并发数（API 通常有 QPS 限制）
    concurrency: int = 4


# --------------------------------------------------------------------------- #
# 提示词策展
# --------------------------------------------------------------------------- #
@dataclass
class PromptCfg:
    """P 阶段：从种子池生成待问的问题。

    核心洞察：黑盒的每次请求都有成本，所以**问什么**比**问多少次**更重要。
    随机从语料抽问题的做法，会把预算浪费在学生已经会了的内容上。
    """

    seed_file: str = os.path.join(PROJECT_ROOT, "dataset", "sft_t2t_mini.jsonl")
    num_prompts: int = 200

    # ---- 选题策略 ----
    # "difficulty"：按学生当前 loss 排序，优先补最不会的（推荐）
    # "random"    ：随机，仅作消融基线
    # "diverse"   ：按 embedding 聚类后每簇均匀取，保证覆盖面
    strategy: str = "difficulty"

    # 难度分档：把候选按学生 loss 分位数切成若干档，
    # 每轮从指定档位采样（档位由 scheduler 动态调整）
    num_difficulty_bins: int = 5

    # 每条种子生成几个变体（改写/加约束），提升单条种子的信息产出
    variants_per_seed: int = 1

    # ---- 系统提示词 ----
    # 黑盒蒸馏里 system prompt 会显著影响教师输出风格。
    # 用统一的、偏"教学解释"的风格，避免教师输出和学生目标格式不匹配。
    system_prompt: str = (
        "你是一个严谨的助手。回答要准确、简洁，直接给出结论再补充必要说明。"
        "不确定的内容明确说不知道，不要编造。"
    )


# --------------------------------------------------------------------------- #
# 质量与难度筛选
# --------------------------------------------------------------------------- #
@dataclass
class FilterCfg:
    """R 阶段：把 n 次采样变成一条可用训练样本。

    黑盒蒸馏最大的质量风险是**教师幻觉**。白盒蒸馏里教师的软标签自带
    "不确定"信号（低概率），黑盒只给文本，错的时候和队的时候长得一样。

    解法是**自一致性投票**：同一问题采 n 次，答案高度一致的说明教师确信，
    这种样本可信；n 次答案互相矛盾的，要么问题本身歧义，要么教师不会，
    两类都不该拿来教学生。
    """

    # ---- 一致性判定 ----
    # 一致性度量："exact"（去空白后全等）| "fuzzy"（token F1 阈值）| "semantic"
    consistency_mode: str = "fuzzy"
    # fuzzy 模式下的 token F1 阈值（同一簇的最低相似度）
    cluster_threshold: float = 0.75
    # 最大簇至少占多少比例，才认为这条样本可信
    min_agreement_ratio: float = 0.4

    # ---- 难度判定 ----
    # 用学生在该答案上的 loss 作为难度代理。
    # 太低 → 学生已经会了，没信息量；太高 → 学生学不动，浪费样本。
    min_student_loss: float = 0.8
    max_student_loss: float = 6.0

    # ---- 长度过滤 ----
    min_answer_tokens: int = 8
    max_answer_tokens: int = 480

    # ---- 拒绝采样 ----
    # 教师答案里出现这些模式说明大概率是拒答或幻觉，直接丢
    reject_patterns: tuple[str, ...] = (
        "作为一个AI", "作为AI语言模型", "我无法", "我不能回答",
        "I cannot", "I'm sorry, but", "As an AI",
    )


# --------------------------------------------------------------------------- #
# 能力匹配调度
# --------------------------------------------------------------------------- #
@dataclass
class SchedulerCfg:
    """S 阶段：动态调整"问多难的问题"。

    这是黑盒蒸馏比白盒更关键的一环：白盒能实时看教师分布的熵来判断
    "这个位置教师有多确信"，黑盒看不到，只能靠**离线统计**推断难度。

    调度逻辑（课程学习 + 能力匹配）：
      * 学生在该档位的平均 loss 下降 → 升档
      * loss 停滞或上升 → 降档
      * 每档保持"略高于当前能力"（zone of proximal development）
    """

    enabled: bool = True
    # 每隔多少轮重新评估档位
    reassess_every_rounds: int = 1
    # 目标 loss 区间：低于下限说明太简单，高于上限说明太难
    target_loss_low: float = 1.0
    target_loss_high: float = 2.5
    # 升/降档的步长
    step: int = 1
    # 初始档位（0 最简单，num_bins-1 最难）
    start_bin: int = 1


# --------------------------------------------------------------------------- #
# 学生
# --------------------------------------------------------------------------- #
@dataclass
class StudentCfg:
    model_dir: str = os.path.join(PROJECT_ROOT, "model")
    weight_dir: str = os.path.join(PROJECT_ROOT, "out")
    from_weight: str = "full_sft"
    hidden_size: int = 512
    num_hidden_layers: int = 8
    vocab_size: int = 6400
    use_moe: bool = False
    save_dir: str = os.path.join(CHORUS_ROOT, "out")
    save_weight: str = "chorus_kd"


# --------------------------------------------------------------------------- #
# 多教师集成
# --------------------------------------------------------------------------- #
@dataclass
class EnsembleCfg:
    """V 阶段：多个黑盒教师按能力域路由。

    为什么需要：单个黑盒教师有系统性偏差（某家擅长代码、某家擅长写作）。
    但集成不是"全都问"——黑盒按量计费，全问等于成本翻倍。

    策略是**域路由**：先判断问题属于哪个域，只问该域最强的那个教师。
    需要多个教师都有数据时，用少量预算做交叉验证。
    """

    enabled: bool = False
    # 域 -> 教师名列表（按优先级）
    routing: dict = field(default_factory=lambda: {
        "code": ["coder"],
        "math": ["reasoner"],
        "chat": ["default"],
        "writing": ["default"],
    })
    # 是否对高价值样本做多教师交叉验证（成本高，仅少量）
    cross_check_ratio: float = 0.1
    # 交叉验证时，教师间不一致的样本如何处理："drop" | "keep_majority"
    on_disagreement: str = "drop"


# --------------------------------------------------------------------------- #
# 在线策略
# --------------------------------------------------------------------------- #
@dataclass
class OnPolicyCfg:
    """U 阶段：学生自采样 + 教师偏好打分。

    黑盒做不了 GKD 那种"教师给学生 prefix 打分布"，因为拿不到分布。
    替代方案：让学生对同一问题生成 k 个答案，让教师**排序**（不是打分，
    因为绝对分数在不同请求间不可比），用排序结果做 DPO / 或选出最好的
    当 SFT 目标。

    成本很高（每题 k+1 次请求），所以只对少量关键样本做。
    """

    enabled: bool = True
    # 每题让学生生成几个候选
    num_candidates: int = 4
    # 教师排序的呈现方式："pairwise"（两两比，贵）| "listwise"（一次给全部）
    ranking_mode: str = "listwise"
    # 抽样做在线策略的比例
    sample_ratio: float = 0.15
    # student 采样温度
    sample_temperature: float = 0.95


# --------------------------------------------------------------------------- #
# 训练
# --------------------------------------------------------------------------- #
@dataclass
class TrainCfg:
    epochs: int = 2
    batch_size: int = 2
    grad_accum: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    log_interval: int = 10
    seed: int = 42
    max_seq_len: int = 512

    # ---- 损失权重 ----
    # 黑盒只有硬标签（文本），所以本质是 SFT。
    # 但有两个可加的软信号：
    #   1. n 次采样的答案都会参与（不只是选中的那条）→ "多正例"
    #   2. 在线策略的偏好对 → DPO 项
    ce_weight: float = 1.0
    # 多正例：选中的主答案权重 1.0，其余一致答案权重 small
    aux_positive_weight: float = 0.3
    # DPO 项权重
    dpo_weight: float = 0.1
    dpo_beta: float = 0.1

    # ---- 数据轮次 ----
    # 每一轮：策展 → 采样 → 筛选 → 训练，然后重新评估学生能力
    num_rounds: int = 2
    # 每轮用多少条样本训练
    samples_per_round: int = 100


@dataclass
class ChorusKDConfig:
    teacher: TeacherAPICfg = field(default_factory=TeacherAPICfg)
    prompt: PromptCfg = field(default_factory=PromptCfg)
    filter: FilterCfg = field(default_factory=FilterCfg)
    scheduler: SchedulerCfg = field(default_factory=SchedulerCfg)
    student: StudentCfg = field(default_factory=StudentCfg)
    ensemble: EnsembleCfg = field(default_factory=EnsembleCfg)
    onpolicy: OnPolicyCfg = field(default_factory=OnPolicyCfg)
    train: TrainCfg = field(default_factory=TrainCfg)