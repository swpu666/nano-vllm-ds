"""投机解码 (chain-style speculative decoding) 的核心算子。

职责边界:
  - 本模块只做**纯张量运算**: 给定 draft/target 的 logits 与候选 token, 算出每个序列
    最终确认的 token 列表。不碰 KV cache / block table / 调度(那些在 ModelRunner 与
    LLMEngine 里), 因此可以单独做正确性测试。
  - 对外三件事:  draft 采样 -> target 验证 -> 汇总"每序列新增了哪些 token"。

两种模式:
  - greedy: draft/target 都取 argmax, 验证规则退化为 "target argmax 是否等于草稿"。
            此时输出必须与不用投机解码的基线**逐 token 完全一致**, 这是本项目的主要
            正确性指标。
  - 采样  : 用标准的 rejection sampling (Leviathan et al. 2023), 保证输出分布与目标模型
            的 token 分布严格一致 —— 接受判据是 r < p(x)/q(x), 被拒绝时从 (p-q)_+ 归一化后
            重采样, 全部接受时额外白送一个 bonus token。

词表长度不等的处理 (Qwen2.5-0.5B=151936 vs Qwen2.5-7B=152064):
  draft 产生的 id 必然小于它的词表长; 但 target 词表更长, 多出的尾部 id 是 padding 条目
  (两边 tokenizer 在公共 id 区间上逐条一致, 由 tests/check_spec_vocab.py 验证)。
  因此把两个分布统一截断到公共前缀 V=min(Vd,Vt) 后重新归一化, 在该空间内做拒绝采样。
  代价是丢弃了 target 落在尾部的概率质量, `tail_mass` 会把它统计出来供核查。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import torch


def _probs(logits: torch.Tensor, temperatures: Optional[torch.Tensor] = None) -> torch.Tensor:
    """logits -> 温度缩放后的概率。始终在 fp32 下算, 避免 fp16 softmax 的精度损失。"""
    x = logits.float()
    if temperatures is not None:
        # logits 可能是 (B, V) 或 (B, γ+1, V), 温度要 reshape 成 (-1, 1) / (-1, 1, 1)
        x = x.div(temperatures.reshape((-1,) + (1,) * (x.dim() - 1)))
    return torch.softmax(x, dim=-1)


def _gumbel_sample(probs: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Gumbel-max 采样, 等价于多项式采样且与 nano-vLLM 原生 Sampler 口径一致。"""
    g = torch.empty_like(probs).exponential_(1, generator=generator).clamp_min_(1e-10)
    return probs.div(g).argmax(dim=-1)


class TailMassMeter:
    """累积统计 target 分布落在公共词表之外的概率质量。"""

    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, target_logits: torch.Tensor, V: int):
        if target_logits.size(-1) <= V:
            return
        p_full = _probs(target_logits[:, 0, :])           # 只统计第一个位置即可
        self.total += float(p_full[:, V:].sum().mean())
        self.count += 1

    @property
    def mean(self) -> float:
        return self.total / self.count if self.count else 0.0


@dataclass
class VerifyOutput:
    """一次 target verify 的结果。

    n_accepted[i]      : 第 i 个序列接受的 draft token 个数 (0..γ)
    accepted_tokens[i] : 该序列本轮最终确认的 token 列表
                         = 接受的草稿 + (被拒处的修正 token 或 全接受时的 bonus token)
    n_new_tokens[i]    : len(accepted_tokens[i]) == n_accepted[i] + 1 (每轮至少推进 1)
    """
    n_accepted: list[int]
    accepted_tokens: list[list[int]]
    n_new_tokens: list[int]


class Speculator:
    """draft 采样 + target 验证。无状态(除统计量), 可被 ModelRunner 直接持有。"""

    def __init__(self, num_spec_tokens: int, tail_mass_meter: Optional[TailMassMeter] = None):
        self.num_spec_tokens = num_spec_tokens
        self.tail_mass = tail_mass_meter or TailMassMeter()
        # 诊断: 累积 accepted 数 / 验证过的候选数 -> 接受率 α
        self.num_accepted_tokens = 0
        self.num_draft_tokens = 0
        self.num_rounds = 0

    # ---------------------------------------------------------------- draft
    def draft_step(
        self,
        draft_logits: torch.Tensor,           # (B, Vd)
        temperatures: Optional[torch.Tensor],
        greedy: bool,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """draft 模型单步采样, 返回 (B,) 候选 token。"""
        if greedy:
            return draft_logits.argmax(dim=-1)
        return _gumbel_sample(_probs(draft_logits, temperatures), generator)

    # --------------------------------------------------------------- verify
    def verify(
        self,
        target_logits: torch.Tensor,          # (B, γ+1, Vt) raw logits
        draft_logits: torch.Tensor,           # (B, γ,   Vd) raw logits
        draft_tokens: torch.Tensor,           # (B, γ) int64
        temperatures: Optional[torch.Tensor],
        greedy: bool,
        generator: Optional[torch.Generator] = None,
    ) -> VerifyOutput:
        B, G = draft_tokens.shape
        assert G == self.num_spec_tokens, (G, self.num_spec_tokens)
        assert target_logits.shape[:2] == (B, G + 1)

        V = min(target_logits.size(-1), draft_logits.size(-1))
        self.tail_mass.update(target_logits, V)

        p_all = _probs(target_logits[..., :V], temperatures)   # (B, γ+1, V)
        if greedy:
            n_accepted, extra = self._verify_greedy(p_all, draft_tokens, G)
        else:
            q_all = _probs(draft_logits[..., :V], temperatures)  # (B, γ, V)
            n_accepted, extra = self._verify_sampling(
                p_all, q_all, draft_tokens, G, generator)

        # 一次性搬到 CPU: 逐个 int()/tolist() 会引入 B 次 GPU->CPU 同步
        tok_by_seq = draft_tokens.tolist()
        extra_list = extra.tolist()
        acc_list = n_accepted.tolist()
        accepted_tokens = []
        for b in range(B):
            toks = tok_by_seq[b][:acc_list[b]]
            toks.append(extra_list[b])
            accepted_tokens.append(toks)

        self.num_accepted_tokens += int(n_accepted.sum())
        self.num_draft_tokens += B * G
        self.num_rounds += B
        return VerifyOutput(
            n_accepted=n_accepted.tolist(),
            accepted_tokens=accepted_tokens,
            n_new_tokens=[len(t) for t in accepted_tokens],
        )

    def _verify_greedy(
        self,
        p_all: torch.Tensor,          # (B, γ+1, V) probabilities
        draft_tokens: torch.Tensor,   # (B, γ)
        G: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """greedy 验证: 草稿 token == target argmax 才接受。

        这保证输出与"每步都让 target 取 argmax"的基线严格一致:
        接受时上下文不变(草稿就是 argmax), 拒绝时替换为 argmax 并终止本轮。
        """
        t_arg = p_all.argmax(dim=-1)                       # (B, γ+1)
        match = t_arg[:, :G] == draft_tokens               # (B, γ)
        prefix = match.cumprod(dim=1).bool()               # 首个 miss 之后全部置 0
        n_accepted = prefix.sum(dim=1)                     # (B,)
        idx = n_accepted.clamp(max=G)                      # 全接受时取 γ+1 位 = bonus
        extra = t_arg.gather(1, idx.unsqueeze(1)).squeeze(1)
        return n_accepted, extra

    def _verify_sampling(
        self,
        p_all: torch.Tensor,          # (B, γ+1, V)
        q_all: torch.Tensor,          # (B, γ, V)
        draft_tokens: torch.Tensor,   # (B, γ)
        G: int,
        generator: Optional[torch.Generator],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """标准 rejection sampling。"""
        device = p_all.device
        tk = draft_tokens.unsqueeze(-1)
        p_cand = p_all[:, :G, :].gather(-1, tk).squeeze(-1)      # (B, γ)
        q_cand = q_all.gather(-1, tk).squeeze(-1)                # (B, γ)

        # q=0 而 p>0 时 ratio=+inf -> 判为接受 (clamp 到 1)
        accept_prob = self.accept_probability(p_cand, q_cand)
        r = torch.rand(p_cand.shape, generator=generator, device=device)
        accept = r < accept_prob
        prefix = accept.cumprod(dim=1).bool()
        n_accepted = prefix.sum(dim=1)                            # (B,)

        # 首个被拒处的修正分布: (p - q)_+ 归一化
        B, V = p_all.size(0), p_all.size(-1)
        # 注意 torch>=2.7 不允许对"新增的前置维度"用 -1, 必须显式给出 B
        idx = n_accepted.clamp(max=G - 1).view(B, 1, 1).expand(B, 1, V)
        p_rej = p_all[:, :G, :].gather(1, idx).squeeze(1)         # (B, V)
        q_rej = q_all.gather(1, idx).squeeze(1)                   # (B, V)
        reject_dist = self.residual_distribution(p_rej, q_rej)
        extra_reject = _gumbel_sample(reject_dist, generator)
        extra_bonus = _gumbel_sample(self.bonus_distribution(p_all, G), generator)   # 全接受时的 bonus
        extra = torch.where(n_accepted == G, extra_bonus, extra_reject)
        return n_accepted, extra

    # --------------------------------------------------------------------------
    # 拒绝采样的三个关键量。抽成方法而不是内联, 是为了让 tests/verify_spec.py 能通过
    # 子类覆写注入常见实现错误, 验证"分布等价性检验"确实抓得住问题 (变异测试)。
    # --------------------------------------------------------------------------
    def accept_probability(self, p_cand: torch.Tensor, q_cand: torch.Tensor) -> torch.Tensor:
        return torch.clamp(p_cand / q_cand.clamp_min(1e-12), max=1.0)

    def residual_distribution(self, p_rej: torch.Tensor, q_rej: torch.Tensor) -> torch.Tensor:
        """第一个被拒位置上的修正分布: (p-q)_+ 归一化后采样。"""
        diff = (p_rej - q_rej).clamp_min(0.0)
        mass = diff.sum(dim=-1, keepdim=True)
        # mass≈0 只在 p==q 的数值边界出现, 此时本该被接受; 退化用 p 不影响分布正确性
        return torch.where(mass <= 1e-9, p_rej, diff / mass.clamp_min(1e-12))

    def bonus_distribution(self, p_all: torch.Tensor, G: int) -> torch.Tensor:
        """全部接受时额外白送的那个 token 的分布 (target 在最后一位的输出)。"""
        return p_all[:, G, :]

    # ------------------------------------------------------------- metrics
    @property
    def acceptance_rate(self) -> float:
        return self.num_accepted_tokens / self.num_draft_tokens if self.num_draft_tokens else 0.0

    @property
    def mean_accepted_length(self) -> float:
        """每轮的期望产出 token 数 (含 bonus), 直接决定加速比。"""
        return (self.num_accepted_tokens + self.num_rounds) / self.num_rounds if self.num_rounds else 0.0

    def summary(self) -> str:
        return (f"alpha(接受率)={self.acceptance_rate:.3f}  "
                f"mean_accepted={self.mean_accepted_length:.2f}  "
                f"rounds={self.num_rounds}  "
                f"target尾部概率质量={self.tail_mass.mean:.2e}")
