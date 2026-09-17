from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Optional
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    quantization: Optional[str] = None
    hf_config: Optional[AutoConfig] = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # 投机解码: draft_model 非空且 num_speculative_tokens > 0 时启用
    draft_model: Optional[str] = None
    num_speculative_tokens: int = 5
    # 动态 γ: 按接受率 + draft/target 耗时比在线选最优 draft 长度, 上界由本字段给定
    max_speculative_tokens: int = 12
    dynamic_gamma: bool = True
    # 多候选 (tree) 验证: 一次 target forward 同时验证 K 条候选链, 取接受最长的那条。
    # K=1 即退化为普通的链式投机解码。
    #
    # ⚠ K>1 目前是**实验特性, 默认关闭**, 有两个已定位的问题(见 docs/spec-decoding.md):
    #   1) paged KV 是"逻辑连续"语义: draft 用 cache_seqlens=L+i 读的是逻辑位置
    #      L+i-1 的 slot, 而多候选时链 c 的节点写在 L+c*G+i-1 —— 两者对不上,
    #      draft 的上下文会错乱, 表现为接受率下降(实测 α 0.74 -> 0.58)。
    #      正确做法要给每条链分配独立 block 并构造虚拟 block_table, 尚未实现。
    #   2) 即使修好, 本项目 c = t_draft/t_target ≈ 0.07, verify 成本占主导,
    #      K 翻倍会直接放大 verify 计算, 而单链在接受率 0.74~0.9 时已接近
    #      接受上限 —— 实测 K=2/4 的吞吐反而更低。
    num_spec_candidates: int = 1
    # draft/target 词表不等长时 (如 Qwen2.5-0.5B 151936 vs 7B 152064),
    # 概率空间统一截断到公共前缀后重归一化; 该项记录被丢弃的尾部 id 数
    spec_common_vocab: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        qcfg = getattr(self.hf_config, "quantization_config", None)
        if qcfg is not None and qcfg.get("quant_method") == "gptq":
            assert qcfg.get("bits") == 4, "only 4-bit GPTQ is supported"
            self.quantization = "gptq"
        assert self.quantization in (None, "gptq")
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
