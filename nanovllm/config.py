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
