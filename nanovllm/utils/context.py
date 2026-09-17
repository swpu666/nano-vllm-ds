from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: Optional[torch.Tensor] = None
    cu_seqlens_k: Optional[torch.Tensor] = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    context_lens: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    # 投机解码的 verify 步骤: 走 prefill 路径(要读历史 KV), 但需要**全部**位置的 logits,
    # 而 ParallelLMHead 在 prefill 时默认只返回每序列最后一个位置的 logits。
    spec_verify: bool = False

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, spec_verify=False):
    global _CONTEXT
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, spec_verify)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
