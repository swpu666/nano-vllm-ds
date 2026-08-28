from __future__ import annotations
import torch
from torch import nn
import triton
import triton.language as tl

from nanovllm.utils.context import get_context

try:
    from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
    _HAS_FLASH = True
except ImportError:
    _HAS_FLASH = False


@triton.jit
def store_kvcache_kernel(
    key_ptr,
    key_stride,
    value_ptr,
    value_stride,
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    D: tl.constexpr,
):
    idx = tl.program_id( 0)
    slot = tl.load(slot_mapping_ptr + idx)
    if slot == -1: return
    key_offsets = idx * key_stride + tl.arange(0, D)
    value_offsets = idx * value_stride + tl.arange(0, D)
    key = tl.load(key_ptr + key_offsets)
    value = tl.load(value_ptr + value_offsets)
    cache_offsets = slot * D + tl.arange(0, D)
    tl.store(k_cache_ptr + cache_offsets, key)
    tl.store(v_cache_ptr + cache_offsets, value)


def store_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    store_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, D)


class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def _sdpa_prefill(self, q, k, v):
        # q/k/v: (total, H, D) / (total, Hk, D); packed varlen -> 逐序列 sdpa
        cu = get_context().cu_seqlens_q
        H, Hk, D = self.num_heads, self.num_kv_heads, self.head_dim
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        cu_list = list(map(int, cu.tolist()))
        outs = []
        for i in range(len(cu_list) - 1):
            s, e = cu_list[i], cu_list[i + 1]
            qq = q[s:e].permute(1, 0, 2).unsqueeze(0)       # (1, H, seqlen, D)
            kk = k[s:e].permute(1, 0, 2).unsqueeze(0)
            vv = v[s:e].permute(1, 0, 2).unsqueeze(0)
            if H != Hk:
                kk = kk.repeat_interleave(H // Hk, dim=1)
                vv = vv.repeat_interleave(H // Hk, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, is_causal=True, scale=self.scale)
            outs.append(out.permute(0, 2, 1, 3).reshape(e - s, H * D))
        return torch.cat(outs, dim=0)

    def _sdpa_decode(self, q, k, v):
        # q: (B, H, D) 每个 seq 一个 token; k/v 已写入 paged cache
        context = get_context()
        B, H, D = q.shape[0], self.num_heads, self.head_dim
        Hk = self.num_kv_heads
        k_cache, v_cache = self.k_cache, self.v_cache      # (num_blocks, block_size, Hk, D)
        ctx_lens = context.context_lens                   # (B,)
        block_tables = context.block_tables                # (B, max_blocks)
        outs = []
        for j in range(B):
            c = int(ctx_lens[j])
            ids = block_tables[j]
            # 取覆盖前 c 个 token 的 block 数
            n_blocks = (c + k_cache.shape[1] - 1) // k_cache.shape[1]
            k_g = k_cache[ids[:n_blocks]].reshape(-1, Hk, D)[:c]  # (c, Hk, D)
            v_g = v_cache[ids[:n_blocks]].reshape(-1, Hk, D)[:c]
            if H != Hk:
                k_g = k_g.repeat_interleave(H // Hk, dim=1)
                v_g = v_g.repeat_interleave(H // Hk, dim=1)
            qq = q[j:j + 1].to(k_g.dtype).unsqueeze(1).permute(0, 2, 1, 3)  # (1, H, 1, D)
            kk = k_g.permute(1, 0, 2).unsqueeze(0)        # (1, H, c, D)
            vv = v_g.permute(1, 0, 2).unsqueeze(0)
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, scale=self.scale)
            outs.append(out.reshape(-1, H * D))
        return torch.cat(outs, dim=0)                     # (B, H*D)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        if context.is_prefill:
            # prefill 使用本次计算的 k/v (不读 paged cache)
            if _HAS_FLASH:
                return flash_attn_varlen_func(
                    q, k, v,
                    max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                    max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                    softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            return self._sdpa_prefill(q, k, v)
        else:                                            # decode
            if _HAS_FLASH:
                return flash_attn_with_kvcache(
                    q.unsqueeze(1), k_cache, v_cache,
                    cache_seqlens=context.context_lens, block_table=  context.block_tables,
                    softmax_scale=self.scale, causal=True).squeeze(1)
            return self._sdpa_decode(q, k, v)
