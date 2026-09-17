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


@triton.jit
def cached_causal_attn_kernel(
    Q, KNEW, VNEW, KCACHE, VCACHE, BT, OUT,
    ROW_SEQ, ROW_LOCAL, ROW_CACHED, ROW_BASE,
    sm_scale,
    stride_q_n, stride_q_h,
    stride_kn_n, stride_kn_h,
    stride_kc_b, stride_kc_s, stride_kc_h,
    stride_vc_b, stride_vc_s, stride_vc_h,
    stride_bt,
    stride_o_n, stride_o_h,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    GQA: tl.constexpr,
):
    """带 paged 前缀的 causal attention, 每个 program 处理一个 (query行, head)。

    为什么不直接用 SDPA:
      PyTorch 的 is_causal 在 seqlen_q < seqlen_k 时等价于 j <= i (朴素因果),
      **不会**自动补偿 (S-L) 的偏移 —— 于是 k/v 里排在最前面的"历史前缀"
      会被整段 mask 掉, 表现为"完全读不到 cache"。实测确认:
      构造 offset=i+(S-L) 的手工掩码才能得到正确答案。
      而显式传 float mask 会退化到 math backend, 复杂度 O(L·S·D) 且中断 flash,
      对投机解码的 verify 来说比省下的时间还贵。
      所以这里直接从 paged cache 读 key/value, 不物化到连续内存。

    掩码约定: query 的第 i 行可以 attend 到 key 的 [0, cached + i]。
    """
    row = tl.program_id(0)
    h = tl.program_id(1)
    kv_h = h // GQA
    d = tl.arange(0, HEAD_DIM)

    seq = tl.load(ROW_SEQ + row)
    li = tl.load(ROW_LOCAL + row)
    cached = tl.load(ROW_CACHED + row)
    base_new = tl.load(ROW_BASE + row)

    q = tl.load(Q + row * stride_q_n + h * stride_q_h + d).to(tl.float32)
    bt_ptr = BT + seq * stride_bt
    off = tl.arange(0, BLOCK_N)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    # ---------------- 前缀部分: 来自 paged KV cache ----------------
    for s in range(0, cached, BLOCK_N):
        j = s + off
        valid = (j < cached)
        bid = tl.load(bt_ptr + j // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
        kp = KCACHE + bid * stride_kc_b + (j % BLOCK_SIZE) * stride_kc_s + kv_h * stride_kc_h
        vp = VCACHE + bid * stride_vc_b + (j % BLOCK_SIZE) * stride_vc_s + kv_h * stride_vc_h
        k = tl.load(kp[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(vp[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k.to(tl.float32), 1) * sm_scale
        qk = tl.where(valid, qk, -1e30)
        m_new = tl.maximum(m_i, tl.max(qk))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m_i = m_new

    # ---------------- 本次新增部分: 来自 contiguous k/v ----------------
    for s in range(0, li + 1, BLOCK_N):
        jn = s + off
        valid = jn <= li
        kp = KNEW + (base_new + jn) * stride_kn_n + kv_h * stride_kn_h
        vp = VNEW + (base_new + jn) * stride_kn_n + kv_h * stride_kn_h
        k = tl.load(kp[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(vp[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k.to(tl.float32), 1) * sm_scale
        qk = tl.where(valid, qk, -1e30)
        m_new = tl.maximum(m_i, tl.max(qk))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m_i = m_new

    tl.store(OUT + row * stride_o_n + h * stride_o_h + d,
             (acc / l_i).to(OUT.dtype.element_ty))


@triton.jit
def _decode_split_kernel(
    Q, KCACHE, VCACHE, BT, CTXLEN, ACC, MAXL,
    sm_scale,
    stride_q_n, stride_q_h,
    stride_kc_b, stride_kc_s, stride_kc_h,
    stride_vc_b, stride_vc_s, stride_vc_h,
    stride_bt,
    stride_acc_n, stride_acc_h, stride_acc_s,
    stride_maxl_n, stride_maxl_h, stride_maxl_s,
    N_SPLITS: tl.constexpr,
    HEAD_DIM: tl.constexpr, BLOCK_SIZE: tl.constexpr, BLOCK_N: tl.constexpr,
    GQA: tl.constexpr,
):
    """flash-decoding 第一阶段: (row, head, kv_split) 处理一段 key, 输出未归一化的部分和。

    为什么要切分 key 维度: decode 时每个序列只有 1 个 query token, 若 grid 只有
    (batch, heads) 个 program, 在 batch 小的时候连一张 3090 的 82 个 SM 都填不满,
    attention 会被访存延迟主导。沿 key 切成 N_SPLITS 段并行再归约是标准做法。
    """
    row = tl.program_id(0)
    h = tl.program_id(1)
    sp = tl.program_id(2)
    kv_h = h // GQA
    d = tl.arange(0, HEAD_DIM)

    c = tl.load(CTXLEN + row)                 # 该序列的 key 总长 (含当前 token)
    per_ctx = tl.cdiv(c, N_SPLITS * BLOCK_N) * BLOCK_N    # 每段覆盖的 key 个数
    start = sp * per_ctx
    end = tl.minimum(start + per_ctx, c)

    q = tl.load(Q + row * stride_q_n + h * stride_q_h + d).to(tl.float32)
    bt_ptr = BT + row * stride_bt
    off = tl.arange(0, BLOCK_N)

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)

    for s in range(start, end, BLOCK_N):
        j = s + off
        valid = j < end
        bid = tl.load(bt_ptr + j // BLOCK_SIZE, mask=valid, other=0).to(tl.int64)
        kp = KCACHE + bid * stride_kc_b + (j % BLOCK_SIZE) * stride_kc_s + kv_h * stride_kc_h
        vp = VCACHE + bid * stride_vc_b + (j % BLOCK_SIZE) * stride_vc_s + kv_h * stride_vc_h
        k = tl.load(kp[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        v = tl.load(vp[:, None] + d[None, :], mask=valid[:, None], other=0.0)
        qk = tl.sum(q[None, :] * k.to(tl.float32), 1) * sm_scale
        qk = tl.where(valid, qk, -1e30)
        m_new = tl.maximum(m_i, tl.max(qk))
        p = tl.exp(qk - m_new)
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p)
        acc = acc * alpha + tl.sum(p[:, None] * v.to(tl.float32), 0)
        m_i = m_new

    base = row * stride_acc_n + h * stride_acc_h + sp * stride_acc_s
    tl.store(ACC + base + d, acc)
    mlbase = row * stride_maxl_n + h * stride_maxl_h + sp * stride_maxl_s
    tl.store(MAXL + mlbase, m_i)
    tl.store(MAXL + mlbase + 1, l_i)


@triton.jit
def _decode_reduce_kernel(
    ACC, MAXL, OUT,
    stride_acc_n, stride_acc_h, stride_acc_s,
    stride_maxl_n, stride_maxl_h, stride_maxl_s,
    stride_o_n, stride_o_h,
    N_SPLITS: tl.constexpr, HEAD_DIM: tl.constexpr,
):
    """flash-decoding 第二阶段: 合并各段的 (m, l, acc)。"""
    row = tl.program_id(0)
    h = tl.program_id(1)
    d = tl.arange(0, HEAD_DIM)

    mlbase = row * stride_maxl_n + h * stride_maxl_h
    m_max = float("-inf")
    for s in range(N_SPLITS):
        m_max = tl.maximum(m_max, tl.load(MAXL + mlbase + s * stride_maxl_s))

    acc = tl.zeros([HEAD_DIM], dtype=tl.float32)
    l_sum = 0.0
    for s in range(N_SPLITS):
        ms = tl.load(MAXL + mlbase + s * stride_maxl_s)
        ls = tl.load(MAXL + mlbase + s * stride_maxl_s + 1)
        w = tl.exp(ms - m_max)
        acc += w * tl.load(ACC + row * stride_acc_n + h * stride_acc_h + s * stride_acc_s + d)
        l_sum += w * ls

    tl.store(OUT + row * stride_o_n + h * stride_o_h + d,
             (acc / l_sum).to(OUT.dtype.element_ty))


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
        """prefill attention。

        ⚠ 这里必须处理"部分 KV 落在 paged cache"的情形:
           本次的 k/v 只是序列的后半段, 前半段要按 block_table 从 cache 里取。
           flash-attn 原生支持这件事 (vLLM 的 prefix caching 就走这条路); 没有 flash-attn
           时原先的 sdpa fallback 完全忽略 block_table, 于是任何依赖它的调用
           (chunked prefill 的续段、投机解码的 verify) 都会丢掉历史上下文。
           这里用 cached_causal_attn_kernel 补上这部分。
        """
        context = get_context()
        cu_q, cu_k = context.cu_seqlens_q, context.cu_seqlens_k
        block_tables = context.block_tables
        H, Hk, D = self.num_heads, self.num_kv_heads, self.head_dim
        q = q.contiguous()
        k = k.contiguous()
        v = v.contiguous()
        cu_q_list = list(map(int, cu_q.tolist()))
        cu_k_list = list(map(int, cu_k.tolist())) if cu_k is not None else None
        cached_list = []
        for i in range(len(cu_q_list) - 1):
            n_q = cu_q_list[i + 1] - cu_q_list[i]
            n_k = (cu_k_list[i + 1] - cu_k_list[i]) if cu_k_list else n_q
            cached_list.append(n_k - n_q)                     # 来自 paged cache 的前缀长度

        if any(c > 0 for c in cached_list):
            return self._prefill_with_cache(q, k, v, block_tables, cu_q_list, cached_list)

        # cached == 0: 本次 k/v 就是全部历史, seqlen_q == seqlen_k, sdpa 的
        # is_causal 语义恰好正确, 直接走原生实现
        outs = []
        for i in range(len(cu_q_list) - 1):
            s, e = cu_q_list[i], cu_q_list[i + 1]
            qq = q[s:e].permute(1, 0, 2).unsqueeze(0)         # (1, H, seqlen, D)
            kk = k[s:e].permute(1, 0, 2).unsqueeze(0)
            vv = v[s:e].permute(1, 0, 2).unsqueeze(0)
            if H != Hk:
                kk = kk.repeat_interleave(H // Hk, dim=1)
                vv = vv.repeat_interleave(H // Hk, dim=1)
            out = torch.nn.functional.scaled_dot_product_attention(
                qq, kk, vv, is_causal=True, scale=self.scale)
            outs.append(out.permute(0, 2, 1, 3).reshape(e - s, H * D))
        return torch.cat(outs, dim=0)

    def _prefill_with_cache(self, q, k, v, block_tables, cu_q_list, cached_list):
        """prefill 且部分 KV 在 paged cache 里 -> 走 Triton kernel。

        每个 query 行需要三个额外信息 (所属序列 / 序列内下标 / 该序列的 cache 前缀长度),
        在 CPU 侧摊平成 tensor 传进去, kernel 里就不必再二分查找序列边界了。
        """
        dev = q.device
        row_seq, row_local, row_cached, row_base = [], [], [], []
        for i in range(len(cu_q_list) - 1):
            s, e = cu_q_list[i], cu_q_list[i + 1]
            for t in range(e - s):
                row_seq.append(i)
                row_local.append(t)
                row_cached.append(cached_list[i])
                row_base.append(s)

        def _t(xs):
            return torch.tensor(xs, dtype=torch.int32, device=dev)

        n_rows = q.shape[0]
        out = torch.empty_like(q)
        assert self.head_dim in (32, 64, 128, 256), "kernel 要求 head_dim 为 2 的幂"
        cached_causal_attn_kernel[(n_rows, self.num_heads)](
            q, k, v, self.k_cache, self.v_cache, block_tables, out,
            _t(row_seq), _t(row_local), _t(row_cached), _t(row_base),
            self.scale,
            q.stride(0), q.stride(1),
            k.stride(0), k.stride(1),
            self.k_cache.stride(0), self.k_cache.stride(1), self.k_cache.stride(2),
            self.v_cache.stride(0), self.v_cache.stride(1), self.v_cache.stride(2),
            block_tables.stride(0),
            out.stride(0), out.stride(1),
            HEAD_DIM=self.head_dim,
            BLOCK_SIZE=self.k_cache.shape[1],
            BLOCK_N=64,
            GQA=self.num_heads // self.num_kv_heads,
            num_warps=4,
        )
        return out.reshape(n_rows, self.num_heads * self.head_dim)

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

    def _decode_paged(self, q, k, v):
        """decode attention: q 是本次的新 token, 历史 K/V 在 paged cache 里
        (Attention.forward 已经先写好了当前 token 的 K/V)。

        用 flash-decoding 替换原来的 sdpa fallback。后者的致命伤是每个 batch 元素
        **每层**都要 int(ctx_lens[j]) 触发一次 GPU->CPU 同步 —— 实测把这个(只有 24 层的)
        0.5B draft 模型单次 forward 拖到 20ms, 比 28 层的 7B target (35ms) 只差一倍,
        投机解码自然不可能赚。这里沿 key 切成 N_SPLITS 段并行再归约, 全程无同步。
        """
        context = get_context()
        ctx_lens, bt = context.context_lens, context.block_tables
        B, dev = q.shape[0], q.device

        # ⚠ split 数只能由**静态形状**决定, 不能读 ctx_lens 的实际值:
        # 这里会被 CUDA graph 捕获 (draft 单步 forward 已 graph 化), 任何
        # GPU->CPU 同步都会让 capture 直接失败。key 比分段更短时, kernel 里的
        # 循环自然不执行, 多余的段会贡献 exp(-inf)=0, 结果仍然正确。
        n_sm = torch.cuda.get_device_properties(dev).multi_processor_count
        want = (4 * n_sm) // max(1, B * self.num_heads)
        n_splits = 1
        for cand in (16, 8, 4, 2, 1):
            if want >= cand:
                n_splits = cand
                break

        acc = torch.empty((B, self.num_heads, n_splits, self.head_dim),
                          dtype=torch.float32, device=dev)
        maxl = torch.empty((B, self.num_heads, n_splits, 2), dtype=torch.float32, device=dev)
        out = torch.empty_like(q)

        _decode_split_kernel[(B, self.num_heads, n_splits)](
            q, self.k_cache, self.v_cache, bt, ctx_lens, acc, maxl,
            self.scale,
            q.stride(0), q.stride(1),
            self.k_cache.stride(0), self.k_cache.stride(1), self.k_cache.stride(2),
            self.v_cache.stride(0), self.v_cache.stride(1), self.v_cache.stride(2),
            bt.stride(0),
            acc.stride(0), acc.stride(1), acc.stride(2),
            maxl.stride(0), maxl.stride(1), maxl.stride(2),
            N_SPLITS=n_splits, HEAD_DIM=self.head_dim,
            BLOCK_SIZE=self.k_cache.shape[1], BLOCK_N=64,
            GQA=self.num_heads // self.num_kv_heads,
            num_warps=4,
        )
        _decode_reduce_kernel[(B, self.num_heads)](
            acc, maxl, out,
            acc.stride(0), acc.stride(1), acc.stride(2),
            maxl.stride(0), maxl.stride(1), maxl.stride(2),
            out.stride(0), out.stride(1),
            N_SPLITS=n_splits, HEAD_DIM=self.head_dim, num_warps=4,
        )
        return out.reshape(B, self.num_heads * self.head_dim)

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
            if self.head_dim in (32, 64, 128, 256):
                return self._decode_paged(q, k, v)
            return self._sdpa_decode(q, k, v)
