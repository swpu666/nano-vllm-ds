from __future__ import annotations
import os
import torch
import triton
import triton.language as tl

# Fused int4 dequant-GEMM (GPTQ symmetric, group_size=128).
#
# 布局约定 (与 HF GPTQ / gptq_linear.py 一致):
#   qweight: (K//8,   N) int32   每 int32 打包 8 个 4-bit 码字 (沿输入维 K)
#   qzeros : (K//gs,  N//8) int32 每 int32 打包 8 个 4-bit 零点 (沿输出维 N)
#   scales : (K//gs,  N) fp16
#   dequant: W[n,k] = (Q[n,k] - (unpack(qzeros)[n, g] + 1)) * scales[g, n], g = k // gs
#
# 与旧版相比修正了两处实质问题:
#   1) 组号 g 由"块起点 k0//GS"改为"逐元素 offs_k//GS"。旧写法只在 BLOCK_K == GS 时正确,
#      因此历史上 "把 BLOCK_K 从 128 改到 256/1024 误差不变" 的实验是在 kernel 本身算错的
#      前提下得到的, 结论无效。现在 BLOCK_K 可以任意取 (不必等于/对齐 GS)。
#   2) 分块策略按 M 自适应: decode(M<=16) 时旧配置 (BM=32,BN=64) 只有 N/64 个 CTA,
#      q_proj 仅 56 个 < 82 个 SM, 硬件三分之二空转 —— 这是 fused 在 decode 上打不过
#      cuBLAS 的主要原因。现在小 M 用 BN=32/64 提高占用。
#
# 数值特征 (实测, 见 tests/verify_gptq.py Part B/B2):
#   * M>=32 (prefill 类形状): 与 cuBLAS 的 fp16 **输出** 逐位一致。
#   * M<=4 (decode 类形状): cuBLAS 会切到 gemv 类实现, 归约顺序与 tl.dot 不同, 落到 fp16
#     输出上有少量元素差 1~4 ulp。
#   * 这点差异**不足以**让贪心解码分歧: 纯 tl.dot 的 fused 在 Part C 实测与
#     vLLM(gptq_marlin) 64/64 一致。


@triton.jit
def _fused_gptq_mm(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, out_ptr,
    M, N, K, GS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """Fused int4 dequant-GEMM kernel (GPTQ symmetric, group_size=GS).

    一次性完成「解包 int4 码字 -> 反量化到 fp16 -> GEMM」, fp16 权重完全不在显存/
    共享内存里物化 (int4 常驻, kernel 内按需解包后直接喂 TensorCore)。
    计算: out[M, N] = x[M, K] @ W[N, K].T, 其中 W 由 qweight/qzeros/scales 反量化得到。

    网格划分: 2D grid, (pid_m, pid_n) 各负责输出矩阵的一个 (BLOCK_M, BLOCK_N) 块,
    即把 C[M,N] 沿行/列切成 BLOCK_M×BLOCK_N 的瓦片, 每个 CTA 算一块。
    K 维在 CTA 内部用 BLOCK_K 循环归约 (BLOCK_K 不必等于 GS)。
    """
    # ---- 1. CTA 负责的输出块坐标 ----
    pid_m = tl.program_id(0)                               # 输出行块索引
    pid_n = tl.program_id(1)                               # 输出列块索引
    # 本 CTA 负责的 (全局) 行/列下标 (含可能越界的部分, 越界由 mask 屏蔽)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)       # (BLOCK_M,) 输出行
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)       # (BLOCK_N,) 输出列
    msk_m = offs_m[:, None] < M                            # (BM,1) 行越界屏蔽
    msk_n = offs_n[None, :] < N                            # (1,BN) 列越界屏蔽

    # ---- 2. qzeros 的 "沿 N 打包" 索引预计算 ----
    # qzeros 形状 (K//GS, N//8): 沿输出维 N 每 8 个 4-bit 零点塞进一个 int32。
    # 对某个输出下标 n, 它属于第 n//8 个 int32 字, 字内偏移 (n%8)*4 位。
    n_div = offs_n // 8                                    # (BLOCK_N,) 字内第几个 int32
    n_sub = (offs_n % 8) * 4                               # (BLOCK_N,) 字内位移(比特)

    # ---- 3. 累加器, fp32 保精度 ----
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ---- 4. K 维 (归约维) 分块循环 ----
    for kk in range(tl.cdiv(K, BLOCK_K)):
        k0 = kk * BLOCK_K                                  # 本块在 K 维的起点
        offs_k = k0 + tl.arange(0, BLOCK_K)               # (BLOCK_K,) 本块覆盖的 K 下标
        k_msk = offs_k < K                                 # (BLOCK_K,) K 越界屏蔽
        # km: 权重码字加载掩码, 同时屏蔽 "K 越界" 与 "输出列越界"
        km = k_msk[:, None] & msk_n                        # (BLOCK_K, BLOCK_N)

        # 4.1 加载激活 (BLOCK_M, BLOCK_K)
        # x 行主序: x[m, k] = x_ptr[m*K + k]。越界处填 0.0 不影响结果。
        x = tl.load(x_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                    mask=msk_m & k_msk[None, :], other=0.0).to(tl.float16)

        # 4.2 解析本块对应的量化组号 g = k // GS, 并加载该组的零点/缩放
        # 关键修正点: 组号必须按 "逐元素 k 下标" 计算, 不能按块起点 k0 算。
        #   - BLOCK_K <= GS 时, 整块落在同一个组里 (g 全相同), 只需取 1 个组号 ->
        #     z/s 形状 (1, BLOCK_N), 省共享内存。
        #   - BLOCK_K >  GS 时, 块内跨多个组, 必须逐元素算 (BLOCK_K,) 组号。
        #   旧版写死 g = k0//GS, 仅在 BLOCK_K == GS 时碰巧正确, 否则整块组号错。
        if BLOCK_K <= GS:
            g_idx = tl.full((1,), k0 // GS, tl.int32)       # (1,) 整块同组
        else:
            g_idx = offs_k // GS                           # (BLOCK_K,) 逐元素组号
        # gm: 零点/缩放加载掩码 (与 g_idx 同形)。g 恒 >= 0, 这里等价于 msk_n 广播。
        gm = (g_idx[:, None] >= 0) & msk_n                 # (G, BLOCK_N)

        # 零点: 从 qzeros[g_idx, n_div] 取出 int32 字, 右移 n_sub 位取低 4 位,
        # 再加 1 得到真实零点 (GPTQ v1 约定 z_true = unpack(qzeros) + 1)。
        qz = tl.load(qz_ptr + g_idx[:, None] * (N // 8) + n_div[None, :], mask=gm, other=0).to(tl.uint32)
        z = ((qz >> n_sub[None, :]) & 0xF).to(tl.float32) + 1.0        # (1|BLOCK_K, BLOCK_N)
        # 缩放: 直接从 scales[g_idx, n] 取, 形状 (1|BLOCK_K, BLOCK_N)。
        s = tl.load(sc_ptr + g_idx[:, None] * N + offs_n[None, :], mask=gm, other=0.0).to(tl.float32)

        # 4.3 解包权重 int4 码字 (BLOCK_K, BLOCK_N)
        # qweight 形状 (K//8, N): 沿 K 维每 8 个 4-bit 码字塞进一个 int32。
        # 对 K 下标 k: 它落在第 r=k//8 个 int32 字, 字内位移 sub=(k%8)*4 位。
        # 用指针算术逐元素解包 (r[:,None]*N + offs_n[None,:]), 不开 tl.cat/reshape/
        # permute 以保证 Triton 能正确推断内存布局 (见文件头说明)。
        r = offs_k // 8                                    # (BLOCK_K,) 字行号
        sub = (offs_k % 8) * 4                             # (BLOCK_K,) 字内位移(比特)
        qw = tl.load(qw_ptr + (r[:, None] * N + offs_n[None, :]), mask=km, other=0).to(tl.uint32)
        codes = (qw >> sub[:, None]) & 0xF                              # (BLOCK_K, BLOCK_N) 0..15

        # 4.4 反量化 + GEMM (核心: 解包与 GEMM 在 kernel 内融合)
        # 反量化全程在 fp32 下做: W = (codes - z) * s, 只在喂 TensorCore 前转 fp16,
        # 这样 fp16 权重不落显存、不落共享内存, 省显存又省一次物化。
        # fp16*fp16 的乘积在 fp32 下可精确表示 (22 位尾数 < 24 位), 故转 fp16 精度损失极小。
        W = ((codes.to(tl.float32) - z) * s).to(tl.float16)
        acc += tl.dot(x, W)                               # (BM,BN) fp32 累加, TensorCore 执行

    # ---- 5. 写回输出 ----
    # acc(fp32) -> fp16, 按 (BM,BN) 块写回, 同时屏蔽行/列越界。
    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.float16), mask=msk_m & msk_n)


@triton.jit
def _ordered_gptq_mm(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, out_ptr,
    M, N, K, GS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    """ordered 变体: dequant 后升到 fp32 做**精确**乘加 (input_precision="ieee")。

    与 _fused_gptq_mm 的区别只在最后一步:
      fused   : tl.dot(fp16 x, fp16 W)            -> tensor core, 归约不可控
      ordered : tl.dot(fp32 x, fp32 W, ieee)      -> 乘积精确 + fp32 FMA 归约

    实测依据:
      * fp16*fp16 的乘积在 fp32 下可精确表示 (11+11=22 位 < 24 位), 所以把乘加放到
        fp32 后误差只剩归约舍入 (~1e-6 相对), 比 fp16 输出的 1 ulp (~5e-4) 小三个
        数量级 -> 任何归约顺序都舍入到同一个 fp16 值。离线扫描 14 种归约顺序
        (顺序/逆序/树形/各种分块) 结果确实完全一致, 证实归约顺序不是误差来源。
      * 反直觉但被实验证实: "更准"并不会与 vLLM 失配 (端到端探针 64/64)。
      * 本核**非必需**: 纯 tl.dot 的 fused 已 64/64 对齐; 而本核没有 tensor core、
        明显更慢, 故默认关闭 (ORDERED_MAX_M=0), 仅作为"fp32 精确累加"的数值参考实现保留。
    """
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    msk_m = offs_m[:, None] < M
    msk_n = offs_n[None, :] < N
    n_div = offs_n // 8
    n_sub = (offs_n % 8) * 4

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for kk in range(tl.cdiv(K, BLOCK_K)):
        k0 = kk * BLOCK_K
        offs_k = k0 + tl.arange(0, BLOCK_K)
        k_msk = offs_k < K
        km = k_msk[:, None] & msk_n

        x = tl.load(x_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                    mask=msk_m & k_msk[None, :], other=0.0)

        if BLOCK_K <= GS:
            g_idx = tl.full((1,), k0 // GS, tl.int32)
        else:
            g_idx = offs_k // GS
        gm = (g_idx[:, None] >= 0) & msk_n
        qz = tl.load(qz_ptr + g_idx[:, None] * (N // 8) + n_div[None, :], mask=gm, other=0).to(tl.uint32)
        z = ((qz >> n_sub[None, :]) & 0xF).to(tl.float32) + 1.0
        s = tl.load(sc_ptr + g_idx[:, None] * N + offs_n[None, :], mask=gm, other=0.0).to(tl.float32)

        r = offs_k // 8
        sub = (offs_k % 8) * 4
        qw = tl.load(qw_ptr + (r[:, None] * N + offs_n[None, :]), mask=km, other=0).to(tl.uint32)
        codes = (qw >> sub[:, None]) & 0xF

        # 先反量化到 fp16 —— 与 cache/stream 喂给 cuBLAS 的权重逐位相同;
        # 再升 fp32, 使乘积精确、归约在 fp32 下可控。
        W = ((codes.to(tl.float32) - z) * s).to(tl.float16).to(tl.float32)

        acc += tl.dot(x.to(tl.float32), W, input_precision="ieee")

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.float16), mask=msk_m & msk_n)


def _pick_config_ordered(M: int, N: int):
    """ordered(fp32) 核的分块。fp32 操作数是 fused 的 2 倍访存/共享内存, 块要更保守。"""
    if M <= 8:
        bm, bn, bk, ns = 16, 64, 32, 2
    else:
        bm, bn, bk, ns = 32, 64, 32, 2
    if N < bn:
        bn = max(triton.next_power_of_2(N), 16)
    return bm, bn, bk, ns


def _pick_config(M: int, N: int):
    """按形状挑分块 (在 RTX 3090 上实测扫描得到, 见 tests/bench_layer.py):

      * BLOCK_K=32 明显好于 128: qweight 的 int32 tile 是 (BLOCK_K, BLOCK_N)*4B,
        这是共享内存大头, 调小后既能塞下更大的 BLOCK_N, 流水线也更容易排。
      * decode(M<=8) 用 BM=16/BN=64: 关键不是单块效率, 而是 CTA 数量要够填满 82 个 SM。
        旧配置 BM=32/BN=64 在 q_proj 上只有 56 个 CTA, 三分之一硬件空转。
      * prefill(M>=128) 用 BM=128/BN=128 提高计算复用。
    """
    if M <= 8:
        bm, bn, bk, ns = 16, 64, 32, 3
    elif M <= 32:
        bm, bn, bk, ns = 32, 64, 64, 3
    elif M <= 64:
        bm, bn, bk, ns = 64, 64, 32, 3
    elif M <= 256:
        bm, bn, bk, ns = 128, 64, 32, 3
    else:
        bm, bn, bk, ns = 128, 128, 32, 2
    if N < bn:                     # 输出维比一块还小 (如 k/v_proj 的 N=512 也够, 保险起见)
        bn = max(triton.next_power_of_2(N), 16)
    return bm, bn, bk, ns


def fused_gptq_linear(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
                      scales: torch.Tensor, M: int, N: int, K: int, GS: int = 128,
                      block_m: int | None = None, block_n: int | None = None,
                      block_k: int | None = None, num_warps: int = 4, num_stages: int | None = None):
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    bmc, bnc, bkc, nsc = _pick_config(M, N)
    bm = block_m or bmc
    bn = block_n or bnc
    bk = block_k or bkc
    ns = num_stages if num_stages is not None else nsc
    # 共享内存不够时逐级降级, 避免直接 OutOfResources
    candidates = [(bm, bn, bk, ns)]
    if block_m is None and block_n is None and block_k is None and num_stages is None:
        candidates += [(bm, max(bn // 2, 32), 32, 2), (64, 64, 32, 2), (32, 32, 32, 2)]
    last = None
    for cm, cn, ck, cns in candidates:
        try:
            grid = (triton.cdiv(M, cm), triton.cdiv(N, cn))
            _fused_gptq_mm[grid](x, qweight, qzeros, scales, out, M, N, K, GS,
                                 BLOCK_M=cm, BLOCK_N=cn, BLOCK_K=ck,
                                 num_warps=num_warps, num_stages=cns)
            return out
        except Exception as e:                                   # OutOfResources 等
            last = e
    raise last


# M <= ORDERED_MAX_M 时走 ordered(fp32 精确) 核, 否则走 tl.dot(fp16)。
#
# 默认 **0 = 关闭**: 已实测纯 tl.dot 就是 64/64 对齐的, 而 ordered 没有 tensor core、
# 明显更慢, 所以默认全部走 tl.dot。保留该开关纯作对照/调试用
# (置为一个大数可强制全程走 fp32 精确累加)。
ORDERED_MAX_M = int(os.getenv("NANOVLLM_GPTQ_ORDERED_MAX_M", "0"))


def ordered_gptq_linear(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
                        scales: torch.Tensor, M: int, N: int, K: int, GS: int = 128,
                        block_m: int | None = None, block_n: int | None = None,
                        block_k: int | None = None, num_warps: int = 4,
                        num_stages: int | None = None):
    """精确变体: dequant 后升 fp32 乘加, 不物化 fp16 权重 (显存与 fused 同为 0.5B/param)。"""
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    bmc, bnc, bkc, nsc = _pick_config_ordered(M, N)
    bm = block_m or bmc
    bn = block_n or bnc
    bk = block_k or bkc
    ns = num_stages if num_stages is not None else nsc
    candidates = [(bm, bn, bk, ns)]
    if block_m is None and block_n is None and block_k is None and num_stages is None:
        candidates += [(16, max(bn // 2, 32), 32, 2), (16, 32, 32, 2)]
    last = None
    for cm, cn, ck, cns in candidates:
        try:
            grid = (triton.cdiv(M, cm), triton.cdiv(N, cn))
            _ordered_gptq_mm[grid](x, qweight, qzeros, scales, out, M, N, K, GS,
                                   BLOCK_M=cm, BLOCK_N=cn, BLOCK_K=ck,
                                   num_warps=num_warps, num_stages=cns)
            return out
        except Exception as e:
            last = e
    raise last
