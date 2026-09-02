from __future__ import annotations
import torch
import triton
import triton.language as tl

# Fused int4 dequant-GEMM (GPTQ symmetric, group_size=128).
# 关键: 权重以 int4 驻留显存, 在 kernel 内解包 + 反量化成 fp16 后才进入 tl.dot,
# 因此 fp16 权重从不物化到 HBM —— 省掉一次权重读回, 这正是 Marlin 比 naive
# "反量化一次缓存 fp16" 快的根本原因 (naive 把 fp16 权重写回显存再读给 GEMM)。
#
# 布局约定 (与 HF GPTQ / gptq_linear.py 一致):
#   qweight: (K//8,   N) int32   每 int32 打包 8 个 4-bit 码字 (沿输入维 K)
#   qzeros : (K//gs,  N//8) int32 每 int32 打包 8 个 4-bit 零点 (沿输出维 N)
#   scales : (K//gs,  N) fp16
#   dequant: W[n,k] = (Q[n,k] - (unpack(qzeros)[n, g] + 1)) * scales[g, n], g = k // gs
#
# 实现要点 (踩坑记录, 全部经单测/真实权重验证):
#   * qweight/qzeros 一律先 .to(tl.uint32) 再做逻辑右移, 否则真实权重高位为 1 时被存成
#     "负数 int32", 有符号 >> 的符号扩展会与 torch 的 &0xF 不一致。
#   * 解包绝不用 tl.cat / tl.permute / tl.reshape 重组 (BK//8,BN,8)->(BK,BN):
#       - tl.cat 在新版 Triton 已移除, 且历史上会重排元素 ("always may reorder")；
#       - tl.reshape 不是行主序重解释, 实测会把 r*8+sub 错位, 整张权重错位 (max 达 7/15)。
#     改用"指针算术逐元素解包": r = k//8, sub = k%8, 直接 qw[pid_k*r + n] >> (sub*4)&0xF,
#     从 (BK,BN) 一步得到, 经 debug_w 验证与 torch 参考逐元素 max 0.0。
#   * 反量化在 fp32 下做 ((code - z)*s), 仅在喂 tl.dot 前 .half(): 否则 fp16 下 (code-z)*s
#     的舍入会随 K 增大累积 (K=18944 时 max diff 达 0.06)。
#   * 输入支持任意前导维度 (_forward_fused 内 reshape 到 2D), 引擎 prefill 时可能传 (B,S,in)。
#   * 已知局限: Triton tl.dot 的 K 维归约顺序与 cuBLAS 不同, 单层 ~1~2 ulp 误差; 在 28 层
#     残差流里逐层放大 (注意力 softmax 对微小 logit 差极敏感), 贪心解码会与 vLLM 分歧。
#     精确贪心匹配请用 cache 模式 (cuBLAS)。详见教程 5.5。

@triton.jit
def _fused_gptq_mm(
    x_ptr, qw_ptr, qz_ptr, sc_ptr, out_ptr,
    M, N, K, GS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    msk_m = offs_m[:, None] < M
    msk_n = offs_n[None, :] < N

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    n_k = tl.cdiv(K, BLOCK_K)
    for kk in range(n_k):
        k0 = kk * BLOCK_K                              # K 起点 (= group 起点, 因 BLOCK_K=GS)
        offs_k = k0 + tl.arange(0, BLOCK_K)            # (BLOCK_K,) 输入维绝对索引

        # 激活 (BLOCK_M, BLOCK_K)
        x = tl.load(x_ptr + (offs_m[:, None] * K + offs_k[None, :]),
                    mask=msk_m & (offs_k[None, :] < K), other=0.0).to(tl.float16)

        # 零点 (每组 g=k0//GS, 沿 N 每 8 输出打包一个 int32): 解包 + 1
        g = k0 // GS
        n_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)        # (BLOCK_N,) 输出维绝对索引
        qz = tl.load(qz_ptr + (g * (N // 8) + n_offs // 8),
                     mask=n_offs < N, other=0).to(tl.uint32)
        z = ((qz >> ((n_offs % 8) * 4)) & 0xF).to(tl.float16) + 1.0   # (BLOCK_N,)

        # scale (K//gs, N)
        s = tl.load(sc_ptr + (g * N + offs_n), mask=offs_n < N, other=0.0).to(tl.float16)

        # 权重码字 (BLOCK_K, BLOCK_N): 输入行 k 对应 qweight 行 k//8, 位移 (k%8)*4
        # —— 用指针算术逐元素解包 (不建 3D block), 彻底避开 tl.cat/tl.reshape/tl.permute ——
        r = offs_k // 8                                 # (BLOCK_K,) int32 行
        sub = offs_k % 8                               # (BLOCK_K,) nibble 位移
        qw = tl.load(qw_ptr + (r[:, None] * N + offs_n[None, :]),
                     mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0).to(tl.uint32)
        w_codes = (qw >> (sub[:, None] * 4)) & 0xF      # (BLOCK_K, BLOCK_N)
        # 反量化在 fp32 下做 (与 _dequant_block 口径一致), 仅在喂 TensorCore 前 .half(),
        # 否则 fp16 下 (code-z)*s 的舍入会随 K 增大累积 (K=18944 时 max diff 达 0.06)。
        W = ((w_codes.to(tl.float32) - z[None, :].to(tl.float32)) *
             s[None, :].to(tl.float32)).to(tl.float16)          # (BLOCK_K, BLOCK_N)
        acc += tl.dot(x, W)                            # (BLOCK_M, BLOCK_N), K=BLOCK_K>=16

    tl.store(out_ptr + offs_m[:, None] * N + offs_n[None, :],
             acc.to(tl.float16), mask=msk_m & msk_n)


def fused_gptq_linear(x: torch.Tensor, qweight: torch.Tensor, qzeros: torch.Tensor,
                      scales: torch.Tensor, M: int, N: int, K: int, GS: int = 128):
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 64))
    _fused_gptq_mm[grid](
        x, qweight, qzeros, scales, out,
        M, N, K, GS,
        BLOCK_M=32, BLOCK_N=64, BLOCK_K=128,
    )
    return out
