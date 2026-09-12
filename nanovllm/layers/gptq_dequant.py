from __future__ import annotations
"""把 GPTQ int4 权重快速反量化成 fp16 (N, K) —— 用于 "int4 常驻显存 + cuBLAS 计算" 的
streaming 模式: 权重始终以 int4 存放 (7B 约 3.5GB), 每次 forward 只在一个复用的
staging buffer 里临时展开成 fp16, 算完即弃, 因此:

    显存 = int4 权重 (0.5B/param) + 一个最大层大小的临时 buffer (7B 约 136MB)
    精度 = 与 cache 模式逐位相同 (喂给 cuBLAS 的是同一份 fp16 权重)
    速度 = 比 cache 模式多一次 dequant 的 HBM 写 + cuBLAS 读, 实测见 bench_layer.py

对比旧的三条路径:
    naive(torch 版 dequant)  慢: 中间产生 (in//8,out,8) int32 等大张量, 访存量是权重的 ~30 倍
    cache(dequant 一次缓存)   显存回到 fp16 (14GB), 等于放弃了量化的运行期收益
    fused(Triton dequant-GEMM) 不物化 fp16 但归约顺序与 cuBLAS 不同, decode 形状下
                               与 cuBLAS/vLLM 有 1~4 ulp 差异 (见 tests/verify_gptq.py)
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _dequant_gptq(qw_ptr, qz_ptr, sc_ptr, out_ptr,
                  N, K, GS: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    """每个 program 负责 (BLOCK_N 个输出通道) x (BLOCK_K 个输入维) 的一块。
    BLOCK_K 取 GS(128) 时整块同组, 零点/scale 只需 (BLOCK_N,) 两个向量。
    输出布局 (N, K) —— 与 cache 模式完全一致, 因此 `x @ W.t()` 的 cuBLAS 调用逐位相同。"""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)      # 行: 输出通道
    k0 = pid_k * BLOCK_K
    offs_k = k0 + tl.arange(0, BLOCK_K)                   # 列: 输入维
    msk_n = offs_n < N
    msk_k = offs_k < K

    # 码字: qweight 是 (K//8, N), 行 r = k//8, nibble 位移 (k%8)*4
    r = offs_k // 8
    sub = (offs_k % 8) * 4
    qw = tl.load(qw_ptr + r[None, :] * N + offs_n[:, None],
                 mask=msk_n[:, None] & msk_k[None, :], other=0).to(tl.uint32)
    codes = (qw >> sub[None, :]) & 0xF                    # (BLOCK_N, BLOCK_K)

    # 零点/scale: 整块同一组 g (BLOCK_K == GS)
    g = k0 // GS
    qz = tl.load(qz_ptr + g * (N // 8) + offs_n // 8, mask=msk_n, other=0).to(tl.uint32)
    z = ((qz >> ((offs_n % 8) * 4)) & 0xF).to(tl.float32) + 1.0     # (BLOCK_N,)
    s = tl.load(sc_ptr + g * N + offs_n, mask=msk_n, other=0.0).to(tl.float32)

    W = ((codes.to(tl.float32) - z[:, None]) * s[:, None]).to(tl.float16)
    tl.store(out_ptr + offs_n[:, None] * K + offs_k[None, :], W,
             mask=msk_n[:, None] & msk_k[None, :])


_SCRATCH: dict[tuple, torch.Tensor] = {}


def scratch_buffer(numel: int, device) -> torch.Tensor:
    """全局复用的一块 fp16 暂存区 (按元素个数向上取到已有的规格), 避免每层都分配。
    同一 CUDA stream 上顺序使用, 上一层算完之后才会被下一层覆写, 因此安全。"""
    buf = _SCRATCH.get(device)
    if buf is None or buf.numel() < numel:
        buf = torch.empty(max(numel, 1 << 20), dtype=torch.float16, device=device)
        _SCRATCH[device] = buf
    return buf


def dequantize_gptq(qweight: torch.Tensor, qzeros: torch.Tensor, scales: torch.Tensor,
                    group_size: int = 128, out: torch.Tensor | None = None,
                    block_n: int = 64, block_k: int | None = None, num_warps: int = 4):
    """qweight (K//8, N) int32 / qzeros (K//gs, N//8) int32 / scales (K//gs, N) fp16
    -> W (N, K) fp16。"""
    K, N = qweight.shape[0] * 8, qweight.shape[1]
    if out is None:
        out = torch.empty((N, K), dtype=torch.float16, device=qweight.device)
    else:
        out = out[: N * K].view(N, K)
    bk = block_k or group_size            # 默认一块 = 一个 group, 组内零点/scale 常数
    grid = (triton.cdiv(N, block_n), triton.cdiv(K, bk))
    _dequant_gptq[grid](qweight, qzeros, scales, out, N, K, group_size,
                        BLOCK_N=block_n, BLOCK_K=bk, num_warps=num_warps, num_stages=2)
    return out
