"""单层 GEMM 微基准: 对比 GPTQ 四条路径在真实权重上的耗时 / 等效带宽。

    python tests/bench_layer.py

也顺便验证 stream 模式与 cache 模式**逐位一致** (喂给 cuBLAS 的是同一份 fp16 权重)。
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.gptq_linear import GPTQColumnParallelLinear
from nanovllm.layers.gptq_dequant import dequantize_gptq
from nanovllm.layers.gptq_triton import fused_gptq_linear
from tests.verify_gptq import load_real_layers, make_layer_from_packed, deq, DEV, GS

WARMUP, ITER = 5, 30


def bench(fn, *a):
    for _ in range(WARMUP):
        fn(*a)
    torch.cuda.synchronize()
    st, en = torch.cuda.Event(True), torch.cuda.Event(True)
    st.record()
    for _ in range(ITER):
        fn(*a)
    en.record()
    torch.cuda.synchronize()
    return st.elapsed_time(en) / ITER


def main():
    print(f"{'layer':<34}{'M':>5}{'cache(ms)':>11}{'stream(ms)':>11}{'fused(ms)':>11}"
          f"{'fused旧(ms)':>12}{'torch(ms)':>11}{'等效带宽GB/s':>14}")
    for base, t in load_real_layers(("q_proj", "gate_proj", "down_proj")).items():
        qw = t[".qweight"].to(DEV); qz = t[".qzeros"].to(DEV); sc = t[".scales"].to(DEV)
        K, N = qw.shape[0] * 8, qw.shape[1]
        layer = make_layer_from_packed(qw, qz, sc, N, K)
        W = deq(layer).half()                       # (N,K) fp16 —— cache 模式的权重
        buf = torch.empty(N * K, dtype=torch.float16, device=DEV)
        for M in (1, 4, 32, 256):
            torch.manual_seed(0)
            x = torch.randn(M, K, device=DEV).half()

            t_cache = bench(lambda: x @ W.t())
            t_stream = bench(lambda: x @ dequantize_gptq(qw, qz, sc, GS, out=buf).t())
            t_fused = bench(lambda: fused_gptq_linear(x, qw, qz, sc, M, N, K, GS))
            t_fused_old = bench(lambda: fused_gptq_linear(x, qw, qz, sc, M, N, K, GS,
                                                          block_m=32, block_n=64, block_k=128))
            t_torch = bench(lambda: layer.forward(x) if layer.mode == "torch" else None) \
                if layer.mode == "torch" else float("nan")

            # 权重字节数 + 读回的 fp16 字节数, 用来估等效带宽 (stream 路径)
            gb = (qw.numel() * 4 + N * K * 2 * 2) / 1e9
            print(f"{base:<34}{M:>5}{t_cache:>11.3f}{t_stream:>11.3f}{t_fused:>11.3f}"
                  f"{t_fused_old:>12.3f}{t_torch:>11.3f}{gb / (t_stream / 1e3):>14.1f}")

            # 一致性: stream 与 cache 必须逐位相同
            o1 = x @ W.t()
            o2 = x @ dequantize_gptq(qw, qz, sc, GS, out=buf).t()
            same = bool(torch.equal(o1, o2))
            o3 = fused_gptq_linear(x, qw, qz, sc, M, N, K, GS)
            ndiff = int((o3 != o1).sum())
            print(f"{'':<34}{'':>5}  stream==cache: {'是' if same else '否 ← BUG'}   "
                  f"fused vs cache 不一致元素: {ndiff}/{o1.numel()}")


if __name__ == "__main__":
    main()
