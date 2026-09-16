"""单层 GEMM 微基准: 对比 GPTQ 两条路径 (fused / torch) 在真实权重上的耗时 / 等效带宽。

    python tests/bench_layer.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.gptq_linear import GPTQColumnParallelLinear
from nanovllm.layers.gptq_triton import fused_gptq_linear
from tests.verify_gpt import load_real_layers, make_layer_from_packed, deq, DEV, GS

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
    print(f"{'layer':<34}{'M':>5}{'fused(ms)':>11}{'fused旧(ms)':>12}{'torch(ms)':>11}"
          f"{'等效带宽GB/s':>14}")
    for base, t in load_real_layers(("q_proj", "gate_proj", "down_proj")).items():
        qw = t[".qweight"].to(DEV); qz = t[".qzeros"].to(DEV); sc = t[".scales"].to(DEV)
        K, N = qw.shape[0] * 8, qw.shape[1]
        layer = make_layer_from_packed(qw, qz, sc, N, K)
        for M in (1, 4, 32, 256):
            torch.manual_seed(0)
            x = torch.randn(M, K, device=DEV).half()

            t_fused = bench(lambda: fused_gptq_linear(x, qw, qz, sc, M, N, K, GS))
            t_fused_old = bench(lambda: fused_gptq_linear(x, qw, qz, sc, M, N, K, GS,
                                                          block_m=32, block_n=64, block_k=128))
            t_torch = bench(lambda: layer.forward(x) if layer.mode == "torch" else None) \
                if layer.mode == "torch" else float("nan")

            # int4 权重字节数 + 读回的 fp16 字节数, 用来估等效带宽 (dequant 后喂 cuBLAS)
            gb = (qw.numel() * 4 + N * K * 2 * 2) / 1e9
            print(f"{base:<34}{M:>5}{t_fused:>11.3f}{t_fused_old:>12.3f}{t_torch:>11.3f}"
                  f"{gb / (t_fused / 1e3):>14.1f}")

            # 一致性: torch (dequant 后 matmul) 与 fused 必须逐位相同
            o1 = deq(layer).half().to(DEV)
            o3 = fused_gptq_linear(x, qw, qz, sc, M, N, K, GS)
            ndiff = int((o3 != x @ o1.t()).sum())
            print(f"{'':<34}{'':>5}  fused vs torch 不一致元素: {ndiff}/{o1.numel()}")


if __name__ == "__main__":
    main()
