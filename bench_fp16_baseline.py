"""fp16 基线: 用**原版 nano-vLLM**(/home/cdzk/WR/nano-vllm, 未加任何量化改动)
跑未量化的 Qwen2.5-7B-Instruct, 作为量化前的对照基准。

测量口径与 bench.py 完全一致(同样的 4 条 prompt / max_tokens / 并发数 / TTFT 与 Decode 定义),
因此可以直接和 GPTQ-Int4 的结果对比。

为什么要分开测 eager / CUDA Graph:
  我们的 GPTQ 路径被引擎强制 enforce_eager=True(动态反量化与 CUDA Graph 不兼容),
  而原版 fp16 默认开 CUDA Graph。只比"原版默认"会把"强制 eager 的代价"算进量化收益里。
  所以两个都测, 才能把"量化本身的收益"和"eager 惩罚"分开。

用法:
  python bench_fp16_baseline.py                 # 原版默认 (CUDA Graph)
  python bench_fp16_baseline.py --eager         # 强制 eager, 与 GPTQ 同条件
"""
import argparse
import os
import sys
from time import perf_counter

# 必须插到最前, 否则会 import 到本仓库(nano-vllm-ds)里那份改过的 nanovllm
ORIG_REPO = os.getenv("NANOVLLM_ORIG_REPO", "/home/cdzk/WR/nano-vllm")
sys.path.insert(0, ORIG_REPO)

import torch  # noqa: E402

DEFAULT_MODEL = "/nas_data/WR/qwen/models/Qwen2.5-7B-Instruct"

PROMPTS = [
    "Please explain the difference between TCP and UDP in detail.",
    "Write a Python function to compute the Fibonacci sequence recursively.",
    "What are the main causes of climate change and its impacts?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]
TEMPERATURE = 0.8


def weights_gib(llm) -> float:
    """模型权重实际占用的显存 (不含 KV cache —— 后者按剩余显存自动分配, 不可比)。"""
    m = getattr(getattr(llm, "model_runner", None), "model", None)
    if m is None:
        return float("nan")
    total = 0
    for mod in m.modules():
        for t in list(mod.parameters(recurse=False)) + list(mod.buffers(recurse=False)):
            total += t.numel() * t.element_size()
    return total / 2 ** 30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-tokens", type=int, default=128)
    ap.add_argument("--eager", action="store_true", help="强制 eager (与 GPTQ 同条件)")
    ap.add_argument("--gpu-mem-util", type=float, default=0.9,
                    help="显存占用比例; CUDA Graph 捕获失败时可调低")
    args = ap.parse_args()

    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    assert os.path.isdir(args.model), f"模型不存在: {args.model}"
    llm = LLM(args.model, tensor_parallel_size=1, max_num_batched_tokens=2048,
              max_num_seqs=4, max_model_len=1024, enforce_eager=args.eager,
              gpu_memory_utilization=args.gpu_mem_util)
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=args.max_tokens)
    llm.generate(["warm-up prompt"], sp, use_tqdm=False)

    ttfts, decodes = [], []
    for p in PROMPTS:
        llm.add_request(p, sp)
        t0 = perf_counter()
        llm.step()                       # 首个 step = prefill
        ttft = (perf_counter() - t0) * 1000
        n_decode = 0
        t_dec = perf_counter()
        while not llm.is_finished():
            _, nt = llm.step()
            if nt < 0:
                n_decode += -nt
        elapsed = perf_counter() - t_dec
        ttfts.append(ttft)
        if elapsed > 0 and n_decode > 0:
            decodes.append(n_decode / elapsed)

    t0 = perf_counter()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    total = sum(len(o["token_ids"]) for o in outs)
    tput = total / (perf_counter() - t0)

    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0

    tag = "强制 eager" if args.eager else "原版默认 (CUDA Graph)"
    print(f"=== 原版 nano-vLLM + fp16 未量化 [{tag}] ===")
    print(f"  model      : {args.model}")
    print(f"  TTFT(avg)  : {avg(ttfts):8.1f} ms")
    print(f"  Decode(avg): {avg(decodes):8.1f} tok/s (单请求)")
    print(f"  Throughput : {tput:8.1f} tok/s (4 请求并发)")
    print(f"  权重显存   : {weights_gib(llm):8.2f} GiB")
    print(f"  (权重显存不含 KV cache; 引擎来自 {ORIG_REPO})")


if __name__ == "__main__":
    main()
