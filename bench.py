import time
import argparse

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams


def bench_nanovllm(model, prompts, max_tokens):
    """用 add_request + step 精确测量 TTFT 与 decode 吞吐。"""
    llm = LLM(model, tensor_parallel_size=1, max_num_batched_tokens=256,
              max_num_seqs=4, max_model_len=1024)
    sp = SamplingParams(temperature=0.8, max_tokens=max_tokens)
    # 预热一次, 避免首次 CUDA 初始化影响计时
    llm.generate(["warm-up prompt"], sp, use_tqdm=False)

    ttfts, decode_tps, outputs = [], [], []
    for p in prompts:
        t_req = time.perf_counter()
        llm.add_request(p, sp)
        ttft = None
        decode_start = None
        decode_num = 0
        out_ids = []
        while not llm.is_finished():
            outputs_step, num_tokens = llm.step()
            if num_tokens > 0 and ttft is None:
                ttft = time.perf_counter() - t_req       # prefill 完成即首 token
                decode_start = time.perf_counter()
            elif num_tokens < 0:
                decode_num += -num_tokens
            for seq_id, token_ids in outputs_step:
                out_ids = token_ids
        outputs.append(out_ids)
        ttfts.append(ttft * 1000 if ttft else 0.0)
        if decode_num > 1 and decode_start:
            decode_tps.append((decode_num - 1) / (time.perf_counter() - decode_start))
    del llm
    return outputs, ttfts, decode_tps


def bench_vllm(model, prompts, max_tokens):
    from vllm import LLM as VLLM
    from vllm import SamplingParams as VSP
    llm = VLLM(model=model, quantization="gptq", dtype="half",
               gpu_memory_utilization=0.92, enforce_eager=True,
               max_model_len=1024, max_num_seqs=4)
    vsp = VSP(temperature=0.8, top_p=0.95, max_tokens=max_tokens)
    ttfts, decode_tps, outputs = [], [], []
    for p in prompts:
        t_req = time.perf_counter()
        # 单请求测量 TTFT
        ttft = None
        gen = llm.generate([p], vsp, use_tqdm=False)
        # vLLM 不直接暴露 TTFT, 用整体耗时近似(单请求下 ≈ prefill+decode)
        ttft = time.perf_counter() - t_req
        out_ids = gen[0].outputs[0].token_ids
        outputs.append(out_ids)
        ttfts.append(ttft * 1000)
    # decode 吞吐: 所有输出 token / 总耗时
    total_tokens = sum(len(o) for o in outputs)
    t0 = time.perf_counter()
    llm.generate(prompts, vsp, use_tqdm=False)
    decode_tps.append(total_tokens / (time.perf_counter() - t0))
    return outputs, ttfts, decode_tps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="/nas_data/WR/models/DeepSeek-R1-Distill-Qwen-32B-GPTQ-Int4")
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--engine", default="nanovllm", choices=["nanovllm", "vllm", "both"])
    args = parser.parse_args()

    prompts = [
        "Please explain the difference between TCP and UDP in detail.",
        "Write a Python function to compute the Fibonacci sequence recursively.",
        "What are the main causes of climate change and its impacts?",
        "Summarize the plot of Romeo and Juliet in three sentences.",
    ]

    if args.engine in ("nanovllm", "both"):
        print("=== nano-vllm-ds (GPTQ naive dequant, TP=1) ===")
        outputs, ttfts, decode_tps = bench_nanovllm(args.model, prompts, args.max_tokens)
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(decode_tps) / len(decode_tps) if decode_tps else 0.0
        for p, o in zip(prompts, outputs):
            print(f"  [prompt] {p[:40]!r}\n    -> {o[:40]}")
        print(f"  Avg TTFT: {avg_ttft:.1f} ms | Avg decode: {avg_tps:.1f} tok/s")

    if args.engine in ("vllm", "both"):
        print("=== vLLM (GPTQ Marlin) ===")
        outputs, ttfts, decode_tps = bench_vllm(args.model, prompts, args.max_tokens)
        avg_ttft = sum(ttfts) / len(ttfts)
        avg_tps = sum(decode_tps) / len(decode_tps) if decode_tps else 0.0
        print(f"  Avg TTFT: {avg_ttft:.1f} ms | Avg decode: {avg_tps:.1f} tok/s")


if __name__ == "__main__":
    main()
