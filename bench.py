"""
GPTQ 部署性能对比: nano-vllm-ds (朴素 dequant)  vs  vLLM (gptq_marlin)

指标定义(两引擎口径一致, 保证可比):
  TTFT      : 单请求从提交到产出首个 token 的时延 (ms)
  Decode    : 单请求在首 token 之后的解码速度 (tok/s)
  Throughput: 多请求并发时的总生成吞吐 (tok/s), 体现调度/批处理能力
"""
import os
import argparse
from time import perf_counter

DEFAULT_MODEL = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"

PROMPTS = [
    "Please explain the difference between TCP and UDP in detail.",
    "Write a Python function to compute the Fibonacci sequence recursively.",
    "What are the main causes of climate change and its impacts?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]

# nano-vLLM 的 SamplingParams 禁止 greedy, 用小 temperature 近似确定性采样
TEMPERATURE = 0.8


def _stats(name, ttfts, decodes, throughput=None, extra=""):
    def avg(xs):
        return sum(xs) / len(xs) if xs else 0.0
    line = f"  TTFT(avg)  : {avg(ttfts):8.1f} ms"
    if ttfts:
        line += f"   [min {min(ttfts):.1f} / max {max(ttfts):.1f}]"
    print(line)
    if decodes:
        print(f"  Decode(avg): {avg(decodes):8.1f} tok/s (单请求)")
    if throughput:
        print(f"  Throughput : {throughput:8.1f} tok/s ({len(PROMPTS)} 请求并发) {extra}")
    return avg(ttfts), avg(decodes), throughput


def bench_nanovllm(model, max_tokens):
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    llm = LLM(model, tensor_parallel_size=1, max_num_batched_tokens=2048,
              max_num_seqs=4, max_model_len=1024)
    sp = SamplingParams(temperature=TEMPERATURE, max_tokens=max_tokens)
    # 预热, 消除首次 CUDA/显存分配的影响
    llm.generate(["warm-up prompt"], sp, use_tqdm=False)

    ttfts, decodes = [], []
    for p in PROMPTS:
        llm.add_request(p, sp)
        t0 = perf_counter()
        _, num_tokens = llm.step()          # 首个 step = prefill, 产出首 token
        ttft = (perf_counter() - t0) * 1000
        n_decode = 0
        t_dec = perf_counter()
        while not llm.is_finished():
            _, nt = llm.step()
            if nt < 0:
                n_decode += -nt             # decode: num_tokens = -len(seqs)
        elapsed = perf_counter() - t_dec
        ttfts.append(ttft)
        if elapsed > 0 and n_decode > 0:
            decodes.append(n_decode / elapsed)

    # 并发吞吐
    t0 = perf_counter()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    total = sum(len(o["token_ids"]) for o in outs)
    tput = total / (perf_counter() - t0)

    mode = "fused int4 dequant-GEMM (不物化 fp16 权重)" if os.getenv("NANOVLLM_GPTQ_FUSED") == "1" \
        else ("权重缓存(fp16, dequant 一次)" if os.getenv("NANOVLLM_GPTQ_CACHE") == "1" \
        else "朴素 on-the-fly dequant(每次 forward 重算)")
    print(f"=== nano-vllm-ds (GPTQ, TP=1) [{mode}] ===")
    _stats("nanovllm", ttfts, decodes, tput)
    del llm
    return ttfts, decodes, tput


def _vllm_ttft(llm, vsp_cls, prompt):
    """单请求 TTFT: 优先用 vLLM 内建 metrics, 否则退化为 max_tokens=1 计时。"""
    from vllm import SamplingParams as VSP
    sp1 = VSP(temperature=0.0, max_tokens=1)
    out = llm.generate([prompt], sp1, use_tqdm=False)[0]
    m = getattr(out, "metrics", None)
    if m is not None:
        ft, at = getattr(m, "first_token_time", None), getattr(m, "arrival_time", None)
        if ft and at:
            return (ft - at) * 1000
    # 退化方案: max_tokens=1 的端到端耗时 ≈ prefill 时延
    t0 = perf_counter()
    llm.generate([prompt], sp1, use_tqdm=False)
    return (perf_counter() - t0) * 1000


def bench_vllm(model, max_tokens):
    from vllm import LLM as VLLM
    from vllm import SamplingParams as VSP

    llm = VLLM(model=model, quantization="gptq_marlin", dtype="half",
               gpu_memory_utilization=0.5, enforce_eager=True,
               max_model_len=1024, max_num_seqs=4)
    vsp = VSP(temperature=TEMPERATURE, top_p=0.95, max_tokens=max_tokens)
    # 预热
    llm.generate(["warm-up prompt"], vsp, use_tqdm=False)

    ttfts = [_vllm_ttft(llm, VSP, p) for p in PROMPTS]

    # 单请求 decode 速度: 用整体耗时近似 (vLLM 离线 API 不拆分首 token 边界)
    decodes = []
    for p in PROMPTS:
        t0 = perf_counter()
        out = llm.generate([p], vsp, use_tqdm=False)[0]
        n = len(out.outputs[0].token_ids)
        elapsed = perf_counter() - t0
        if n > 1 and elapsed > 0:
            decodes.append((n - 1) / elapsed)

    # 并发吞吐
    t0 = perf_counter()
    outs = llm.generate(PROMPTS, vsp, use_tqdm=False)
    total = sum(len(o.outputs[0].token_ids) for o in outs)
    tput = total / (perf_counter() - t0)

    print("=== vLLM (gptq_marlin, enforce_eager) ===")
    _stats("vllm", ttfts, decodes, tput)
    return ttfts, decodes, tput


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max_tokens", type=int, default=128)
    parser.add_argument("--engine", default="both",
                        choices=["nanovllm", "nanovllm-cache", "nanovllm-fused", "vllm", "both", "all"])
    parser.add_argument("--json", action="store_true", help="机器可读输出, 供子进程间传递结果")
    args = parser.parse_args()

    # 单引擎: 直接测量
    if args.engine == "nanovllm":
        ttfts, decodes, tput = bench_nanovllm(args.model, args.max_tokens)
        _emit_json(args, ttfts, decodes, tput) if args.json else None
        return
    if args.engine == "nanovllm-cache":
        os.environ["NANOVLLM_GPTQ_CACHE"] = "1"
        ttfts, decodes, tput = bench_nanovllm(args.model, args.max_tokens)
        _emit_json(args, ttfts, decodes, tput) if args.json else None
        return
    if args.engine == "nanovllm-fused":
        os.environ["NANOVLLM_GPTQ_FUSED"] = "1"
        ttfts, decodes, tput = bench_nanovllm(args.model, args.max_tokens)
        _emit_json(args, ttfts, decodes, tput) if args.json else None
        return
    if args.engine == "vllm":
        ttfts, decodes, tput = bench_vllm(args.model, args.max_tokens)
        _emit_json(args, ttfts, decodes, tput) if args.json else None
        return

    # 多引擎: 各起独立子进程, 避免 CUDA 显存/上下文互相干扰
    engines = ["nanovllm", "nanovllm-cache", "nanovllm-fused", "vllm"] if args.engine == "all" \
        else ["nanovllm", "vllm"]
    res = {}
    for eng in engines:
        print(f"########## 运行 {eng} (独立子进程) ##########")
        res[eng] = _run_subprocess(eng, args)
        print()
    _compare(res)


def _emit_json(args, ttfts, decodes, tput):
    import json
    print("JSON_RESULT " + json.dumps({"ttfts": ttfts, "decodes": decodes, "tput": tput}))


def _run_subprocess(engine, args):
    import json
    import subprocess
    import sys
    cmd = [sys.executable, __file__, "--engine", engine, "--model", args.model,
           "--max_tokens", str(args.max_tokens), "--json"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    print(proc.stdout.rstrip())
    if proc.returncode != 0:
        print(f"[WARN] {engine} 失败 (rc={proc.returncode}):")
        print((proc.stderr or "")[-1500:])
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT "):
            return json.loads(line[len("JSON_RESULT "):])
    return None


def _compare(res):
    rows = []
    for name, r in res.items():
        if not r:
            continue
        tt = sum(r["ttfts"]) / len(r["ttfts"]) if r["ttfts"] else 0.0
        dc = sum(r["decodes"]) / len(r["decodes"]) if r["decodes"] else 0.0
        rows.append((name, tt, dc, r["tput"]))
    if len(rows) < 2:
        return
    print("=== 对比汇总 ===")
    print(f"{'engine':<16}{'TTFT(ms)':>12}{'Decode(t/s)':>14}{'Throughput(t/s)':>18}")
    for name, tt, dc, tp in rows:
        print(f"{name:<16}{tt:>12.1f}{dc:>14.1f}{tp:>18.1f}")
    base = next((r for r in rows if r[0] == "vllm"), None)
    if base:
        print("\n(相对 vLLM 的倍数, TTFT 越低越好, 吞吐越高越好)")
        for name, tt, dc, tp in rows:
            if name == "vllm":
                continue
            print(f"  {name:<16} TTFT {tt / base[1] if base[1] else 0:6.2f}x | "
                  f"Decode {dc / base[2] if base[2] else 0:6.2f}x | "
                  f"Throughput {tp / base[3] if base[3] else 0:6.2f}x")


if __name__ == "__main__":
    main()
