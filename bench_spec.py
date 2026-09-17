"""投机解码性能基准: 与 bench.py 同口径, 对比"不用投机解码"的基线。

指标 (与 bench.py 保持一致):
  TTFT      : 单请求从提交到产出首个 token 的时延 (ms)
  Decode    : 单请求在首 token 之后的解码速度 (tok/s)
  Throughput: 4 请求并发时的总生成吞吐 (tok/s)
附加投机解码专属指标:
  alpha     : 接受率 = 被接受的草稿 token / 提交的候选 token
  mean_acc  : 每轮平均确认的 token 数 (含 bonus), 理论加速的上界就是这个值
另外检查 KV cache 是否泄漏: 投机解码每轮都会"先占位 γ 个 block 再回滚",
如果 trim 没把跨出去的 block 还回去, 长跑几个 step 就会耗尽 KV cache。

用法:
  python bench_spec.py                       # γ = 0(baseline) / 3 / 5 / 7 全跑
  python bench_spec.py --gammas 0,4,6
  python bench_spec.py --mode single --gamma 4 --json   # 子进程使用
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
from time import perf_counter

DEFAULT_TARGET = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
DEFAULT_DRAFT = "/nas_data/LLM/qwen/qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Please explain the difference between TCP and UDP in detail.",
    "Write a Python function to compute the Fibonacci sequence recursively.",
    "What are the main causes of climate change and its impacts?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]

# nano-vLLM 原本不支持 greedy; 用很小的 temperature 近似确定性采样
TEMPERATURE = 0.8   # 可用 --temperature 覆盖


def _avg(xs):
    return sum(xs) / len(xs) if xs else 0.0


def _weights_gib(llm) -> float:
    """target + draft 的权重显存 (KV cache 是按剩余显存自动分配的, 不参与对比)。"""
    total = 0
    for mod in getattr(llm.model_runner, "model", None), getattr(llm.model_runner, "draft_model", None):
        if mod is None:
            continue
        for t in list(mod.parameters()) + list(mod.buffers()):
            total += t.numel() * t.element_size()
    return total / 2 ** 30


def _check_kv_leak(llm, temperature=TEMPERATURE):
    """跑三轮生成, 结束后空闲 block 数必须回到起点, 否则 block 回收有 bug。"""
    from nanovllm.sampling_params import SamplingParams
    sp = SamplingParams(temperature=temperature, max_tokens=48)
    bm = llm.scheduler.block_manager
    llm.generate(["warm-up prompt for kv cache account"], sp, use_tqdm=False)
    free0 = len(bm.free_block_ids)
    for _ in range(3):
        llm.generate(PROMPTS, sp, use_tqdm=False)
    free1 = len(bm.free_block_ids)
    return free0, free1


def bench_one(target: str, draft: str | None, gamma: int, max_tokens: int,
              dynamic: bool = True, max_gamma: int = 12, K: int = 1,
              temperature: float = TEMPERATURE):
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams

    kwargs = dict(tensor_parallel_size=1, enforce_eager=True,
                  max_num_seqs=len(PROMPTS), max_model_len=2048,
                  max_num_batched_tokens=8192)
    gamma = 0 if draft is None else gamma
    if draft is not None:
        kwargs.update(draft_model=draft, num_speculative_tokens=gamma,
                      dynamic_gamma=dynamic, max_speculative_tokens=max_gamma,
                      num_spec_candidates=K)
    llm = LLM(target, **kwargs)
    sp = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    llm.generate(["warm-up prompt"], sp, use_tqdm=False)

    # ---- TTFT + 单请求 decode ----
    ttfts, decodes = [], []
    for p in PROMPTS:
        llm.add_request(p, sp)
        t0 = perf_counter()
        _, num_tokens = llm.step()               # 首步 = prefill
        ttft = (perf_counter() - t0) * 1000
        n_decode, t_dec = 0, perf_counter()
        while not llm.is_finished():
            _, nt = llm.step()
            if nt < 0:
                n_decode += -nt
        elapsed = perf_counter() - t_dec
        ttfts.append(ttft)
        if elapsed > 0 and n_decode > 0:
            decodes.append(n_decode / elapsed)

    # ---- 并发吞吐 ----
    t0 = perf_counter()
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    tput = sum(len(o["token_ids"]) for o in outs) / (perf_counter() - t0)

    stats = {}
    if draft is not None:
        spec = llm.model_runner.speculator
        stats = {"alpha": spec.acceptance_rate, "mean_acc": spec.mean_accepted_length,
                 "tail_mass": spec.tail_mass.mean,
                 "gamma_final": llm.model_runner.current_gamma,
                 "cost_ratio": llm.model_runner._cost_ratio}
        free0, free1 = _check_kv_leak(llm, temperature)
        stats["kv_leak"] = free0 != free1
        stats["free_blocks"] = f"{free1}/{free0}"

    result = {
        "gamma": gamma,
        "ttft": _avg(ttfts),
        "decode": _avg(decodes),
        "tput": tput,
        "weights_gib": _weights_gib(llm),
        **stats,
    }
    del llm
    return result


def _run_child(target, draft, gamma, max_tokens, dynamic, max_gamma, K, temp):
    cmd = [sys.executable, __file__, "--mode", "single", "--gamma", str(gamma),
           "--max_tokens", str(max_tokens), "--target", target, "--json",
           "--dynamic", str(int(dynamic)), "--max-gamma", str(max_gamma),
           "--candidates", str(K), "--temperature", str(temp)]
    if draft:
        cmd += ["--draft", draft]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] gamma={gamma} 子进程退出码 {proc.returncode}")
        print((proc.stderr or "")[-2000:])
        return None
    for line in proc.stdout.splitlines():
        if line.startswith("JSON_RESULT "):
            return json.loads(line[len("JSON_RESULT "):])
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--draft", default=None)
    ap.add_argument("--gammas", default="0,3,5,7")
    ap.add_argument("--dynamic", type=int, default=1,
                    help="1=按接受率与耗时比在线选 γ; 0=固定用 --gammas 给的 γ")
    ap.add_argument("--max-gamma", type=int, default=12)
    ap.add_argument("--candidates", type=int, default=1,
                    help="K: 一次 verify 同时验证的候选链数 (tree verification)")
    ap.add_argument("--temperature", type=float, default=TEMPERATURE)
    ap.add_argument("--max_tokens", type=int, default=128)
    ap.add_argument("--mode", default="parent", choices=["parent", "single"])
    ap.add_argument("--gamma", type=int, default=0)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    if args.mode == "single":
        draft = None if args.gamma == 0 else (args.draft or DEFAULT_DRAFT)
        r = bench_one(args.target, draft, args.gamma, args.max_tokens,
                      dynamic=bool(args.dynamic), max_gamma=args.max_gamma,
                      K=args.candidates)
        print("JSON_RESULT " + json.dumps(r))
        return

    draft = args.draft or DEFAULT_DRAFT
    rows = []
    for g in [int(x) for x in args.gammas.split(",")]:
        print(f"########## γ={g} ({'基线' if g == 0 else '投机解码'}) ##########")
        r = _run_child(args.target, draft, g, args.max_tokens,
                       bool(args.dynamic), args.max_gamma, args.candidates,
                       args.temperature)
        if r:
            rows.append(r)
            extra = ""
            if g > 0:
                leak = "❌ 泄漏" if r.get("kv_leak") else "✅ 无泄漏"
                ginfo = f"  收敛 γ={r.get('gamma_final')}  c={r.get('cost_ratio', 0):.3f}" \
                    if r.get("gamma_final") is not None else ""
                extra = (f"\n  接受率 alpha : {r['alpha']:.3f}   平均接受长度: {r['mean_acc']:.2f}{ginfo}"
                         f"\n  KV block 回收: {leak} (空闲 {r.get('free_blocks', '-')})")
            print(f"  TTFT        : {r['ttft']:.1f} ms"
                  f"\n  Decode      : {r['decode']:.1f} tok/s (单请求)"
                  f"\n  Throughput  : {r['tput']:.1f} tok/s ({len(PROMPTS)} 并发)"
                  f"\n  权重显存    : {r['weights_gib']:.2f} GiB{extra}\n")

    if len(rows) < 2:
        return
    print("=== 对比汇总 (相对 γ=0 基线) ===")
    print(f"{'γ':>3}{'TTFT(ms)':>12}{'Decode(t/s)':>14}{'Throughput(t/s)':>18}{'加速':>8}")
    base = next(r for r in rows if r["gamma"] == 0)
    for r in rows:
        sp = r["decode"] / base["decode"] if base["decode"] else 0
        print(f"{r['gamma']:>3}{r['ttft']:>12.1f}{r['decode']:>14.1f}{r['tput']:>18.1f}{sp:>8.2f}x")


if __name__ == "__main__":
    main()
