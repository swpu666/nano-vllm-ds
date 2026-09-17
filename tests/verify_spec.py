"""投机解码的正确性验证。

分三部分, 从强到弱:
  Part A  greedy 模式 -> 输出必须与"逐步 argmax"的基线逐 token 相同 (确定性强对照)
  Part B  采样模式 -> 输出的 token 分布必须严格等于 target 分布 (rejection sampling 的核心承诺)
          + 变异测试: 注入 4 种常见实现错误, 上述统计检验必须一个个抓出来。
          没有变异测试, "分布一致"这种结论是没有含金量的 —— 弱检验会因为阈值太松放过 bug。
  Part C  端到端 -> 同一 prompts 下, 开/关投机解码的 greedy 输出必须完全相同;
          同时统计接受率 α 与每轮平均接受长度。

运行:
  python tests/verify_spec.py                      # A + B (纯张量, 秒级)
  python tests/verify_spec.py --e2e                # A + B + C (C 需加载模型, 起子进程)
  python tests/verify_spec.py --mode gen --gamma 4 # 子进程内部使用
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nanovllm.engine.speculator import Speculator, TailMassMeter, _probs, _gumbel_sample  # noqa: E402

DEFAULT_TARGET = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
DEFAULT_DRAFT = "/nas_data/LLM/qwen/qwen/Qwen2.5-0.5B-Instruct"

PROMPTS = [
    "Please explain the difference between TCP and UDP in detail.",
    "Write a Python function to compute the Fibonacci sequence recursively.",
    "What are the main causes of climate change and its impacts?",
    "Summarize the plot of Romeo and Juliet in three sentences.",
]


# ============================================================ Part A: greedy
def part_a(verbose: bool = True):
    """greedy 模式下投机解码的输出必须退化成"每步让 target 取 argmax"的结果。"""
    ok = True
    V, G = 32, 4
    torch.manual_seed(0)
    # 每个位置用不同的 logits, 这样"用错位置的 logits"会被立刻暴露
    target_logits = torch.randn(1, G + 1, V)
    baseline = target_logits[0].argmax(dim=-1).tolist()      # γ+1 个参考 token
    dummy_draft_logits = torch.zeros(1, G, V)                # greedy 模式只用 argmax, 不看 q

    spec = Speculator(G)

    # A1: 草稿完全等于 target argmax -> 全部接受 + bonus
    draft = torch.tensor([baseline[:G]])
    out = spec.verify(target_logits, dummy_draft_logits, draft, None, greedy=True)
    want = baseline[:G] + [baseline[G]]
    ok &= _check("A1 全接受: 输出 == argmax(0..γ)+bonus", out.accepted_tokens[0] == want,
                 f"got={out.accepted_tokens[0]} want={want}")

    # A2: 在第 j 位故意给出错误的草稿 -> 只接受前 j 个, 并用该位置的 argmax 修正
    for j in range(G):
        bad = list(baseline[:G])
        bad[j] = (bad[j] + 1) % V          # 保证 != baseline[j]
        out = spec.verify(target_logits, dummy_draft_logits, torch.tensor([bad]), None, greedy=True)
        want = bad[:j] + [baseline[j]]
        ok &= _check(f"A2 第 {j} 位被拒: 接受 {j} 个并修正",
                     out.n_accepted[0] == j and out.accepted_tokens[0] == want,
                     f"n={out.n_accepted[0]} got={out.accepted_tokens[0]} want={want}")

    # A3: 草稿全错时也必须产出 1 个 token, 否则引擎会原地空转
    bad = [(baseline[idx] + 1) % V for idx in range(G)]
    out = spec.verify(target_logits, dummy_draft_logits, torch.tensor([bad]), None, greedy=True)
    ok &= _check("A3 草稿全错时仍至少产出 1 token",
                 len(out.accepted_tokens[0]) == 1 and out.n_accepted[0] == 0,
                 f"got={out.accepted_tokens[0]}")

    return ok


# =================================================== Part B: 分布等价性 + 变异
def _dist_equiv(spec_cls, base_target, base_draft, N, G):
    """返回 (首 token 分布的 TVD, bonus token 分布的 TVD, 全接受样本数)。

    一次性把 N 个样本 pack 成 batch (它们共享同一份 target/draft 分布), 远比逐样本
    调用快。理论上两条硬约束:
      - 每轮输出的**首 token** 边际分布必须严格等于 p(pos0)
      - 全接受时送出的 bonus token 必须服从 p(posG)
    TVD = 0.5 * Σ|empirical - target|, 是两个分布之间的总变差距离。
    """
    V = base_target.size(-1)
    spec = spec_cls(G)
    target_logits = base_target.expand(N, G + 1, V)
    draft_logits = base_draft.expand(N, G, V)
    # draft token 必须按 q 采样: 拒绝采样的接受概率依赖 q, 采样错了结论就不成立
    draft_tokens = _gumbel_sample(_probs(draft_logits))               # (N, G)
    out = spec.verify(target_logits, draft_logits, draft_tokens, None, greedy=False)

    head = torch.zeros(V)
    bonus = torch.zeros(V)
    n_full = 0
    for i in range(N):
        head[out.accepted_tokens[i][0]] += 1
        if out.n_accepted[i] == G:                                    # 走的是 bonus 分支
            n_full += 1
            bonus[out.accepted_tokens[i][-1]] += 1

    p0 = _probs(base_target[:, 0, :])[0]                              # (V,)
    pG = _probs(base_target[:, G, :])[0]
    tv_head = 0.5 * (head / head.sum() - p0).abs().sum().item()
    tv_bonus = 0.5 * (bonus / bonus.sum().clamp(min=1) - pG).abs().sum().item()
    return tv_head, tv_bonus, n_full


def part_b(verbose: bool = True):
    V, G, N = 16, 2, 40000
    torch.manual_seed(1)
    # 位置 0: p0 != q0 (会频繁拒绝); 位置 1..G: draft 与 target 完全一致 -> 必被接受
    target_logits = torch.randn(1, G + 1, V)
    draft_logits = torch.cat([torch.randn(1, 1, V), target_logits[:, 1:G, :]], dim=1)

    tv_head, tv_bonus, n_full = _dist_equiv(Speculator, target_logits, draft_logits,
                                            N, G)
    # N=40k, V=16: 多项式噪声的标准差 ~ sqrt(p/N) -> TVD 期望噪声量级 ~1e-3
    TH = 0.02
    ok = True
    ok &= _check(f"B1 首 token 分布 == target p(pos0)  [TVD={tv_head:.4f} < {TH}]", tv_head < TH)
    ok &= _check(f"B2 bonus 分布 == target p(pos{G})  [TVD={tv_bonus:.4f} < {TH}, n={n_full}]",
                 tv_bonus < TH and n_full > 1000)
    if verbose:
        print(f"    (全接受样本 {n_full}/{N} = {n_full / N:.3f})")

    # ---------------- 变异测试: 检验必须能抓住下面每种错误 ----------------
    mutants = {
        "忘了除以 q (accept 概率直接用 p)":
            {"accept_probability": lambda self, p, q: p},
        "残差分布用了 p 而非 (p-q)_+":
            {"residual_distribution": lambda self, p, q: p},
        "残差分布直接用了 draft 的 q":
            {"residual_distribution": lambda self, p, q: q},
        "bonus 取错位置 (用 pos0 而非 posG)":
            {"bonus_distribution": lambda self, p_all, G: p_all[:, 0, :]},
    }
    for name, overrides in mutants.items():
        mutant_cls = type("Mutant", (Speculator,), overrides)
        h, b, _ = _dist_equiv(mutant_cls, target_logits, draft_logits, N, G)
        ok &= _check(f"B3 变异被抓住: {name}  [TVD head={h:.4f} bonus={b:.4f}]",
                     h > TH or b > TH,
                     note="该错误未被分布检验捕获 -> 检验力度不足")

    if verbose:
        print("    (变异测试的 TVD 越大代表错误越容易被上面的阈值判定为异常)")
    return ok


# ============================================================ Part C: 端到端
def _run_child(mode: str, gamma: int, max_tokens: int, target: str, draft: str, K: int = 1):
    import json
    cmd = [sys.executable, __file__, "--mode", "gen", "--gamma", str(gamma),
           "--max_tokens", str(max_tokens), "--target", target, "--candidates", str(K)]
    if draft:
        cmd += ["--draft", draft]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"[FAIL] 子进程 ({mode}) 退出码 {proc.returncode}")
        print((proc.stderr or "")[-3000:])
        return None, None
    result = stats = None
    for line in proc.stdout.splitlines():
        if line.startswith("RESULT "):
            result = json.loads(line[len("RESULT "):])
        elif line.startswith("STATS "):
            stats = json.loads(line[len("STATS "):])
    print(proc.stdout.rstrip())
    return result, stats


def part_c(target: str, draft: str, gamma: int, max_tokens: int, K: int = 1):
    print("\n=== Part C: 端到端 greedy 对齐 (model loading, 每个子进程独立) ===")
    base_result, _ = _run_child("baseline", 0, max_tokens, target, None)
    spec_result, spec_stats = _run_child("spec", gamma, max_tokens, target, draft, K)
    print()
    if base_result is None or spec_result is None:
        return False

    ok = True
    n = min(len(base_result), len(spec_result))
    identical = sum(1 for i in range(n) if base_result[i] == spec_result[i])
    ok &= _check(f"C1 {n} 条 greedy 输出逐 token 完全一致", identical == n,
                 f"一致 {identical}/{n}")

    # 逐条给出差异行, 便于定位
    for i in range(n):
        if base_result[i] != spec_result[i]:
            a, b = base_result[i], spec_result[i]
            j = next((x for x in range(min(len(a), len(b))) if a[x] != b[x]), min(len(a), len(b)))
            print(f"    [diff] prompt#{i} 首个不同位置 {j}: baseline={a[j]} spec={b[j]}")

    if spec_stats:
        print(f"    接受率 alpha   = {spec_stats['acceptance_rate']:.3f}")
        print(f"    平均接受长度   = {spec_stats['mean_accepted']:.2f} (γ={gamma})")
        print(f"    target 尾部概率质量被丢弃 = {spec_stats['tail_mass']:.2e}")
        ok &= _check("C2 target 落在公共词表之外的概率质量可忽略 (<1e-4)",
                     spec_stats["tail_mass"] < 1e-4)
    return ok


# ================================================================ 子进程模式
def mode_gen(target: str, draft: str | None, gamma: int, max_tokens: int, K: int = 1,
             dynamic: bool = True):
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams
    kwargs = dict(tensor_parallel_size=1, enforce_eager=True,
                  max_num_seqs=len(PROMPTS), max_model_len=2048, max_num_batched_tokens=8192)
    if draft:
        kwargs.update(draft_model=draft, num_speculative_tokens=gamma,
                      num_spec_candidates=K, dynamic_gamma=dynamic)
    llm = LLM(target, **kwargs)
    sp = SamplingParams(greedy=True, max_tokens=max_tokens, ignore_eos=False)
    outs = llm.generate(PROMPTS, sp, use_tqdm=False)
    print("RESULT " + json.dumps([o["token_ids"] for o in outs]))

    if draft:
        spec = llm.model_runner.speculator
        print("STATS " + json.dumps({
            "acceptance_rate": spec.acceptance_rate,
            "mean_accepted": spec.mean_accepted_length,
            "tail_mass": spec.tail_mass.mean,
        }))
    del llm


def _check(name: str, cond: bool, extra: str = "", note: str = "") -> bool:
    """extra: 总是显示(对照信息); note: 仅失败时显示(诊断提示)。"""
    line = f"  [{'PASS' if cond else 'FAIL'}] {name}"
    if extra:
        line += f"   {extra}"
    if note and not cond:
        line += f"   {note}"
    print(line)
    return bool(cond)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="verify", choices=["verify", "gen"])
    ap.add_argument("--target", default=DEFAULT_TARGET)
    # 默认必须为 None: 不给 --draft 就代表"跑不带投机解码的基线"
    ap.add_argument("--draft", default=None)
    ap.add_argument("--gamma", type=int, default=4)
    ap.add_argument("--candidates", type=int, default=1,
                    help="K: 一次 verify 同时验证的候选链数 (tree verification)")
    ap.add_argument("--dynamic", type=int, default=1)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--e2e", action="store_true")
    args = ap.parse_args()

    if args.mode == "gen":
        mode_gen(args.target, args.draft, args.gamma, args.max_tokens)
        return

    print("=== Part A: greedy 一致性 ===")
    ok_a = part_a()
    print("\n=== Part B: 采样分布的等价性 + 变异测试 ===")
    ok_b = part_b()
    ok = ok_a and ok_b

    if args.e2e:
        # --draft 默认是 None (不带 draft == 跑基线), e2e 需要显式给出 draft 路径
        draft = args.draft or DEFAULT_DRAFT
        ok &= part_c(args.target, draft, args.gamma, args.max_tokens, args.candidates)

    print("\n" + ("=" * 60))
    print("总结:", "✅ 全部通过" if ok else "❌ 存在失败项")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
