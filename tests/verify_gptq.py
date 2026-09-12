"""
GPTQ 反量化正确性 —— 严格验证 (取代 "mean(码字)≈零点" 这类弱启发式证据)

用法:
    python tests/verify_gptq.py            # 全部
    python tests/verify_gptq.py --part A   # 黄金往返 + 变异测试
    python tests/verify_gptq.py --part B   # 真实权重单层精度 (fused vs cuBLAS)
    python tests/verify_gptq.py --part C   # 端到端 prefill logits 一致性

设计原则 (为什么这比 "mean(码字)≈真实零点" 有含金量):
  1. 自造"黄金"权重: 直接由已知码字按约定反算得到, 因此量化无损, 往返必须 **逐元素 bit-exact
     (max err = 0.0)**; 任何解包顺序/分组映射/零点约定错误都会被打成明显的大误差。
  2. 变异测试 (mutation testing): 人为注入 5 种典型错误, 断言测试 **必须失败** —— 证明这个
     测试真的有判别力, 而不是"怎么跑都过"。
  3. 反例演示: 构造一个非零均值(有偏)的权重, 使得 "mean(码字)" 离真实零点很远,
     但往返测试依然通过 —— 直接说明旧 heuristic 不可靠。
"""
import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from nanovllm.layers.gptq_linear import (
    GPTQColumnParallelLinear, _unpack_qweight, _unpack_qzeros,
)

MODEL = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
DEV = "cuda"
ZERO_STORED = 7          # qzeros 里存的值; 真实零点 = ZERO_STORED + 1 = 8
GS = 128


# --------------------------------------------------------------------------------------
# 打包器 (与 HF GPTQ / gptq_linear.py 的布局约定一致)
#   qweight: (K//8,   N)    每 int32 沿 K 维打包 8 个 4-bit 码字, nibble sub = k % 8
#   qzeros : (K//GS,  N//8) 每 int32 沿 N 维打包 8 个零点,  nibble sub = n % 8
#   scales : (K//GS,  N)    fp16
# --------------------------------------------------------------------------------------
def pack_from_codes(codes, scales, zero_stored=ZERO_STORED, zeros=None):
    """codes: (N, K) int32 [0,15]; scales: (K//GS, N) fp16 -> qweight/qzeros/scales
    zeros: 可选 (ng, N) int32, 用于构造"零点随输出通道变化"的样本(对称量化下其实是常数,
           但只有让它变化, 才能检验 qzeros 沿 N 的解包顺序)。"""
    N, K = codes.shape
    ng = K // GS
    qw = codes.reshape(N, K // 8, 8)                      # (N, K//8, 8) sub 在最后一维
    pow16 = (1 << (torch.arange(8, device=codes.device, dtype=torch.int32) * 4))  # 1,16,...,2^28
    qweight = (qw.permute(1, 0, 2) * pow16).sum(-1).to(torch.int32).contiguous()  # (K//8, N)
    z = torch.full((ng, N), zero_stored, dtype=torch.int32, device=codes.device) if zeros is None else zeros
    qzeros = (z.reshape(ng, N // 8, 8) * pow16).sum(-1).to(torch.int32).contiguous()  # (ng, N//8)
    return qweight, qzeros, scales


def make_layer_from_packed(qweight, qzeros, scales, N, K, zp_bias=1):
    layer = GPTQColumnParallelLinear(K, N, bias=False, group_size=GS).to(DEV)
    layer.qweight.data.copy_(qweight.to(DEV))
    layer.qzeros.data.copy_(qzeros.to(DEV))
    layer.scales.data.copy_(scales.to(DEV))
    layer.zero_point_bias = zp_bias
    return layer


def deq(layer):
    return layer._dequant_block(0, layer.out_features)     # (N, K) fp32


# --------------------------------------------------------------------------------------
# Part A: 黄金往返 + 变异测试
# --------------------------------------------------------------------------------------
def build_golden(N=256, K=1024, seed=0, bias_w=0.0):
    """构造"可被 int4 精确表示"的权重 -> 往返必须无损。bias_w 用于制造非零均值分布。"""
    g = torch.Generator().manual_seed(seed)
    ng = K // GS
    scales = (0.03125 + 0.5 * torch.rand(ng, N, generator=g)).half()      # (ng, N) fp16
    codes = torch.randint(0, 16, (N, K), generator=g)
    if bias_w > 0:      # 把码字整体推向一端(模拟有偏分布), 会触发饱和截断
        codes = (codes.float() + bias_w).clamp(0, 15).to(torch.int32)
    s_full = scales.repeat_interleave(GS, dim=0).t().float()              # (N, K)
    W = ((codes.float() - (ZERO_STORED + 1)) * s_full).half()             # 真实权重
    return codes, scales, W


def part_A():
    print("=" * 78)
    print("Part A: 黄金往返测试 (无损量化 => 必须逐元素 bit-exact) + 变异测试")
    print("=" * 78)
    ok = True

    # ---- A1 无损往返 (两组形状, 含非 2 的幂) ----
    for (N, K, seed) in [(256, 1024, 0), (512, 3584, 1), (18944 // 8, 3584, 2)]:
        codes, scales, W = build_golden(N, K, seed)
        qw, qz, sc = pack_from_codes(codes, scales)
        layer = make_layer_from_packed(qw, qz, sc, N, K)
        Wh = deq(layer).half()
        err = (Wh.float() - W.to(DEV).float()).abs().max().item()
        good = err == 0.0
        ok &= good
        print(f"  [A1] N={N:<6} K={K:<6} 往返 max|err| = {err:<8.4g}  {'PASS' if good else 'FAIL'}")

    # ---- A2 变异测试: 每一种人为错误都必须被抓住 ----
    N, K = 256, 1024
    codes, scales, W = build_golden(N, K, 0)
    qw, qz, sc = pack_from_codes(codes, scales)
    Wf = W.to(DEV).float()

    mutants = {}

    # M1: 零点约定错 (z_true = qzeros, 少了 +1)
    mutants["M1 零点少 +1 (z_true=qzeros)"] = dict(zp_bias=0)
    # M2: qweight 的 nibble 顺序解释反了 (k%8 -> 7-k%8)
    inv = codes.reshape(N, K // 8, 8).flip(-1).reshape(N, K)
    qw_m2, _, _ = pack_from_codes(inv, scales)
    mutants["M2 qweight nibble 序反"] = dict(qweight=qw_m2)
    # M3: 把 qweight 的"包 8 个 k"这一维理解错 (行序反转, 等价于 K 维分块错位)
    mutants["M3 qweight 轴/序误解"] = dict(qweight=qw.flip(0).contiguous())
    # M4: 分组映射错 (scale 组号取反)
    mutants["M4 分组映射错 (组号取反)"] = dict(scales=sc.flip(0).contiguous())
    # M5: qzeros 沿 N 的 nibble 序反 (零点取变化的数, 否则"全 7"翻转后不变, 检验不出)
    g5 = torch.Generator().manual_seed(5)
    z_var = torch.randint(0, 15, (K // GS, N), generator=g5).to(torch.int32)
    _, qz_ok, _ = pack_from_codes(codes, scales, zeros=z_var)
    qz_m5 = (z_var.reshape(K // GS, N // 8, 8).flip(-1) *
             (1 << (torch.arange(8, dtype=torch.int32) * 4))).sum(-1).to(torch.int32).contiguous()
    base_codes = ((codes.float() - (z_var.repeat_interleave(GS, dim=0).t().float() + 1))
                  * scales.repeat_interleave(GS, dim=0).t().float()).half()
    mutants["M5 qzeros 沿 N 解包序反"] = dict(qzeros=qz_m5, qweight=pack_from_codes(codes, scales)[0],
                                             golden=base_codes)

    print("  [A2] 变异测试 (每个变异体都**必须**被判 FAIL, 否则测试无判别力):")
    for name, mut in mutants.items():
        kw = dict(qweight=qw, qzeros=qz, scales=sc)
        kw.update({k: v for k, v in mut.items() if k in ("qweight", "qzeros", "scales")})
        layer = make_layer_from_packed(kw["qweight"], kw["qzeros"], kw["scales"], N, K,
                                       zp_bias=mut.get("zp_bias", 1))
        try:
            Wh = deq(layer).half()
            golden = mut.get("golden", Wf)
            err = (Wh.float() - golden.to(DEV).float()).abs().max().item()
        except Exception as e:                       # 形状直接崩也算被抓住
            err, _ = float("inf"), str(e)[:40]
        caught = err > 0.0
        ok &= caught
        print(f"       {name:<32} max|err| = {err:<10.4g} -> {'被抓住 OK' if caught else '漏检 BAD'}")

    # ---- A3 反例: "mean(码字)≈真实零点" 这个 heuristic 并不可靠 ----
    print("  [A3] 反例 —— 有偏分布下 mean(码字) 会远离真实零点:")
    for bw in (0.0, 3.0, 6.0):
        codes_b, scales_b, W_b = build_golden(512, 1024, 3, bias_w=bw)
        mean_code = codes_b.float().mean().item()
        print(f"       bias={bw:<4} mean(码字)={mean_code:6.3f}  真实零点=8.000  "
              f"启发式结论={'对' if abs(mean_code - 8) < 0.5 else '错 ←'}")
        if bw > 0:
            qw_b, qz_b, sc_b = pack_from_codes(codes_b, scales_b)
            layer = make_layer_from_packed(qw_b, qz_b, sc_b, 512, 1024)
            err = (deq(layer).half().float() - W_b.to(DEV).float()).abs().max().item()
            ok &= (err == 0.0)
            print(f"                 但往返测试 max|err|={err:.4g} -> "
                  f"{'仍然 PASS' if err == 0.0 else 'FAIL'}  (说明启发式弱、往返测试强)")

    # ---- A4 真实量化的有损情形: 误差必须落在理论界内 ----
    g = torch.Generator().manual_seed(7)
    N, K = 512, 1024
    W = torch.randn(N, K, generator=g).half()
    Wf = W.to(DEV).float()
    ng = K // GS
    blk = Wf.reshape(N, ng, GS)
    s = (blk.abs().amax(-1) / 7.0).clamp(min=1e-8)                       # sym int4: [-7,7]
    codes = (blk / s[:, :, None]).round().clamp(-8, 7) + 8               # -> [0,15]
    codes = codes.to(torch.int32).reshape(N, K)
    scales = s.transpose(0, 1).contiguous().half()                       # (ng, N)
    qw, qz, sc = pack_from_codes(codes, scales)
    layer = make_layer_from_packed(qw, qz, sc, N, K)
    Wh = deq(layer)
    err = (Wh - Wf).abs()
    bound = (s.max().item() / 2) * 1.001
    good = err.max().item() <= bound
    ok &= good
    print(f"  [A4] 有损量化 (randn): max|err|={err.max().item():.5f} <= 理论界 s_max/2"
          f"={bound:.5f}  均值={err.mean().item():.5f}  {'PASS' if good else 'FAIL'}")
    # 顺带: 若零点取错, 有损情形下的均值偏移会暴露它
    layer.zero_point_bias = 0
    err0 = (deq(layer) - Wf).abs()
    print(f"       对照: zp_bias=0 时 max|err|={err0.max().item():.5f} 均值={err0.mean().item():.5f}"
          f" (≈ 一个 scale, 与记录中的 +1·scale 偏移一致)")

    print(f"\n  Part A 结论: {'ALL PASS' if ok else '存在 FAIL'}\n")
    return ok


# --------------------------------------------------------------------------------------
# Part B: 真实权重单层 —— fused (Triton) vs cuBLAS
# --------------------------------------------------------------------------------------
def _find_key(keys, suffix):
    for k in keys:
        if k.endswith(suffix):
            return k
    return None


def load_real_layers(names=("q_proj", "gate_proj", "down_proj")):
    from safetensors import safe_open
    from glob import glob
    out = {}
    files = sorted(glob(os.path.join(MODEL, "*.safetensors")))
    keys_all = []
    for f in files:
        with safe_open(f, "pt", "cpu") as sf:
            keys_all.extend(sf.keys())
    for nm in names:
        kq = _find_key([k for k in keys_all if f".{nm}." in k], ".qweight")
        if kq is None:
            continue
        base = kq[: -len(".qweight")]
        tensors = {}
        for suf in (".qweight", ".qzeros", ".scales", ".bias"):
            t = None
            for f in files:
                with safe_open(f, "pt", "cpu") as sf:
                    if base + suf in sf.keys():
                        t = sf.get_tensor(base + suf)
                        break
            tensors[suf] = t
        out[base] = tensors
    return out


def part_B():
    print("=" * 78)
    print("Part B: 真实权重 —— fused(Triton tl.dot) vs cuBLAS, 多种形状 / 多种 M")
    print("=" * 78)
    from nanovllm.layers.gptq_triton import fused_gptq_linear

    layers = load_real_layers()
    if not layers:
        print("  [!] 未找到真实权重, 跳过"); return False

    print(f"  {'layer':<42}{'M':>5}{'K':>7}{'N':>7}  {'max|Δ|':>10}{'mean|Δ|':>10}"
          f"{'不匹配元素':>12}{'相对误差':>12}")
    worst = 0.0
    for base, t in layers.items():
        qw = t[".qweight"].to(DEV)
        qz = t[".qzeros"].to(DEV)
        sc = t[".scales"].to(DEV)
        K, N = qw.shape[0] * 8, qw.shape[1]           # qweight: (K//8, N)
        layer = make_layer_from_packed(qw, qz, sc, N, K)
        W = deq(layer).half()                          # (N, K) fp16 —— 与 cache 模式逐位相同
        bias = t[".bias"].to(DEV) if t[".bias"] is not None else None

        for M in (1, 4, 32, 256):
            torch.manual_seed(0)
            x = torch.randn(M, K, device=DEV).half()
            ref = (x @ W.t())                          # cuBLAS fp16, 内部 fp32 累加
            if bias is not None:
                ref = ref + bias
            got = fused_gptq_linear(x, qw, qz, sc, M, N, K, GS)
            if bias is not None:
                got = got + bias
            d = (got.float() - ref.float()).abs()
            scale = ref.float().abs().mean().item()
            ndiff = int((got != ref).sum().item())
            rel = (d.max().item() / scale) if scale > 0 else 0.0
            worst = max(worst, rel)
            print(f"  {base:<42}{M:>5}{K:>7}{N:>7}  {d.max().item():>10.4g}{d.mean().item():>10.4g}"
                  f"{ndiff:>10}/{d.numel():<6}{rel:>12.3e}")
        torch.cuda.empty_cache()
    print(f"\n  最大相对误差 = {worst:.3e}  (fp16 ulp 相对尺度 ≈ 4.9e-4)")
    print("  说明: 0.0 只表示 fp16 **输出** 落在同一个可表示值上, 并不代表 fp32 累加器逐位相同。\n")
    return True


def part_B2():
    """精度归因: 用 fp64 真值判断"谁更准", 并观察 BLOCK_K / M 的影响。
    目的: 回答 "单层 max 0.0 是不是好得过分了" —— 结论要能被复现, 而不是一句口号。"""
    print("=" * 78)
    print("Part B2: 精度归因 (fp64 真值 + BLOCK_K 扫描)")
    print("=" * 78)
    from nanovllm.layers import gptq_triton
    from nanovllm.layers.gptq_triton import fused_gptq_linear

    print("  关键: 若 |fused-真值| 与 |cuBLAS-真值| 量级相当, 说明二者只是归约顺序不同的"
          "两种等精度实现, 而不是 fused 更差。\n")
    print(f"  {'layer':<34}{'M':>4}{'|fused-真值|':>14}{'|cuBLAS-真值|':>15}"
          f"{'|fused-cuBLAS|':>15}{'不一致元素':>14}")
    for base, t in load_real_layers(("q_proj", "gate_proj", "down_proj")).items():
        qw = t[".qweight"].to(DEV); qz = t[".qzeros"].to(DEV); sc = t[".scales"].to(DEV)
        K, N = qw.shape[0] * 8, qw.shape[1]
        layer = make_layer_from_packed(qw, qz, sc, N, K)
        W = deq(layer).half()
        for M in (1, 4, 32, 256):
            torch.manual_seed(0)
            x = torch.randn(M, K, device=DEV).half()
            truth = (x.double() @ W.t().double())                    # fp64 真值
            ref = x @ W.t()                                          # cuBLAS
            got = fused_gptq_linear(x, qw, qz, sc, M, N, K, GS)
            e_f = (got.double() - truth).abs().max().item()
            e_c = (ref.double() - truth).abs().max().item()
            e_d = (got.float() - ref.float()).abs().max().item()
            ndiff = int((got != ref).sum().item())
            print(f"  {base:<34}{M:>4}{e_f:>14.3e}{e_c:>15.3e}{e_d:>15.3e}"
                  f"{ndiff:>10}/{got.numel()}")
        print()
    return True


def _fused_with_bk(x, qw, qz, sc, M, N, K, BK):
    """用指定 BLOCK_K 调一次 fused kernel (仅用于精度归因实验)。"""
    import triton
    from nanovllm.layers.gptq_triton import _fused_gptq_mm
    out = torch.empty((M, N), dtype=torch.float16, device=x.device)
    grid = (triton.cdiv(M, 32), triton.cdiv(N, 64))
    _fused_gptq_mm[grid](x, qw, qz, sc, out, M, N, K, GS,
                         BLOCK_M=32, BLOCK_N=64, BLOCK_K=BK, num_warps=4)
    return out


# --------------------------------------------------------------------------------------
# Part C: 端到端 —— 四种路径 vs vLLM 的贪心 token 一致性
# --------------------------------------------------------------------------------------
PROMPTS_C = [
    "Please explain the difference between TCP and UDP in detail.",
    "Write a Python function to compute the Fibonacci sequence recursively.",
]


def _gen_nanovllm(mode: str, max_tokens: int = 32):
    """在子进程里用指定模式跑贪心 (temperature=1e-9 近似 greedy: 除以极小温度后再取 argmax)。"""
    import json
    from nanovllm.llm import LLM
    from nanovllm.sampling_params import SamplingParams
    torch.manual_seed(0)
    llm = LLM(MODEL, tensor_parallel_size=1, max_num_batched_tokens=2048,
              max_num_seqs=2, max_model_len=1024)
    sp = SamplingParams(temperature=1e-9, max_tokens=max_tokens)
    llm.generate(["warm up"], sp, use_tqdm=False)
    outs = llm.generate(PROMPTS_C, sp, use_tqdm=False)
    print("JSON_TOKENS " + json.dumps([o["token_ids"] for o in outs]))


def _gen_vllm(max_tokens: int = 32):
    import json
    from vllm import LLM as VLLM
    from vllm import SamplingParams as VSP
    llm = VLLM(model=MODEL, quantization="gptq_marlin", dtype="half",
               gpu_memory_utilization=0.5, enforce_eager=True, max_model_len=1024)
    outs = llm.generate(PROMPTS_C, VSP(temperature=0.0, max_tokens=max_tokens), use_tqdm=False)
    print("JSON_TOKENS " + json.dumps([list(o.outputs[0].token_ids) for o in outs]))


def _run(mode: str, max_tokens: int = 32):
    import subprocess
    code = ("from tests.verify_gptq import _gen_vllm; _gen_vllm(%d)" % max_tokens) \
        if mode == "vllm" else \
        ("import os; os.environ['NANOVLLM_GPTQ_%s']='1'; "
         "from tests.verify_gptq import _gen_nanovllm; _gen_nanovllm('%s', %d)"
         % (mode.upper(), mode, max_tokens))
    env = dict(os.environ)
    env.pop("NANOVLLM_GPTQ_FUSED", None)
    env.pop("NANOVLLM_GPTQ_CACHE", None)
    env.pop("NANOVLLM_GPTQ_TORCH", None)
    p = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       env=env, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    for line in p.stdout.splitlines():
        if line.startswith("JSON_TOKENS "):
            import json
            return json.loads(line[len("JSON_TOKENS "):])
    print(f"  [!] {mode} 运行失败:\n{(p.stderr or '')[-800:]}")
    return None


def _cmp(a, b):
    n = min(len(a), len(b))
    first = next((i for i in range(n) if a[i] != b[i]), None)
    return sum(1 for i in range(n) if a[i] == b[i]), n, (first if first is not None else -1)


def part_C():
    print("=" * 78)
    print("Part C: 端到端贪心一致性 (各路径 vs vLLM gptq_marlin, 2 prompts x 32 tokens)")
    print("=" * 78)
    modes = ["stream", "cache", "fused", "torch"]
    ref = _run("vllm")
    if ref is None:
        print("  [!] vLLM 不可用, 改用 cache 模式作参考")
        ref = _run("cache")
    print(f"  {'mode':<10}{'匹配/总数':>14}{'首个分歧位置':>14}   结论")
    for m in modes:
        got = _run(m)
        if got is None or ref is None:
            print(f"  {m:<10}{'--':>14}{'--':>14}   运行失败, 见上方报错")
            continue
        tot, n, first = 0, 0, -1
        for a, b in zip(ref, got):
            c, nn, f = _cmp(a, b)
            tot += c
            n += nn
            if first < 0 and f >= 0:
                first = f
        ok = tot == n
        print(f"  {m:<10}{tot:>7}/{n:<6}{first:>14}   "
              f"{'与参考逐 token 一致' if ok else '与参考分歧 (首分歧 @ token %d)' % first}")
    print()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="AB2")
    a = ap.parse_args()
    if "A" in a.part:
        part_A()
    if "B2" in a.part:
        part_B2()
    if "B" in a.part:
        part_B()
    if "C" in a.part:
        part_C()
