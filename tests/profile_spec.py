"""投机解码耗时分解: 用 CUDA event 分别测 target decode / draft decode / target verify。

目的: 判断 1 轮投机 (~γ+1 次 draft + 1 次 verify, 产出 mean_acc 个 token)
与"产出同样多 token 需要多少次普通 decode"的盈亏关系:
    加速比 = mean_acc * t_decode_target / ((γ+1) * t_draft + t_verify)
如果这个式子的估算值和实际端到端加速一致, 就说明瓶颈定位正确。
"""
import os
import sys
from time import perf_counter
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.context import reset_context, set_context

TARGET = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
DRAFT = "/nas_data/LLM/qwen/qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Please explain the difference between TCP and UDP in detail."


def _time(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(n):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / n


def main(gamma: int = 5):
    llm = LLM(TARGET, draft_model=DRAFT, num_speculative_tokens=gamma,
              enforce_eager=True, tensor_parallel_size=1,
              max_num_seqs=1, max_model_len=2048, max_num_batched_tokens=8192)
    mr = llm.model_runner
    tok = AutoTokenizer.from_pretrained(TARGET)

    llm.add_request(PROMPT, SamplingParams(greedy=True, max_tokens=256))
    seqs, is_prefill = llm.scheduler.schedule()
    token_ids = mr.call("run", seqs, True)
    llm.scheduler.postprocess(seqs, token_ids, True)
    seq = seqs[0]

    # --- target 普通单步 decode ---
    @torch.inference_mode()
    def t_decode():
        input_ids, positions = mr.prepare_decode([seq])
        mr.run_model(input_ids, positions, False)
        reset_context()

    t_dec = _time(t_decode)

    # --- draft 单步 decode ---
    seq.append_spec_tokens([0] * gamma)
    llm.scheduler.block_manager.may_append_n(seq, gamma)
    for i in range(gamma):
        seq.token_ids[len(seq) - gamma + i] = token_ids[0]
    step = {"i": 0}

    @torch.inference_mode()
    def d_decode():
        input_ids, positions, sm, cl, bt = mr.prepare_draft_decode([seq], step["i"], gamma)
        mr._draft_forward_graph(input_ids, positions, sm, cl, bt)

    t_dft = _time(d_decode)

    # --- target verify (γ+1 个 query) ---
    @torch.inference_mode()
    def t_verify():
        input_ids, positions = mr.prepare_spec_verify([seq], gamma)
        mr.model.compute_logits(mr.model(input_ids, positions))
        reset_context()

    # --- kernel 级 breakdown: 找一个 draft forward 里最耗时的 op ---
    from torch.profiler import ProfilerActivity, profile as _prof
    _input_ids, _positions = None, None

    @torch.inference_mode()
    def _dbody_prep():
        nonlocal _input_ids, _positions
        _input_ids, _positions = mr.prepare_draft_decode([seq], 0, gamma)[:2]
        return None

    @torch.inference_mode()
    def _dbody_only():
        i, p, sm, cl, bt = mr.prepare_draft_decode([seq], 0, gamma)
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
        mr.draft_model(i, p)
        reset_context()

    _dbody_prep()
    with _prof(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
        for _ in range(5):
            _dbody_only()
    print("\n  ---- draft forward 的 kernel 级耗时 top15 ----")
    tbl = prof.key_averages().table(sort_by="cuda_time_total", row_limit=15)
    for line in tbl.splitlines():
        print("   ", line)

    t_ver = _time(t_verify)

    # --- draft forward 拆分: 主干 vs lm_head ---
    @torch.inference_mode()
    def d_body():
        input_ids, positions, sm, cl, bt = mr.prepare_draft_decode([seq], 0, gamma)
        set_context(False, slot_mapping=sm, context_lens=cl, block_tables=bt)
        hidden = mr.draft_model(input_ids, positions)
        reset_context()
        return hidden

    hidden = d_body()
    t_dbody = _time(d_body)

    @torch.inference_mode()
    def d_head():
        mr.draft_model.compute_logits(hidden)

    t_dhead = _time(d_head)
    print(f"\n  [draft 拆分] 主干(B=1,24 层): {t_dbody:7.2f} ms   "
          f"lm_head(151936x896): {t_dhead:7.2f} ms")

    # --- 各层一次的 attention 开销 (估算) ---
    import time as _t
    layer = mr.draft_model.model.layers[0]
    q = torch.randn(1, 14, 64, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    t0 = _t.perf_counter()
    for _ in range(100):
        kk = torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
        torch.randn(1, 2, 64, device="cuda", dtype=torch.float16)
    torch.cuda.synchronize()
    print(f"  [参考] 100 次空 op 的 launch 开销 ≈ {(perf_counter() - t0):.2f} ms")

    # --- 只算 attention 前后的差异: 单独测一次 full forward 的 GPU 时间 ---
    llm.scheduler.block_manager.trim(seq, gamma)

    print(f"\ngamma = {gamma}")
    print(f"  target 单步 decode        : {t_dec:8.2f} ms")
    print(f"  draft  单步 decode        : {t_dft:8.2f} ms   ({(gamma + 1) * t_dft:8.2f} ms / 轮, γ+1 次)")
    print(f"  target 一次 verify        : {t_ver:8.2f} ms")
    per_round = (gamma + 1) * t_dft + t_ver
    print(f"  ---- 1 轮总耗时           : {per_round:8.2f} ms")
    spec = llm.model_runner.speculator
    mean_acc = spec.mean_accepted_length or 1.0
    print(f"  实测平均接受长度 mean_acc : {mean_acc:8.2f} token/轮")
    print(f"  等价的普通 decode 耗时    : {mean_acc * t_dec:8.2f} ms")
    print(f"  ==> 单 inline 限预估加速  : {mean_acc * t_dec / per_round:8.2f} x")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 5)
