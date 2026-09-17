"""不变式回归测试: 读 paged KV cache 的路径必须与"整段重算"等价。

这个测试来自一次真实的定位过程: 投机解码的 verify 一开始输出全是重复乱码。
当时很容易误判成"投机解码的数值问题", 但把它拆成两个更基础的不变式后立刻定位:

    A. 整段 prefill (完全不读 cache)                     -> 黄金参考
    B. 分两段 chunked prefill (第二段读 paged cache)      -> 必须 == A
    C. 投机解码的 verify forward (同样读 paged cache)     -> 必须 == A

B 完全不涉及投机解码代码。如果 B 就不成立, 那就是框架级 bug, 而不是新模块的问题。
——先做同进程 A/B、把问题收敛到"具体路径 + 具体偏差形态", 再看**同一层**的
hidden states, 比跨进程端到端对比有效得多。

根因记录:
  1) 没有 flash-attn 时, sdpa fallback 完全忽略 block_table, 历史 KV 根本没被读;
  2) 补上 gather 之后仍不对, 因为 PyTorch SDPA 的 is_causal 在 seqlen_q < seqlen_k
     时等价于 j <= i, **不会**自动补偿 (S-L) 的偏移, 于是前缀被整段 mask 掉。
     最终用 layers/attention.py 里的 cached_causal_attn_kernel 解决。

用法:
  python tests/check_prefill_cache.py
"""
from __future__ import annotations
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoTokenizer

from nanovllm.llm import LLM
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.utils.context import reset_context, set_context

TARGET = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
DRAFT = "/nas_data/LLM/qwen/qwen/Qwen2.5-0.5B-Instruct"
PROMPT = "Please explain the difference between TCP and UDP in detail."
G = 4
TOL = 0.2          # fp16 logits 尺度下的经验阈值 (logits 峰值 ~30)


def ref_full_prefill(mr, token_ids):
    """整段 prefill, slot 全 -1 (只算不写), 返回最后一个位置的 logits。"""
    L = len(token_ids)
    input_ids = torch.tensor(token_ids, dtype=torch.int64, device="cuda")
    positions = torch.arange(L, dtype=torch.int64, device="cuda")
    cu = torch.tensor([0, L], dtype=torch.int32, device="cuda")
    slot = torch.full((L,), -1, dtype=torch.int32, device="cuda")
    set_context(True, cu, cu, L, L, slot, None, None)
    with torch.inference_mode():
        logits = mr.model.compute_logits(mr.model(input_ids, positions)).float()
    reset_context()
    return logits[-1]


def _slots(seq, bs, start, end):
    return [seq.block_table[t // bs] * bs + t % bs for t in range(start, end)]


def chunked_prefill(mr, seq, bs, split):
    """两段 chunked prefill: [0, split) 写 cache, [split, L) 读 cache。"""
    token_ids = seq.token_ids
    L = len(token_ids)
    with torch.inference_mode():
        # 第一段: 无 cache
        ids = torch.tensor(token_ids[:split], dtype=torch.int64, device="cuda")
        pos = torch.arange(split, dtype=torch.int64, device="cuda")
        cu1 = torch.tensor([0, split], dtype=torch.int32, device="cuda")
        sm1 = torch.tensor(_slots(seq, bs, 0, split), dtype=torch.int32, device="cuda")
        set_context(True, cu1, cu1, split, split, sm1, None, None)
        mr.model(ids, pos)
        reset_context()
        # 第二段: 读 cache
        ids = torch.tensor(token_ids[split:], dtype=torch.int64, device="cuda")
        pos = torch.arange(split, L, dtype=torch.int64, device="cuda")
        nq = L - split
        cu_q = torch.tensor([0, nq], dtype=torch.int32, device="cuda")
        cu_k = torch.tensor([0, L], dtype=torch.int32, device="cuda")
        sm2 = torch.tensor(_slots(seq, bs, split, L), dtype=torch.int32, device="cuda")
        bt = torch.tensor([seq.block_table], dtype=torch.int32, device="cuda")
        set_context(True, cu_q, cu_k, nq, L, sm2, None, bt)
        logits = mr.model.compute_logits(mr.model(ids, pos)).float()
        reset_context()
    return logits[-1]


def _check(name: str, cond: bool, extra: str = "") -> bool:
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"   {extra}" if extra else ""))
    return bool(cond)


def main():
    llm = LLM(TARGET, draft_model=DRAFT, num_speculative_tokens=G,
              enforce_eager=True, tensor_parallel_size=1,
              max_num_seqs=1, max_model_len=2048, max_num_batched_tokens=8192)
    mr = llm.model_runner
    bs = mr.block_size
    token_ids = AutoTokenizer.from_pretrained(TARGET).encode(PROMPT)
    print(f"prompt tokens = {len(token_ids)}   block_size = {bs}")

    seq = Sequence(list(token_ids), SamplingParams(greedy=True, max_tokens=32))
    llm.scheduler.block_manager.allocate(seq, 0)

    ref = ref_full_prefill(mr, token_ids)
    ok = True

    for split in (len(token_ids) - 1, len(token_ids) // 2):
        got = chunked_prefill(mr, seq, bs, split)
        d = (ref - got).abs().max().item()
        ok &= _check(f"chunked prefill (split={split}) == 整段 prefill  [max|Δ|={d:.4e}]",
                     d < TOL)

    # verify forward 同样必须等于黄金参考 (它的 query 起点就是 x_{L-1})
    seq.append_spec_tokens([0] * G)
    llm.scheduler.block_manager.may_append_n(seq, G)
    for i in range(G):
        seq.token_ids[len(seq) - G + i] = token_ids[-1]
    with torch.inference_mode():
        input_ids, positions = mr.prepare_spec_verify([seq], G)
        logits = mr.model.compute_logits(mr.model(input_ids, positions)).float()
        reset_context()
    v = logits.view(1, G + 1, -1)[0, 0]
    d = (ref - v).abs().max().item()
    ok &= _check(f"spec verify(pos0) == 整段 prefill  [max|Δ|={d:.4e}]", d < TOL)
    llm.scheduler.block_manager.trim(seq, G)

    print("\n总结:", "✅ 读 paged cache 的路径与整段重算等价" if ok else "❌ 存在不等价路径")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
