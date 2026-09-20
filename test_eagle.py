"""EAGLE 推理冒烟测试。

验证:
  1. greedy 下开 EAGLE 与不开投机的解码结果逐 token 相同 (正确性由拒绝采样保证,
     与草稿接受率 α 无关 —— α 再低也必须是 4/4)。
  2. 打印接受率 α / 平均每轮确认 token 数 (spec.summary)。

用法:
  CUDA_VISIBLE_DEVICES=1 python test_eagle.py --eagle_head /tmp/eagle_smoke.pt
"""

from __future__ import annotations

import argparse

from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams


MODEL = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
PROMPTS = [
    "请介绍一下你自己。",
    "中国的首都是哪里？",
    "用一句话解释什么是深度学习。",
    "1加1等于几？请说明原因。",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle_head", required=True, help="训练好的 EAGLE head 权重")
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=64)
    args = ap.parse_args()

    params = SamplingParams(greedy=True, max_tokens=args.max_tokens)

    # 1) 基线 (不开投机)
    base = LLM(MODEL, gpu_memory_utilization=0.45)
    out_base = base.generate(PROMPTS, params)
    base.exit()

    # 2) EAGLE 投机解码
    eagle = LLM(MODEL, eagle_head=args.eagle_head, num_speculative_tokens=args.gamma,
                gpu_memory_utilization=0.45)
    out_eagle = eagle.generate(PROMPTS, params)
    print("\n=== EAGLE spec.summary ===")
    print(eagle.model_runner.speculator.summary())
    eagle.exit()

    # 3) 逐 token 比对 (必须 4/4)
    ok = True
    for p, b, e in zip(PROMPTS, out_base, out_eagle):
        bt = b["token_ids"]
        et = e["token_ids"]
        if bt != et:
            ok = False
            print(f"[MISMATCH] {p}\n  base : {b['text'][:60]!r}\n  eagle: {e['text'][:60]!r}")
        else:
            print(f"[4/4 OK] len={len(bt)}  {b['text'][:30]!r}")
    print("\nALL 4/4 ✓" if ok else "\nFAILED ✗")


if __name__ == "__main__":
    main()
