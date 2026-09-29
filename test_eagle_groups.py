"""分域测 EAGLE 接受率 α: 看 α 是否随"prompt 与训练分布匹配度"变化。

训练语料是英文 Wikipedia 正文, 所以预期: 英文百科 prompt 的 α 最高。
"""
import argparse
from nanovllm import LLM
from nanovllm.sampling_params import SamplingParams

MODEL = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"

GROUPS = {
    "英文百科(in-domain)": [
        "The Walking Dead is an American post-apocalyptic horror television series",
        "The theory of general relativity was developed by Albert Einstein in the early",
        "Deep learning is a subfield of machine learning that is based on artificial neural",
        "The city of Paris is the capital and most populous city of France, located on the",
    ],
    "英文技术": [
        "Explain what speculative decoding is and why it speeds up LLM inference.",
        "What is the difference between GPTQ and AWQ quantization methods?",
    ],
    "中文助手": [
        "请介绍一下你自己。",
        "中国的首都是哪里？",
    ],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eagle_head", required=True)
    ap.add_argument("--gamma", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=64)
    ap.add_argument("--gpu_mem", type=float, default=0.35)
    args = ap.parse_args()
    params = SamplingParams(greedy=True, max_tokens=args.max_tokens)

    eagle = LLM(MODEL, eagle_head=args.eagle_head, num_speculative_tokens=args.gamma,
                gpu_memory_utilization=args.gpu_mem)
    spec = eagle.model_runner.speculator
    print(f"\n=== EAGLE 分域接受率 (head={args.eagle_head}, γ={args.gamma}) ===")
    for name, prompts in GROUPS.items():
        a0, d0, r0 = spec.num_accepted_tokens, spec.num_draft_tokens, spec.num_rounds
        out = eagle.generate(prompts, params)
        da, dd, dr = (spec.num_accepted_tokens - a0, spec.num_draft_tokens - d0,
                      spec.num_rounds - r0)
        alpha = da / dd if dd else 0.0
        mean = (da + dr) / dr if dr else 0.0
        print(f"  [{name:20s}] alpha={alpha:.3f}  mean_accepted={mean:.2f}  rounds={dr}")
        # 第 0 步草稿是否被接受(最纯的指标: head 首步 vs target)
        print(f"       sample: {out[0]['text'][:60]!r}")
    eagle.exit()


if __name__ == "__main__":
    main()
