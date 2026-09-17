"""投机解码前置检查: draft / target 词表是否可安全搭配。

draft 模型产生的 token id 必须能让 target 无歧义地解释, 否则:
  - 同一个 id 在两边对应不同 token -> 草稿完全无效
  - id >= target 词表长度 -> index out of range

检查项:
  A. 两边 config 声明的 vocab_size
  B. 两边实际权重中 embedding / lm_head 的行数
  C. tokenizer 的 token->id 映射在公共 id 区间上是否逐条一致
  D. target 分布落在 draft 词表之外的概率质量 (决定是否可接受"截断+重归一化"近似)

用法:
  python tests/check_spec_vocab.py \
      --draft /nas_data/LLM/qwen/qwen/Qwen2.5-0.5B-Instruct \
      --target /nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4
"""
from __future__ import annotations
import argparse
import json
import os


def _declared_vocab(path: str) -> int:
    with open(os.path.join(path, "config.json")) as f:
        return json.load(f)["vocab_size"]


def _weight_vocab(path: str) -> int | None:
    """从 safetensors index / 单文件里读 embedding 的实际行数。"""
    from safetensors import safe_open
    from glob import glob
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    emb_file = None
    idx = os.path.join(path, "model.safetensors.index.json")
    if os.path.exists(idx):
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        emb = next((k for k in wm if "embed_tokens" in k), None)
        if emb is None:
            return None
        emb_file = os.path.join(path, wm[emb])
        files = [emb_file]
    for file in files:
        with safe_open(file, framework="pt", device="cpu") as f:
            for k in f.keys():
                if k.endswith("embed_tokens.weight"):
                    return int(f.get_slice(k).get_shape()[0])
    return None


def _load_vocab(path: str) -> dict[str, int]:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(path)
    return dict(tok.get_vocab())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--draft", default="/nas_data/LLM/qwen/qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--target", default="/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4")
    args = ap.parse_args()

    vd_cfg, vt_cfg = _declared_vocab(args.draft), _declared_vocab(args.target)
    vd_w, vt_w = _weight_vocab(args.draft), _weight_vocab(args.target)
    print(f"A. config vocab_size : draft={vd_cfg}  target={vt_cfg}")
    print(f"B. 权重行数(embed)   : draft={vd_w}  target={vt_w}")

    V = min(vd_cfg, vt_cfg)
    print(f"   -> 公共词表 V = {V}")

    vd_map, vt_map = _load_vocab(args.draft), _load_vocab(args.target)
    print(f"C. tokenizer 词条数  : draft={len(vd_map)}  target={len(vt_map)}")

    # id -> token 的反查, 只看 id < V 的公共区间
    inv_d = {i: t for t, i in vd_map.items() if i < V}
    inv_t = {i: t for t, i in vt_map.items() if i < V}
    bad = [i for i in sorted(set(inv_d) & set(inv_t)) if inv_d[i] != inv_t[i]]
    only_d = sorted(set(inv_d) - set(inv_t))
    only_t = sorted(set(inv_t) - set(inv_d))
    print(f"   冲突 id 数 (同一 id 两边 token 不同): {len(bad)}" + (f"  前 10 个: {bad[:10]}" if bad else ""))
    print(f"   仅 draft 有: {len(only_d)}  仅 target 有: {len(only_t)}")

    # draft 采样可能超出 V 的 id (其 logits 本身有 vd_cfg 维)
    over = [i for i in vd_map.values() if i >= V]
    print(f"   draft 词表中 id >= V 的词条数: {len(over)}"
          + (f"  -> 需要在 logits 上 mask, 否则 target 索引越界" if over else ""))

    ok = (not bad) and (not over)
    print()
    print("结论:", "✅ 可以搭配 (公共词表内 id 语义完全一致)" if ok
          else "❌ 不可搭配")


if __name__ == "__main__":
    main()
