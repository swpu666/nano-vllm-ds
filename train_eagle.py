"""训练 EAGLE draft head。

思路 (与 nanovllm/models/eagle.py 的 docstring 对应):
  1. 用 target 模型 (GPTQ-Int4) 在语料上跑 prefill, 取最后一层隐藏状态 H
     (每个位置 t 的 H[t] 就是"预测第 t+1 个 token 时的内部表征")。
  2. 监督 head: 输入 (x_t 的 embedding, H[t]), 让它同时预测
        - 下一 token  x_{t+1}   (CE loss)
        - 下一位置的真实隐藏状态 H[t+1]  (MSE loss)
     后者正是 EAGLE 比"独立 draft 模型"接受率更高的关键 —— head 显式拟合了
     target 的隐藏层动态, 而不只是去逼近 token 分布。
  3. 只存 head 自有参数 (fc + 一个 transformer block); embed / lm_head 复用 target,
     不保存。

用法:
  # 用 dense 基模最后一层初始化 block (推荐, 收敛快), 在语料上训练
  python train_eagle.py \
      --model /nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4 \
      --base_model /nas_data/WR/qwen/models/Qwen2.5-7B-Instruct \
      --data data/eagle_train.txt \
      --out eagle_head.pt --steps 2000 --max_len 256 --lr 1e-4

  # 不加载基模 (默认随机初始化 block), 仅做快速冒烟
  python train_eagle.py --model <gptq> --data <txt> --out eagle_head.pt --steps 50 --no_base
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

from transformers import AutoTokenizer, AutoModelForCausalLM

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import set_context, reset_context
from nanovllm.models.eagle import EAGLEHead


def build_target(model_path: str) -> ModelRunner:
    """用 ModelRunner 托管 target: 它负责加载权重、分配 KV、绑定注意力上下文。

    训练时我们只要它的 `model` (跑 prefill 拿隐藏状态), KV 仅用于让注意力
    前向正常走完, 不参与梯度。
    """
    # 训练时 target 只跑单条 chunk 的 prefill, 不需要大 KV cache, 把显存让给 head + 优化器
    cfg = Config(model_path, max_model_len=1024, gpu_memory_utilization=0.05)
    if cfg.quantization != "gptq":
        print(f"[warn] target 不是 GPTQ ({cfg.quantization}), 继续但 head 监督目标仍为隐藏状态")
    return ModelRunner(cfg, 0, [])


@torch.no_grad()
def target_hidden(mr: ModelRunner, ids: list[int]) -> torch.Tensor:
    """跑一次 prefill, 返回 target 最后一层隐藏状态 (T, H) fp16。

    复用训练期 KV cache 的 slot 0..T-1 (每个 chunk 独立, 互相覆盖无妨)。
    """
    T = len(ids)
    input_ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
    positions = torch.arange(T, dtype=torch.int64, device="cuda")
    cu_q = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    cu_k = torch.tensor([0, T], dtype=torch.int32, device="cuda")
    slot = torch.arange(T, dtype=torch.int32, device="cuda")
    set_context(True, cu_q, cu_k, T, T, slot, None, None)
    H = mr.model(input_ids, positions)
    reset_context()
    return H.detach()


def load_corpus(path: str, tokenizer, max_len: int, max_chunks: int = 200000):
    """把语料切成长度 max_len 的 token chunk (丢弃不足 max_len 的尾部)。"""
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = tokenizer.encode(text, add_special_tokens=False)
    chunks = [ids[i:i + max_len] for i in range(0, len(ids) - max_len + 1, max_len)]
    if max_chunks:
        chunks = chunks[:max_chunks]
    print(f"[data] {path}: {len(ids)} tokens -> {len(chunks)} chunks @ {max_len}")
    return chunks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="target GPTQ 模型目录")
    ap.add_argument("--base_model", default="/nas_data/WR/qwen/models/Qwen2.5-7B-Instruct",
                    help="dense 基模 (用于初始化 head block); 传空字符串则随机初始化")
    ap.add_argument("--data", required=True, help="训练语料 (纯文本)")
    ap.add_argument("--out", default="eagle_head.pt", help="head 权重输出路径")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--max_len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--hidden_loss_w", type=float, default=1.0, help="MSE(隐藏状态) 的权重")
    ap.add_argument("--no_base", action="store_true", help="不加载基模, 随机初始化 block")
    ap.add_argument("--max_chunks", type=int, default=0, help="0=全部; 否则最多取前 N 个 chunk")
    args = ap.parse_args()

    # CUDA_VISIBLE_DEVICES 已把目标卡映射为 cuda:0
    torch.cuda.set_device(0)

    mr = build_target(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    hf = mr.config.hf_config

    head = EAGLEHead(
        mr.model,
        hidden_size=hf.hidden_size,
        num_heads=hf.num_attention_heads,
        num_kv_heads=getattr(hf, "num_key_value_heads", hf.num_attention_heads),
        head_dim=getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
        intermediate_size=hf.intermediate_size,
        eps=getattr(hf, "rms_norm_eps", 1e-6),
    ).cuda()
    head.train()
    # 训练时把可训练 block 提升到 fp32 (master weights), 共享的 embed/lm_head 保持 fp16
    # (target 是 fp16, 不能被改)。这样CE+MSE 在 fp32 下稳定收敛, 推理时再 .half()。
    for name, p in head.named_parameters():
        if p.requires_grad:
            p.data = p.data.float()

    if not args.no_base and args.base_model:
        print(f"[init] 从 dense 基模加载 block: {args.base_model}")
        base = AutoModelForCausalLM.from_pretrained(args.base_model, torch_dtype=torch.float16,
                                                    device_map="cpu")
        with torch.no_grad():
            head.init_from_dense(base)
        del base
        torch.cuda.empty_cache()

    params = [p for p in head.parameters() if p.requires_grad]
    print(f"[train] 可训练参数 {sum(p.numel() for p in params)/1e6:.1f}M, "
          f"lr={args.lr}, hidden_loss_w={args.hidden_loss_w}")
    optimizer = torch.optim.AdamW(params, lr=args.lr)

    chunks = load_corpus(args.data, tokenizer, args.max_len, args.max_chunks or None)
    assert chunks, "语料为空"

    step = 0
    while step < args.steps:
        for chunk in chunks:
            if step >= args.steps:
                break
            ids = chunk
            tokens = torch.tensor([ids], dtype=torch.int64, device="cuda")
            with torch.no_grad():
                H = target_hidden(mr, ids)                     # (T, H) fp16
            Hf = H.unsqueeze(0).float()                        # (1, T, H) 训练目标(fp32)
            optimizer.zero_grad(set_to_none=True)
            logits, hpred = head.forward_sequence(tokens, Hf.half())
            loss_tok = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, logits.size(-1)),
                tokens[:, 1:].reshape(-1))
            loss_hid = F.mse_loss(hpred[:, :-1].float(), Hf[:, 1:])
            loss = loss_tok + args.hidden_loss_w * loss_hid
            loss.backward()
            optimizer.step()
            step += 1
            if step % 20 == 0:
                print(f"step {step:5d}  loss={loss.item():.3f} "
                      f"(tok={loss_tok.item():.3f} hid={loss_hid.item():.4f})")

    head.eval()
    head.save_head(args.out)
    print(f"[done] head 已保存到 {args.out}")


if __name__ == "__main__":
    main()
