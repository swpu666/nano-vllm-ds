import torch
from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.utils.context import set_context, reset_context
from nanovllm.models.eagle import EAGLEHead
from transformers import AutoTokenizer

MODEL = "/nas_data/WR/models/Qwen2.5-7B-Instruct-GPTQ-Int4"
HEAD = "/tmp/eagle_v3.pt"
PROMPT = "请介绍一下你自己。"

mr = ModelRunner(Config(MODEL, max_model_len=1024, gpu_memory_utilization=0.05), 0, [])
tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok.encode(PROMPT)
T = len(ids)
input_ids = torch.tensor(ids, dtype=torch.int64, device="cuda")
positions = torch.arange(T, dtype=torch.int64, device="cuda")
cu_q = torch.tensor([0, T], dtype=torch.int32, device="cuda")
cu_k = torch.tensor([0, T], dtype=torch.int32, device="cuda")
slot = torch.arange(T, dtype=torch.int32, device="cuda")
set_context(True, cu_q, cu_k, T, T, slot, None, None)
with torch.inference_mode():
    H = mr.model(input_ids, positions)
reset_context()
with torch.inference_mode():
    tgt_logits = mr.model.compute_logits(H[-1:])   # context 已清空 -> 返回完整 logits
H_last = H[-1]

hf = mr.config.hf_config
head = EAGLEHead(mr.model, hf.hidden_size, hf.num_attention_heads,
                 getattr(hf, "num_key_value_heads", hf.num_attention_heads),
                 getattr(hf, "head_dim", hf.hidden_size // hf.num_attention_heads),
                 hf.intermediate_size, getattr(hf, "rms_norm_eps", 1e-6)).cuda().half().eval()
head.load_head(HEAD)
head.reset_chain(1, 1)
with torch.inference_mode():
    lg, h = head.forward_one(torch.tensor([ids[-1]], dtype=torch.int64, device="cuda"),
                              H_last.unsqueeze(0).half())

print("last token:", tok.decode([ids[-1]]))
print("head  argmax:", tok.decode([lg[0].argmax(-1).item()]))
print("target argmax:", tok.decode([tgt_logits[0].argmax(-1).item()]))
print("head  top5:", [(tok.decode([i]), round(float(lg[0, i]), 1)) for i in lg[0].topk(5).indices.tolist()])
print("target top5:", [(tok.decode([i]), round(float(tgt_logits[0, i]), 1)) for i in tgt_logits[0].topk(5).indices.tolist()])
# 看 head 预测的隐藏状态与 target 隐藏状态差距 (应该很小才说明 head 学对了)
print("||h_head - H_last|| / ||H_last|| =",
      float((h[0] - H_last).norm() / H_last.norm()))

# ---- 诊断: head 是否真用了 target_hidden? 对比真实 hidden vs 零 hidden ----
head.reset_chain(1, 1)
with torch.inference_mode():
    lg_zero, _ = head.forward_one(
        torch.tensor([ids[-1]], dtype=torch.int64, device="cuda"),
        torch.zeros_like(H_last).unsqueeze(0).half())
print("head argmax (真实hidden):", tok.decode([lg[0].argmax(-1).item()]))
print("head argmax (零hidden)  :", tok.decode([lg_zero[0].argmax(-1).item()]))
print("两 hidden 下 logits 差异 norm:", float((lg - lg_zero).norm()))
