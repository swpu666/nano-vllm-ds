"""EAGLE 风格的 draft head: 用 target 的隐藏层监督训练。

设计 (与 EAGLE 原论文一致):
  - 输入 = (token 的 embedding, target 在上一位置的隐藏状态),
    输出 = (下一 token 的 logits, 预测的下一位置 target 隐藏状态)。
  - 预测出的隐藏状态会作为下一步的输入, 从而**自回归地**生成候选链
    (起草阶段完全不需要 target 参与); 整条链由 target 一次性 verify。
  - embed / lm_head 直接复用 target (同一对象, 冻结), 不额外占词表参数;
    只额外一个 transformer block (注意力 + MLP) + 一个 fc 投影, 把
    concat(embed(token), target_hidden) 投回 hidden_size。
  - 训练目标 = 在 target 隐藏状态上做监督:
        logits  用 CE        预测下一 token  x_{t+1}
        pred_hidden 用 MSE   逼近 target 在下一位置真实的隐藏状态 H[t+1]
    这样 head 既学会了"下一个 token 长什么样", 也学会了"target 此时的内部
    表征长什么样", 正是 EAGLE 比普通 draft-model 接受率更高的根源。

head 自带的 chain KV cache 每轮重置 (短链 ≤ γ+1), 不参与 paged KV 管理。
"""

from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F

from nanovllm.layers.layernorm import RMSNorm


class EAGLEHead(nn.Module):

    def __init__(
        self,
        target: nn.Module,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        intermediate_size: int,
        eps: float = 1e-6,
        dtype: torch.dtype = torch.float16,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.q_size = num_heads * head_dim
        self.kv_size = num_kv_heads * head_dim
        self.n_rep = num_heads // num_kv_heads

        # 共享 target 的 embed / lm_head / 最后的 final norm (同一对象, 冻结, 不计入可训练参数)
        # 注意: model() 返回的隐藏状态已经过 self.norm, 所以 head 输出也必须再过一次
        # 同样的 norm, 否则 lm_head (在 post-norm 空间训练) 收到的 hidden 处于错误归一化空间,
        # 草稿 logits 会全部失真 (实测 α 直接掉到 0)。
        self.embed = target.model.embed_tokens
        self.lm_head = target.lm_head
        self.norm = target.model.norm
        for p in self.embed.parameters():
            p.requires_grad = False
        for p in self.lm_head.parameters():
            p.requires_grad = False
        for p in self.norm.parameters():
            p.requires_grad = False

        # 额外可训练部分: fc 投影 + 一个 transformer block
        self.fc = nn.Linear(2 * hidden_size, hidden_size, bias=False, dtype=dtype)
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.q_proj = nn.Linear(hidden_size, self.q_size, bias=False, dtype=dtype)
        self.k_proj = nn.Linear(hidden_size, self.kv_size, bias=False, dtype=dtype)
        self.v_proj = nn.Linear(hidden_size, self.kv_size, bias=False, dtype=dtype)
        self.o_proj = nn.Linear(self.q_size, hidden_size, bias=False, dtype=dtype)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False, dtype=dtype)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False, dtype=dtype)

        # 运行期 chain KV cache (每轮 reset_chain), 不持久化为参数
        self._kcache = self._vcache = None
        self._step = 0

    # ---------------------------------------------------------------- 初始化
    @torch.no_grad()
    def init_from_dense(self, dense_model: nn.Module):
        """用 dense 基模的**最后一层**初始化 block (EAGLE 标准做法, 收敛快)。

        仅复制 transformer block 的权重; fc / 两个 norm 用默认初始化即可
        (fc 负责把 (embed, hidden) 对齐到 block 输入分布, 学起来很快)。
        """
        layer = dense_model.model.layers[-1]

        def cp(src, dst):
            dst.weight.data.copy_(src.weight.data.to(dst.weight.dtype))

        cp(layer.self_attn.q_proj, self.q_proj)
        cp(layer.self_attn.k_proj, self.k_proj)
        cp(layer.self_attn.v_proj, self.v_proj)
        cp(layer.self_attn.o_proj, self.o_proj)
        cp(layer.mlp.gate_proj, self.gate_proj)
        cp(layer.mlp.up_proj, self.up_proj)
        cp(layer.mlp.down_proj, self.down_proj)
        self.input_layernorm.weight.data.copy_(layer.input_layernorm.weight.data)
        self.post_attention_layernorm.weight.data.copy_(layer.post_attention_layernorm.weight.data)

    # ------------------------------------------------------------- chain KV
    # 不装饰 inference_mode: 训练时 (forward_sequence) 要写入普通张量;
    # 推理时由 draft_logits 的 inference_mode 上下文覆盖。
    def reset_chain(self, batch_size: int, max_len: int, device="cuda"):
        self._step = 0
        # 用 list 累积 K/V (而非原地写入共享 buffer), 避免训练时跨步 autograd 报
        # "variable modified by inplace operation"。G 很小, 每步 stack 开销可忽略。
        self._klist: list[torch.Tensor] = []
        self._vlist: list[torch.Tensor] = []

    # --------------------------------------------------------------- 单步
    # 不装饰 inference_mode: 训练时 (forward_sequence) 需要走 autograd;
    # 推理时由 draft_logits 的 inference_mode 上下文覆盖, 行为一致。
    def forward_one(self, token: torch.Tensor, target_hidden: torch.Tensor):
        """(token, target_hidden) -> (logits, pred_hidden)。自回归地推进 chain。

        token           : (B,) int64
        target_hidden   : (B, H) 上一位置的 target 隐藏状态 (teacher forcing 用真值,
                          推理时则是上一 head 步预测出的 hidden)
        返回 logits (B, V) 与 pred_hidden (B, H)。
        """
        B = token.size(0)
        # 把输入对齐到 block 权重的 dtype (训练 fp32 / 推理 fp16), 共享的 embed 输出恒 fp16
        dtype = self.fc.weight.dtype
        x = self.embed(token).to(dtype)                             # (B, H)
        target_hidden = target_hidden.to(dtype)
        f = self.fc(torch.cat([x, target_hidden], dim=-1))          # (B, H)
        # 复刻 Qwen2DecoderLayer 的残差约定 (add-rms-norm 融合)
        h_attn = self.input_layernorm(f)                            # residual <- f
        q = self.q_proj(h_attn).view(B, self.num_heads, self.head_dim)
        k = self.k_proj(h_attn).view(B, self.num_kv_heads, self.head_dim)
        v = self.v_proj(h_attn).view(B, self.num_kv_heads, self.head_dim)
        # 累积本步 K/V 到 list (query 只看到 0..step, 天然因果, 无需 mask)
        self._klist.append(k)
        self._vlist.append(v)
        Kc = torch.stack(self._klist, dim=1)                        # (B, step+1, n_kv, hd)
        Vc = torch.stack(self._vlist, dim=1)
        Kc = Kc.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)   # (B, heads, step+1, hd)
        Vc = Vc.repeat_interleave(self.n_rep, dim=2).transpose(1, 2)
        q_ = q.unsqueeze(2)                                         # (B, heads, 1, hd)
        attn = F.scaled_dot_product_attention(q_, Kc, Vc)           # (B, heads, 1, hd)
        attn = attn.squeeze(2).flatten(1)                           # (B, q_size)
        a = self.o_proj(attn)
        h_mlp, residual = self.post_attention_layernorm(a, f)       # add_rms(fused)
        m = self.mlp(h_mlp)
        h_out = m + residual                                       # 末层 residual 加和 (pre-final-norm)
        h_out = self.norm(h_out)                                   # 与 target 一致: 最后一层之后还有 final norm
        logits = self.lm_head(h_out)
        self._step += 1
        return logits, h_out

    def mlp(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

    # ---------------------------------------------------- 推理: 自回归起草
    @torch.inference_mode()
    def draft_logits(self, start_token, start_hidden, G: int) -> tuple[torch.Tensor, torch.Tensor]:
        """从 (start_token, start_hidden) 自回归生成 G 个候选, 返回 (logits, tokens)。

        start_token   : (B,) int64   (= 上一轮最后接受的 token)
        start_hidden  : (B, H)       (= target 在上一轮最后接受位置的真实隐藏状态)
        返回 draft_logits (B, G, V) 与 draft_tokens (B, G) —— 采样交给调用方
        (Speculator.draft_step, 与现有 draft-model 路径完全一致, 保证拒绝采样口径相同)。
        """
        B = start_token.size(0)
        self.reset_chain(B, G, device=start_hidden.device)
        logits_per_step, tokens = [], []
        hidden = start_hidden
        tok = start_token
        for _ in range(G):
            lg, hidden = self.forward_one(tok, hidden)
            logits_per_step.append(lg)
            tokens.append(lg.argmax(dim=-1))        # 占位, 真实采样由 Speculator 完成
            tok = tokens[-1]
        return torch.stack(logits_per_step, dim=1), torch.stack(tokens, dim=1)

    # ---------------------------------------------- 训练: 整条序列前向
    # 不装饰 inference_mode: 需要 autograd。
    def forward_sequence(self, tokens: torch.Tensor, target_hiddens: torch.Tensor):
        """给定一条序列的 token 与每一步的 target 隐藏状态, 返回每步的 (logits, pred_hidden)。

        tokens          : (B, T) int64
        target_hiddens  : (B, T, H)  —— 第 t 个位置 target 的真实隐藏状态 H[t],
                          作为 forward_one 第 t 步的输入 (teacher forcing)
        返回 logits (B, T, V) 与 pred_hidden (B, T, H)。
        训练目标:
            logits[:, t]     预测 tokens[:, t+1]
            pred_hidden[:, t] 逼近 target_hiddens[:, t+1]
        """
        B, T = tokens.shape
        self.reset_chain(B, T, device=tokens.device)
        logits_list, hpred_list = [], []
        for t in range(T):
            lg, h = self.forward_one(tokens[:, t], target_hiddens[:, t])
            logits_list.append(lg)
            hpred_list.append(h)
        return torch.stack(logits_list, dim=1), torch.stack(hpred_list, dim=1)

    # ------------------------------------------------------------- 存/取
    @torch.no_grad()
    def save_head(self, path: str):
        """只存 head 自有参数 (不含共享的 embed / lm_head)。"""
        state = {k: v for k, v in self.state_dict().items()
                 if not k.startswith("embed.") and not k.startswith("lm_head.")
                 and not k.startswith("norm.")}
        torch.save(state, path)

    @torch.no_grad()
    def load_head(self, path: str):
        self.load_state_dict(torch.load(path, map_location="cpu"), strict=False)
