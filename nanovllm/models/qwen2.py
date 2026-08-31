from __future__ import annotations
import torch
from torch import nn
import torch.distributed as dist
from transformers import Qwen2Config

from nanovllm.layers.attention import Attention
from nanovllm.layers.layernorm import RMSNorm
from nanovllm.layers.rotary_embedding import get_rope
from nanovllm.layers.embed_head import VocabParallelEmbedding, ParallelLMHead
from nanovllm.layers.gptq_linear import GPTQColumnParallelLinear, GPTQRowParallelLinear


class Qwen2Attention(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        rope_theta: float = 10000,
        rope_scaling: dict | None = None,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        tp_size = dist.get_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        assert self.total_num_kv_heads % tp_size == 0
        self.num_kv_heads = self.total_num_kv_heads // tp_size
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5

        # Qwen2 仅 q/k/v 带 bias; o_proj 与 MLP 均无 bias (已核对 checkpoint 键名)
        self.q_proj = GPTQColumnParallelLinear(hidden_size, self.q_size, bias=qkv_bias, group_size=group_size)
        self.k_proj = GPTQColumnParallelLinear(hidden_size, self.kv_size, bias=qkv_bias, group_size=group_size)
        self.v_proj = GPTQColumnParallelLinear(hidden_size, self.kv_size, bias=qkv_bias, group_size=group_size)
        self.o_proj = GPTQRowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            group_size=group_size,
        )
        if isinstance(rope_scaling, dict):
            rope_theta = rope_scaling.get("rope_theta", rope_theta)
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            self.num_kv_heads,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        q = self.q_proj(hidden_states).view(-1, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(-1, self.num_kv_heads, self.head_dim)
        q, k = self.rotary_emb(positions, q, k)
        o = self.attn(q, k, v)
        output = self.o_proj(o.flatten(1, -1))
        return output


class Qwen2MLP(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        group_size: int = 128,
    ) -> None:
        super().__init__()
        self.gate_proj = GPTQColumnParallelLinear(hidden_size, intermediate_size, bias=False, group_size=group_size)
        self.up_proj = GPTQColumnParallelLinear(hidden_size, intermediate_size, bias=False, group_size=group_size)
        self.down_proj = GPTQRowParallelLinear(intermediate_size, hidden_size, bias=False, group_size=group_size)
        assert hidden_act == "silu"

    def forward(self, x):
        # 与 SiluAndMul 一致: silu(gate) * up (注意不是 gate * silu(up))
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        x = torch.nn.functional.silu(gate) * up
        x = self.down_proj(x)
        return x


class Qwen2DecoderLayer(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.self_attn = Qwen2Attention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            max_position=config.max_position_embeddings,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", True),
            head_dim=getattr(config, "head_dim", None),
            rope_theta=getattr(config, "rope_theta", 10000.0),
            rope_scaling=getattr(config, "rope_scaling", None),
            group_size=getattr(config, "group_size", 128),
        )
        self.mlp = Qwen2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            group_size=getattr(config, "group_size", 128),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # 沿用 nano-vllm 原始约定: 残差加法与归一化融合 (add-rms-norm),
        # 层内不做最后的加法, 交由下一层的 input_layernorm (或最终 norm) 完成。
        if residual is None:
            hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(positions, hidden_states)
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class Qwen2Model(nn.Module):

    def __init__(
        self,
        config: Qwen2Config,
    ) -> None:
        super().__init__()
        self.embed_tokens = VocabParallelEmbedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([Qwen2DecoderLayer(config) for _ in range(config.num_hidden_layers)])
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)   # fp16, 该模型权重健康无需 fp32
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(positions, hidden_states, residual)
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states


class Qwen2ForCausalLM(nn.Module):

    def __init__(
        self,
        config: Qwen2Config
    ) -> None:
        super().__init__()
        self.model = Qwen2Model(config)
        self.lm_head = ParallelLMHead(config.vocab_size, config.hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        return self.model(input_ids, positions)

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.lm_head(hidden_states)
