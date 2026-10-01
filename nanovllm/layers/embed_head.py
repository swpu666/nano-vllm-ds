from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill and not context.spec_verify:
            # 普通 prefill: 只需要每个序列**最后一个**位置的 logits 来采样第一个生成 token,
            # 提前把 x 截到 [cu_q[i+1]-1] 这 B 行, 省掉前面所有 prompt token 的 lm_head 矩阵乘。
            last_indices = context.cu_seqlens_q[1:] - 1
            x = x[last_indices].contiguous()
        # 投机解码的 verify 分支 (spec_verify=True): 此时走的是 prefill 路径(要读历史 KV),
        # 但 query 是 [x_{L-1}, d_1, ..., d_γ] 共 γ+1 个位置 —— **每一个位置**都要算 logits:
        #   - 前 γ 个位置的 logits 用来做拒绝采样判据 (p(d_i)/q(d_i))
        #   - 第 γ+1 个位置的 logits 是 bonus token 的分布 (全接受时额外白送的那一个)
        # 所以不能按 "只取最后一行" 截断, 必须保留全部行 -> 也就是不进上面的 if 分支。
        # 这一行就是 model_runner.prepare_spec_verify 里 set_context(..., spec_verify=True)
        # 在 kernel 之外真正生效的地方, 是 verify 能拿到完整 logits 的开关。
        # fp16 matmul (lm_head 输入经 norm 归一化, std 小); 权重为完整 fp16 占显存大, 不全量转 fp32
        logits = torch.nn.functional.linear(x.half(), self.weight)
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
        return logits.to(torch.float16)
