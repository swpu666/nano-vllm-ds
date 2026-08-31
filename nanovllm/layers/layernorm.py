from __future__ import annotations
import torch
from torch import nn


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        # 与输入 dtype 一致, 避免 fp32 权重导致输出提升为 float32
        self.weight = nn.Parameter(torch.ones(hidden_size, dtype=torch.get_default_dtype()))

    @torch.inference_mode()
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # 归一化在 fp32 下计算(精度), 输出还原为输入 dtype(fp16), 与 vLLM 一致
        dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.mul(self.weight.float()).to(dtype)

    @torch.inference_mode()
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dtype = x.dtype
        x = x.float().add(residual.float())
        residual = x.clone().to(dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        return x.mul(self.weight.float()).to(dtype), residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
