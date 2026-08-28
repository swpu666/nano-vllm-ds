from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F


def _unpack_qweight(qw: torch.Tensor) -> torch.Tensor:
    """
    qweight: (in//8, out) int32, 每个 int32 复用 8 个 4-bit 权重 (沿输入维打包).
    返回 (out, in) float, 直接可用于 F.linear(x, W).
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw.device)
    w = (qw.unsqueeze(-1) >> shifts) & 0xF            # (in//8, out, 8)
    w = w.permute(1, 0, 2).reshape(qw.shape[1], -1)   # (out, in)
    return w


def _unpack_qzeros(qz: torch.Tensor) -> torch.Tensor:
    """
    qzeros: (n_groups, out//8) int32, 每个 int32 复用 8 个 4-bit zero (沿输出维打包).
    返回 (out, n_groups) float.
    """
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qz.device)
    z = (qz.unsqueeze(-1) >> shifts) & 0xF             # (n_groups, out//8, 8)
    z = z.reshape(qz.shape[0], -1)                    # (n_groups, out)
    z = z.transpose(0, 1)                             # (out, n_groups)
    return z


class GPTQColumnParallelLinear(nn.Module):
    """GPTQ 4-bit 线性层 (TP=1 退化为整块加载)。"""
    def __init__(self, input_size: int, output_size: int, bias: bool = False, group_size: int = 128):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.group_size = group_size
        self.n_groups = input_size // group_size
        # 存储布局与 HF GPTQ 一致:
        #   qweight: (in//8, out)
        #   qzeros : (n_groups, out//8)
        #   scales : (n_groups, out)
        self.qweight = nn.Parameter(torch.zeros(input_size // 8, output_size, dtype=torch.int32), requires_grad=False)
        self.qzeros = nn.Parameter(torch.zeros(self.n_groups, output_size // 8, dtype=torch.int32), requires_grad=False)
        self.scales = nn.Parameter(torch.zeros(self.n_groups, output_size, dtype=torch.float16), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float16), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        # 经数值验证: 该 gptqmodel 新格式 qzeros 直接存真实 zero point (sym 下≈7),
        # 无需偏移, 直接 W=(Q-Z)*S。若换用 exllama v1 旧格式改为 8。
        self.zero_offset = 0

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, shard_id=None):
        param.data.copy_(loaded_weight)

    def _dequant_block(self, out_start: int, out_end: int) -> torch.Tensor:
        """反量化 [out_start, out_end) 行, 返回 (rows, in_features) fp32。"""
        nb = out_end - out_start
        qw = self.qweight[:, out_start:out_end]           # (in//8, rows)
        qz = self.qzeros[:, out_start // 8:(out_end + 7) // 8]  # (ngroups, ceil(rows/8))
        sc = self.scales[:, out_start:out_end]            # (ngroups, rows)
        w = _unpack_qweight(qw)                            # (rows, in)
        z = _unpack_qzeros(qz).float().unsqueeze(-1)       # (rows, ngroups, 1)
        s = sc.transpose(0, 1).float().unsqueeze(-1)       # (rows, ngroups, 1)
        w = w.float().reshape(nb, self.n_groups, self.group_size)
        w = (w - (z - self.zero_offset)) * s
        return w.reshape(nb, self.in_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 分块 dequant + fp32 matmul, 控制峰值显存 (退化模型激活 std 大, 需 fp32)
        out_features = self.out_features
        xf = x.float()
        block = 2048 if out_features > 4096 else out_features
        parts = []
        for start in range(0, out_features, block):
            end = min(start + block, out_features)
            wb = self._dequant_block(start, end)           # (block, in) fp32
            parts.append(xf @ wb.t())
        out = torch.cat(parts, dim=-1)
        if self.bias is not None:
            out = out + self.bias.float()
        return out.to(x.dtype)


class GPTQRowParallelLinear(GPTQColumnParallelLinear):
    """行并行线性; TP=1 时与 Column 等价 (不切分)。"""
