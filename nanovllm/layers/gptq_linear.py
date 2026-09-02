from __future__ import annotations
import os
import torch
from torch import nn
import torch.nn.functional as F

# 反量化缓存开关 (也可由环境变量 NANOVLLM_GPTQ_CACHE=1 打开)。
# 关闭: 每个 forward 都重新 dequant -> 显存省(0.5B/param), 但受限于显存带宽, 极慢;
# 开启: 只 dequant 一次并缓存 fp16 -> 快, 但显存回到 2B/param (量化只省加载/磁盘)。
_GPTQ_CACHE = os.getenv("NANOVLLM_GPTQ_CACHE", "0") == "1"

# fused dequant-GEMM (Triton): 权重以 int4 驻留, kernel 内解包反量化直接 tl.dot,
# 不物化 fp16 权重 -> 省显存与一次 HBM 往返, 是对标 Marlin "免显存往返"的实现。
# 注意实测: 本硬件(3090/7B)上瓶颈是算力+注意力而非权重带宽, 故 fused 吞吐(77.9 tok/s)
# 略低于 cache 模式(87.7, cuBLAS 更优化); 且 tl.dot 归约顺序 != cuBLAS, 深残差模型下
# 贪心解码会与 vLLM 分歧 -> 精确贪心匹配请用 cache 模式。详见教程 5.5。
try:
    from nanovllm.layers.gptq_triton import fused_gptq_linear
    _HAS_TRITON = True
except Exception:
    _HAS_TRITON = False
_FUSED = os.getenv("NANOVLLM_GPTQ_FUSED", "0") == "1" and _HAS_TRITON


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
    def __init__(self, input_size: int, output_size: int, bias: bool = False, group_size: int = 128,
                 cache_dequant: bool | None = None):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.group_size = group_size
        self.n_groups = input_size // group_size
        self.cache_dequant = _GPTQ_CACHE if cache_dequant is None else cache_dequant
        self._w_cache = None          # 反量化后的 fp16 权重缓存 (惰性构建)
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
        # GPTQ 标准格式: qzeros 存的是 (真实零点 - 1), 故 真实零点 = qzeros + 1。
        # 实测验证(sym=true): 本模型 qzeros 恒为 7, 而解包码字 mean(Q)≈7.998≈8,
        # 即真实零点=8=qzeros+1; 若直接用 qzeros(=7) 反量化, 权重会整体偏移 +1·scale
        # (实测 mean(W)=+0.00746 ≈ scales.mean()=0.00747, 与偏移一个 scale 完全吻合)。
        self.zero_point_bias = 1

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
        w = (w - (z + self.zero_point_bias)) * s
        return w.reshape(nb, self.in_features)

    def _cached_weight(self) -> torch.Tensor:
        """首次调用时反量化整块权重并缓存 fp16, 随后释放 int4 打包权重以省显存。"""
        if self._w_cache is None:
            w = self._dequant_block(0, self.out_features).half()
            for name in ("qweight", "qzeros", "scales"):
                p = getattr(self, name, None)
                if isinstance(p, nn.Parameter):
                    p.data = torch.empty(0, dtype=p.dtype, device=p.device)
            self._w_cache = w
        return self._w_cache

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.half()
        if _FUSED:
            out = self._forward_fused(xf)
        elif self.cache_dequant:
            out = xf @ self._cached_weight().t()
        else:
            # 分块 dequant + fp16 matmul (TensorCore, 内部 fp32 累加), 控制峰值显存。
            # 权重反量化在 fp32 下完成再转 fp16, 保证精度; 激活全程 fp16, 与 vLLM 对齐。
            out_features = self.out_features
            block = 2048 if out_features > 4096 else out_features
            parts = []
            for start in range(0, out_features, block):
                end = min(start + block, out_features)
                wb = self._dequant_block(start, end).half()  # (block, in)
                parts.append(xf @ wb.t())
            out = torch.cat(parts, dim=-1)
        if self.bias is not None:
            out = out + self.bias
        return out

    def _forward_fused(self, xf: torch.Tensor) -> torch.Tensor:
        """fused int4 dequant-GEMM (Triton): 不物化 fp16 权重, 兼顾速度与显存。
        支持任意前导维度 (引擎在 prefill 时可能传 (batch, seq, in))。"""
        xc = xf.contiguous()
        *lead, K = xc.shape
        x2 = xc.reshape(-1, K)
        M = x2.shape[0]
        out = fused_gptq_linear(x2, self.qweight, self.qzeros, self.scales,
                                M, self.out_features, K, self.group_size)
        out = out.reshape(*lead, self.out_features)
        if self.bias is not None:
            out = out + self.bias
        return out


class GPTQRowParallelLinear(GPTQColumnParallelLinear):
    """行并行线性; TP=1 时与 Column 等价 (不切分)。"""
