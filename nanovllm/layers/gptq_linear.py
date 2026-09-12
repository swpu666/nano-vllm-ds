from __future__ import annotations
import os
import torch
from torch import nn
import torch.nn.functional as F

# 四条执行路径 (环境变量在 import 时读取; 默认 fused, 见下方说明):
#
#   fused (默认)  Triton dequant-GEMM, fp16 权重完全不物化 (对标 Marlin, 这才是真正
#                的"重写 GEMM")。显存 5.2GiB (0.5B/param, 量化的真实收益), 吞吐 ~77 tok/s,
#                Part C 与 vLLM(gptq_marlin) 64/64 对齐 —— 同显存下比 stream 快 1.3x。
#                与 cuBLAS 相比: prefill(M>=32) 逐位一致, decode(M<=4) 有 1~4 ulp 差异,
#                但这点差异**不足以**造成贪心分歧 (实测仍 64/64)。
#   stream        int4 常驻显存, 每次 forward 用 Triton kernel 把权重展开到一个**复用**的
#                fp16 暂存区, 再交给 cuBLAS。显存同 fused, 但多了展开+回读的带宽开销
#                (~59 tok/s)。数值上与 cache 模式逐位相同, 适合作为 fused 的对照基准。
#   cache         只 dequant 一次并长期缓存 fp16。最快 (~89 tok/s), 但显存回到 2B/param
#                (14.2GiB), 放弃了量化的运行期收益 —— 只作为"速度上界"参照。
#   torch         旧朴素实现 (torch 张量运算解包), 访存量约为权重的 30 倍, 仅作对照。
#
# 重要历史教训: fused 曾长期 0/64, 一度被归因为"tl.dot 归约顺序与 cuBLAS 不同"。
# 实测证伪 —— 真因是 bias 被加了两次 (_forward_fused 内一次, forward() 又一次),
# 而 Qwen2 中只有 q/k/v 投影带 bias, 所以只有这三个投影整体偏移一个 bias。
# 归约顺序造成的 1~4 ulp 并不足以让贪心解码分歧。详见 _forward_fused 的注释。
_GPTQ_CACHE = os.getenv("NANOVLLM_GPTQ_CACHE", "0") == "1"
_GPTQ_TORCH = os.getenv("NANOVLLM_GPTQ_TORCH", "0") == "1"
_GPTQ_STREAM = os.getenv("NANOVLLM_GPTQ_STREAM", "0") == "1"

try:
    from nanovllm.layers.gptq_triton import (fused_gptq_linear, ordered_gptq_linear,
                                             ORDERED_MAX_M)
    _HAS_TRITON = True
except Exception:
    ORDERED_MAX_M = 0
    _HAS_TRITON = False
_FUSED = os.getenv("NANOVLLM_GPTQ_FUSED", "0") == "1" and _HAS_TRITON
_HAS_DEQUANT = True
try:
    from nanovllm.layers.gptq_dequant import dequantize_gptq, scratch_buffer
except Exception:
    _HAS_DEQUANT = False


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
        self.mode = ("fused" if _FUSED else
                     "cache" if self.cache_dequant else
                     "torch" if _GPTQ_TORCH or not _HAS_DEQUANT else
                     "stream" if _GPTQ_STREAM else
                     "fused" if _HAS_TRITON else
                     "stream")
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
        if self.mode == "fused":
            out = self._forward_fused(xf)
        elif self.mode == "cache":
            out = xf @ self._cached_weight().t()
        elif self.mode == "stream":
            out = self._forward_stream(xf)
        else:
            # 旧朴素路径: 分块 dequant + fp16 matmul (TensorCore, 内部 fp32 累加)。
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

    def _forward_stream(self, xf: torch.Tensor) -> torch.Tensor:
        """int4 常驻显存: Triton 把权重展开到复用的 fp16 暂存区, 再用 cuBLAS 做 GEMM。
        展开结果与 cache 模式逐位一致 -> 数值行为与 cache/vLLM 对齐;
        显存只多出一个最大层大小的 buffer (7B 约 136MB), 而不是整模型的 14GB。"""
        W = dequantize_gptq(self.qweight, self.qzeros, self.scales,
                            group_size=self.group_size,
                            out=scratch_buffer(self.out_features * self.in_features, xf.device))
        return xf @ W.t()

    def _forward_fused(self, xf: torch.Tensor) -> torch.Tensor:
        """int4 dequant-GEMM 混合核: 不物化 fp16 权重 (显存 0.5B/param), 且数值与 vLLM 对齐。

          M <= ORDERED_MAX_M (decode + 短 prefill) -> ordered 核: dequant 后升 fp32 精确乘加
          M >  ORDERED_MAX_M (长 prefill)         -> tl.dot 核: 已验证与 cuBLAS 逐位一致

        短 prompt 的 prefill M 只有 10~30, 必须走 ordered —— 否则 prefill 就错了。
        支持任意前导维度 (引擎在 prefill 时可能传 (batch, seq, in))。"""
        xc = xf.contiguous()
        *lead, K = xc.shape
        x2 = xc.reshape(-1, K)
        M = x2.shape[0]
        fn = ordered_gptq_linear if M <= ORDERED_MAX_M else fused_gptq_linear
        out = fn(x2, self.qweight, self.qzeros, self.scales,
                 M, self.out_features, K, self.group_size)
        out = out.reshape(*lead, self.out_features)
        # 注意: bias 由 forward() 统一加, 这里**不能**再加一次。
        # 历史上这里多加了一次 -> q/k/v (Qwen2 中唯一带 bias 的投影) 结果整体偏移一个 bias,
        # 这正是 fused 模式端到端 0/64 的真正主因 (而非 tl.dot 的归约顺序)。
        return out


class GPTQRowParallelLinear(GPTQColumnParallelLinear):
    """行并行线性; TP=1 时与 Column 等价 (不切分)。"""
