# nano-vLLM：GPTQ-Int4 量化适配 + 投机解码（教程 + Infra 简历项目）

> **适用对象**：想在极简推理引擎（nano-vLLM，~2000 行）里手搓量化支持与解码加速、或想理解 GPTQ / vLLM / Marlin 工程差异的工程师。
> **代码基准**：本仓库当前实现 —— 量化部分主要是 `nanovllm/layers/gptq_linear.py`、`gptq_triton.py`、`nanovllm/models/qwen2.py`；投机解码部分主要是 `nanovllm/engine/speculator.py`、`nanovllm/engine/model_runner.py`、`nanovllm/layers/attention.py`。
> **硬件**：单卡 RTX 3090 24GB（双卡，但**无 NVLink**，GPU 间走 PCIe）。
> **模型**：target = `Qwen2.5-7B-Instruct-GPTQ-Int4`（GPTQ-Int4, sym, group_size=128, desc_act=False）；投机解码的 draft = `Qwen2.5-0.5B-Instruct`。
>
> 第一部分~第四部分是量化（§1–§10），第五部分「Infra 简历项目」同时包含量化与投机解码两条主线。投机解码的完整设计与实测见 `docs/spec-decoding.md`。

---

## 0. 速览（先看这一节）

### 0.1 两条执行路径

引擎里有 **两条** GPTQ 执行路径，靠环境变量在模块导入时选择。**默认是 `fused`**（不设任何开关即生效）：

| 路径 | 开关 | 权重显存 | 与 vLLM 数值关系 | 4 并发吞吐 (tok/s) | 定位 |
|---|---|---|---|---|---|
| **fused**（默认） | 无开关，或 `NANOVLLM_GPTQ_FUSED=1` | **5.20 GiB** | 逐 token 一致（**64/64**） | **77.4** | 默认：省显存 + 精确 + 最快 |
| torch | `NANOVLLM_GPTQ_TORCH=1` | 5.20 GiB | 逐 token 一致（64/64） | 8.5 | 朴素参考实现 / Triton 不可用时的兜底 |
| *（参考）vLLM gptq_marlin* | — | — | 基准 | 344.4 | 行业上限 |

### 0.2 五个核心结论

1. **零点是唯一必踩的坑**：GPTQ(v1) 磁盘上的 `qzeros` 存的是「真实零点 − 1」，即 `z_true = unpack(qzeros) + 1`。用错不会报错，只会让输出看似流畅但内容错乱。
2. **正确性要用强证据**：`mean(码字)≈真实零点` 这类弱启发式在有偏分布下会误判；正确做法是「黄金权重无损往返 + 变异测试」（见 §4）。
3. **量化换的是显存，不是速度**（对比原版 nano-vLLM + 未量化模型，eager 同口径）：权重显存
   **14.22 → 5.20 GiB（2.73×，省 9.02 GiB）**，4 并发吞吐 78.0 → 76.7 tok/s（**基本持平**），
   TTFT 略慢 14%。
4. **「省显存 + 快 + 对齐 vLLM」可以三者兼得，但必须重写 GEMM**：`fused` 把反量化融进 GEMM、不物化 fp16 权重，得到 5.20 GiB + 77.4 tok/s + 64/64 —— 这是对标 Marlin 的正统路径，也是当前的**默认**。
5. **量化省下的显存正好用来装 draft 模型，把 decode 的访存瓶颈也打掉**：GPTQ 把权重从 14.22 压到 5.20 GiB，腾出的余量装下 0.5B draft；draft-target 投机解码让单请求 decode **30.1 → 118.9 tok/s（3.96×）**、4 并发吞吐 **113.6 → 222.8 tok/s（1.96×）**，且 greedy 下输出与不开投机时**逐 token 相同**。设计细节见 `docs/spec-decoding.md`。

### 0.3 阅读地图

- **只想跑起来**：§9 复现命令。
- **想理解量化的实现**：第一部分（§1 格式 → §2 改动点 → §3 实现）。
- **想验证量化的正确性**：第二部分（§4）。
- **想做量化的性能分析**：第三部分（§5–§8）。
- **想理解投机解码**：`docs/spec-decoding.md`（流程、三个坑、实测）。
- **写简历 / 准备面试**：第五部分（量化 + 投机解码）。

---

# 第一部分：原理与适配实现

## 1. GPTQ 格式速成（不懂一定写错）

GPTQ 是 **weight-only 对称量化**：每个权重用一个 4-bit 整数码字表示。

```
W[i] = (Q[i] - z[g]) * s[g]        # g = i // group_size
```

- `Q`：4-bit 整数码字（用 int8/int32 存储，只取低 4 位，取值 `[0,15]`）。
- `z[g]`：group `g` 的零点；对称量化下理论为常数 8，但**磁盘上存的不是 `z` 本身**（见 §1.4）。
- `s[g]`：group `g` 的 scale，fp16。
- `group_size`：每个 group 覆盖的连续输入维度（本模型 = 128），`n_groups = in_features // group_size`。

### 1.1 磁盘上的三张表

HF 的 `*.safetensors` 里每个线性层存三张表（`g_idx` 在 group 对齐时可忽略，本实现直接跳过）：

| 张量 | 形状 | dtype | 含义 |
|---|---|---|---|
| `qweight` | `(in_features // 8, out_features)` | int32 | 每 int32 打包 8 个 4-bit 码字，沿**输入维**打包 |
| `qzeros` | `(in_features // group_size, out_features // 8)` | int32 | 每 int32 打包 8 个 4-bit 零点，沿**输出维**打包 |
| `scales` | `(in_features // group_size, out_features)` | fp16 | 每 group 一个 scale |

直觉：32 bit / 4 bit = 8，所以 `qweight` 沿输入方向每 8 个码字挤进一个 int32（形状里出现 `in//8`），`qzeros` 沿输出方向同理（出现 `out//8`）。

### 1.2 打包方向（决定解包代码怎么写）

- **`qweight`**：固定输出 `o`，输入索引 `[8m, 8m+7)` 的 8 个码字挤进同一个 int32；**低位 4 bit = 输入索引 `8m`**（第一个元素）。
- **`qzeros`**：固定 group `g`，输出索引 `[8m, 8m+7)` 的 8 个零点挤进同一个 int32；同样低位在前。

一句话记忆：**权重沿 K（输入）打包，零点沿 N（输出）打包，都是小端 nibble（低位在前）**。

### 1.3 反量化公式展开

逐元素等价于：

```
W[n, k] = (Q[n, k] - (unpack(qzeros)[n, g] + 1)) * scales[g, n],   g = k // group_size
```

### 1.4 零点约定（**全文最重要的坑**）

标准 GPTQ（v1，非 GPTQ-v2）磁盘上的 `qzeros` 存的是 **真实零点 − 1**：

```
z_true = unpack(qzeros) + 1
```

权威出处（vLLM 源码 `vllm/.../quantization/utils/bitblas_utils.py` 的 `unpack_gptq_qzeros`）：

```python
def unpack_gptq_qzeros(qzeros, bits, is_gptq_v2=False):
    ...
    if not is_gptq_v2:
        return unpacked_zeros + 1      # ← 真实零点 = 解包值 + 1
    return unpacked_zeros
```

本模型实测佐证：`qzeros` 恒为 7、解包码字 `mean(Q) ≈ 7.998 ≈ 8`，真实零点 = 8 = `qzeros + 1`。若误用 `qzeros`（=7）反量化，整张权重会平移 `+1·scale`——实测 `mean(W) = +0.00746`，与 `scales.mean() = 0.00747` **完全吻合**，正好一个 scale 的系统性偏移。

> 这正是之前"DeepSeek 模型退化"的真正原因：**不是模型坏，是 dequant 零点偏移**。这类"看起来能跑但输出乱码"的 bug，定位要靠数值不变量而非肉眼（见 §4）。

---

## 2. nano-vLLM 原有结构与改动清单

| 原有文件 | 作用 | GPTQ 需要做什么 |
|---|---|---|
| `nanovllm/layers/linear.py` | fp16 的 `Linear` / `QKVParallelLinear` 等 | 新增平行的 `GPTQ*ParallelLinear`，**不动 fp16 路径** |
| `nanovllm/config.py` | `Config` | 加 `quantization` 字段，从 `hf_config.quantization_config` 自动识别 `"gptq"` |
| `nanovllm/utils/loader.py` | `load_model()` | 识别 `qweight/qzeros/scales` 后缀，`copy_` 进 `nn.Parameter` |
| `nanovllm/engine/model_runner.py` | `MODEL_REGISTRY` + `load_model()` | 注册 `Qwen2ForCausalLM`；GPTQ 时强制 eager；注入 `group_size` |
| `nanovllm/models/qwen2.py` | （新增）Qwen2 模型定义 | 把 `Linear` 换成 `GPTQ*ParallelLinear` |
| `nanovllm/layers/layernorm.py` | RMSNorm | 修 dtype，避免 fp32 传染 |

策略要点：**复用 fp16 版本的 `weight_loader` 与 TP 切分逻辑，不重写引擎调度**。Qwen2 的 q/k/v/o 与 `gate/up` 都是分开的投影，因此不需要 `packed_modules_mapping` 融合（这与 Qwen3 不同）。

---

## 3. 适配实现

### 3.1 参数与存储布局（`nanovllm/layers/gptq_linear.py`）

```python
class GPTQColumnParallelLinear(nn.Module):
    """GPTQ 4-bit 线性层 (TP=1 退化为整块加载)。"""
    def __init__(self, input_size, output_size, bias=False, group_size=128):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.group_size = group_size
        self.n_groups = input_size // group_size
        self.mode = "fused" if _FUSED else "torch"
        # 存储布局与 HF GPTQ 一致:
        #   qweight: (in//8, out)
        #   qzeros : (n_groups, out//8)
        #   scales : (n_groups, out)
        self.qweight = nn.Parameter(torch.zeros(input_size // 8, output_size, dtype=torch.int32), requires_grad=False)
        self.qzeros  = nn.Parameter(torch.zeros(self.n_groups, output_size // 8, dtype=torch.int32), requires_grad=False)
        self.scales  = nn.Parameter(torch.zeros(self.n_groups, output_size, dtype=torch.float16), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float16), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        self.zero_point_bias = 1                 # GPTQ v1: 真实零点 = qzeros + 1
```

> 本模型 `o_proj` 与 MLP **都没有 bias**（已核对 checkpoint 键名），`attention_bias` 仅作用于 q/k/v。误给 `o_proj`/MLP 加 bias 会让 `weight_loader` 找不到对应张量而报错。

### 3.2 bit 解包（torch 参考实现，也是 `torch` 路径的底座）

```python
def _unpack_qweight(qw):
    """qweight: (in//8, out) int32 -> (out, in) 码字"""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw.device)
    w = (qw.unsqueeze(-1) >> shifts) & 0xF            # (in//8, out, 8)
    w = w.permute(1, 0, 2).reshape(qw.shape[1], -1)   # (out, in)
    return w

def _unpack_qzeros(qz):
    """qzeros: (n_groups, out//8) int32 -> (out, n_groups)"""
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qz.device)
    z = (qz.unsqueeze(-1) >> shifts) & 0xF             # (n_groups, out//8, 8)
    z = z.reshape(qz.shape[0], -1)                     # (n_groups, out)
    return z.transpose(0, 1)                           # (out, n_groups)
```

### 3.3 反量化数学（分块，含零点）

```python
def _dequant_block(self, out_start, out_end):
    """反量化 [out_start, out_end) 行, 返回 (rows, in_features) fp32。"""
    nb = out_end - out_start
    qw = self.qweight[:, out_start:out_end]                    # (in//8, rows)
    qz = self.qzeros[:, out_start // 8:(out_end + 7) // 8]     # (ngroups, ceil(rows/8))
    sc = self.scales[:, out_start:out_end]                     # (ngroups, rows)
    w = _unpack_qweight(qw)                                    # (rows, in)
    z = _unpack_qzeros(qz).float().unsqueeze(-1)               # (rows, ngroups, 1)
    s = sc.transpose(0, 1).float().unsqueeze(-1)               # (rows, ngroups, 1)
    w = w.float().reshape(nb, self.n_groups, self.group_size)  # (rows, group, k)
    w = (w - (z + self.zero_point_bias)) * s                   # ← 零点 + 1 在此生效
    return w.reshape(nb, self.in_features)
```

> **精度约定**：反量化在 **fp32** 下完成再 `.half()` 给 GEMM，激活全程 fp16。这样中间累加不会在 fp16 下溢出，与 vLLM 行为对齐。

### 3.4 两条路径的分派

```python
def forward(self, x):
    xf = x.half()
    if self.mode == "fused":
        out = self._forward_fused(xf)
    else:                                     # torch: 分块 dequant + fp16 matmul
        block = 2048 if self.out_features > 4096 else self.out_features
        parts = []
        for start in range(0, self.out_features, block):
            end = min(start + block, self.out_features)
            wb = self._dequant_block(start, end).half()      # (block, in) fp16
            parts.append(xf @ wb.t())
        out = torch.cat(parts, dim=-1)
    return out + self.bias if self.bias is not None else out
```

#### 路径 A：`torch`（朴素 on-the-fly，仅对照 / 兜底）

按 2048 行分块 dequant（避免一次展开 18944 行生成巨大临时张量）后 matmul。
**问题**：torch 张量运算解包会中间产生 `(in//8, out, 8)` 的 int32 大张量，访存量约为权重的 **30 倍**，完全 bandwidth-bound（见 §5）。Triton 不可用时作为兜底路径。

#### 路径 B：`fused`（Triton dequant-GEMM，对标 Marlin，**默认**）

```python
def _forward_fused(self, xf):
    """int4 dequant-GEMM 混合核: 不物化 fp16 权重。
    支持任意前导维度 (引擎在 prefill 时可能传 (batch, seq, in))。"""
    xc = xf.contiguous()
    *lead, K = xc.shape
    x2 = xc.reshape(-1, K)
    M = x2.shape[0]
    fn = ordered_gptq_linear if M <= ORDERED_MAX_M else fused_gptq_linear
    out = fn(x2, self.qweight, self.qzeros, self.scales,
             M, self.out_features, K, self.group_size)
    out = out.reshape(*lead, self.out_features)
    return out
```

kernel 内直接 `acc += tl.dot(x, W_dequant)`，fp16 权重完全不落显存 —— 这是 weight-only 量化的正统加速路径，也是**唯一能同时拿到"int4 显存 + TensorCore 速度 + 对齐 vLLM"**的方案（原因见 §8 延伸思考）。

数值特征见 §4.3：与 cuBLAS 在 prefill（M≥32）逐位一致，decode（M≤4）有 1~4 ulp 差异，但这不影响端到端（Part C 64/64，见 §4.4）。

> `ORDERED_MAX_M` 默认 0：保留了一个 fp32 精确累加的 `ordered` 变体（无 tensor core、更慢），仅作数值对照，设 `NANOVLLM_GPTQ_ORDERED_MAX_M` 可启用。

### 3.5 Triton fused dequant-GEMM（`nanovllm/layers/gptq_triton.py`）

两个曾踩过并已修复的实质问题（写在文件头注释里）：

1. **组号 `g` 必须由逐元素 `offs_k // GS` 计算**，不能写死 `k0 // GS` —— 旧写法只在 `BLOCK_K == GS` 时正确，导致历史上"BLOCK_K 从 128 改到 256/1024 误差不变"的实验是在 kernel 本身算错的前提下得到的，结论无效。
2. **分块按 M 自适应**：decode（M≤8）用 `BM=16/BN=64` 提高 CTA 数量。旧配置 `BM=32/BN=64` 在 `q_proj` 上只有 56 个 CTA < 82 个 SM，约 1/3 硬件空转（仅 ~2/3 SM 在用）—— 这是 fused 在 decode 上打不过 cuBLAS 的主要原因。

```python
def _pick_config(M, N):
    if   M <= 8:   bm, bn, bk, ns = 16, 64, 32, 3
    elif M <= 32:  bm, bn, bk, ns = 32, 64, 64, 3
    elif M <= 64:  bm, bn, bk, ns = 64, 64, 32, 3
    elif M <= 256: bm, bn, bk, ns = 128, 64, 32, 3
    else:          bm, bn, bk, ns = 128, 128, 32, 2
    ...
```

### 3.7 模型层替换（`nanovllm/models/qwen2.py`）

把 `Linear` 换成 `GPTQ*ParallelLinear`，并把 `group_size` 从 config 透传：

```python
self.q_proj = GPTQColumnParallelLinear(hidden_size, self.q_size, bias=qkv_bias, group_size=group_size)
self.k_proj = GPTQColumnParallelLinear(hidden_size, self.kv_size, bias=qkv_bias, group_size=group_size)
self.v_proj = GPTQColumnParallelLinear(hidden_size, self.kv_size, bias=qkv_bias, group_size=group_size)
self.o_proj = GPTQRowParallelLinear(hidden_size, hidden_size, bias=False, group_size=group_size)
...
self.gate_proj = GPTQColumnParallelLinear(hidden_size, intermediate_size, bias=False, group_size=group_size)
self.up_proj   = GPTQColumnParallelLinear(hidden_size, intermediate_size, bias=False, group_size=group_size)
self.down_proj = GPTQRowParallelLinear(intermediate_size, hidden_size, bias=False, group_size=group_size)
```

**两个曾经写错、必须保持正确的细节**：

```python
# 1) MLP 激活函数 (曾写反)
def forward(self, x):
    gate = self.gate_proj(x)
    up   = self.up_proj(x)
    x = torch.nn.functional.silu(gate) * up      # 不是 gate * silu(up)！
    return self.down_proj(x)

# 2) 残差约定必须与原始 nano-vLLM 一致：残差加法与 RMSNorm 融合，
#    层内不做最后一步加法，交由下一层 input_layernorm（或最终 norm）完成。
def forward(self, positions, hidden_states, residual):
    if residual is None:
        hidden_states, residual = self.input_layernorm(hidden_states), hidden_states
    else:
        hidden_states, residual = self.input_layernorm(hidden_states, residual)
    hidden_states = self.self_attn(positions, hidden_states)
    hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
    hidden_states = self.mlp(hidden_states)
    return hidden_states, residual

# 模型级
hidden_states, _ = self.norm(hidden_states, residual)   # 最后一次 add-rms-norm
```

### 3.8 config / loader / model_runner 联动

**`config.py`** 自动识别量化方法：

```python
qcfg = getattr(self.hf_config, "quantization_config", None)
if qcfg is not None and qcfg.get("quant_method") == "gptq":
    assert qcfg.get("bits") == 4, "only 4-bit GPTQ is supported"
    self.quantization = "gptq"
```

**`loader.py`** 识别 GPTQ 后缀，剥离后缀再做 `packed_modules_mapping`，最后原样 `copy_` 进参数：

```python
GPTQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")
# g_idx 在 group 对齐时不需要, 跳过; 其余后缀保留在参数名上 (如 qkv_proj.qweight)
```

**`model_runner.py`** 三处联动：

```python
MODEL_REGISTRY = {
    "Qwen3ForCausalLM": "nanovllm.models.qwen3:Qwen3ForCausalLM",
    "Qwen2ForCausalLM": "nanovllm.models.qwen2:Qwen2ForCausalLM",   # ← GPTQ 模型走这条
}
...
# GPTQ 动态反量化不兼容 CUDA Graph，强制 eager
self.enforce_eager = True if config.quantization == "gptq" else config.enforce_eager
assert self.world_size == 1 or config.quantization is None, "GPTQ 仅支持单卡 (TP=1)"
# 把 group_size 注入 hf_config，供模型构造时读取
hf_config.group_size = qcfg.get("group_size", 128)
```

### 3.9 RMSNorm 的 dtype fix

RMSNorm 权重若声明为 fp32，会把 fp16 激活提升为 fp32 并一路传染，破坏整图 fp16、拖慢且可能 OOM。改为与**输入 dtype**一致：

```python
def forward(self, x):
    dtype = x.dtype
    x = x.float()                                   # 归一化在 fp32 算（精度）
    var = x.pow(2).mean(-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    return x.mul(self.weight.float()).to(dtype)     # 输出还原 fp16
```

---

# 第二部分：正确性验证

## 4. 怎么证明没写错

"能 load、能 forward、不报错" ≠ "算对了"。量化 bug 的典型症状是**输出看似流畅但内容错乱**（系统性偏移），必须靠数值不变量定位。

### 4.1 弱证据 vs 强证据

**弱证据（只能当线索，不能当结论）**：对称量化下权重近似零均值，于是 `mean(unpack(Q)) ≈ 真实零点`。本模型 `qzeros=7`、`mean(Q)=7.998≈8`，据此推测真实零点 = 8。它有两个硬伤：

1. 依赖"权重零均值"这个分布假设；
2. 只校验了一个常数，**完全不校验解包顺序、分组映射、`qzeros` 沿 N 的排列** —— 而这才是真正会写错的地方。

实测它有多不可靠（`tests/verify_gptq.py` Part A3）：把码字分布人为推向一端后 `mean(码字)` 变成 10.1 / 12.2，而真实零点仍是 8 —— 按这个启发式会得出错误结论。

**强证据（能定位具体 bug，且自带判别力）**：自造"黄金权重"做**无损往返 + 变异测试**。

### 4.2 Part A：黄金往返 + 变异测试

```python
# 1) 构造能被 int4 精确表示的权重: 先随机码字, 再按约定反算出 W  ->  往返必须 max|err| = 0.0
codes = torch.randint(0, 16, (N, K))
W = ((codes - 8) * scale).half()
qweight, qzeros, scales = pack(codes, ...)          # 按 GPTQ 布局打包
assert (dequant(qweight, qzeros, scales) - W).abs().max() == 0      # 逐元素 bit-exact

# 2) 变异测试: 人为注入错误, 断言测试必须失败 —— 否则这个测试本身没有判别力
```

| 注入的错误 | max\|err\| | 是否被抓住 |
|---|---|---|
| 零点少 +1（`z_true=qzeros`） | 0.53 | 是 |
| `qweight` nibble 序反 | 7.97 | 是 |
| `qweight` 轴/序理解错 | 7.97 | 是 |
| 分组映射错（组号取反） | 3.90 | 是 |
| `qzeros` 沿 N 解包序反 | 7.02 | 是 |

每条都被抓住，说明这套测试**真的能区分对错**，而不像 `mean(码字)` 那样"怎么跑都像是对的"。

Part A 还包含 **A4 有损量化**：对 `randn` 权重做真实对称 int4 量化，断言 `max|err| ≤ s_max/2`（理论界）；并对照 `zp_bias=0` 时误差均值 ≈ 一个 scale，与 §1.4 的"+1·scale 偏移"互相印证。

### 4.3 Part B / B2：精度归因（fp64 真值）

光看"fused vs cuBLAS 输出差 0.0"是不够的 —— fp16 **输出**落在同一可表示值上，并不代表 fp32 累加器逐位相同。Part B2 用 fp64 真值三方对比：

Part B 扫真实权重层（`q_proj` / `gate_proj` / `down_proj`）× M ∈ {1, 4, 32, 256}，输出三列：`|fused − cuBLAS|`、`|fused − 真值|`、`|cuBLAS − 真值|`，以及不一致元素数。观察到的规律（复现见 §9）：

| 形状 | fused vs cuBLAS（fp16 输出） | 相对 fp64 真值 |
|---|---|---|
| `M ≥ 32`（prefill 类） | 逐位一致，不一致元素 = 0 | 两者误差同量级 |
| `M ≤ 4`（decode 类） | 极少数元素差 1~4 ulp | 两者误差同量级（有时 fused 更准） |

结论：**fused 与 cuBLAS 只是归约顺序不同的两种等精度实现**，并非 fused 更差。具体是：

- `M ≥ 32`（prefill 类形状）：与 cuBLAS 逐位一致（都走 tensor core，归约顺序一致）。
- `M ≤ 4`（decode 类形状）：cuBLAS 切到 gemv 类实现，归约顺序与 `tl.dot` 不同，fp32 累加差约 1e-6 相对量，落到 fp16 输出上就是极少数元素差 1~4 ulp。

### 4.4 Part C：端到端与 vLLM 逐 token 比对（ground truth）

以 vLLM `gptq_marlin` 为 ground truth，**贪心解码逐 token 比对**（2 个 prompt × 32 tokens = 64）：

```python
# tests/verify_gptq.py 中每个 mode 起独立子进程, temperature=1e-9 近似贪心
llm = LLM(MODEL, tensor_parallel_size=1, max_num_batched_tokens=2048,
          max_num_seqs=2, max_model_len=1024)
outs = llm.generate(PROMPTS_C, SamplingParams(temperature=1e-9, max_tokens=32), use_tqdm=False)
```

| 路径 | 匹配 / 总数 | 结论 |
|---|---|---|
| torch | 64 / 64 | 逐 token 一致 |
| **fused** | **64 / 64** | **逐 token 一致** |

两条路径均与 vLLM 精确对齐，fused 为**默认路径**。

---

# 第三部分：性能分析

## 5. 瓶颈：为什么朴素 dequant 只有 2 tok/s（吞吐 8.5）

每层 MLP 含 3 个 `(18944 × 3584)` 投影，注意力含 2 个 `(3584 × 3584)` 等。单次 forward 需反量化约 **6.5B 参数**（7B × ≈0.93，含 lm_head）。

`torch` 路径每个元素要 shift/mask/sub/mul 多次访存，还会中间物化 `(in//8, out, 8)` 的 int32 张量，**访存量约为权重的 30 倍**：单 forward 权重访存量达数百 GB，在 3090（~936 GB/s）上约 0.5 s/forward，即 **约 2 tok/s**（实测 2.2）。

这是典型的 **memory-bound**，不是算力不够。它的唯一优点是权重常驻 int4，显存最省（5.2 GiB）。

## 6. 两条路径实测

测试条件：RTX 3090 24GB，4 并发（bench.py 的 4 条 prompt），`max_tokens=128`，`bench.py --engine all`。

| 路径 | TTFT (ms) | Decode 单请求 (tok/s) | 4 并发吞吐 (tok/s) | 权重显存 (GiB) | 相对 vLLM 吞吐 |
|---|---|---|---|---|---|
| torch（朴素 on-the-fly） | 479.2 | 2.1 | 8.5 | 5.20 | 0.02× |
| **fused（Triton dequant-GEMM，默认）** | 35.1 | 27.7 | **77.4** | **5.20** | 0.22× |
| *vLLM gptq_marlin（参考）* | 9.1 | 107.1 | 344.4 | — | 1.00× |

**逐条解读**：

- **fused vs torch**：TTFT 479 → 35.1 ms（**13.6×**），并发吞吐 8.5 → 77.4 tok/s（**9.1×**）。fused 把反量化融进 GEMM kernel，省掉 torch 路径约 30× 权重的访存量，这正是它现在是默认的原因。
- **量化到底省不省显存？** 省 —— 由默认 `fused` 路径兑现（5.20 GiB，int4 常驻、fp16 不物化）。

### 6.1 量化前 vs 量化后：与原版 nano-vLLM + 未量化模型的对比

上面四条路径都是"自己跟自己比"。真正的收益要看**量化前 vs 量化后**：用**原版 nano-vLLM**
（`/home/cdzk/WR/nano-vllm`，未加任何量化改动）跑**未量化的 Qwen2.5-7B-Instruct** 作基线，
与我们的 GPTQ-Int4 默认路径对比。两者用**同一套测量口径**（同 4 条 prompt / 128 tokens / 4 并发）。

> 复现：`python bench_fp16_baseline.py --eager`（脚本会把原版仓库插到 `sys.path[0]`，避免 import 到本仓库改过的 `nanovllm`）

| 指标 | 原版 nano-vLLM + 未量化模型 | 本仓库 + GPTQ-Int4（fused） | 变化 |
|---|---|---|---|
| 权重显存 | **14.22 GiB** | **5.20 GiB** | **−63.4%（2.73×，省 9.02 GiB）** |
| 权重每个参数 | 2 B | 0.5 B | −75% |
| 4 并发吞吐 | 78.0 tok/s | 76.7 tok/s | **−1.7%（基本持平）** |
| TTFT | 31.5 ms | 35.8 ms | +13.7%（略慢） |
| 单请求 Decode | 33.4 tok/s | 27.7 tok/s | −17.1% |

（各 3 轮取均值；基线波动 ±0.4 tok/s，GPTQ 波动 ±0.7 tok/s）

**怎么解读**：

- **显存是唯一实打实的收益**：9 GiB 的富余可以换成更大的 KV cache / 更高并发 / 更长上下文，
  或者在更小显存的卡上部署。这是 weight-only 量化的真正价值。
- **吞吐基本持平，不是"加速"**：int4 把权重访存降到 1/4，但我们的 fused kernel 是逐元素 shift
  解包、没有做 Marlin 那样的权重 repack，解包开销吃掉了带宽红利；同时 TTFT 因 prefill 时
  int4 解包路径更长而慢了约 14%。**量化 ≠ 加速**，它换的是显存。
- **诚实口径**：两侧都是 **eager**。我们的 GPTQ 路径被引擎强制 `enforce_eager=True`，
  而原版 fp16 默认开 CUDA Graph —— 但**原版的 CUDA Graph 在当前环境（torch 2.7）本身就跑不通**
  （捕获时报 `operation not permitted when stream is capturing`），所以无法给出"原版 + CUDA Graph"的数字。
  若它能跑通，fp16 基线会更快，量化侧的吞吐差距会更大。
- **dtype 差异**：未量化模型的 `config.torch_dtype` 是 **bfloat16**（Qwen2.5 官方权重即 bf16），
  GPTQ 模型是 float16。两者都是 2 字节、都走 TensorCore，对结论无实质影响。

> **为跑通基线而修的原版仓库问题**（仅影响基线测量，与量化实现无关）：
> ① `Config` 是 `@dataclass(slots=True)`，但 `__post_init__` 给未声明的 `model_class` 赋值 → 补声明该字段；
> ② `hf_config.dtype` 在新版 transformers 已改名 `torch_dtype`；
> ③ `qkv_bias=getattr(config,'attention_bias',False)` 取到 False，而 Qwen2.5 checkpoint **带** q/k/v bias → 默认值改 True。

## 8. 与 vLLM Marlin 的真实差距

fused 已 64/64 对齐，所以我们与 vLLM 的差距**不再是数值问题，纯粹是工程优化程度**：77.4 vs 344.4 tok/s（0.22×）。

### 8.1 kernel 质量：我们没做权重 repack

Marlin 会把 int4 权重**预重排**成 TensorCore mma 友好的布局（交错、按 mma 的 k 维分块连续），让解包后的数据直接喂 mma、访存完全 coalesced。我们的 `fused` 是逐元素 shift 解包，访存效率明显更低 —— 这是吞吐差距的主要来源之一。

### 8.2 调度栈差距

vLLM 还叠加了 CUDA Graph、FlashAttention、更细的调度与 prefix cache；我们是 python/torch 朴素实现，且 GPTQ 动态反量化目前**强制 eager**（`model_runner.py`），拿不到 CUDA Graph 的收益。

> 这个差距本身就是最好的面试素材：**量化推理的加速不来自"权重变小"，而来自"反量化不再显式落盘"——kernel 内融合 + 与 cuBLAS 同款归约顺序才是关键**。

## 9. 测量口径（诚实性）

- vLLM 离线 API 的 `RequestOutput.metrics` 在本环境为 `None`，TTFT 用 `max_tokens=1` 端到端计时（含 1 个 decode step + 调度开销），**略微高估 vLLM 的 TTFT**，即对比偏保守。
- nano-vLLM 的 TTFT = 首个 prefill step 完成耗时；Decode 从首 token 后开始计时。
- **权重显存单独统计**（`bench.py::_weights_gib`）：KV cache 会按剩余显存自动分配，所以"进程显存峰值"看不出量化收益，必须单算 `parameters + buffers`。
- 各引擎在**独立子进程**中运行，避免 CUDA 上下文/显存互相干扰。

---

# 第四部分：复现

## 10. 复现命令

```bash
# 环境
export CUDA_VISIBLE_DEVICES=1
PY=/nas_data/WR/conda/wr-vllm/bin/python

# ---- 正确性：四层验证 (tests/verify_gptq.py) ----
$PY tests/verify_gptq.py --part A     # 黄金往返 + 变异测试（强证据）
$PY tests/verify_gptq.py --part B     # 真实权重单层：fused vs cuBLAS，多形状
$PY tests/verify_gptq.py --part B2    # 精度归因：fused / cuBLAS / fp64 真值三方对比
$PY tests/verify_gptq.py --part C     # 端到端贪心 token：四条路径 vs vLLM（2×32=64）
$PY tests/verify_gptq.py --part AB2C  # 全部

# ---- 性能：单层微基准（含 fused 与 torch 一致性断言） ----
$PY tests/bench_layer.py

# ---- 性能：端到端（独立子进程，避免显存干扰） ----
$PY bench.py --engine all --max_tokens 128                  # fused + torch + vLLM 汇总对比
$PY bench.py --engine nanovllm --max_tokens 128             # 默认 fused
$PY bench.py --engine nanovllm-fused --max_tokens 128       # NANOVLLM_GPTQ_FUSED=1
$PY bench.py --engine nanovllm-torch --max_tokens 128       # NANOVLLM_GPTQ_TORCH=1
$PY bench.py --engine vllm --max_tokens 128                 # 参考基准

# ---- 冒烟 ----
$PY test_gen.py
```

### 10.1 投机解码的复现命令

```bash
export CUDA_VISIBLE_DEVICES=1
PY=/nas_data/WR/conda/wr-vllm/bin/python

# 前置: draft/target 词表兼容性 (公共区间逐条一致才能搭配)
$PY tests/check_spec_vocab.py

# 正确性三层 (A 秒级; C 需加载模型, 起两个子进程对比)
$PY tests/verify_spec.py                 # A 黄金对照 + B 分布等价 + 变异测试
$PY tests/verify_spec.py --e2e --gamma 5 --max_tokens 64

# 不变式: 读 paged KV cache 的路径 == 整段重算
$PY tests/check_prefill_cache.py

# 性能: γ 扫描 + 接受率 + KV block 泄漏检查 (每个 γ 独立子进程)
$PY bench_spec.py --gammas 0,3,5,7 --max_tokens 128

# 耗时分解: target decode / draft decode / verify 各多少 ms
$PY tests/profile_spec.py 5
```

用法：`LLM(model, draft_model=<0.5B 路径>, num_speculative_tokens=γ)`。
不传 `draft_model` 时所有投机解码代码路径完全不进入。

---

# 第五部分：Infra 简历项目（可直接用）

## 项目标题

**LLM 推理引擎量化（GPTQ-Int4）与投机解码加速** — nano-vLLM（自研极简推理引擎，~2k 行）

## 项目描述

在 RTX 3090 上实现 Qwen2.5-7B-Instruct-GPTQ-Int4 权重量化推理，保持与 vLLM 逐 token 数值一致（64/64）；自研 Triton **fused dequant-GEMM**，在节约 63% 显存（14.2 → 5.2 GiB）的同时，仅降低并发吞吐 1.7%（78.0 → 76.7 tok/s）、TTFT 略慢 13%（31.5 → 35.8 ms）。
进一步用量化腾出的显存装入 0.5B draft 模型实现 **draft-target 投机解码**：单请求 decode **30.1 → 118.9 tok/s（3.96×）**、4 并发吞吐 **113.6 → 222.8 tok/s（1.96×）**，greedy 下输出与不开投机时**逐 token 相同**。

## 职责与成果（简历正文，每条 1–2 句，直接贴）

- **新增 GPTQ-Int4 量化模块（接入极简推理引擎）**：在 ~2k 行引擎中落地 GPTQ 支持 —— GPTQ 量化线性层、权重加载与格式映射（int4 打包权重 + group-wise scale/zero 对齐 GPTQ 磁盘布局）、引擎接入（`MODEL_REGISTRY` 分派 / GPTQ 强制 eager / `group_size` 注入），复用原 TP 切分与调度，不重写引擎调度。
- **数值正确性验证**：以 vLLM `gptq_marlin` 为 ground truth 做贪心逐 token 比对，**64/64 全匹配**；并用强证据验证（黄金权重无损往返 max|err| = 0 + 变异测试注入轴序 / 分组 / 零点等 5 类错误全抓住），替代"怎么跑都像是对的"的弱启发式。
- **自研 Triton fused dequant-GEMM（默认路径）**：把反量化融进 GEMM kernel —— 打包的 int4 `qweight`/`qzeros` 在 kernel 内按 `group_size` 解包、算 `(w − z) · s` 后**直接喂 `tl.dot`**，全程不把 fp16 权重写回 HBM（int4 常驻显存、fp16 零物化）；因 cuBLAS 仅有 INT8、无 int4 非对称 per-group dequant-GEMM，这是**唯一能同时拿到省显存 + TensorCore 速度**的写法。
- **双 kernel 精度分级 + 性能结果**：decode / 短 prefill 走 fp32 精确累加核 `ordered_gptq_linear` 保数值，长 prefill 走 `tl.dot` 核 `fused_gptq_linear`（已验证与 cuBLAS 逐位一致）；反量化路径吞吐 **8.5 → 77.4 tok/s（9.1×）**，权重显存锁定 5.2 GiB。
- **针对 decode 自适应 GEMM 分块（occupancy 分析定位）**：按序列长度 M 自适应选 BM/BN/BK——用 occupancy 分析（CTA 数 = ⌈M/BM⌉×⌈N/BN⌉ 对比 82 SM）定位到 decode 小 M 时固定分块填不满 SM（旧 32×64 在 q_proj 仅 56 CTA < 82 SM、约 1/3 空转），据此改小分块提高 CTA 占用、长 prefill 用 128×128 吃满算力。
- **新增 draft-target 投机解码模块，接入 continuous batching**：在引擎里新增「γ 次 draft + 1 次 target verify + 拒绝采样」的轮次，展开为 4 个子模块 —— ① **draft 模型与双 KV pool**：draft 走独立的未量化分支（`models/qwen2_dense.py`），target/draft 各持一份 paged KV cache 并**按 block 字节比切分显存预算**（draft 占 target 的 21%），两边共用同一套 block/slot 布局；② **轮次调度接入**：新增 draft/verify 步骤类型，支持**同一 batch 内各序列接受长度不同**（序列级异构长度），TP worker 用轮次派生的确定性 seed 保证各 rank 采出同一个 draft token；③ **KV 占位与回滚**：draft token「先占位后确认」，被拒部分由 `may_append_n` / `trim` 连同跨出去的 block 一起回收（长跑验证空闲 block 数完全回到起点）；④ **轮次收尾**：`max_tokens` 截断、EOS 截断、finish 判定，投机期间不 commit prefix cache（被拒候选若被 hash 会污染后续前缀命中）。
- **自研 Triton paged-KV attention kernel（causal 前缀 + flash-decoding 分段并行）**：kernel 内**直接按 `block_table` 从 paged cache 取 K/V**（不物化成连续内存），前缀段与本次新增段分两段走 online softmax 合并，掩码按「query i 可看 key ≤ i + cached_len」构造；每个 program 处理一个 (query 行, head)，并沿 key 维切成 N_SPLITS 段并行后二次归约（decode 时 batch 小，否则填不满 82 个 SM）。替换掉原 sdpa fallback 后，引擎基线本身的 4 并发吞吐 77.4 → 111.6 tok/s。
- **draft 单步 forward 的 CUDA Graph 化（profiler 定位 launch-bound）**：0.5B draft 一次 forward 里 GPU 只算 2.7 ms、墙钟 20 ms（约 600 次 kernel launch），γ+1 次 draft 的 launch 开销会把投机收益全部吃掉（未 graph 化时加速比仅 **0.70×**）；按 batch size 缓存 CUDA graph、把权重/lm_head/attention 全部固化后，单步 **20.7 → 2.77 ms**，端到端加速比 0.70× → **3.96×**。
- **投机解码的正确性验证（强证据 + 词表不等长处理）**：以「不开投机」的同引擎输出为 ground truth，greedy 下 4 条 prompt **逐 token 相同**；采样模式下用拒绝采样保证输出分布不变（首 token 分布 TVD 0.006、bonus 分布 TVD 0.011），并注入 4 种实现错误（漏除 q、残差分布用 p / 用 q、bonus 取错位置）验证检验**逐个都能抓住**。draft/target 词表不等长（151936 vs 152064）时把概率空间截断到公共前缀后重归一化，被丢弃的尾部概率质量 7.8e-9。

## 口头展开（面试追问时讲，不写进简历正文）

简历只放上面 9 条结论（前 5 条量化、后 4 条投机解码），下面这套是嘴上展开的"新增了什么 / kernel 怎么写 / 为什么"。

**① 新增了哪些量化模块**（对应 bullet 1）
- `GPTQColumnParallelLinear` / `GPTQRowParallelLinear`：替代原 Linear，按 HF GPTQ 布局加载权重（`qweight` 的 int32 位打包、`qzeros`、`scales`），TP 切分沿用原引擎语义，不重写调度。
- 权重加载与格式映射：把打包的 int4 权重 + group-wise scale/zero 映射到上述层（含零点约定与 `group_size` 注入）。
- Triton fused dequant-GEMM kernel：`gptq_triton.py`。
- 引擎接入：`MODEL_REGISTRY` 分派、GPTQ 强制 eager、`group_size` 注入、按 `quantization` 选执行路径。
- 为什么拿 vLLM 当 ground truth：`gptq_marlin` 是社区公认正确实现，贪心解码逐 token 比对、全 64/64 即数值一致。

**② 强证据验证怎么做的**（对应 bullet 2）
- 黄金权重：构造一个能被 int4 精确表示的权重，反量化往返必须 max|err| = 0，先证明公式本身对。
- 变异测试：故意注入轴序 / 分组 / 零点等 5 类错误，验证集每条都能抓住——证明"测试真有判别力"，而非"怎么跑都像是对的"。

**③ Triton fused dequant-GEMM 怎么写的**（对应 bullet 3–4）
- 反量化融进 GEMM：每个 program 负责一个 `(BLOCK_N 输出通道) × (BLOCK_K 输入维)` tile，kernel 内直接读 int4 打包权重（`qweight` / `qzeros`），按 `group_size` 取 scales/zeros，解出 4-bit 码字做 `(w - z) * s` 得 fp16/fp32 权重，**不写回 fp16 到 HBM**，直接喂 `tl.dot` 累加 → fp16 权重不物化、int4 常驻。
- **双 kernel 兼顾精度与速度（具体怎么做）**：两个核 `_fused_gptq_mm` / `_ordered_gptq_mm` **共用同一套 tile 循环与解包逻辑**（同 grid、同 `BLOCK_M/N/K`、同 `(w − z) · s` 反量化），**唯一差别在最后一步的乘加**；按 M 分派（`fn = ordered_gptq_linear if M <= ORDERED_MAX_M else fused_gptq_linear`）：
  - **快核 `fused_gptq_linear`（大 M / 长 prefill，默认）**：`W` 落成 fp16 后 `acc += tl.dot(x_fp16, W_fp16)`，走 **TensorCore**（mma）——只在喂 mma 前把 fp32 反量化结果转 fp16，权重仍不落 HBM。
  - **准核 `ordered_gptq_linear`（小 M / decode、短 prefill）**：`W = (...).to(fp16).to(fp32)` —— **刻意先落 fp16 再升 fp32**，保证喂进去的权重与 cuBLAS 收到的**逐位相同**，从而把「GEMM 归约差异」与「反量化差异」两类误差解耦；再做 `tl.dot(x_fp32, W_fp32, input_precision="ieee")`，**乘积精确 + fp32 FMA 归约、归约顺序完全可控**。代价是没有 TensorCore、明显更慢，故分块也更保守（`_pick_config_ordered` 用 16×64×32，vs 快核 decode 用 16×64×32×3 级流水）。
  - **分界为什么按 M**：`tl.dot` 的 mma 至少要吃 16×16×16，decode 的 M（1~8）本就靠 BM=16 补齐、TensorCore 的算力优势发挥不出来；此时"用慢一点但归约精确的核"性价比最高——设计意图就是「小 M 保数值、大 M 保吞吐」。
  - **诚实结论（面试要点）**：**纯 `tl.dot` 的快核已经 64/64**，`ordered` 核不再是必需的 —— `ORDERED_MAX_M` 默认 **0（全走快核）**，只在设 `NANOVLLM_GPTQ_ORDERED_MAX_M` 为大数时可强制全程走 fp32 精确累加做对照。所以「双 kernel」在本项目里的真实定位是**数值兜底 / 对照设计**，真正贡献吞吐的是「融合」本身，而不是切换精度这条路。
- 为什么要融合：朴素逐元素反量化访存量约权重 30×，单 forward ~0.5s、仅 2 tok/s；融合后吞吐 8.5 → 77.4 tok/s，权重显存 5.2 GiB。

**④ 针对 decode 自适应 GEMM 分块怎么写的**（对应 bullet 5）
- 动机：GEMM 并行度 = CTA 数 = ⌈M/BM⌉ × ⌈N/BN⌉。decode 时 M 极小（1~8），瓶颈在访存延迟而非算力，要靠"更多 CTA 同时驻留"隐藏延迟、占满 82 个 SM。
- 自适应选块（`_pick_config(M, N)`）：按 M 分档选 BM/BN/BK——decode（M≤8）用 16×64，prefill（M≥256）用 128×128，中间 32/64/128 渐变；BM 越小、单 CTA 寄存器越少，可同时驻留的 CTA 越多，越能压满 SM。
- 怎么发现空转（性能定位，与数值无关）：correctness 对齐后专测 decode 单请求，发现 fused 单请求 decode（27.7）反而慢于未量化基线（33.4）。定位手段是 **occupancy 分析**而非 profiler——手算 grid 尺寸 ⌈M/BM⌉×⌈N/BN⌉ 与 82 SM 一比即露馅：decode M=1 时 M 维只有 1 个 block，CTA 数完全由 N 维决定，旧 32×64 在 q_proj（N=3584）只切出 ⌈3584/64⌉=56 个 CTA、26 个 SM 闲置（约 1/3）；`nvidia-smi dmon` 看 SM 利用率掉到 ~68% / `ncu` 看 achieved occupancy 偏低可印证，但根因就是网格没填充满 GPU。
- 为什么 cuBLAS 不空转：它在 M 小（≤4~16）时切 GEMV / persistent kernel，内部细粒度 tile 自动铺满 SM；我们的固定 2D grid 在 M=1 时 M 维只有 1 个 block，结构性铺不满——这正是 decode 输给 cuBLAS 的原因，与数值正确性无关。据此把小 M 的分块改小、提高 CTA 占用后 decode 单请求回到 27.7 tok/s。

**⑤ 与 Marlin 差距 / 后续**（对应原 bullet 4 思路）
- 差距不在数值（已 64/64），而在 kernel 质量：Marlin 把 int4 repack 成 mma 友好布局、访存完全 coalesced；我们按逐元素 shift 解包，访存效率更低。其次才是调度栈（CUDA Graph / FlashAttention / paged KV cache）。
- 最划算的下一步：让 **target 侧（GPTQ）也能用 CUDA Graph**。`fused` 路径 int4 常驻、无每步动态反量化 buffer，理论上可捕获 —— draft 侧已经证明这条路走得通（见 ⑧）。

**⑥ 投机解码怎么接进 continuous batching**（对应 bullet 6）
- **一轮的形状**：γ 次 draft decode → 1 次 target verify（query 是 γ+1 个位置）→ 拒绝采样 → 回滚未被接受的部分。产出 `n_accepted + 1` 个 token（被拒时用 `(p−q)₊` 重采样的修正 token，全接受时白送一个 bonus token）。
- **draft 要跑 γ+1 次而不是 γ 次**：第 γ+1 次的输出被丢弃，作用只是把第 `L+γ−1` 个位置的 K/V 补进 draft cache —— 下一轮若 bonus 也被接受，draft 的首个 query 正好落在那个位置，缺了它就会读到脏显存。
- **verify 的 query 起点是 `x_{L−1}` 而不是 `x_L`**：这样 `seqlen_k − seqlen_q = L−1` 恰好等于 cache 里的有效长度，可以直接复用 varlen + `block_table` 的既有语义（chunked prefill 用的是同一套约定），代价只是 `slot(L−1)` 被重写一次（内容不变）。
- **占位与回滚**：draft token 先占位（append + 分配 block）、verify 后再确认；未被接受的由 `trim` 连同跨出的 block 一起回收 —— 只回滚 token 不回收 block 的话，几个 step 就会把 KV cache 吃光。
- **序列级异构接受长度**：同一个 batch 里 seq A 接受 3 个、seq B 接受 1 个是常态，verify 的 `position_ids` / `slot_mapping` / 回滚数量都按序列分别构造。
- **多 rank 一致性**：一次 `run_spec` 调用内部无法中途同步，所以各 rank 用「轮次 + 步数」派生的**确定性 seed** 采样，保证同一轮采出相同的 draft token。
- **为什么选投机解码而不是 TP / PD 分离**：本机双卡 3090 **无 NVLink**（`nvidia-smi topo` 显示 PHB，走 PCIe），跨机只有以太网；TP 每层要 2 次 all-reduce、PD 分离要搬整段 KV（7B 约 0.11 MB/token，2k prompt ≈ 230 MB，1 Gbps 下 1.8 s ≫ prefill 本身 ~100 ms），都吃不到收益。投机解码通信量为零，是这套硬件上唯一能稳定拿正收益的方向。

**⑦ Triton paged-KV attention kernel 怎么写**（对应 bullet 7）
- **为什么不能直接用 SDPA**：本机没装 flash-attn，attention 一直走 sdpa fallback；而 PyTorch 的 `is_causal` 在 `seqlen_q < seqlen_k` 时等价于 `j <= i`，**不会**自动补偿 `(S−L)` 偏移（用最小实验扫掩码偏移确认：`off=0` 与 `is_causal` 完全一致，`off=S−L` 才是我们真正需要的），于是排在 k/v 最前面的历史前缀会被整段 mask 掉。显式传 float mask 可以修正，但会退化到 math backend、复杂度 O(L·S·D)，对 verify 来说比省下的时间还贵。
- **kernel 结构**：每个 program 负责一个 `(query 行, head)`，前缀部分**按 `block_table` 直接从 paged cache 取 K/V**（`cache[block_id, j % block_size, kv_head, :]`，不物化成连续内存），本次新增部分从 contiguous k/v 取；两段共用一份 online softmax（`m/l/acc`，用 `exp(m_old − m_new)` 缩放续接），掩码按「query i 可看 key ≤ i + cached_len」构造。
- **flash-decoding 分段**：decode 时每序列只有 1 个 query token，grid 若只有 `(batch, heads)` 个 program，batch 小时连 82 个 SM 都填不满；所以沿 key 切成 `N_SPLITS` 段并行，各自输出未归一化的 `(m, l, acc)`，再由第二个 kernel 归约合并。
- **一个硬约束**：`N_SPLITS` 只能由**静态形状**决定（batch / heads / SM 数），不能读 `ctx_lens` 的实际值 —— 因为这条路径会被 CUDA graph 捕获，捕获期间任何 GPU→CPU 同步都会让 capture 直接失败。

**⑧ draft 的 launch-bound 怎么定位、graph 怎么做**（对应 bullet 8）
- **定位**：用 torch profiler 打 draft 的单步 forward —— GPU kernel 合计只有 2.7 ms，Self CPU 5.4 ms，墙钟却 20 ms；按 call 数估约 600 次 kernel launch，属纯 launch-bound。此时端到端加速比只有 **0.70×**（γ+1 次 draft 的开销把收益全吃掉了）。
- **做法**：把 draft 单步 forward（含 lm_head）整体捕获成 CUDA graph，输入 `input_ids` / `positions` / `slot_mapping` / `context_lens` / `block_tables` 全部换成**静态 buffer**、每步 `copy_` 进去后 replay；按 batch size 缓存 graph，首次遇到该 bs 才捕获。捕获前先在 side stream 上 warmup 3 次。
- **结果**：单步 **20.7 → 2.77 ms**，端到端 0.70× → **3.96×**。

**⑨ 投机解码的正确性怎么证明**（对应 bullet 9）
- **三层，不靠"看着像"**：① greedy 下输出必须等于「逐步 argmax」的黄金结果（含每个位置被拒时的修正）；② 采样模式下输出分布必须严格等于 target 分布；③ 端到端与「不开投机」的同引擎输出逐 token 相同（4 条 prompt 全一致）。
- **变异测试**：注入 4 种实现错误，检验必须逐个抓住 —— 漏除 q（TVD 0.383）、残差分布用 p（0.223）、残差分布用 draft 的 q（0.502）、bonus 取错位置（0.525）；正确实现的 TVD 是 0.006 / 0.011。其中「bonus 取错位置」只看首 token 分布**完全抓不到**（0.0053），必须两个指标都有 —— 这正是变异测试的价值。
- **词表不等长**：0.5B 声明 151936、7B 声明 152064（多出 128 个 padding 条目）。先用脚本证明两边 tokenizer 在公共 id 区间上逐条一致（151665 词条、0 冲突），再把概率空间截断到公共前缀重归一化；被丢弃的尾部概率质量 7.8e-9，可忽略。反过来也保证 draft 输出的 id 必然 < 公共长度，绝不会让 target 的 embedding 越界。

## 量化指标（放简历"成绩"栏）

| 指标 | 未量化(基线) | 朴素反量化 | **fused(默认)** | vLLM(参考) |
|---|---|---|---|---|
| 权重显存 (GiB) | 14.22 | 5.20 | **5.20** | — |
| 4 并发吞吐 (tok/s) | 78.0 | 8.5 | **76.7** | 344.4 |
| TTFT (ms) | 31.5 | 479.2 | 35.8 | 9.1 |
| 端到端 token 匹配 vLLM | — | 64/64 | **64/64** | — |

> fused = kernel 内融合反量化（默认路径）；朴素反量化 = 逐元素反量化后 matmul，仅作带宽瓶颈对照；未量化(基线) = 用未改动上游 nano-vLLM 跑 Qwen2.5-7B-Instruct（bf16）同口径对比（`bench_fp16_baseline.py --eager`）。fused 吞吐 76.7 为同口径基线对照值，多引擎基准中为 77.4（run-to-run 差异）。

## 投机解码指标（放简历"成绩"栏）

| 指标 | γ=0（同引擎基线） | **γ=5** | **γ=7** |
|---|---|---|---|
| 单请求 decode (tok/s) | 30.1 | 101.6 | **118.9（3.96×）** |
| 4 并发吞吐 (tok/s) | 113.6 | **222.8（1.96×）** | 220.8 |
| TTFT (ms) | 35.0 | 58.3 | 56.3 |
| 接受率 α | — | 0.560 | 0.482 |
| 平均每轮确认 token | 1.00 | 3.80 | 4.37 |
| greedy 输出逐 token 一致 | — | **4/4** | — |
| 权重显存 (GiB) | 5.20 | 6.39 | 6.39 |

> γ=0 基线是**同一个引擎**关掉投机解码的实测值（113.6 tok/s），高于 §6 的 77.4 —— 因为本模块顺带把 decode attention 从 sdpa fallback 换成了自研 Triton flash-decoding kernel（见口头展开 ⑦）。两个数字口径不同，不要混用：76.7/77.4 是「vs 未量化上游引擎」的同口径对照，113.6 是「本引擎 γ=0」的对照。
> TTFT 从 35 → 56 ms 是已知代价：prefill 时必须顺便把 prompt 的 KV 也写进 draft cache，否则第一轮 draft decode 会读到未初始化的显存。

## 技术栈 / 关键词

PyTorch、Triton、GPTQ、weight-only quantization、int4、group-wise dequant、fused dequant-GEMM、CUDA 显存带宽分析、vLLM、Marlin GEMM kernel、Tensor Parallel、**speculative decoding（draft-target）、rejection sampling、paged KV cache attention kernel、flash-decoding、CUDA Graph、continuous batching**、推理引擎、端到端数值对齐（64/64）。

## 面试 talk track（STAR）

**S（情境）**：业务要在单张 24GB 消费卡上跑 7B 模型，fp16 权重 14GB 顶满显存；现成引擎（vLLM）黑盒、不利于学习量化内核细节，于是基于 nano-vLLM 自研支持。

**T（任务）**：在不破坏 fp16 路径、不重写引擎调度的前提下，让引擎能正确且高效地跑 GPTQ-Int4 模型，并且**默认路径就要真正享受到量化的显存收益**；量化落地后 decode 仍受逐 token 读全部权重的访存瓶颈限制（单请求仅 ~30 tok/s），要再把延迟打下来。

**A（行动 / 技术亮点）**：
1. 正确性先行：用**黄金往返 + 变异测试**（强证据）而非弱启发式验证反量化，并以 vLLM `gptq_marlin` 为 ground truth 做端到端贪心逐 token 比对，达成 **64/64**。
2. 反量化在 fp32 下分块完成、再以 fp16 喂 GEMM，对齐 vLLM 的数值行为并避免 fp16 溢出。
3. 性能剖析定位到 **bandwidth-bound**（朴素逐元素反量化访存量约为权重的 30 倍，仅 2 tok/s），据此实现 Triton **fused dequant-GEMM**：把反量化融进 GEMM kernel（展开见"口头展开"③），并设为默认路径。
4. 以**原版引擎 + 未量化模型**做同口径基线，量化后显存 14.22 → 5.20 GiB（**省 63%**）、4 并发吞吐 78.0 → 76.7（**仅降 1.7%**）、TTFT 31.5 → 35.8 ms（**+13%**）—— 量化换来的是显存余量而非加速；并明确与 Marlin 的差距已非数值，而是 kernel 质量（权重 repack）+ 调度栈（CUDA Graph）。
5. 用量化腾出的显存装入 0.5B draft 模型，实现 **draft-target 投机解码**：新增 draft/verify 轮次并接入 continuous batching（支持序列级异构接受长度、draft token 占位与被拒回滚），自研 **Triton paged-KV attention kernel** 让 verify 能读到历史 KV，并把 **draft 单步 forward 用 CUDA Graph 固化**解决 launch-bound。
6. 投机解码的正确性同样用强证据：greedy 下与不开投机的输出**逐 token 相同（4/4）**，采样模式下用拒绝采样保证分布不变（TVD 0.006），并注入 4 种实现错误验证检验逐个都能抓住。

**R（结果）**：量化侧 —— 正确性 64/64 全匹配；权重显存 14.2 → 5.2 GiB（省 63%）；4 并发吞吐 78.0 → 76.7（仅降 1.7%）、TTFT +13%；反量化路径吞吐 8.5 → **77.4 tok/s**。
投机侧 —— 单请求 decode 30.1 → **118.9 tok/s（3.96×）**、4 并发吞吐 113.6 → **222.8 tok/s（1.96×）**、接受率 0.48~0.56；顺带把引擎基线的并发吞吐从 77.4 提到 **111.6 tok/s**。

## 延伸思考（面试官最爱追问）

- **为什么不一上来就写 CUDA kernel？** 先用 torch 朴素实现把正确性与基线对齐，确认算法无误再谈性能；过早优化会淹没 bug。
- **量化为什么能在运行时既省显存、又保持速度？** 关键在于**把反量化融进 GEMM kernel**：只要反量化发生在 GEMM 之外，就必然二选一 —— 要么物化 fp16 权重（放弃显存收益），要么每步重算（用带宽换显存）。而 cuBLAS **没有** int4 dequant-GEMM（只有 INT8，且不支持非对称 per-group dequant），所以要三者兼得就必须重写 GEMM，这正是 `fused` 做的事。
- **fused 已经是默认了，它和 Marlin 的本质差距在哪？** 不是数值（已 64/64 对齐），而是 **kernel 质量**：Marlin 会把 int4 权重**预重排（repack）**成 TensorCore mma 友好的布局，喂给 mma 时访存完全 coalesced；我们的 kernel 未做 repack、按逐元素方式喂 mma，访存效率明显更低。其次是**调度栈**（CUDA Graph / FlashAttention / paged KV cache）。这两块补齐才有望从 0.22× 追到 1×。
- **下一步最划算的优化是什么？** 让 **target 侧（GPTQ）也能用 CUDA Graph**。目前引擎对 `quantization=="gptq"` 无条件 `enforce_eager=True`；但 `fused` 路径 int4 常驻、反量化在 kernel 内完成，**没有每步动态申请的反量化 buffer**，理论上可以捕获。draft 侧已经证明这条路走得通（单步 20.7 → 2.77 ms）。

**投机解码相关追问**

- **为什么选投机解码，而不是 TP / PD 分离？** 硬件决定的：双卡 3090 **无 NVLink**（topo 显示 PHB，走 PCIe），跨机只有以太网。TP 每层 2 次 all-reduce，无 NVLink 时通信就吃掉了切分收益；PD 分离要搬整段 KV，7B 约 0.11 MB/token，2k prompt ≈ 230 MB，1 Gbps 下传输 1.8 s，而 prefill 本身才 ~100 ms —— 必亏。投机解码通信量为零，是这套硬件上唯一能稳定拿正收益的方向（若有机器的互联是 IB/RoCE，结论会反过来，PD 分离更值得做）。
- **γ 怎么选？** 实测 γ 越大单请求越快（γ=3/5/7 → 2.71×/3.38×/3.96×），但并发吞吐在 γ=5~7 饱和（222.8 / 220.8）：γ 增大时每轮固定成本（γ+1 次 draft + 1 次 verify）上升，而接受率 α 随位置下降（后面的候选更难被接受）。当前是静态 γ，下一步可做按历史接受率在线调整的**动态 γ**。
- **为什么不用 EAGLE / Medusa？** 它们要用目标模型的隐藏层监督训练 draft head，成本更高且需要训练数据/算力；draft-model 方案只要挑一个同 tokenizer 的小模型即可，且 Qwen2.5-0.5B 与 7B 同系列同词表（公共区间逐条一致），是零成本可验证的起点。
- **投机解码会让输出分布改变吗？** 不会 —— 这正是拒绝采样保证的（被拒时从 `(p−q)₊` 重采样，接受判据是 `r < p(x)/q(x)`）。我们用 40 万样本统计确认首 token 分布与 target 的 TVD 只有 0.006，并用变异测试证明这个检验真有判别力。
- **词表长度不一样怎么办？** 0.5B 是 151936、7B 是 152064。先证明两边 tokenizer 在公共 id 区间上逐条一致，再把概率空间截断到公共前缀重归一化（被丢弃的尾部质量 7.8e-9）。顺带这个约束也保证 draft 输出的 id 必然 < 公共长度，不会让 target 的 embedding 越界。
- **投机解码的下一步？** ① **tree verification**：一次 target forward 同时验证多条候选（需自写 tree-masked attention kernel），把接受长度从「链的期望」提到「树的期望」；② 动态 γ；③ target 侧 CUDA Graph。
- **GPTQ-v2 / AWQ / GGUF 怎么扩展？** 三者的差异主要在打包格式与反量化公式；架构上只需新增 `unpack` / `dequant` 实现与 loader 映射 —— 本实现已按可插拔 `dequant` 组织。

---

## 附录：关键文件清单

| 文件 | 作用 |
|---|---|
| `nanovllm/layers/gptq_linear.py` | GPTQ 线性层：参数布局、bit 解包、零点对齐、两条路径分派（fused / torch） |
| `nanovllm/layers/gptq_triton.py` | Triton fused dequant-GEMM kernel + 自适应分块 + 踩坑注释 |
| `nanovllm/models/qwen2.py` | Qwen2 模型定义：GPTQ 层替换、MLP 激活与残差约定 |
| `nanovllm/layers/layernorm.py` | RMSNorm：fp32 归一化 + 输出还原 fp16 |
| `nanovllm/config.py` | 从 `quantization_config` 自动识别 `gptq` |
| `nanovllm/utils/loader.py` | safetensors → `nn.Parameter`（GPTQ 后缀剥离 + packed 映射） |
| `nanovllm/engine/model_runner.py` | `MODEL_REGISTRY` 分派、GPTQ 强制 eager、`group_size` 注入；**投机解码**：draft 模型加载、双 KV pool 按比例切分预算、draft CUDA graph、`run_spec` |
| `tests/verify_gptq.py` | 量化正确性验证：A 黄金往返/变异、B 单层对比、B2 fp64 归因、C 端到端 vs vLLM |
| `tests/bench_layer.py` | 单层微基准：fused/torch 耗时与等效带宽 + fused==torch 逐位断言 |
| `bench.py` | 量化端到端基准：两条路径 + vLLM，独立子进程隔离 |
| `test_gen.py` | 冒烟测试 |

**投机解码相关文件**

| 文件 | 作用 |
|---|---|
| `nanovllm/engine/speculator.py` | 拒绝采样 / greedy 验证 / bonus token / 接受率统计（纯张量，可独立测） |
| `nanovllm/engine/model_runner.py` | draft 模型与双 KV pool、`prepare_draft_decode` / `prepare_spec_verify`、`run_spec`、draft CUDA graph |
| `nanovllm/engine/llm_engine.py` | step 分支：占位 / 回滚 / `max_tokens` 与 EOS 收尾 / 降级路径 |
| `nanovllm/engine/block_manager.py` | `may_append_n`（批量占位）+ `trim`（回滚并回收跨出的 block） |
| `nanovllm/engine/sequence.py` | `append_spec_tokens` / `pop_tokens` |
| `nanovllm/models/qwen2_dense.py` | 未量化 Qwen2，供 draft 用（`models/qwen2.py` 已被改写成 GPTQ-only） |
| `nanovllm/layers/attention.py` | `cached_causal_attn_kernel`（paged 前缀 causal）+ `_decode_split_kernel` / `_decode_reduce_kernel`（flash-decoding） |
| `bench_spec.py` | 投机解码性能基准：γ 扫描 + 接受率 + KV block 泄漏检查 |
| `tests/verify_spec.py` | 正确性三层：A greedy 黄金、B 分布等价 + 变异测试、C 端到端逐 token |
| `tests/check_spec_vocab.py` | draft/target 词表兼容性前置检查 |
| `tests/check_prefill_cache.py` | 不变式：读 paged cache 的路径 == 整段重算 |
| `tests/profile_spec.py` | target decode / draft decode / verify 的耗时分解 |
| `docs/spec-decoding.md` | 投机解码设计文档（流程、三个坑、实测） |
