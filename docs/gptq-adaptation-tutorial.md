# nano-vLLM 适配 GPTQ 4-bit 量化模型：从原理到落地（教程 + Infra 简历项目）

> 适用对象：想在极简推理引擎（nano-vLLM，~2000 行）里手搓量化支持、或想理解 GPTQ/vLLM/Marlin 工程差异的工程师。
> 代码基准：本仓库当前实现（`nanovllm/layers/gptq_linear.py`、`nanovllm/models/qwen2.py` 等）。
> 硬件：单卡 RTX 3090 24GB。模型：`Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4`（GPTQ-Int4, sym, group_size=128, desc_act=False）。

---

# 第一部分：nano-vLLM 如何适配量化模型

## 0. 背景与动机

nano-vLLM 本来只支持 **fp16 全精度**权重（`nanovllm/layers/linear.py` 里的 `Linear` 直接 `F.linear(x, w.half())`）。把量化模型塞进去，价值是：

- **显存**：7B 模型 fp16 约 14GB 权重；GPTQ-Int4 仅约 3.5GB（0.5 byte/param），单机 24GB 卡能放下更大的模型/更长的 KV cache。
- **带宽**：权重读取量降到 1/4，解码阶段受限于权重带宽（memory-bound），理论上解码吞吐可大幅提升。
- **学习价值**：量化推理的真正难点不在“矩阵乘”，而在**反量化（dequant）怎么和 GEMM 融合**、**零点/打包约定怎么对齐**——这正是 vLLM Marlin、AWQ kernel 的核心。

本教程覆盖：GPTQ 磁盘格式 → bit 解包 → 零点约定（最易踩坑）→ 反量化数学 → 与引擎各模块的对接 → 正确性验证 → 性能瓶颈与优化。

---

## 1. GPTQ 格式速成（必须懂，否则一定写错）

GPTQ 是 **weight-only 对称量化**：每个权重用一个 4-bit 整数码字表示，反量化公式

```
W[i] = (Q[i] - z[g]) * s[g]        # g = i // group_size
```

- `Q`：4-bit 整数码字（int8 存储但只取低 4 位）。
- `z[g]`：group `g` 的**零点（zero point）**。对称量化下 GPTQ 的 `z` 由码字统计决定（`z ≈ round(mean(Q))`），但**磁盘上存的不是 `z` 本身**，见 1.3 的“零点约定”坑。
- `s[g]`：group `g` 的 scale，fp16。
- `group_size`：每个 group 覆盖的连续输入维度（本模型=128）。`n_groups = in_features // group_size`。

### 1.1 磁盘上的三张表及其形状

HF 的 `*.safetensors` 里，每个线性层存三张表（`g_idx` 在 `group_size` 对齐时可忽略，本实现直接跳过）：

| 张量 | 形状 | dtype | 含义 |
|---|---|---|---|
| `qweight` | `(in_features // 8, out_features)` | int32 | 每 int32 打包 8 个 4-bit 码字，沿**输入维**打包 |
| `qzeros`  | `(in_features // group_size, out_features // 8)` | int32 | 每 int32 打包 8 个 4-bit 零点，沿**输出维**打包 |
| `scales`  | `(in_features // group_size, out_features)` | fp16 | 每 group 一个 scale |

> 直觉：`qweight` 的 `(in//8, out)` 是因为 32 bit / 4 bit = 8，把输入方向每 8 个权重的 4-bit 码字挤进一个 int32；`qzeros` 是 `(out//8)` 同理但沿输出方向。

### 1.2 打包方向（决定解包代码怎么写）

- **`qweight`**：固定输出 `o` 不变，输入索引 `[8m, 8m+7)` 的 8 个码字挤进同一个 int32。低位 4 bit = 输入索引 `8m`（第一个元素）。
- **`qzeros`**：固定 group `g` 不变，输出索引 `[8m, 8m+7)` 的 8 个零点挤进同一个 int32。同样低位在前。

### 1.3 零点约定（**全文最重要的坑**）

标准 GPTQ（v1，非 GPTQ-V2）磁盘上的 `qzeros` 存的是 **真实零点 − 1**：

```
z_true = unpack(qzeros) + 1
```

权威出处（vLLM 源码）：`vllm/.../quantization/utils/bitblas_utils.py` 的 `unpack_gptq_qzeros`：

```python
def unpack_gptq_qzeros(qzeros, bits, is_gptq_v2=False):
    ...
    if not is_gptq_v2:
        return unpacked_zeros + 1      # ← 真实零点 = 解包值 + 1
    return unpacked_zeros
```

**实测佐证（可直接复用的诊断法）**：对称量化下 `mean(解包后的码字 Q) ≈ 真实零点`。本模型 `qzeros` 恒为 7，而 `mean(Q) ≈ 7.998 ≈ 8`，说明真实零点 = 8 = `qzeros + 1`。
若误用 `qzeros`（=7）反量化，整张权重会平移 `+1·scale`——实测 `mean(W) = +0.00746`，与 `scales.mean() = 0.00747` **完全吻合**，正好是一个 scale 的系统性偏移。这种偏移不会让 loss 爆炸，但会让生成结果彻底错乱。

> 这正是之前“DeepSeek 模型退化”的真正原因：**不是模型坏，是 dequant 零点偏移**。可见“看起来能跑但输出乱码”的 bug，定位要靠数值不变量而非肉眼。

---

## 2. nano-vLLM 原有结构（要改哪些地方）

- `nanovllm/layers/linear.py`：fp16 的 `Linear` / `QKVParallelLinear` / `ColumnParallelLinear` / `RowParallelLinear`。
- `nanovllm/engine/model_runner.py`：`MODEL_REGISTRY` 把 HF `architectures[0]` 映射到模型类；`load_model()` 加载权重。
- `nanovllm/config.py`：`Config.quantization` 字段，从 `hf_config.quantization_config` 自动识别 `"gptq"`。
- `nanovllm/utils/loader.py`：`load_model()`，把 safetensors 张量 `copy_` 到 `nn.Parameter`。

我们的策略是**不破坏 fp16 路径**，新增一套 `GPTQ*ParallelLinear`（复用 fp16 版本的 `weight_loader` 与 TP 切分逻辑），并新增 `Qwen2ForCausalLM` 模型类（Qwen2 与 Qwen3 主要差异是 Qwen2 的 q/k/v/o 是**分开的**投影、MLP 也是 `gate/up` 分开，因此不需要 `packed_modules_mapping` 融合）。

---

## 3. 适配实现（核心代码）

### 3.1 参数与存储布局

`nanovllm/layers/gptq_linear.py`：

```python
class GPTQColumnParallelLinear(nn.Module):
    def __init__(self, input_size, output_size, bias=False, group_size=128,
                 cache_dequant=None):
        super().__init__()
        self.in_features = input_size
        self.out_features = output_size
        self.group_size = group_size
        self.n_groups = input_size // group_size
        self.cache_dequant = _GPTQ_CACHE if cache_dequant is None else cache_dequant
        self._w_cache = None
        # 与 HF GPTQ 对齐的存储布局
        self.qweight = nn.Parameter(torch.zeros(input_size // 8, output_size, dtype=torch.int32), requires_grad=False)
        self.qzeros  = nn.Parameter(torch.zeros(self.n_groups, output_size // 8, dtype=torch.int32), requires_grad=False)
        self.scales  = nn.Parameter(torch.zeros(self.n_groups, output_size, dtype=torch.float16), requires_grad=False)
        self.register_parameter("bias", None) if not bias else \
            self.bias = nn.Parameter(torch.zeros(output_size, dtype=torch.float16), requires_grad=False)
        self.zero_point_bias = 1      # 见 1.3：真实零点 = qzeros + 1
```

> 注意：本模型 `o_proj` 与 MLP **都没有 bias**（已核对 checkpoint 键名），所以 `attention_bias` 仅作用于 q/k/v。误给 `o_proj`/MLP 加 bias 会让 `weight_loader` 找不到对应张量而报错。

### 3.2 bit 解包

```python
def _unpack_qweight(qw):
    # qw: (in//8, out) int32 -> (out, in) 码字
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qw.device)
    w = (qw.unsqueeze(-1) >> shifts) & 0xF          # (in//8, out, 8)
    return w.permute(1, 0, 2).reshape(qw.shape[1], -1)   # (out, in)

def _unpack_qzeros(qz):
    # qz: (n_groups, out//8) int32 -> (out, n_groups)
    shifts = torch.arange(0, 32, 4, dtype=torch.int32, device=qz.device)
    z = (qz.unsqueeze(-1) >> shifts) & 0xF           # (n_groups, out//8, 8)
    z = z.reshape(qz.shape[0], -1).transpose(0, 1)    # (out, n_groups)
    return z
```

- `qweight` 解包：`unsqueeze(-1)>>shifts` 把每个 int32 拆成 8 个码字 → `(in//8, out, 8)`，再 `permute(1,0,2).reshape(out, in)` 得到 `(out, in)`。
- `qzeros` 解包：拆成 `(n_groups, out//8, 8)` 后 `reshape`+`transpose` 得到 `(out, n_groups)`。

### 3.3 反量化数学（含零点 + 分块）

```python
def _dequant_block(self, out_start, out_end):
    nb = out_end - out_start
    qw = self.qweight[:, out_start:out_end]                       # (in//8, nb)
    qz = self.qzeros[:, out_start // 8:(out_end + 7) // 8]         # (ngroups, ceil(nb/8))
    sc = self.scales[:, out_start:out_end]                        # (ngroups, nb)
    w = _unpack_qweight(qw)                                       # (nb, in)
    z = _unpack_qzeros(qz).float().unsqueeze(-1)                  # (nb, ngroups, 1)
    s = sc.transpose(0, 1).float().unsqueeze(-1)                  # (nb, ngroups, 1)
    w = w.float().reshape(nb, self.n_groups, self.group_size)     # (nb, group, k)
    w = (w - (z + self.zero_point_bias)) * s                      # ← 零点 + 1 在此生效
    return w.reshape(nb, self.in_features)
```

逐元素等价于：`W[o, g*gs + k] = (Q[o, g*gs+k] - (z_true[o,g])) * s[o,g]`，其中 `z_true = unpack(qzeros)[o,g] + 1`。

> **精度细节**：反量化在 fp32 下做，再 `.half()` 给 GEMM；激活也保持 fp16。这样归一化/反量化的中间累加都在 fp32，避免 fp16 溢出，与 vLLM 行为对齐。

### 3.4 分块 dequant 控制峰值显存（naive 模式）

`out_features`（如 18944）一次解包会生成巨大临时张量。按 2048 行分块：

```python
def forward(self, x):
    xf = x.half()
    if self.cache_dequant:
        return xf @ self._cached_weight().t()
    out_features = self.out_features
    block = 2048 if out_features > 4096 else out_features
    parts = []
    for start in range(0, out_features, block):
        end = min(start + block, out_features)
        wb = self._dequant_block(start, end).half()      # (block, in) fp16
        parts.append(xf @ wb.t())
    out = torch.cat(parts, dim=-1)
    return out + self.bias if self.bias is not None else out
```

### 3.5 模型层替换（`nanovllm/models/qwen2.py`）

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

**MLP 激活函数必须正确**（曾写反的 bug）：

```python
def forward(self, x):
    gate = self.gate_proj(x)
    up   = self.up_proj(x)
    x = torch.nn.functional.silu(gate) * up      # 不是 gate * silu(up)！
    return self.down_proj(x)
```

**残差约定必须与原始 nano-vLLM 一致**（曾写错的 bug）：残差加法与 RMSNorm 融合，层内**不做最后一步加法**，交由下一层 `input_layernorm`（或最终 `norm`）完成：

```python
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

### 3.6 loader / config 联动

`config.py` 自动识别量化方法：

```python
qcfg = getattr(self.hf_config, "quantization_config", None)
if qcfg is not None and qcfg.get("quant_method") == "gptq":
    assert qcfg.get("bits") == 4, "only 4-bit GPTQ is supported"
    self.quantization = "gptq"
```

`loader.py` 对 GPTQ 权重走 `weight_loader`（默认 `param.data.copy_`），safetensors 的 `qweight/qzeros/scales` 直接 `copy_` 进 `nn.Parameter`：

```python
GPTQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")
# g_idx 在 group 对齐时不用，跳过；其余后缀原样映射参数名
```

`model_runner.py` 通过 `MODEL_REGISTRY` 把 `Qwen2ForCausalLM` 映射到本模型类即可，无需改引擎调度。

### 3.7 RMSNorm 的 dtype fix

RMSNorm 权重若声明为 fp32，会把 fp16 激活提升为 fp32 并一路传染，破坏整图 fp16、拖慢且可能 OOM。改为与**输入 dtype**一致：

```python
def rms_forward(self, x):
    dtype = x.dtype
    x = x.float()                                   # 归一化在 fp32 算（精度）
    var = x.pow(2).mean(-1, keepdim=True)
    x.mul_(torch.rsqrt(var + self.eps))
    return x.mul(self.weight.float()).to(dtype)     # 输出还原 fp16
```

---

## 4. 正确性验证（怎么证明没写错）

“能 load、能 forward、不报错”≠“算对了”。量化 bug 的典型症状是**输出看似流畅但内容错乱**（系统性偏移）。两步验证：

### 4.1 数值不变量自检（零点）

对称量化下 `mean(unpack(Q)) ≈ 真实零点`。用这招可**不依赖任何外部实现**就定出 `zero_point_bias`：
- 本模型 `qzeros=7`，`mean(Q)=7.998≈8` → 真实零点=8 → `zero_point_bias=1`。
- 反证：若 `bias=0`，`mean(W)=+0.00746` 恰等于 `scales.mean()`，说明整体被平移一个 scale，方向完全吻合“少减了 1”。

### 4.2 与权威实现逐 token 对比（ground truth）

以 vLLM `gptq_marlin` 为 ground truth（其 kernel 经过严格验证），**贪心解码逐 token 比对**：

```python
# /tmp/correctness_test.py（节选）
engine = LLMEngine(MODEL, tensor_parallel_size=1, max_num_batched_tokens=2048, max_num_seqs=4, max_model_len=1024)
sp = SamplingParams(temperature=1e-3, max_tokens=48)   # temperature 极小 ≈ 贪心（SamplingParams 禁止纯贪心）
outs = engine.generate(PROMPTS, sp, use_tqdm=False)
# 与 vLLM 记录的 greedy token 序列逐位比对
```

结果：**4 个 prompt × 48 token 全部 48/48 匹配**。这是 dequant 公式正确的决定性证据。

---

## 5. 性能分析与优化

### 5.1 naive 模式的真实瓶颈：显存带宽

每层 MLP 含 3 个 `(18944 × 3584)` 投影，注意力含 2 个 `(3584 × 3584)` 等。单次 forward 需反量化约 **6.5B 参数**（7B × ≈0.93，含 lm_head）。每个元素反量化要 shift/mask/sub/mul 多次访存，单 forward 的权重访存量 ≈ 数百 GB，在 3090（~936 GB/s）上 ≈ **0.5 s/forward**，即 ≈ **1.9 tok/s** 的惨烈解码速度。这是典型的 **memory-bound**，不是算力不够。

naive 模式的优点：权重常驻 int4（≈3.5GB），显存最省。

### 5.2 优化：权重缓存（dequant 一次，缓存 fp16）

```python
def _cached_weight(self):
    if self._w_cache is None:
        w = self._dequant_block(0, self.out_features).half()   # 只解包一次
        for name in ("qweight", "qzeros", "scales"):            # 解包后释放 int4 权重
            getattr(self, name).data = torch.empty(0, ...)
        self._w_cache = w
    return self._w_cache
```

开关注环境变量 `NANOVLLM_GPTQ_CACHE=1`。收益：**TTFT 510ms → 31ms（17×），并发吞吐 7.2 → 69.4 tok/s（9.6×）**。代价：显存回到 fp16 的 ≈14GB（量化只省了“加载/磁盘”，运行期不再省）。

### 5.3 与 vLLM Marlin 的差距（核心 insight）

| engine | TTFT(ms) | Decode(t/s) | 并发吞吐(t/s) |
|---|---|---|---|
| nano-vLLM 朴素 dequant | 530.2 | 1.9 | 7.2 |
| nano-vLLM 权重缓存 fp16 | 31.3 | 27.5 | 69.4 |
| vLLM gptq_marlin | 11.1 | 93.0 | 303.1 |

即使缓存到 fp16，我们仍只有 vLLM 的 **~23% 吞吐**。原因：
- **Marlin 把反量化融进 GEMM kernel**，不把 fp16 权重物化到显存（省一次 HBM 往返），这正是 weight-only 量化的“正统”加速路径。
- vLLM 还叠加了 CUDA Graph、FlashAttention、更细的调度与 prefix cache。
- 我们只是 python/torch 朴素实现：每次 matmul 都要先把 fp16 权重从显存读进来。

> 这个差距本身就是最好的面试素材：**量化推理的加速不来自“权重变小”，而来自“反量化不再显式落盘”——kernel 内融合才是关键**。

### 5.4 测量口径（诚实性）

- vLLM 离线 API 的 `RequestOutput.metrics` 在本环境为 `None`，TTFT 用 `max_tokens=1` 端到端计时（含 1 个 decode step + 调度开销），**略微高估 vLLM 的 TTFT**，即对比偏保守。
- nano-vLLM 的 TTFT = 首个 prefill step 完成耗时；Decode 从首 token 后开始计时。

---

## 6. 复现命令

```bash
# 环境
export CUDA_VISIBLE_DEVICES=1
PY=/nas_data/WR/conda/wr-vllm/bin/python

# 正确性：nano-vLLM vs vLLM(gptq_marlin) 贪心逐 token 对比
$PY /tmp/correctness_test.py

# 性能：naive / 缓存 / vLLM 三引擎对比（独立子进程，避免显存干扰）
$PY bench.py --engine all --max_tokens 128

# 仅跑某一引擎
NANOVLLM_GPTQ_CACHE=1 $PY bench.py --engine nanovllm --max_tokens 128
```

---

# 第二部分：Infra 简历项目（可直接用）

## 项目标题（一）

**LLM 推理引擎量化（GPTQ-Int4）支持与性能优化** — nano-vLLM（自研极简推理引擎，~2k 行）

## 项目描述（简历正文，约 60 字）

在自研极简推理引擎中实现 GPTQ-4bit 权重量化推理，打通 bit 解包→零点对齐→融合反量化的完整链路；通过数值不变量 + 权威引擎逐 token 比对验证正确性，并定位带宽瓶颈，吞吐从 7 tok/s 优化到 69 tok/s，定位出与 Marlin 融合 kernel 的真实差距。

## 职责与成果（bullet，可直接贴）

- 在 ~2k 行的极简推理引擎中落地 GPTQ-Int4 支持：实现 int32 位解包、零点约定对齐（`z_true = qzeros + 1`）、group-wise 反量化与分块 matmul，复用原有 TP 切分与权重加载路径，无需改动引擎调度。
- 设计正确性验证方案：用“对称量化 `mean(码字) ≈ 真实零点`”数值不变量定位零点偏移 bug，并以 vLLM `gptq_marlin` 为 ground truth 做贪心逐 token 比对，4 个 prompt × 48 token **全匹配**。
- 定位 naive 反量化的**显存带宽瓶颈**（单 forward 反量化 6.5B 参数 ≈ 0.5s，仅 1.9 tok/s），实现“反量化一次缓存 fp16”优化，TTFT 17×、并发吞吐 9.6×（7.2 → 69.4 tok/s）。
- 通过系统剖析指出与 vLLM 约 4× 吞吐差距的根因：Marlin 将反量化**融进 GEMM kernel**（不物化 fp16 权重），量化收益来自“免显存往返”而非“权重变小”。

## 量化指标（放简历“成绩”栏）

| 指标 | 优化前 | 优化后 |
|---|---|---|
| 解码吞吐（单请求） | 1.9 tok/s | 27.5 tok/s |
| 4 并发总吞吐 | 7.2 tok/s | 69.4 tok/s |
| 首 token 时延 TTFT | 530 ms | 31 ms |
| 与 vLLM 吞吐比 | 0.02× | 0.23× |
| 权重显存 | 3.5 GB (int4) | 14 GB (fp16) |

## 技术栈 / 关键词

PyTorch、GPTQ、weight-only quantization、int4 位解包、group-wise dequant、CUDA 显存带宽分析、vLLM、Marlin GEMM kernel、RMSNorm fp32 归一化、Tensor Parallel、推理引擎。

## 面试 talk track（STAR 展开）

**S（情境）**：业务要在单张 24GB 消费卡上跑 7B 模型，fp16 权重 14GB 顶满显存；现成引擎（vLLM）黑盒、不利于学习量化内核细节，于是基于 nano-vLLM 自研支持。

**T（任务）**：在不破坏 fp16 路径、不重写引擎调度的前提下，让引擎能正确且不太慢地跑 GPTQ-Int4 模型。

**A（行动 / 技术亮点）**：
1. 先把 GPTQ 格式吃透：码字打包方向、`(in//8, out)` / `(ngroups, out//8)` 形状、`z_true = qzeros + 1` 的零点约定。
2. 用数值不变量（`mean(Q) ≈ z_true`）发现并修复零点偏移——这是“能跑但输出错乱”的静默 bug，肉眼看不出来。
3. 分块反量化 + fp32 中间累加 + fp16 输出，对齐 vLLM 的数值行为，避免 fp16 溢出。
4. 正确性用**双保险**验证：数值不变量自检 + vLLM ground truth 逐 token 比对。
5. 性能剖析定位 bandwidth-bound，做 dequant-cache 优化；并进一步剖析出与 Marlin 的差距根因。

**R（结果）**：正确性 48/48 全匹配；吞吐 7→69 tok/s；并形成对“量化加速本质 = kernel 内融合反量化”的系统性认知。

## 延伸思考（加分项，面试官最爱追问）

- **为什么不一上来就写 CUDA kernel？** 先用 torch 朴素实现 + 正确性与基线对齐，确认算法正确再谈性能；过早优化会淹没 bug。
- **下一步怎么做才能追平 vLLM？** 用 Triton/CK 写 fused dequant-GEMM kernel（int4 直接进 TensorCore 的 mma 指令，不物化 fp16），并接入 CUDA Graph 与 paged KV cache。
- **为什么权重缓存反而显存变大？** 量化只省“磁盘/加载”，naive 运行时仍常驻 int4 省显存；cache 模式用 fp16 换带宽，是容量↔带宽的 trade-off。
- **GPTQ-v2 / AWQ / GGUF 怎么扩展？** 零点是 `qzeros` 本身（不需 +1）即 GPTQ-v2；AWQ 是 `W = (Q) * s + z` 的反向缩放 + act-order；GGUF 是另一套打包（需转 `q8_0` 等）。架构上只需新增 `unpack`/`dequant` 与 loader 映射。

---

## 附录：关键文件清单

| 文件 | 作用 |
|---|---|
| `nanovllm/layers/gptq_linear.py` | GPTQ 线性层：bit 解包、零点对齐、分块/缓存反量化 |
| `nanovllm/models/qwen2.py` | Qwen2 模型定义，GPTQ 层替换、MLP/残差正确性 |
| `nanovllm/layers/layernorm.py` | RMSNorm，fp32 归一化 + 输出还原 fp16 |
| `nanovllm/config.py` | 从 `quantization_config` 自动识别 `gptq` |
| `nanovllm/utils/loader.py` | safetensors → `nn.Parameter`（`qweight/qzeros/scales` 直拷） |
| `nanovllm/engine/model_runner.py` | `MODEL_REGISTRY` 分派 `Qwen2ForCausalLM` |
| `bench.py` | 三引擎（naive / cache / vLLM）对比基准 |
| `/tmp/correctness_test.py` | 逐 token 正确性验证 |
