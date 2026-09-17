# 投机解码 (Speculative Decoding) 设计与实测

## 0. 为什么做这个

已有的 GPTQ-Int4 模块把 7B 权重从 14.22 GiB 压到 5.20 GiB（−63%），但吞吐基本持平：
量化换到的是**显存**，不是速度。decode 仍然是逐 token 读全部权重的访存瓶颈，
单请求只能到 ~30 tok/s。

投机解码直接攻击的就是这个瓶颈，而且和量化能形成闭环：

> 量化省下的 9 GiB → 装下 0.5B 的 draft 模型 → draft 反过来把 decode 的访存瓶颈打掉。
> **省下来的显存变成了算力。**

硬件前提：两张 RTX 3090 **无 NVLink**（`nvidia-smi topo` 显示 PHB，GPU 间走 PCIe）。
任何依赖高频 GPU 间通信的方案（TP=2、跨机 PD 分离）在这套硬件上都吃不到收益；
投机解码的通信量为零，是唯一能稳定拿到正收益的方向。

## 1. 模型搭配

| | target | draft |
|---|---|---|
| 模型 | Qwen2.5-7B-Instruct-GPTQ-Int4 | Qwen2.5-0.5B-Instruct |
| vocab | 152064 | 151936 |
| 权重显存 | 5.20 GiB | 1.19 GiB |

词表长度**不相等**，这是个真问题（Qwen2.5-0.5B 的 HF config 声明 151936，
7B 声明 152064，多出的 128 个是 padding 条目）。处理方式：

- `tests/check_spec_vocab.py` 先证明两边 tokenizer 在公共 id 区间上**逐条一致**
  （151665 个词条，0 冲突），draft 词表中没有 id ≥ 公共长度的条目；
- 概率空间统一截断到 `V = min(Vd, Vt)` 后重新归一化，在该空间内做拒绝采样；
- 实测 target 落在尾部（被丢弃）的概率质量是 **7.8e-09**，可以忽略；
- 附带约束：draft 输出 id 必然 < V，绝不会让 target 的 embedding 越界。

> 顺带一个可考的点：Qwen3 系列 vocab = 151936 且语义与 Qwen2.5 不同，
> **不能**混用作 draft/target，同系列 + 同 tokenizer 是硬要求。

## 2. 一轮投机解码的流程

```
1) draft:  γ+1 次 forward (第 γ+1 次只为把 prefix 的 KV 补到与 target 对齐)
2) target: 1 次 verify forward, query = [x_{L-1}, d_1, ..., d_γ] 共 γ+1 个位置
3) 拒绝采样: 逐位判接受; 首位被拒时从 (p-q)_+ 重采样; 全接受时白送一个 bonus token
4) 回滚:    未被接受的 draft token 连同它跨出去的 KV block 一起回收
```

几个容易做错、也确实做错了的地方：

**a) draft 必须跑 γ+1 次，不是 γ 次。**
第 γ+1 次的输出被丢弃，它的作用只是把第 `L+γ-1` 个位置的 K/V 写进 draft cache。
下一轮若 bonus 也被接受，draft 的首个 query 正好落在那个位置——缺了它就会读到脏显存。

**b) verify 的 query 起点是 `x_{L-1}` 而不是 `x_L`。**
这样 `seqlen_k - seqlen_q = L-1` 恰好等于 cache 里的有效长度，可以直接复用
flash-attn varlen + block_table 的既有语义（chunked prefill 用的是同一套约定）。
代价是 `slot(L-1)` 会被重写一次，但内容不变，换来的是不需要为"少读一个 token"特判掩码。

**c) 占位 token 会让 `num_completion_tokens` 虚增 γ。**
`max_tokens` 的余量必须按**本轮开始时**的完成数算，否则 γ 一旦吃满就会出现
"每轮确认 0 个 token"，序列永远不 finish，`generate()` 直接死循环（真踩过）。

**d) 投机期间不 commit prefix cache。**
被拒绝的候选一旦被 hash，后续相同前缀的请求就会命中到错误的 block。
代价是这些序列被抢占后要从更早的位置重算——慢，但不会错。

## 3. 三个性能/正确性坑

### 3.1 没有 flash-attn 时，sdpa fallback 根本不读 paged cache

本机环境**没有安装 flash_attn**，`Attention` 一直走 `_sdpa_prefill` / `_sdpa_decode`
fallback。前者的实现完全忽略 `block_table`，于是任何"部分 KV 落在 cache 里"的调用
（chunked prefill 续段、投机解码的 verify）都会**丢掉整个历史上下文**。

### 3.2 PyTorch SDPA 的 `is_causal` 在 `L ≠ S` 时不等价于 offset 因果

补上 gather 之后仍然不对。用最小实验扫描掩码偏移确认：

```
L=5 S=15:
  off=0  (j <= i)          -> max|Δ| = 0.000e+00   <== is_causal 的实际行为
  off=10 (j <= i + S-L)    -> max|Δ| = 2.564e+00   <== 我们真正需要的
```

也就是说 gather 出来的前缀被整段 mask 掉了。显式传 float mask 可以修正，但会退化到
math backend，复杂度 O(L·S·D) 且中断 flash，对 verify 来说比省下的时间还贵。

**解决**：`layers/attention.py` 新增 `cached_causal_attn_kernel`，直接从 paged cache
读 key/value（不物化到连续内存），掩码按 `j <= i + cached` 手工处理。

### 3.3 draft 单步 forward 被 launch 开销主导

profiler 显示 0.5B 的 draft 一次 forward 里 GPU 只算了 **2.7 ms**，墙钟却要 **20 ms**，
CPU 侧 5.4 ms —— 一次 forward 约 600 次 kernel launch，纯 launch-bound。
γ+1 次 draft 的开销会把投机解码的收益全部吃回去（不做这步时加速比只有 **0.70x**）。

**解决**：draft 的单步 forward 用 CUDA graph 捕获（按 batch size 缓存）。
→ 单步 **20.7 ms → 2.77 ms**。

### 3.4 顺带：decode attention 的每层 GPU→CPU 同步

`_sdpa_decode` 里 `int(ctx_lens[j])` 每个 batch 元素**每层**触发一次同步，
这也是 0.5B draft 慢到 20 ms 的一部分原因。改用 flash-decoding 风格的两阶段 Triton
kernel（沿 key 分段并行 + 归约）后，基线的 4 并发吞吐也从 77.4 → **111.6 tok/s**。

## 4. 正确性验证

三层，从强到弱，`tests/verify_spec.py`：

| Part | 内容 | 结果 |
|---|---|---|
| A | greedy 下输出 == 逐步 argmax 的黄金结果（含各位置被拒的修正） | 6/6 |
| B | 采样下首 token 分布 == target 分布；bonus 分布 == p(posG) | TVD 0.006 / 0.011 |
| B | **变异测试**：注入 4 种实现错误，检验必须逐个抓住 | 4/4 抓住 |
| C | 端到端 greedy：开/关投机解码 4 条 prompt **逐 token 相同** | 4/4 |

变异测试的 TVD 对照（阈值 0.02）：

| 注入的错误 | 首 token TVD | bonus TVD |
|---|---|---|
| 正确实现 | 0.0061 | 0.0110 |
| 忘了除以 q | **0.3829** | 0.1615 |
| 残差分布用 p 而非 (p−q)₊ | **0.2234** | 0.0103 |
| 残差分布用 draft 的 q | **0.5017** | 0.0102 |
| bonus 取错位置 | 0.0053 | **0.5252** |

最后一行说明：只看"首 token 分布"抓不到 bonus 取错位置的 bug，**两个指标都要有**
——这正是变异测试的价值，否则"分布一致"这个结论根本站不住。

另有 `tests/check_prefill_cache.py` 守住框架级不变式：
"读 paged cache 的路径 == 整段重算"（chunked prefill、spec verify 两条路径）。

## 5. 实测（RTX 3090 24GB，Qwen2.5-7B-GPTQ-Int4 + Qwen2.5-0.5B）

### 5.1 装上 flash-attn 之后（当前口径）

环境装上 flash-attn 2.8.3 后，**接受率从 0.48~0.56 跳到 0.74~0.90**：之前自研的 decode
attention kernel 存在细微精度损失，把 draft 的输出质量拖低了。这从侧面说明 fallback
实现再小心也难达到 flash-attn 的数值质量 —— 能装就装。

| γ | TTFT | 单请求 decode | 4 并发吞吐 | 接受率 α | 平均接受长度 |
|---|---|---|---|---|---|
| 0（基线） | 34.9 ms | 32.6 tok/s | 121.1 tok/s | — | — |
| 5 | 56.6 ms | 101.5 tok/s | 225.4 tok/s | — | — |
| 7 | 57.8 ms | **118.5 tok/s** | **289.3 tok/s** | 0.741 | 6.19 |
| 7（叠动态 γ） | 56.5 ms | 119.1 tok/s | **294.0 tok/s** | — | — |

- 单请求 decode **3.64×**，4 并发吞吐 **2.39×**
- TTFT 从 35 → 57 ms：prefill 时必须顺便把 prompt 的 KV 写进 draft cache，
  否则第一轮 draft decode 会读到未初始化的显存。这是已知代价。
- KV block 回收：连跑 3 轮 × 4 并发，空闲 block 数完全回到起点（无泄漏）
- 未装 flash-attn 的旧口径（基线 113.6 / γ=7 时 220.8，1.94×）保留作对照，
  **两个口径的基线不同，不要混用**。

### 5.2 未装 flash-attn 时的旧口径

| γ | TTFT | 单请求 decode | 4 并发吞吐 | 接受率 α | 平均接受长度 |
|---|---|---|---|---|---|
| 0（基线） | 35.0 ms | 30.1 tok/s | 113.6 tok/s | — | — |
| 3 | 55.8 ms | 81.5 tok/s | 172.0 tok/s | — | — |
| 5 | 58.3 ms | 101.6 tok/s | **222.8 tok/s** | 0.560 | 3.80 |
| 7 | 56.3 ms | **118.9 tok/s** | 220.8 tok/s | 0.482 | 4.37 |

## 6. 代码结构

```
nanovllm/
  engine/speculator.py     拒绝采样 / greedy 验证 / 接受率统计 (纯张量, 可独立测)
  engine/model_runner.py   draft 模型加载 + 独立 KV cache + CUDA graph + run_spec
  engine/llm_engine.py     step 分支、占位/回滚、max_tokens 与 eos 收尾
  engine/block_manager.py  may_append_n / trim  (占位与回滚)
  engine/sequence.py       append_spec_tokens / pop_tokens
  models/qwen2_dense.py    未量化 Qwen2, 供 draft 用 (models/qwen2.py 已是 GPTQ-only)
  layers/attention.py      paged cache prefill kernel + flash-decoding decode kernel
bench_spec.py              性能基准 (含 KV block 泄漏检查)
tests/verify_spec.py       A/B/C 三层正确性
tests/check_spec_vocab.py  draft/target 词表兼容性前置检查
tests/check_prefill_cache.py  读 cache 路径 == 整段重算
tests/profile_spec.py      target decode / draft decode / verify 的耗时分解
```

开关：`LLM(model, draft_model=..., num_speculative_tokens=γ)`。
不传 `draft_model` 时所有投机解码代码路径完全不进入。

---

# 第二部分：动态 γ 与多候选（tree）验证

## 7. 动态 γ

### 7.1 为什么需要

固定 γ 只能在一个工作点上最优，而接受率 α 会随 prompt 分布、并发数、温度漂移。
γ 太小浪费"多验证几个也不亏"的机会，γ 太大则给 draft 白烧时间。

### 7.2 怎么选

用一阶延迟模型（Leviathan et al. 2023 同款）：

```
E[每轮确认 token 数] = (1 − α^(γ+1)) / (1 − α)
E[每轮耗时(以 target 单步为单位)] = 1 + γ·c      # 1 次 verify + γ+1 次 draft
→ γ* = argmax  (1 − α^(γ+1)) / ((1 − α)(1 + γ·c))
```

`c = t_draft_step / t_target_step` 由 ModelRunner 在线测（实测本配置 **c ≈ 0.07**，
即一次 draft 只相当于 target 的 7% —— 这是 0.5B vs 7B 加上 CUDA graph 后的结果）。

### 7.3 两个实现细节（都是踩出来的）

**a) 计时必须只测纯 GPU 时间。**
draft 循环里每步都有 `tok.tolist()`（GPU→CPU 同步），把 CPU 侧时间也算进 t_draft
会让 c 被高估 2~3 倍，控制器就会把 γ 压得过小 —— 实测这一条让单请求 decode
从 100 掉到 74。现在改成：每 16 轮做一次「额外 replay 一次 draft + 正式那次 verify」
的同时间线测量，且读的是**上一轮**的事件（那时已自然完成，不额外插入同步）。

**b) 用实测的 γ→接受长度表，而不是几何模型 + 切换滞后。**
接受率随候选位置下降，恒定 α 的几何模型会系统性高估长链 —— 实测它把 γ 从 7 推到 8
反而变慢。现在维护 `gamma -> EMA(每轮确认 token 数)`，测过 ≥3 轮就用实测值，
没测过才用几何模型兜底；并且新 γ 必须比当前 γ 好 10% 才切换，否则会在阈值附近横跳、
每次切换都打断统计。

### 7.4 效果（诚实版）

- 在固定负载下（T=0.8）动态 γ 收敛到 7，吞吐 294.0 vs 静态 γ=7 的 289.3，**+1.6%**；
  在 T=0.3 / T=1.2 下最优 γ 也都落在 7 附近（因为 c 很小，大 γ 一直划算）。
  也就是说**在这套配置下动态 γ 的收益很小**，因为人工调好的静态 γ 已经接近最优。
- 它的真正价值在**换配置后不用重新调参**：c 一旦上升（比如换成更大的 draft、或 draft
  没做 CUDA graph），最优 γ 会明显下降。按上面的公式，α=0.74 时：
  - `c=0.07` → 最优 γ ≈ 7~8
  - `c=0.30` → 最优 γ ≈ 2
  固定 γ 换硬件就要重调，动态 γ 自动跟上。

## 8. 多候选（tree）验证：已实现，但当前配置下不划算

### 8.1 做了什么

一次 target forward 同时验证 **K 条候选链**（`num_spec_candidates=K`），取接受最长的那条：

- draft 阶段每步处理 B×K 行（K 条链并行 decode），第 0 步按 draft 分布的 **top-K 分叉**
  （不是独立采样 K 次 —— 同分布下独立采样的 K 条链会在第一个 token 就重合，实测
  K=2 的接受长度反而低于 K=1：5.95 vs 6.19）；
- verify 阶段把 K 条链当成 B×K 个 varlen "序列"，每条链内部仍是链式的，
  所以可以复用已有的 causal 掩码，不需要树形掩码 kernel；
- 拒绝采样在 B×K 行上独立做，然后每个 seq 取接受最长的那条链。

greedy 下 K=2 的端到端输出仍然与基线**逐 token 相同（4/4）**。

### 8.2 为什么不划算（两个原因）

**a) paged KV 的「逻辑连续」语义会让多链共享 slot 区间时上下文错乱。**
draft decode 用 `cache_seqlens = L+i` 去读逻辑位置 `L+i−1` 的 slot，而多候选时链 c 的
节点写在 `L + c*G + i − 1` —— 两者对不上，链 c 读到的是别的链（或未初始化）的内容。
表现为接受率下降（T=0.3 实测 α 0.743 → 0.577）。
注意这不破坏 greedy 的正确性（被接受的 token 一定等于 target argmax，候选质量只影响
接受率），所以单看"输出对不对"是发现不了的 —— 必须盯接受率。

> 正确做法是给每条链分配**独立的 block** 并构造虚拟 `block_table`
> （`seq.block_table[:ceil(L/256)] + [chain_block]`），让链 c 的逻辑位置 `L+i`
> 映射到 `chain_block[i]`。需要 BlockManager 支持借还临时 block，尚未实现。

**b) 就算修好，本配置下 K>1 也不划算：verify 成本占主导。**
本项目 c ≈ 0.07，一轮的成本 ≈ `1 + γ·c`，主要花在 target 的 verify 上；
而 K 翻倍会**直接把 verify 的计算量翻倍**（query 数从 B×8 变 2B×8），
收益却受限于"单链已经接近接受上限"—— α 0.74~0.9 时单链 mean_acc 已到 6.19/8。

| T=0.8, γ=7 | α | 平均接受长度 | 4 并发吞吐 |
|---|---|---|---|
| K=1 | 0.741 | 6.19 | **289.3** |
| K=2（top-K 分叉） | 0.648 | 5.54 | 277.4 |
| K=2（独立采样） | 0.708 | 5.95 | 207.9 |

### 8.3 什么时候值得做

tree verification 的适用条件是：
1. **接受率中等**（α 0.3~0.6）—— 单链频繁在很早的位置被拒，多候选才有救；
   α 已经 0.9 时再加候选只是浪费。
2. **verify 的增量成本要低** —— 也就是 target 相对 draft 越"贵"越不划算。
   本项目 target 7B / draft 0.5B，verify 是主要开销，加大 K 直接放大它。
3. 真正共享前缀的 tree attention 还需要给 kernel 传**显式的祖先 key 列表**
   （本次实现是按链组织、共享读了 K 次前缀），这部分留作后续。

`num_spec_candidates` 因此**默认 1**，K>1 标记为实验特性并在运行时打印提示。
