from __future__ import annotations
import os
import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.speculator import Speculator
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

MODEL_REGISTRY = {
    "Qwen3ForCausalLM": "nanovllm.models.qwen3:Qwen3ForCausalLM",
    "Qwen2ForCausalLM": "nanovllm.models.qwen2:Qwen2ForCausalLM",
}

# draft 模型始终走未量化的 dense 分支 (models/qwen2.py 已被改写成 GPTQ-only)
DRAFT_REGISTRY = {
    "Qwen2ForCausalLM": "nanovllm.models.qwen2_dense:Qwen2DenseForCausalLM",
}

# 投机解码随机流的 seed 基数: 各 rank 必须独立算出同一个 seed, 见 ModelRunner._make_generator
_SPEC_GEN_BASE = 20240916


def _import_class(path: str):
    import importlib
    module_path, cls_name = path.split(":")
    module = importlib.import_module(module_path)
    return getattr(module, cls_name)


class ModelRunner:

    def __init__(self, config: Config, rank: int,  event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        # GPTQ 动态反量化不兼容 CUDA Graph，强制 eager
        self.enforce_eager = True if config.quantization == "gptq" else config.enforce_eager
        self.world_size = config.tensor_parallel_size
        assert self.world_size == 1 or config.quantization is None, "GPTQ 仅支持单卡 (TP=1)"
        self.rank = rank
        self.event = event

        # 把 GPTQ group_size 注入 hf_config，供模型构造时读取
        if config.quantization == "gptq":
            qcfg = hf_config.quantization_config
            hf_config.group_size = qcfg.get("group_size", 128)

        # 端口可用 NANOVLLM_MASTER_PORT 覆盖: 同机并行跑多个实例/测试时避免 2333 冲突
        port = os.getenv("NANOVLLM_MASTER_PORT", "2333")
        dist.init_process_group("nccl", f"tcp://localhost:{port}",
                                world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch_dtype = hf_config.torch_dtype
        if isinstance(torch_dtype, str):
            torch_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[torch_dtype]
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device("cuda")
        arch = hf_config.architectures[0]
        model_cls = _import_class(MODEL_REGISTRY.get(arch, MODEL_REGISTRY["Qwen3ForCausalLM"]))
        self.model = model_cls(hf_config)
        load_model(self.model, config.model)
        # 投机解码相关 state (未启用时保持 None, 现有路径零开销)
        self.speculator = None
        self.draft_model = None
        self.draft_kv_cache = None
        self.spec_round = 0
        if config.draft_model:
            self._init_draft_model(config)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def _init_draft_model(self, config: Config):
        """加载 draft 模型。

        draft 需与 target 共用 tokenizer (例如 Qwen2.5-0.5B-Instruct 配
        Qwen2.5-7B-Instruct-GPTQ-Int4)。词表长度允许不等 (151936 / 152064),
        Speculator 会把概率空间截断到公共前缀后重归一化, 其安全性由
        tests/check_spec_vocab.py 保证 (公共区间上 token->id 映射逐条一致)。
        """
        assert self.world_size == 1, "投机解码当前仅支持单卡 (TP=1)"
        draft_config = AutoConfig.from_pretrained(config.draft_model)
        arch = draft_config.architectures[0]
        assert arch in DRAFT_REGISTRY, \
            f"draft 模型架构 {arch} 不受支持, 可选 {list(DRAFT_REGISTRY)}"
        assert config.num_speculative_tokens >= 1
        # draft 的 config.torch_dtype 常常与 target 不一致 (Qwen2.5-0.5B 声明 bfloat16,
        # target 是 float16)。必须统一到**当前运行 dtype**: 否则 KV cache 会按 draft
        # 的 bf16 分配, 与 fp16 的权重/激活混算时直接报 dtype 不匹配, 而且两个模型的
        # 数值口径也会被扯开。
        draft_config.torch_dtype = torch.get_default_dtype()
        self.draft_model = _import_class(DRAFT_REGISTRY[arch])(draft_config)
        load_model(self.draft_model, config.draft_model)
        self.speculator = Speculator(config.num_speculative_tokens)
        print(f"[spec] draft={config.draft_model} "
              f"γ={config.num_speculative_tokens} "
              f"vocab(draft/target)={draft_config.vocab_size}/{self.config.hf_config.vocab_size}")

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        # GPTQ fp32 dequant 峰值高, 用小 token 热身以控制显存
        max_num_batched_tokens = min(self.config.max_num_batched_tokens, 256)
        max_model_len = min(self.config.max_model_len, max_num_batched_tokens)
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * seq_len) for _ in range(num_seqs)]
        for seq in seqs:
            seq.num_scheduled_tokens = seq_len
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        block_bytes = self._block_bytes(hf_config)
        # draft 复用同一套 block/slot 布局 (两边 block_size 相同), 因此它也要按比例
        # 分走一份 KV 预算; 不先扣掉的话 KV cache 会按 target 独占来算 -> 直接 OOM
        draft_ratio = 0.0
        if self.draft_model is not None:
            draft_ratio = self._block_bytes(self.draft_model.config) / block_bytes
        budget = int(total * config.gpu_memory_utilization - used - peak + current)
        config.num_kvcache_blocks = int(budget / (block_bytes * (1.0 + draft_ratio)))
        if config.num_kvcache_blocks <= 0:
            # 显存估算不足(GPTQ dequant 峰值高), 用保守默认值兜底
            config.num_kvcache_blocks = 64
        assert config.num_kvcache_blocks > 0
        self.kv_cache = self._alloc_kv_cache(hf_config, config.num_kvcache_blocks)
        self._bind_kv_cache(self.model, self.kv_cache)
        if self.draft_model is not None:
            self.draft_kv_cache = self._alloc_kv_cache(self.draft_model.config, config.num_kvcache_blocks)
            self._bind_kv_cache(self.draft_model, self.draft_kv_cache)

    def _kv_shape_and_dtype(self, hf_config):
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        dtype = hf_config.torch_dtype
        if isinstance(dtype, str):
            dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype]
        return num_kv_heads, head_dim, dtype

    def _block_bytes(self, hf_config) -> int:
        num_kv_heads, head_dim, dtype = self._kv_shape_and_dtype(hf_config)
        return 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * dtype.itemsize

    def _alloc_kv_cache(self, hf_config, num_blocks: int) -> torch.Tensor:
        num_kv_heads, head_dim, dtype = self._kv_shape_and_dtype(hf_config)
        return torch.empty(2, hf_config.num_hidden_layers, num_blocks,
                           self.block_size, num_kv_heads, head_dim, dtype=dtype)

    def _bind_kv_cache(self, model: torch.nn.Module, kv_cache: torch.Tensor):
        layer_id = 0
        for module in model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = kv_cache[0, layer_id]
                module.v_cache = kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            start = seq.num_cached_tokens
            seqlen_q = seq.num_scheduled_tokens
            end = start + seqlen_q
            seqlen_k = end
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = seq.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = seq.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = seq.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = [seq.temperature for seq in seqs]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(input_ids, positions, is_prefill)
        if is_prefill and self.draft_model is not None:
            # prefill 必须同时把 draft 的 KV 写进去: 第一轮 draft decode 是增量 decode,
            # 如果 prompt 的 KV 不在 draft cache 里, 它读到的就是未初始化的显存。
            # 这里的 context 还是上面 prepare_prefill 设的那份, slot 布局两边完全一致, 直接复用。
            self._draft_forward(input_ids, positions)
        greedy = bool(seqs) and all(seq.greedy for seq in seqs)
        token_ids = self.sampler(logits, temperatures, greedy).tolist() if self.rank == 0 else None
        reset_context()
        return token_ids

    # ------------------------------------------------------------ 投机解码
    def _slot_of(self, seq: Sequence, token_idx: int) -> int:
        """第 token_idx 个 token 落在 paged KV cache 的哪个 slot。draft/target 共用。"""
        return seq.block_table[token_idx // self.block_size] * self.block_size + token_idx % self.block_size

    @torch.inference_mode()
    def _draft_forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        """draft 模型 forward -> logits (eager)。复用调用方已经 set 好的 context。"""
        hidden = self.draft_model(input_ids, positions)
        return self.draft_model.compute_logits(hidden)

    @torch.inference_mode()
    def _draft_forward_graph(self, input_ids, positions, slot_mapping, context_lens, block_tables):
        """draft 单步 forward 走 CUDA graph。

        为什么必须 graph 化: 0.5B 的 draft 一次 forward 只有约 600 次 kernel launch,
        GPU 实际只算 2.7ms, 但墙钟要 20ms —— launch 开销占了九成。γ+1 次 draft 的
        launch 开销会把投机解码的收益全吃回去 (实测不做这步时加速比只有 0.7x)。
        """
        bs = input_ids.size(0)
        graph, gv = self._get_draft_graph(bs)
        gv["input_ids"].copy_(input_ids)
        gv["positions"].copy_(positions)
        gv["slot_mapping"].copy_(slot_mapping)
        gv["context_lens"].copy_(context_lens)
        gv["block_tables"][:, :block_tables.size(1)].copy_(block_tables)
        graph.replay()
        return gv["logits"]

    @torch.inference_mode()
    def _get_draft_graph(self, bs: int):
        """按 batch size 缓存 draft 的 decode graph (首次遇到该 bs 时才捕获)。"""
        graphs = self.__dict__.setdefault("_draft_graphs", {})
        if bs in graphs:
            return graphs[bs]
        hf = self.draft_model.config
        max_num_blocks = (self.config.max_model_len + self.block_size - 1) // self.block_size
        # 必须显式 device="cuda": ModelRunner.__init__ 末尾把默认 device 改回
        # 了 cpu (避免 Python 侧构造张量时白占显存)
        gv = {
            "input_ids": torch.zeros(bs, dtype=torch.int64, device="cuda"),
            "positions": torch.zeros(bs, dtype=torch.int64, device="cuda"),
            "slot_mapping": torch.zeros(bs, dtype=torch.int32, device="cuda"),
            "context_lens": torch.zeros(bs, dtype=torch.int32, device="cuda"),
            "block_tables": torch.zeros(bs, max_num_blocks, dtype=torch.int32, device="cuda"),
            "logits": torch.zeros(bs, hf.vocab_size, device="cuda"),
        }
        graph = torch.cuda.CUDAGraph()

        def _body():
            gv["logits"] = self.draft_model.compute_logits(
                self.draft_model(gv["input_ids"], gv["positions"]))

        set_context(False, slot_mapping=gv["slot_mapping"], context_lens=gv["context_lens"],
                    block_tables=gv["block_tables"])
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):                       # warmup: 让 Triton/显存池先稳定
                _body()
        torch.cuda.current_stream().wait_stream(side)
        with torch.cuda.graph(graph):
            _body()
        reset_context()
        graphs[bs] = (graph, gv)
        return graphs[bs]

    def _make_generator(self, step: int) -> torch.Generator:
        """为 draft/verify 采样生成**跨进程一致**的随机数流。

        TP>1 时只有 rank0 会把采样结果写回 Sequence, 但每个 rank 在**同一轮内部**
        都要知道 draft token 才能做下一步 forward (一次 call 无法中途同步)。
        所以用轮次 + 步数派生的确定性 seed, 保证所有 rank 采出同一个 token。
        """
        gen = torch.Generator(torch.device("cuda"))
        gen.manual_seed(_SPEC_GEN_BASE + self.spec_round * 1024 + step)
        return gen

    def prepare_draft_decode(self, seqs: list[Sequence], i: int, G: int):
        """draft 的第 i 步 (i = 0..G), 每个 seq 处理第 L-1+i 个 token。

        i=0     输入已确认的最后一个 token -> 产出 d_1
        i>=1    输入第 i 个草稿 token      -> 产出 d_{i+1}
        i==G    输出**丢弃**: 只为把 prefix 的 KV 补到与 target 对齐。
                下一轮若 bonus 也被接受, draft 的首个 query 会落在新增的那个位置上,
                届时该位置必须已有正确的 K/V, 否则会读到脏显存。
        """
        input_ids, positions, slot_mapping, context_lens = [], [], [], []
        for seq in seqs:
            L = len(seq) - G
            idx = L - 1 + i
            input_ids.append(seq[idx])
            positions.append(idx)
            context_lens.append(idx + 1)
            slot_mapping.append(self._slot_of(seq, idx))
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        # 这里不 set_context: draft 走 CUDA graph, context 只在捕获时用到,
        # replay 时 Python 侧完全不执行。返回值交给 _draft_forward_graph 拷进静态 buffer。
        return input_ids, positions, slot_mapping, context_lens, block_tables

    def prepare_spec_verify(self, seqs: list[Sequence], G: int):
        """构造 target 的 verify forward。

        query = [x_{L-1}, d_1, ..., d_G]  共 γ+1 个位置, 位置编号 L-1 .. L+G-1
        cache = 前 L-1 个 token (之前轮次已经写进去的)

        走 prefill 分支, 是为了复用 flash-attn varlen + block_table 的"部分 KV 落在
        paged cache"语义 —— chunked prefill 用的就是同一套约定:
        cache 里放前 seqlen_k-seqlen_q 个 token, query 部分本次算出并写回。

        slot(L-1) 会被重新写一次但内容不变, 换来的是 query 起点与 KV 布局严丝合缝,
        不需要为"少读一个 token"特判注意力掩码。

        spec_verify=True 让 lm_head 保留全部位置的 logits, 而不是只取每序列最后一个。
        """
        input_ids, positions, slot_mapping = [], [], []
        cu_seqlens_q, cu_seqlens_k = [0], [0]
        max_seqlen_q = max_seqlen_k = 0
        for seq in seqs:
            L = len(seq) - G
            start = L - 1
            end = start + G + 1
            input_ids.extend(seq[start:end])
            positions.extend(range(start, end))
            slot_mapping.extend(self._slot_of(seq, t) for t in range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + end - start)
            cu_seqlens_k.append(cu_seqlens_k[-1] + end)
            max_seqlen_q = max(max_seqlen_q, end - start)
            max_seqlen_k = max(max_seqlen_k, end)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k,
                    slot_mapping, None, block_tables, spec_verify=True)
        return input_ids, positions

    @torch.inference_mode()
    def run_spec(self, seqs: list[Sequence], num_spec_tokens: int, greedy: bool):
        """投机解码一轮: γ 次 draft decode + 1 次 target verify + 拒绝采样。

        前置条件: 调用方(调度层)已把 γ 个占位 token 追加到每个 seq 尾部并分配好 block,
        这些占位值会在 draft 阶段被真实输出覆盖。
        后置条件: 返回每 seq 本轮确认的 token 列表; **本函数不回收任何 KV block**,
        被拒绝的部分由调度层按返回值调用 BlockManager.trim 回滚。
        """
        G, B = num_spec_tokens, len(seqs)
        self.spec_round += 1
        temperatures = self.prepare_sample(seqs)

        draft_logits_per_step, draft_tokens_per_step = [], []
        for i in range(G + 1):
            input_ids, positions, sm, cl, bt = self.prepare_draft_decode(seqs, i, G)
            logits = self._draft_forward_graph(input_ids, positions, sm, cl, bt)   # (B, Vd)
            if i == G:
                break          # 第 G 步只写 KV, 输出丢弃 (见 prepare_draft_decode 注释)
            tok = self.speculator.draft_step(logits, temperatures, greedy,
                                             self._make_generator(i))
            # 一次 tolist 而不是逐个 int(): 每个 token 一次 GPU->CPU 同步太贵
            toks = tok.tolist()
            # 覆盖占位值: 第 i 个草稿 token 位于 index L+i
            for b, seq in enumerate(seqs):
                seq.token_ids[len(seq) - G + i] = toks[b]
            draft_logits_per_step.append(logits)
            draft_tokens_per_step.append(tok)

        input_ids, positions = self.prepare_spec_verify(seqs, G)
        target_logits = self.model.compute_logits(self.model(input_ids, positions))   # (B*(G+1), Vt)
        reset_context()

        result = None
        if self.rank == 0:
            draft_logits = torch.stack(draft_logits_per_step, dim=1)                  # (B, G, Vd)
            draft_tokens = torch.stack(draft_tokens_per_step, dim=1)                  # (B, G)
            target_logits = target_logits.view(B, G + 1, -1)                          # (B, G+1, Vt)
            result = self.speculator.verify(
                target_logits, draft_logits, draft_tokens, temperatures, greedy,
                self._make_generator(G))
        return result

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
