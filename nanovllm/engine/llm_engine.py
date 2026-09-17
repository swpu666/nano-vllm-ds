from __future__ import annotations
import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        Sequence.block_size = config.kvcache_block_size
        self.config = config
        self.spec_enabled = config.draft_model is not None and config.num_speculative_tokens > 0
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if not hasattr(self, "model_runner"):   # 幂等, 避免 atexit 重复调用报错
            return
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)

    def step(self):
        seqs, is_prefill = self.scheduler.schedule()
        if not (self.spec_enabled and not is_prefill):
            num_tokens = sum(seq.num_scheduled_tokens for seq in seqs) if is_prefill else -len(seqs)
            token_ids = self.model_runner.call("run", seqs, is_prefill)
            self.scheduler.postprocess(seqs, token_ids, is_prefill)
            outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
            return outputs, num_tokens

        # 投机解码: 一轮产出 1~γ+1 个 token / seq
        # num_tokens 沿用约定 —— decode 返回负值, generate() 据此算 decode 吞吐
        num_tokens = -self._step_spec(seqs)
        outputs = [(seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished]
        return outputs, num_tokens

    def _step_spec(self, seqs: list[Sequence]) -> int:
        """跑一轮投机解码, 返回本轮**实际确认**的 token 总数。

        职责划分: ModelRunner 只算, 显存回收/block 管理全部在这里做 ——
        draft token 是"先占位后确认"的, 一旦调整紫部分必须把占位连同跨出去的
        block 一起还回去, 否则 KV cache 会被逐步吃光。
        """
        # 动态 γ: current_gamma 由 ModelRunner 每轮按"接受率 + draft/target 耗时比"更新。
        # 注意只取 rank0(主进程) 的那一份 —— worker 不做采样统计, 它的估计值不可信,
        # 而 γ 必须对所有 rank 一致(实际是从主进程作为参数传下去的)。
        G = self.model_runner.current_gamma if self.config.dynamic_gamma \
            else self.config.num_speculative_tokens
        block_manager = self.scheduler.block_manager

        # 同一 batch 内 greedy 必须一致 (Speculator 按 batch 处理),
        # 不一致就降级成普通 decode, 而不是让部分序列走错采样方式
        greedy = seqs[0].greedy
        if not all(seq.greedy == greedy for seq in seqs):
            return self._step_spec_fallback(seqs, [])

        # 多候选: 一次 verify 同时验证 K 条候选链, 占位也要 γ*K 个
        K = self.config.num_spec_candidates
        num_slots = G * K

        # 1) 占位 draft token 并分配 block
        # ⚠ 占位 token 会立刻让 num_completion_tokens 虚增 γ*K, 而 max_tokens 余量必须按
        #   **本轮开始时**的完成数来算。否则余量会被少算, 一旦吃满就会出现
        #   "每轮确认 0 个 token" -> 序列永远不 finish -> generate() 死循环。
        base_completion = [seq.num_completion_tokens for seq in seqs]
        appended = []
        ok = True
        for seq in seqs:
            # spec 期间保持 is_prefill=True, 让 Sequence.__getstate__ 把完整
            # token_ids 传给 TP worker (worker 需要尾部位置构造 verify 输入)
            seq.is_prefill = True
            seq.append_spec_tokens([0] * num_slots)
            ok &= block_manager.may_append_n(seq, num_slots)
            appended.append(seq)
        if not ok:
            return self._step_spec_fallback(seqs, appended)

        # 2) draft γ 步 + target verify + 拒绝采样
        result = self.model_runner.call("run_spec", seqs, G, greedy, K)

        # 3) 按接受结果回滚 / 落盘
        total = 0
        for i, seq in enumerate(seqs):
            accepted = None
            if result is not None:
                # 受 max_tokens 限制: 本轮可能只能收下一部分
                remaining = seq.max_tokens - base_completion[i]
                accepted = result.accepted_tokens[i][:max(0, remaining)]
                # 截到 eos 为止
                if not seq.ignore_eos and self.scheduler.eos in accepted:
                    cut = accepted.index(self.scheduler.eos) + 1
                    accepted = accepted[:cut]

            if accepted is None:          # worker 不产出采样结果 -> 全部回滚
                block_manager.trim(seq, num_slots)
                seq.num_scheduled_tokens = 0
                continue

            k = result.n_accepted[i]
            m = len(accepted)
            keep = min(m, k)
            if K > 1:
                # 多候选: 占位布局是 [链0 的 γ 个][链1 的 γ 个]..., 被选中的链
                # 未必是链 0 —— 先把它的 token 搬到序列头部, 再统一 trim
                L = len(seq) - num_slots
                c = result.chain_id[i] if result.chain_id else 0
                for j in range(keep):
                    seq.token_ids[L + j] = seq.token_ids[L + c * G + j]
            # 尾部现在有 num_slots 个占位(被选中链的前 k 个已是真 draft token),
            # 只保留前 keep 个; 若 m==k+1 还要补 extra token
            block_manager.trim(seq, num_slots - keep)
            if m > k:
                assert block_manager.can_append(seq), "投机解码追加 token 时 KV 不足"
                block_manager.may_append(seq)
                seq.append_token(accepted[-1])
            seq.num_scheduled_tokens = 0
            total += m

            # m == 0 只可能是 max_tokens 已经用尽 (这是唯一能让 accepted 变空的路径),
            # 必须在这里判 finish, 否则序列会一直留在 running 里 -> generate() 空转
            if m == 0 or (not seq.ignore_eos and seq.last_token == self.scheduler.eos) \
                    or seq.num_completion_tokens >= seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                block_manager.deallocate(seq)
                if seq in self.scheduler.running:
                    self.scheduler.running.remove(seq)

        # 投机期间的 token 不 commit 进 prefix cache: 被拒绝的候选一旦被 hash,
        # 后续相同前缀的请求就会命中到错误的 block。代价是这些 seq 若被抢占,
        # 会从更早的位置重新 prefill —— 慢, 但不会错。
        # (因此这里不调用 scheduler.postprocess / hash_blocks)
        return total

    def _step_spec_fallback(self, seqs: list[Sequence], appended: list[Sequence]) -> int:
        """降级路径: 回滚所有占位, 退回普通单步 decode。"""
        for seq in appended:
            self.scheduler.block_manager.trim(
                seq, self.config.num_speculative_tokens * self.config.num_spec_candidates)
        token_ids = self.model_runner.call("run", seqs, False)
        self.scheduler.postprocess(seqs, token_ids, False)
        return len(seqs)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[str]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            output, num_tokens = self.step()
            if num_tokens > 0:
                prefill_throughput = num_tokens / (perf_counter() - t)
            else:
                decode_throughput = -num_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids} for token_ids in outputs]
        return outputs
