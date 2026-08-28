from nanovllm.engine.llm_engine import LLMEngine
from nanovllm.sampling_params import SamplingParams

MODEL = "/nas_data/WR/models/DeepSeek-R1-Distill-Qwen-32B-GPTQ-Int4"

engine = LLMEngine(
    MODEL,
    tensor_parallel_size=1,
    max_num_batched_tokens=2048,
    max_num_seqs=8,
    max_model_len=2048,
)
sp = SamplingParams(temperature=0.8, max_tokens=48)

outputs = engine.generate(
    ["The capital of France is", "2 + 2 ="],
    sp,
    use_tqdm=False,
)
for o in outputs:
    print("GEN:", repr(o["text"]))
print("DONE")
engine.exit()
