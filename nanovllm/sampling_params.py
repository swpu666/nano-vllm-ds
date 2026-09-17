from __future__ import annotations
from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    # greedy=True 时忽略 temperature 走 argmax。
    # 单独开关(而非 temperature=0)是为了不破坏原有"禁止 temperature<=1e-10"的断言语义,
    # 且投机解码的正确性验证需要严格 greedy 的参考输出。
    greedy: bool = False

    def __post_init__(self):
        assert self.temperature > 1e-10, "greedy sampling is not permitted"
