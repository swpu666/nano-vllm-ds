from __future__ import annotations
import torch
from torch import nn


class Sampler(nn.Module):

    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor, greedy: bool = False):
        """greedy 时直接取 argmax。

        单独开关而不是靠 temperature->0, 是为了让投机解码的 verify 有一个
        **确定性强对照**: greedy 下它的输出必须与完全不投机解码的基线逐 token 相同。
        """
        if greedy:
            return logits.argmax(dim=-1)
        return self._forward(logits, temperatures)

    @torch.compile
    def _forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        logits = logits.float().div_(temperatures.unsqueeze(dim=1))
        probs = torch.softmax(logits, dim=-1)
        sample_tokens = probs.div_(torch.empty_like(probs).exponential_(1).clamp_min_(1e-10)).argmax(dim=-1)
        return sample_tokens
