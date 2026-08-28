from __future__ import annotations
import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


GPTQ_SUFFIXES = (".qweight", ".qzeros", ".scales", ".g_idx")


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                # GPTQ packed 权重: 后缀剥离后再做 packed 映射
                suffix = ""
                for s in GPTQ_SUFFIXES:
                    if weight_name.endswith(s):
                        suffix = s
                        break
                # g_idx 在 group_size 对齐时不需要, 跳过
                if suffix == ".g_idx":
                    continue
                base_name = weight_name[: -len(suffix)] if suffix else weight_name
                for k in packed_modules_mapping:
                    if k in base_name:
                        v, shard_id = packed_modules_mapping[k]
                        # 映射后的名字保留 GPTQ 后缀 (如 qkv_proj.qweight)
                        param_name = (base_name.replace(k, v)) + suffix
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader")
                        weight_loader(param, f.get_tensor(weight_name), shard_id)
                        break
                else:
                    param = model.get_parameter(weight_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))
