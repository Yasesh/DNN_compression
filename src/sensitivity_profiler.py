import torch
import torch.nn as nn
from src.svd_wrapper import SVDLinear, count_parameters


def profile_layer_sensitivity(
    model: nn.Module, layer_idx: int, rank_ratio: float = 0.5
) -> dict:
    """Isolates a specific Transformer encoder layer in BERT, applies SVD recursively,

    and returns metrics to evaluate parameter sensitivity.
    """
    target_layer = model.encoder.layer[layer_idx]
    orig_layer_params = count_parameters(target_layer)

    # Recursively traverse nested submodules (attention, intermediate, output)
    def recurse_replace(module):
        for name, child in module.named_children():
            if isinstance(child, nn.Linear):
                out_features, in_features = child.weight.shape
                if min(out_features, in_features) >= 128:
                    target_rank = int(min(out_features, in_features) * rank_ratio)
                    setattr(module, name, SVDLinear(child, rank=target_rank))
            else:
                recurse_replace(child)

    recurse_replace(target_layer)
    comp_layer_params = count_parameters(target_layer)
    param_drop_pct = (
        (orig_layer_params - comp_layer_params) / orig_layer_params
    ) * 100

    return {
        "Layer Index": layer_idx,
        "Original Params": orig_layer_params,
        "Compressed Params": comp_layer_params,
        "Reduction (%)": param_drop_pct,
    }