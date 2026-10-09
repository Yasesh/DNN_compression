import torch
import torch.nn as nn

class SVDLinear(nn.Module):
    def __init__(self, original_layer: nn.Linear, rank: int):
        super(SVDLinear, self).__init__()
        device = original_layer.weight.device
        weight = original_layer.weight.data.float().cpu()
        out_features, in_features = weight.shape

        max_rank = min(out_features, in_features)
        rank = min(rank, max_rank)

        orig_params = out_features * in_features
        has_bias = original_layer.bias is not None
        factorized_params = (in_features * rank) + (rank * out_features) + (out_features if has_bias else 0)

        # Safeguard against parameter bloat or full-rank splitting
        if rank >= max_rank or factorized_params >= orig_params:
            self.layer1 = nn.Identity()
            self.layer2 = original_layer
            return

        U, S, Vh = torch.linalg.svd(weight, full_matrices=False)

        U_r = U[:, :rank]
        S_r = S[:rank]
        Vh_r = Vh[:rank, :]

        W1 = torch.diag(S_r) @ Vh_r
        W2 = U_r

        self.layer1 = nn.Linear(in_features, rank, bias=False).to(device)
        self.layer1.weight = nn.Parameter(W1.to(device))

        self.layer2 = nn.Linear(rank, out_features, bias=has_bias).to(device)
        if has_bias:
            self.layer2.bias = nn.Parameter(original_layer.bias.data.clone().to(device))
        self.layer2.weight = nn.Parameter(W2.to(device))

    def forward(self, x):
        return self.layer2(self.layer1(x))

def replace_linear_with_svd(model: nn.Module, rank_ratio: float = 0.5, min_dim: int = 128):
    def recurse_and_replace(module):
        for name, child in list(module.named_children()):
            if isinstance(child, nn.Linear):
                out_features, in_features = child.weight.shape
                if min(out_features, in_features) >= min_dim:
                    target_rank = int(min(out_features, in_features) * rank_ratio)
                    setattr(module, name, SVDLinear(child, rank=target_rank))
            else:
                recurse_and_replace(child)
    recurse_and_replace(model)
    return model

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)