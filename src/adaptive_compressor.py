import torch
import torch.nn as nn
from src.svd_wrapper import SVDLinear, count_parameters

def calculate_energy_rank(weight_matrix: torch.Tensor, energy_threshold: float) -> int:
    """
    Calculates the minimum rank needed to retain a specific percentage of matrix energy.
    This mimics the core dynamic rank allocation of GitHub AdaSVD frameworks.
    """
    # Perform SVD to extract singular values (S)
    _, S, _ = torch.linalg.svd(weight_matrix.float().cpu(), full_matrices=False)
    
    # Square the singular values to get the energy/variance
    squared_s = S ** 2
    total_energy = torch.sum(squared_s)
    cumulative_energy = torch.cumsum(squared_s, dim=0)
    
    # Find the rank where cumulative energy meets our threshold
    target_energy = energy_threshold * total_energy
    
    # torch.searchsorted finds the index where the threshold is crossed
    rank = torch.searchsorted(cumulative_energy, target_energy).item() + 1
    
    return int(rank)

def apply_adaptive_svd(model: nn.Module, energy_threshold: float = 0.90, min_dim: int = 128):
    """Recursively applies adaptive rank SVD utilizing the existing SVDLinear wrapper."""
    ranks_chosen = []

    def recurse_and_replace(module, name_prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            
            if isinstance(child, nn.Linear):
                out_f, in_f = child.weight.shape
                if min(out_f, in_f) >= min_dim:
                    # 1. Calculate dynamic rank based on the layer's unique energy distribution
                    target_rank = calculate_energy_rank(child.weight.data, energy_threshold)
                    ranks_chosen.append((full_name, target_rank, min(out_f, in_f)))
                    
                    # 2. Wrap using YOUR existing SVD framework
                    setattr(module, name, SVDLinear(child, rank=target_rank))
            else:
                recurse_and_replace(child, full_name)
                
    recurse_and_replace(model)
    return model, ranks_chosen