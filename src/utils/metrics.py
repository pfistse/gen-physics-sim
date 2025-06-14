import torch

def mean_absolute_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(pred - target), dim=(1, 2, 3))

def root_mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean((pred - target) ** 2, dim=(1, 2, 3)))

def mean_squared_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Return mean squared error per sample."""
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))

def pearson_correlation(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_flat = pred.view(pred.size(0), -1)
    target_flat = target.view(target.size(0), -1)
    pred_mean = pred_flat.mean(dim=1, keepdim=True)
    target_mean = target_flat.mean(dim=1, keepdim=True)
    cov = ((pred_flat - pred_mean) * (target_flat - target_mean)).mean(dim=1)
    pred_std = pred_flat.std(dim=1)
    target_std = target_flat.std(dim=1)
    return cov / (pred_std * target_std + 1e-8)

def rate_of_change(tensor: torch.Tensor) -> torch.Tensor:
    return torch.abs(tensor[1:] - tensor[:-1]).mean(dim=(1, 2, 3))

def rate_of_change_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.abs(rate_of_change(pred) - rate_of_change(target))
