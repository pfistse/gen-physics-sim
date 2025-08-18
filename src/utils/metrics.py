import torch
import numpy as np
from typing import Dict, Any, List
import pytorch_lightning as pl
from utils.log import get_logger
from utils.wandb import log_samples_video

logger = get_logger("utils.metrics")


def calculate_mse(
    model: pl.LightningModule,
    num_seqs: int = 10,
    seq_len: int = 20,
    warmup_len: int = 500,
    channel_titles: List[str] = None,
    num_steps: int = 1,
    **kwargs
) -> Dict[str, Any]:
    """Calculate deviation from ground truth physics data for generated sequences."""
    model.eval()
    device = model.device
    
    datamodule = model.trainer.datamodule
    data = datamodule.test_set
    
    S = datamodule.ctx_len
    C = datamodule.num_channels
    F = C - len(datamodule.sim_params)
    
    seqs_solver = data.load_sequences(
        start_frame=warmup_len,
        length=S+seq_len,
        num_seq=num_seqs
    )

    conds, targets = [], []
    for seq_solver in seqs_solver:
        cond = seq_solver[:S]
        target = seq_solver[S:]

        conds.append(cond)
        targets.append(target)

    mse_time = []
    results = []
    
    with torch.no_grad():
        for cond, target in list(zip(conds, targets)):
            target = target.to(device)
            cond = cond.unsqueeze(0).to(device)  # [1, S, C, H, W]

            # Model sequence generation
            seq_model = model.generate_sequence(
                cond,
                seq_len=seq_len,
                num_steps=num_steps,
            ).squeeze(0)  # [L, C, H, W]

            # Calculate metric
            seq_model_fields = seq_model[:, :F]
            target_fields = target[:, :F]

            mse_time_seq = mse_metric(seq_model_fields, target_fields).cpu().tolist()
            mse_time.append(mse_time_seq)

    return {
        "mse_time": mse_time,
        "mse_mean": float(np.mean(mse_time)),
    }


def calculate_vrs(
    model: pl.LightningModule,
    num_samples: int,
    warmup_len: int,
    delta: int,
    seed: int,
    **kwargs
):

    data = model.trainer.datamodule.train_set
    assert hasattr(
        data, 'load_multiple_sequences'), f"VRS metric requires dataset to have function `load_multiple_sequences`"

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():

        # Solver sequence generation
        warmup, seqs_solver = data.load_multiple_sequences(
            num_seq=num_samples, start_frame=warmup_len, len=delta, seed=seed)

        seqs_solver = seqs_solver.to(model.device)
        warmup = warmup.to(model.device)

        samples_solver = seqs_solver[:, -1:].squeeze(1)

        # Model sequence generation
        cond = warmup[-model.ctx_len:].unsqueeze(0).repeat(num_samples, 1, 1, 1, 1)

        seqs_model = model.generate_sequence(
            cond,
            seq_len=delta,
            num_steps=model.num_steps_eval,
        )

        samples_model = seqs_model[:, -1:].squeeze(1)

        # Calculate metric
        mean_solver = samples_solver.mean(dim=0)
        mean_model = samples_model.mean(dim=0)

        var_solver = samples_solver.var(dim=0, unbiased=False)
        var_model = samples_model.var(dim=0, unbiased=False)

        vrs = vrs_metric(var_model, var_solver)
        mean_diff = (mean_solver - mean_model).abs().mean().item()

    return {
        "vrs": vrs,
        "var_solver": var_solver.mean().item(),
        "var_model": var_model.mean().item(),
        "mean_diff": mean_diff,
    }

# METRICS

def mse_metric(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Computes the mean squared error per sample."""
    return torch.mean((pred - target) ** 2, dim=(1, 2, 3))


def vrs_metric(var_model: torch.Tensor, var_solver: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes a 0–1 similarity score for how closely `var_model` variance matches `var_solver` variance."""
    return torch.exp(-(torch.log((var_model + eps) / (var_solver + eps))).abs().mean()).item()
