import numpy as np
from typing import Any, Dict, List
import pytorch_lightning as pl
import torch
from numpy.linalg import norm as L2
import logging

logger = logging.getLogger(__name__)
EPS: float = float(torch.finfo(torch.float32).eps)


def _l2(value: Any) -> float:
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    return float(L2(value))


def calculate_mse(
    model: pl.LightningModule,
    cfg: Any,
    channel_titles: List[str] = None,
    num_steps: int = 1,
    progress: Any = None,
) -> Dict[str, Any]:
    """Calculate deviation from ground truth physics data for generated sequences."""
    model.eval()
    device = model.device

    datamodule = model.trainer.datamodule
    data = getattr(datamodule, "sim_dataset", datamodule.test_set)

    S = datamodule.ctx_len
    C = datamodule.num_channels
    F = C - len(datamodule.sim_params)

    seq_iter = data.load_sequences(
        length=S + cfg.seq_len,
        num_seq=cfg.num_seqs,
    )

    conds, targets = [], []
    for seq_solver in seq_iter:
        cond = seq_solver[:S]
        target = seq_solver[S:]

        conds.append(cond)
        targets.append(target)

    mse_time = []

    task_id = None
    if progress is not None:
        task_id = progress.add_task("calculate_mse", total=len(conds))

    with torch.no_grad():
        for cond, target in list(zip(conds, targets)):
            target = target.to(device)
            cond = cond.unsqueeze(0).to(device)  # [1, S, C, H, W]

            seq_model = model.generate_sequence(
                cond,
                seq_len=cfg.seq_len,
                num_steps=num_steps,
            ).squeeze(
                0
            )  # [L, C, H, W]

            seq_model_fields = seq_model[:, :F]
            target_fields = target[:, :F]

            mse_time_seq = (
                (seq_model_fields - target_fields).square().mean(dim=(1, 2, 3))
            )

            mse_time.append(mse_time_seq.cpu().tolist())
            if task_id is not None:
                progress.advance(task_id, 1)

    return {
        "mse_time": mse_time,
        "mse_mean": float(np.mean(mse_time)),
    }


def calculate_mean_std(
    model: pl.LightningModule,
    cfg: Any,
    entry: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    model.eval()
    device = model.device

    with torch.no_grad():
        seed = int(entry.get("seed", 0))
        warmup = entry["warmup"].to(device)
        seqs_solver = entry["traj"].to(device)

        seqs_solver = seqs_solver[:, -1:].squeeze(1)

        assert (
            seqs_solver.shape[0] == cfg.num_samples
        ), f"Expected {cfg.num_samples} samples, got {seqs_solver.shape[0]}"

        mean_solver = seqs_solver.mean(dim=0)
        std_solver = seqs_solver.std(dim=0, unbiased=False)

        warmup_ctx = warmup[:, -model.ctx_len :]
        if warmup_ctx.shape[0] == 1:
            cond = warmup_ctx.expand(cfg.num_samples, -1, -1, -1, -1)
        else:
            assert (
                warmup_ctx.shape[0] == cfg.num_samples
            ), f"cond batch mismatch, expected {cfg.num_samples}, got {warmup_ctx.shape[0]}"
            cond = warmup_ctx

        seqs_model = model.generate_sequence(
            cond,
            seq_len=cfg.delta,
            num_steps=model.num_steps_eval,
        )
        samples_model = seqs_model[:, -1:].squeeze(1)

        mean_model = samples_model.mean(dim=0)
        std_model = samples_model.std(dim=0, unbiased=False)

        mean_rel_err = _l2(mean_solver - mean_model) / (_l2(mean_solver) + EPS)
        std_rel_err = _l2(std_solver - std_model) / (_l2(std_solver) + EPS)

        mean_rel_overshoot = (mean_model - mean_solver).mean() / (
            mean_solver.abs().mean() + EPS
        )
        std_rel_overshoot = (std_model - std_solver).mean() / (
            std_solver.abs().mean() + EPS
        )

    return {
        "seed": seed,
        "_std_solver_arr": std_solver.cpu().numpy(),
        "_std_model_arr": std_model.cpu().numpy(),
        "_mean_solver_arr": mean_solver.cpu().numpy(),
        "_mean_model_arr": mean_model.cpu().numpy(),
        "_std_solver": float(std_solver.mean().item()),
        "_std_model": float(std_model.mean().item()),
        "std_rel_err": float(std_rel_err),
        "std_rel_overshoot": float(std_rel_overshoot.item()),
        "_mean_solver": float(mean_solver.mean().item()),
        "_mean_model": float(mean_model.mean().item()),
        "mean_rel_err": float(mean_rel_err),
        "mean_rel_overshoot": float(mean_rel_overshoot.item()),
    }


def calculate_enstrophy(
    model: pl.LightningModule,
    cfg: Any,
    progress: Any = None,
) -> Dict[str, Any]:
    """Enstrophy spectrum and total enstrophy."""
    model.eval()
    device = model.device

    datamodule = model.trainer.datamodule
    data = getattr(datamodule, "sim_dataset", datamodule.test_set)

    S = datamodule.ctx_len
    C = datamodule.num_channels
    P = len(datamodule.sim_params)
    F = C - P
    assert F == 1, f"enstrophy requires single field (F=1), got F={F}"

    seqs_solver = data.load_sequences(
        length=S + cfg.seq_len,
        num_seq=cfg.num_seqs,
    )

    spectra_model: List[np.ndarray] = []
    spectra_solver: List[np.ndarray] = []
    total_model: List[float] = []
    total_solver: List[float] = []

    task_id = None
    if progress is not None:
        total = len(seqs_solver)
        task_id = progress.add_task("calculate_enstrophy", total=total)

    with torch.no_grad():
        for cond_seq_solver in seqs_solver:
            cond = cond_seq_solver[:S].to(device)
            seq_solver = cond_seq_solver[S:].to(device)

            seq_model = model.generate_sequence(
                cond=cond.unsqueeze(0),
                seq_len=cfg.seq_len,
                num_steps=model.num_steps_eval,
            ).squeeze(0)

            seq_model = seq_model[:, :F]
            seq_solver = seq_solver[:, :F]

            for t in range(cfg.seq_len):
                frame_model = seq_model[t, 0]
                frame_solver = seq_solver[t, 0]

                total_model.append(_total_enstrophy(frame_model))
                total_solver.append(_total_enstrophy(frame_solver))

                spectrum_model = _radial_enstrophy_spectrum(frame_model)
                spectrum_solver = _radial_enstrophy_spectrum(frame_solver)

                spectra_model.append(
                    spectrum_model.cpu()
                )  # TODO implement _radial_enstrophy_spectrum with np
                spectra_solver.append(spectrum_solver.cpu())

            if task_id is not None:
                progress.advance(task_id, 1)

    spectra_model = np.mean(spectra_model, axis=0)
    spectra_solver = np.mean(spectra_solver, axis=0)
    total_model = np.stack(total_model).mean(axis=0)
    total_solver = np.stack(total_solver).mean(axis=0)

    avg_error = float(np.mean(np.abs(spectra_model - spectra_solver)))

    return {
        "_spec_model": spectra_model,
        "_spec_solver": spectra_solver,
        "total_enstr_model": total_model,
        "total_sentr_solver": total_solver,
        "avg_enstr_err": avg_error,
    }


def _total_enstrophy(frame: torch.Tensor) -> float:
    return float((frame.square().sum(dim=(0, 1))).item())


def _radial_enstrophy_spectrum(field: torch.Tensor) -> torch.Tensor:
    """Radial enstrophy spectrum for 2D vorticity.

    field: [H, W]
    result: [K] where K = floor(sqrt((H//2)**2 + (W//2)**2)) + 1
    """
    assert field.ndim == 2, f"field: [H,W], got {tuple(field.shape)}"
    H, W = field.shape

    F = torch.fft.fft2(field)
    power = (F.real**2 + F.imag**2) / (H * W)
    power = torch.fft.fftshift(power)

    yy = torch.arange(H, device=field.device) - (H // 2)
    xx = torch.arange(W, device=field.device) - (W // 2)
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    r = torch.sqrt(gy.to(torch.float32) ** 2 + gx.to(torch.float32) ** 2)
    bins = torch.floor(r).to(torch.int64)
    K = int(bins.max().item()) + 1

    spec = torch.bincount(bins.view(-1), weights=power.view(-1), minlength=K)
    return spec


def calculate_var_samples(
    model: pl.LightningModule, cfg: Any, progress: Any = None
) -> np.ndarray:
    """Calculate variance samples for a given start frame."""
    data = model.trainer.datamodule.train_set
    assert hasattr(
        data, "generate_sequence"
    ), "`calculate_var_samples` requires dataset to expose `generate_sequence`"

    model.eval()
    device = model.device

    task_id = None
    if progress is not None:
        task_id = progress.add_task("calculate_var_samples", total=1)

    with torch.no_grad():
        cond = data.generate_sequence(start_frame=cfg.warmup_len, len=model.ctx_len)
        cond = cond.to(device)

        cond = cond.unsqueeze(0).repeat(cfg.num_samples, 1, 1, 1, 1)

        seqs_model = model.generate_sequence(
            cond,
            seq_len=cfg.delta,
            num_steps=model.num_steps_eval,
        )

    samples_model = seqs_model[:, -1:].squeeze(1).cpu().numpy()

    if task_id is not None:
        progress.advance(task_id, 1)

    return samples_model


# DEBUG METRICS


def calculate_std_steps(
    model: pl.LightningModule, cfg: Any, progress: Any = None
) -> List[np.ndarray]:
    data = model.trainer.datamodule.train_set
    warmup, _ = data.generate_sequences(
        num_seq=cfg.num_samples,
        start_frame=cfg.warmup_len,
        len=cfg.delta,
        seed=cfg.seed,
        same_warmup=True,
    )

    warmup = warmup.to(model.device)
    cond = warmup[:, -model.ctx_len :].repeat(cfg.num_samples, 1, 1, 1, 1)

    std_steps = []

    task_id = None
    if progress is not None:
        task_id = progress.add_task("calculate_std_steps", total=len(cfg.steps))

    with torch.no_grad():
        for steps in cfg.steps:
            seqs = model.generate_sequence(
                cond=cond, seq_len=cfg.delta, num_steps=steps
            )
            std = seqs[:, -1].std(dim=0).cpu().numpy()
            std_steps.append(std)
            if task_id is not None:
                progress.advance(task_id, 1)

    return std_steps


def calculate_mean_std_rel_err_steps(
    model: pl.LightningModule,
    cfg: Any,
    entry: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    device = model.device

    warmup = entry["warmup"].to(device)
    seqs_solver = entry["traj"].to(device)
    std_solver = seqs_solver[:, cfg.delta - 1].std(dim=0)
    mean_solver = seqs_solver[:, cfg.delta - 1].mean(dim=0)

    cond = warmup[:, -model.ctx_len :].expand(cfg.num_samples, -1, -1, -1, -1)

    std_rel_err_steps = []
    mean_rel_err_steps = []
    with torch.no_grad():
        for num_steps in cfg.steps:
            seqs_model = model.generate_sequence(
                cond=cond, seq_len=cfg.delta, num_steps=num_steps
            )
            std_model = seqs_model[:, -1].std(dim=0)
            mean_model = seqs_model[:, -1].mean(dim=0)

            std_rel_err = _l2(std_solver - std_model) / (_l2(std_solver) + EPS)
            std_rel_err_steps.append(std_rel_err)

            mean_rel_err = _l2(mean_solver - mean_model) / (_l2(mean_solver) + EPS)
            mean_rel_err_steps.append(mean_rel_err)

    return {
        "seed": entry["seed"],
        "std_rel_err_steps": std_rel_err_steps,
        "mean_rel_err_steps": mean_rel_err_steps,
    }


def calculate_std_rel_err_grid(
    model: pl.LightningModule,
    cfg: Any,
    entry: Dict[str, torch.Tensor],
) -> Dict[str, Any]:
    """Rel std error for multiple steps and Foellmer coefficients."""
    device = model.device

    warmup = entry["warmup"].to(device)
    seqs_solver = entry["traj"].to(device)

    assert cfg.delta >= 1, "delta >= 1"
    assert cfg.steps, "steps must be set"
    assert cfg.g_coefs, "g_coefs must be set"

    std_solver = seqs_solver[:, cfg.delta - 1].std(dim=0)

    warmup_ctx = warmup[:, -model.ctx_len :]
    if warmup_ctx.shape[0] == 1:
        cond = warmup_ctx.expand(cfg.num_samples, -1, -1, -1, -1)
    else:
        assert (
            warmup_ctx.shape[0] == cfg.num_samples
        ), f"cond batch mismatch, expected {cfg.num_samples}, got {warmup_ctx.shape[0]}"
        cond = warmup_ctx

    rel_err_grid: List[List[float]] = []
    with torch.no_grad():
        for g_coef in cfg.g_coefs:
            rel_err_steps: List[float] = []
            for num_steps in cfg.steps:
                seqs_model = model.generate_sequence(
                    cond=cond,
                    seq_len=cfg.delta,
                    num_steps=num_steps,
                    g_coef=float(g_coef),
                )
                std_model = seqs_model[:, -1].std(dim=0)
                rel_err = torch.linalg.norm(std_solver - std_model) / (
                    torch.linalg.norm(std_solver) + EPS
                )
                rel_err_steps.append(float(rel_err.item()))
            rel_err_grid.append(rel_err_steps)

    return {
        "seed": int(entry["seed"]),
        "rel_err_grid": rel_err_grid,
    }
