from __future__ import annotations

from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torchfsm._type import FourierTensor, SpatialTensor
from torchfsm.mesh import FourierMesh, MeshGrid
from torchfsm.operator import (
    ImplicitSource,
    Laplacian,
    NonlinearFunc,
    NonlinearOperator,
    VorticityConvection,
)
from torchfsm.traj_recorder import CPURecorder, IntervalController

import os
os.environ["TQDM_DISABLE"] = "1"

__all__ = ["simulate"]


class _RandomSourceCore(NonlinearFunc):
    """Eight‑mode stochastic forcing in physical space."""

    def __init__(self, mesh: MeshGrid) -> None:
        super().__init__()
        x, y = mesh.mesh_grid()
        self.coefs = [
            torch.sin(6 * x),
            torch.cos(7 * x),
            torch.sin(5 * (x + y)),
            torch.cos(8 * (x + y)),
            torch.cos(6 * x),
            torch.sin(7 * x),
            torch.cos(5 * (x + y)),
            torch.sin(8 * (x + y)),
        ]

    def __call__(
        self,
        u_fft: FourierTensor,
        f_mesh: FourierMesh,
        u: Optional[SpatialTensor] = None,
    ) -> FourierTensor:
        if self.coefs[0].device != f_mesh.device:
            self.coefs = [c.to(f_mesh.device) for c in self.coefs]
        shape = (u_fft.shape[0],) + (1,) * (u_fft.ndim - 1)
        src = 0.0
        for c in self.coefs:  # preserve RNG order
            src += torch.randn(shape, device=u_fft.device) * c
        return f_mesh.fft(src)


class RandomSource(NonlinearOperator):
    def __init__(self, mesh: MeshGrid) -> None:
        super().__init__(_RandomSourceCore(mesh))


def _zscore(arr: np.ndarray):
    mean = arr.mean(axis=(0, 2, 3), keepdims=True, dtype=np.float32)
    std = np.maximum(arr.std(axis=(0, 2, 3), keepdims=True, dtype=np.float32), 1e-6)
    return (arr - mean) / std, mean, std


def _integrate(
    eqn, mesh, u0: torch.Tensor, steps: int, dt: float, interval: int, *, include_initial_state: bool = True
) -> np.ndarray:
    rec = CPURecorder(
        IntervalController(interval=interval, start=0),
        include_initial_state=include_initial_state,
    )
    return (
        eqn.integrate(
            u_0=u0,
            mesh=mesh,
            dt=dt,
            step=steps,
            trajectory_recorder=rec,
        )
        .squeeze(0)
        .cpu()
        .numpy()
        .astype(np.float32)
    )


def simulate(
    *,
    warmup_seed: int = 1,
    trajectory_seeds: Optional[Sequence[int]] = None,
    T: int = 50_000,
    N: int = 128,
    L: float = 2 * np.pi,
    dt: float = 1e-3,
    nu: float = 1e-3,
    alpha: float = 0.1,
    epsilon: float = 1.0,
    warmup: int = 500,
    interval: int = 5,
) -> Union[np.ndarray, Tuple[np.ndarray, List[np.ndarray]]]:
    """Simulate Navier–Stokes vorticity and return normalised data."""

    device = "cuda" if torch.cuda.is_available() else "cpu"

    mesh = MeshGrid([(0, L, N), (0, L, N)], device=device)
    eqn = -VorticityConvection() + nu * Laplacian() - alpha * ImplicitSource() - epsilon * RandomSource(mesh)

    # Warm-up
    torch.manual_seed(warmup_seed)
    np.random.seed(warmup_seed)
    warm_traj = _integrate(
        eqn,
        mesh,
        torch.zeros((1, 1, N, N), device=device),
        steps=warmup,
        dt=dt,
        interval=interval,
        include_initial_state=False,
    )
    init_state = torch.from_numpy(warm_traj[-1:]).to(device)

    # Single‑trajectory mode
    if trajectory_seeds is None:
        traj = _integrate(eqn, mesh, init_state, (T - 1) * interval, dt, interval)
        return _zscore(traj)[0]

    # Multi‑trajectory mode
    trajectories: List[np.ndarray] = []
    warm_norm: np.ndarray | None = None
    for idx, seed in enumerate(trajectory_seeds):
        torch.manual_seed(seed)
        np.random.seed(seed)
        traj = _integrate(eqn, mesh, init_state, (T - 1) * interval, dt, interval)
        if idx == 0:
            traj_norm, mean, std = _zscore(traj)
            warm_norm = ((warm_traj - mean) / std).astype(np.float32)
            trajectories.append(traj_norm)
        else:
            trajectories.append(_zscore(traj)[0])

    assert warm_norm is not None
    return warm_norm, trajectories
