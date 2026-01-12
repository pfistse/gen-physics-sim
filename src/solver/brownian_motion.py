import numpy as np
from typing import List, Optional, Sequence, Tuple, Union

def _gaussian2d_periodic(h: int, w: int, cx: float, cy: float, sigma: float) -> np.ndarray:
    y = np.arange(h, dtype=np.float32)[:, None]
    x = np.arange(w, dtype=np.float32)[None, :]
    dx = ((x - cx + w / 2) % w) - w / 2
    dy = ((y - cy + h / 2) % h) - h / 2
    r2 = dx * dx + dy * dy
    return np.exp(-r2 / (2.0 * sigma * sigma)).astype(np.float32)

def _run_sequence(
    *,
    frames: int,
    sigma: float,
    radius: float,
    size: int,
    num_particles: int,
    rng: np.random.Generator,
    cx0: np.ndarray,
    cy0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Core integrator: render ``frames`` steps from initial centres using ``rng``.

    Returns (seq, cx_final, cy_final).
    """
    seq = np.empty((frames, 1, size, size), dtype=np.float32)
    cx = cx0.astype(np.float32).copy()
    cy = cy0.astype(np.float32).copy()
    for t in range(frames):
        field = np.zeros((size, size), dtype=np.float32)
        for b in range(num_particles):
            field += _gaussian2d_periodic(size, size, float(cx[b]), float(cy[b]), sigma)
        seq[t, 0] = field

        r = radius * np.sqrt(rng.random(num_particles, dtype=np.float32))
        th = (2.0 * np.pi) * rng.random(num_particles, dtype=np.float32)
        cx = (cx + r * np.cos(th)) % size
        cy = (cy + r * np.sin(th)) % size
    return seq, cx, cy


def sequence(
    frames: int,
    sigma: float,
    radius: float,
    *,
    size: int = 64,
    seed: int | None = None,
    num_particles: int = 1,
) -> np.ndarray:
    """Generate a Brownian-motion sequence of sum-of-Gaussians.

    Returns an array of shape (L, 1, size, size). If ``num_particles > 1``,
    the frame is a sum of ``num_particles`` identical Gaussians with independent
    Brownian motion (periodic boundary conditions).
    """
    rng = np.random.default_rng(seed)
    cx0 = rng.uniform(0.0, size, size=num_particles).astype(np.float32)
    cy0 = rng.uniform(0.0, size, size=num_particles).astype(np.float32)
    seq, _, _ = _run_sequence(
        frames=frames,
        sigma=sigma,
        radius=radius,
        size=size,
        num_particles=num_particles,
        rng=rng,
        cx0=cx0,
        cy0=cy0,
    )
    return seq

def dataset(
    n: int,
    frames: int,
    sigma: float,
    radius: float,
    *,
    size: int = 64,
    seed: int | None = None,
    num_particles: int = 1,
) -> np.ndarray:
    """Generate a dataset of Brownian-motion sequences.

    Shape: (N, L, 1, size, size). ``num_particles`` controls how many
    independent Gaussians are summed per frame.
    """
    rng = np.random.default_rng(seed)
    data = np.empty((n, frames, 1, size, size), dtype=np.float32)
    for i in range(n):
        data[i] = sequence(
            frames,
            sigma,
            radius,
            size=size,
            seed=int(rng.integers(2**32 - 1)),
            num_particles=num_particles,
        )
    return data


def simulate(
    *,
    warmup_seed: int = 1,
    trajectory_seeds: Optional[Sequence[int]] = None,
    frames: int = 100,
    T: Optional[int] = None,
    size: int = 64,
    sigma: float = 1.0,
    radius: float = 5.0,
    num_particles: int = 1,
    warmup: int = 0,
    start_after_warmup: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Simulate Brownian motion with optional warm‑up and multiple rollouts.

    Mirrors the API shape of ``src/solver/inc_stoc_vort.simulate`` to make
    re‑use from data modules straightforward.

    - If ``trajectory_seeds`` is ``None``, returns a single sequence of shape
      (T, 1, size, size) generated after a warm‑up. When
      ``start_after_warmup=True``, the first returned frame corresponds to one
      Brownian step after the end‑of‑warm‑up state (useful when distinct
      per‑seed first frames are required).
    - Otherwise returns a tuple ``(warmup_seq, trajectories)`` where
      ``warmup_seq`` has shape (1, warmup, 1, size, size) and ``trajectories``
      is a stacked array of shape (B, T, 1, size, size) with
      ``B = len(trajectory_seeds)``, aligning with the inc_stoc generator.
    """

    # Warm‑up using a deterministic RNG
    rng_warm = np.random.default_rng(warmup_seed)
    cx0 = rng_warm.uniform(0.0, size, size=num_particles).astype(np.float32)
    cy0 = rng_warm.uniform(0.0, size, size=num_particles).astype(np.float32)
    warm_seq, cx, cy = _run_sequence(
        frames=warmup,
        sigma=sigma,
        radius=radius,
        size=size,
        num_particles=num_particles,
        rng=rng_warm,
        cx0=cx0,
        cy0=cy0,
    )

    # Choose frame count, keeping backward compatibility for callers using T=
    total_frames = int(T) if T is not None else int(frames)

    # Single‑trajectory mode
    if trajectory_seeds is None:
        rng = np.random.default_rng(warmup_seed + 1)
        frames_needed = total_frames + (1 if start_after_warmup else 0)
        seq, _, _ = _run_sequence(
            frames=frames_needed,
            sigma=sigma,
            radius=radius,
            size=size,
            num_particles=num_particles,
            rng=rng,
            cx0=cx,
            cy0=cy,
        )
        return seq[1:] if start_after_warmup else seq

    # Multi‑trajectory mode continuing from the same end‑of‑warmup state
    trajectories_list: List[np.ndarray] = []
    for i, seed in enumerate(trajectory_seeds):
        rng_i = np.random.default_rng(seed)
        frames_needed = total_frames + (1 if start_after_warmup else 0)
        seq, _, _ = _run_sequence(
            frames=frames_needed,
            sigma=sigma,
            radius=radius,
            size=size,
            num_particles=num_particles,
            rng=rng_i,
            cx0=cx,
            cy0=cy,
        )
        trajectories_list.append(seq[1:] if start_after_warmup else seq)

    # Align shapes with inc_stoc: add batch dim to warm‑up and stack rollouts
    trajectories = np.stack(trajectories_list, axis=0)  # (B, T, 1, size, size)
    warm_batched = warm_seq[None, ...]  # (1, warmup, 1, size, size)

    return warm_batched, trajectories
