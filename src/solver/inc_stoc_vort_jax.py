from __future__ import annotations
from typing import Optional, Tuple

import tqdm.auto

import numpy as np
import jax
import jax.numpy as jnp

Array = jnp.ndarray
PRNGKey = jax.random.PRNGKey


def _wavenumbers(N: int, L: float):
    n = jnp.fft.fftfreq(N, d=1.0 / N).astype(jnp.float32)
    kx = (2 * jnp.pi / L) * n[:, None]
    ky = (2 * jnp.pi / L) * n[None, :]
    k2 = kx**2 + ky**2
    return kx, ky, k2


def _dealias_mask(N: int, method="rect"):
    n = jnp.fft.fftfreq(N, d=1.0 / N).astype(jnp.float32)
    kx = n[:, None]
    ky = n[None, :]
    kc = N / 3.0
    if method == "circ":
        return ((kx * kx + ky * ky) <= kc * kc).astype(jnp.float32)
    elif method == "rect":
        return ((jnp.abs(kx) <= kc) & (jnp.abs(ky) <= kc)).astype(jnp.float32)
    else:
        return jnp.ones((N, N), dtype=jnp.float32)


def _fft(x: Array) -> Array:
    return jnp.fft.fft2(x, axes=(-2, -1))


def _ifft(x: Array) -> Array:
    return jnp.fft.ifft2(x, axes=(-2, -1)).real


def _nonlinear_hat(wh: Array, kx: Array, ky: Array, k2: Array, dealias: Array) -> Array:
    wh_f = wh * dealias  # dealiasing before operation
    k2_inv = jnp.where(k2 > 0, 1.0 / k2, 0.0).astype(jnp.float32)
    psi_hat = -wh_f * k2_inv
    wx, wy = _ifft(1j * kx * wh_f), _ifft(1j * ky * wh_f)
    psix, psiy = _ifft(1j * kx * psi_hat), _ifft(1j * ky * psi_hat)
    return _fft(psix * wy - psiy * wx) * dealias  # dealiasing after operation


def _make_forcing_basis_fft(N: int, L: float) -> Array:
    x = jnp.linspace(0.0, L, N, endpoint=False, dtype=jnp.float32)
    X, Y = jnp.meshgrid(x, x, indexing="ij")
    modes = jnp.stack(
        [
            jnp.sin(6 * X),
            jnp.cos(7 * X),
            jnp.sin(5 * (X + Y)),
            jnp.cos(8 * (X + Y)),
            jnp.cos(6 * X),
            jnp.sin(7 * X),
            jnp.cos(5 * (X + Y)),
            jnp.sin(8 * (X + Y)),
        ],
        axis=0,
    ).astype(jnp.float32)
    return _fft(modes)  # [8, N, N] complex64


def _integrate(
    key: PRNGKey,
    w0: Array,  # [B, N, N] float32
    *,
    steps: int,
    dt: float,
    record_every: int,
    nu: float,
    alpha: float,
    eps: float,
    kx: Array,
    ky: Array,
    k2: Array,
    dealias: Array,
    basis_fft: Array,
    show_progress: bool,
    desc: str,
) -> Array:
    assert steps >= record_every, "steps >= record_every"

    wh = _fft(w0)
    records = []
    sqrt_dt = jnp.sqrt(jnp.asarray(dt, dtype=jnp.float32))

    iter_steps = (
        tqdm.auto.trange(1, steps + 1, desc=desc, leave=False)
        if show_progress
        else range(1, steps + 1)
    )

    for s in iter_steps:
        key, sub = jax.random.split(key)

        wh = wh.at[..., 0, 0].set(
            0.0
        )  # clamp residual numerical drift at the zero mode
        NLh = _nonlinear_hat(wh, kx, ky, k2, dealias)
        det = -NLh - nu * k2 * wh - alpha * wh

        coeff = jax.random.normal(
            sub, (w0.shape[0], basis_fft.shape[0]), dtype=jnp.float32
        )  # [B,8]
        fh = jnp.tensordot(coeff, basis_fft, axes=([1], [0]))  # [B,N,N] complex64

        wh = wh + dt * det + eps * sqrt_dt * fh
        # wh = wh.at[..., 0, 0].set(0.0)  # TODO

        if s % record_every == 0:
            records.append(_ifft(wh)[..., None, :, :])  # [B,1,N,N]

    return jnp.stack(records, axis=1)  # [B, Trec, 1, N, N]


def _downsample_avg(x: Array, target: Optional[int]) -> Array:
    if (target is None) or (x.shape[-1] == target):
        return x
    B, T, C, H, W = x.shape
    f = H // target
    return x.reshape(B, T, C, target, f, target, f).mean(axis=(4, 6))


def _mean_l2(x: Array) -> Array:
    flat = x.reshape(-1, *x.shape[-3:])  # [..., 1, H, W]
    norms = jnp.sqrt(jnp.mean(flat**2, axis=(-3, -2, -1)))
    return jnp.mean(norms)


def simulate(
    T: int = 100,
    N: int = 256,
    L: float = 2 * np.pi,
    dt: float = 1e-4,
    nu: float = 1e-3,
    alpha: float = 0.1,
    epsilon: float = 1.0,
    warmup: int = 100,
    snapshot_interval: float = 0.5,
    downsample_to: Optional[int] = 128,
    seed: int = 1,
    same_warmup: bool = True,
    B: Optional[int] = None,
    show_progress: bool = False,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:

    key = jax.random.PRNGKey(seed)
    kx, ky, k2 = _wavenumbers(N, L)
    dealias = _dealias_mask(N)
    basis_fft = _make_forcing_basis_fft(N, L)

    steps_per_record = int(round(snapshot_interval / dt))
    warmup_steps = warmup * steps_per_record
    total_steps = T * steps_per_record

    bw = 1 if (same_warmup or B is None) else B
    w0 = jnp.zeros((bw, N, N), dtype=jnp.float32)

    warm_rec = jnp.zeros((bw, 0, 1, N, N), jnp.float32)
    if warmup_steps > 0:
        key, sub = jax.random.split(key)
        warm_rec = _integrate(
            sub,
            w0,
            steps=warmup_steps,
            dt=dt,
            record_every=steps_per_record,
            nu=nu,
            alpha=alpha,
            eps=epsilon,
            kx=kx,
            ky=ky,
            k2=k2,
            dealias=dealias,
            basis_fft=basis_fft,
            show_progress=show_progress,
            desc="warmup",
        )
        w0 = warm_rec[:, -1, 0, :, :]

    if same_warmup and B is not None:
        w0 = jnp.repeat(w0, B, axis=0)

    key, sub = jax.random.split(key)
    traj = _integrate(
        sub,
        w0,
        steps=total_steps,
        dt=dt,
        record_every=steps_per_record,
        nu=nu,
        alpha=alpha,
        eps=epsilon,
        kx=kx,
        ky=ky,
        k2=k2,
        dealias=dealias,
        basis_fft=basis_fft,
        show_progress=show_progress,
        desc="simulate",
    )

    warm_rec = _downsample_avg(warm_rec, downsample_to)
    traj = _downsample_avg(traj, downsample_to)

    if normalize:
        mean = (
            _mean_l2(traj) if B is not None and B * T >= 100 else 3.009334
        )  # need at least 100 frames to calculate normalization factor
        mean = jnp.maximum(mean, jnp.asarray(1e-12, dtype=traj.dtype))  # tiny guard
        warm_rec = warm_rec / mean
        traj = traj / mean

    return np.array(warm_rec, dtype=np.float32), np.array(traj, dtype=np.float32)
