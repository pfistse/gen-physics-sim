"""Utility modules for the gen-physics-sim project."""

try:
    from .wandb import log_samples
except Exception:  # pragma: no cover - optional dependency
    def log_samples(*args, **kwargs):
        """Dummy logging function when wandb is unavailable."""
        pass
