from typing import Any, Callable, List, Optional, Sequence

import torch
import torch.distributed as dist

from metrics.utils import (
    calculate_mean_std,
    calculate_mean_std_rel_err_steps,
    calculate_std_rel_err_grid,
)
from utils.wandb import (
    log_acc_metrics_steps,
    log_std_rel_err_grid,
    log_mean_std_grid,
)


class DistributedMetric:
    """Utility for collecting per-rank evaluation entries."""

    def __init__(
        self,
        name: str,
        enabled_fn: Callable[["BaseGenerativeModel"], bool],
        entry_fn: Callable[["BaseGenerativeModel", Any], Optional[Any]],
        finalize_fn: Callable[["BaseGenerativeModel", List[Any]], None],
        filter_fn: Optional[Callable[["BaseGenerativeModel", Any], bool]] = None,
    ) -> None:
        self.name = name
        self.enabled_fn = enabled_fn
        self.entry_fn = entry_fn
        self.finalize_fn = finalize_fn
        self.filter_fn = filter_fn
        self._entries: List[Any] = []

    def reset(self) -> None:
        self._entries = []

    def collect_step(self, model: "BaseGenerativeModel", batch: Any) -> None:
        if not self.enabled_fn(model):
            return
        if self.filter_fn is not None and not self.filter_fn(model, batch):
            return

        entry = self.entry_fn(model, batch)
        if entry is not None:
            self._entries.append(entry)

    def finalize(self, model: "BaseGenerativeModel") -> None:
        if not self.enabled_fn(model):
            return

        entries = self._entries
        self._entries = []

        if dist.is_available() and dist.is_initialized():
            world_size = dist.get_world_size()
            gathered: List[List[Any]] = [None] * world_size  # type: ignore[list-item]
            dist.all_gather_object(gathered, entries)
            entries = [
                entry
                for rank_entries in gathered
                if rank_entries
                for entry in rank_entries
            ]

        if model.trainer.is_global_zero:
            self.finalize_fn(model, entries)


# def _std_rel_err_filter(cfg: Any, batch: Any) -> bool:
#     if not isinstance(batch, dict):
#         return False
#     if "warmup" not in batch or "traj" not in batch:
#         return False

#     seeds_cfg: Sequence[int] = getattr(cfg, "seeds", []) or []
#     if seeds_cfg:
#         batch_seed = int(batch.get("seed", 0))
#         return batch_seed in set(int(seed) for seed in seeds_cfg)
#     return True


def create_mean_std_rel_err_steps_metric(
    *, unbiased: bool = False
) -> DistributedMetric:
    """Create mean_std_rel_err_steps distributed metric."""

    def enabled_fn(model) -> bool:
        return model.eval_config.debug.mean_std_rel_err_steps.enabled

    def filter_fn(model, batch) -> bool:
        cfg = model.eval_config.debug.mean_std_rel_err_steps
        return batch["seed"] in cfg.seeds  # _std_rel_err_filter(cfg, batch)

    def entry_fn(model, batch):
        cfg = model.eval_config.debug.mean_std_rel_err_steps
        return calculate_mean_std_rel_err_steps(model=model, cfg=cfg, entry=batch)

    def finalize_fn(model, entries: List[Any]) -> None:
        cfg = model.eval_config.debug.mean_std_rel_err_steps
        entries.sort(key=lambda item: item["seed"])

        std_steps_ranks = torch.tensor(
            [entry["std_rel_err_steps"] for entry in entries],
            dtype=torch.float32,
        )
        std_mean_steps = std_steps_ranks.mean(dim=0).tolist()
        std_std_steps = std_steps_ranks.std(dim=0, unbiased=unbiased).tolist()

        mean_steps_ranks = torch.tensor(
            [entry["mean_rel_err_steps"] for entry in entries],
            dtype=torch.float32,
        )
        mean_mean_steps = mean_steps_ranks.mean(dim=0).tolist()
        mean_std_steps = mean_steps_ranks.std(dim=0, unbiased=unbiased).tolist()

        log_acc_metrics_steps(
            means=[std_mean_steps, mean_mean_steps],
            stds=[std_std_steps, mean_std_steps],
            steps=list(cfg.steps),
            module=model,
            prefix="debug",
            title="mean_std_rel_err_steps",
        )

    return DistributedMetric(
        name="mean_std_rel_err_steps",
        enabled_fn=enabled_fn,
        filter_fn=filter_fn,
        entry_fn=entry_fn,
        finalize_fn=finalize_fn,
    )


def create_std_rel_err_grid_metric(*, unbiased: bool = False) -> DistributedMetric:
    """Create std_rel_err_grid distributed metric."""

    def _cfg(model):
        return getattr(model.eval_config.debug, "std_rel_err_grid", None)

    def enabled_fn(model) -> bool:
        cfg = _cfg(model)
        return bool(cfg and getattr(cfg, "enabled", False))

    def filter_fn(model, batch) -> bool:
        cfg = _cfg(model)
        if cfg is None:
            return False
        return _std_rel_err_filter(cfg, batch)

    def entry_fn(model, batch):
        cfg = _cfg(model)
        if cfg is None:
            return None
        return calculate_std_rel_err_grid(model=model, cfg=cfg, entry=batch)

    def finalize_fn(model, entries: List[Any]) -> None:
        cfg = _cfg(model)
        if cfg is None or not entries:
            return

        entries.sort(key=lambda item: item["seed"])

        values = torch.tensor(
            [entry["rel_err_grid"] for entry in entries],
            dtype=torch.float32,
        )
        mean = values.mean(dim=0).tolist()
        std = values.std(dim=0, unbiased=unbiased).tolist()
        g_coefs = [float(coef) for coef in cfg.g_coefs]
        log_std_rel_err_grid(
            mean=mean,
            std=std,
            steps=list(cfg.steps),
            g_coefs=g_coefs,
            module=model,
            prefix="debug",
            title="std_rel_err_grid",
        )

    return DistributedMetric(
        name="std_rel_err_grid",
        enabled_fn=enabled_fn,
        filter_fn=filter_fn,
        entry_fn=entry_fn,
        finalize_fn=finalize_fn,
    )


def create_mean_std_metric() -> DistributedMetric:
    """Create mean/std distributed metric handler."""

    def enabled_fn(model) -> bool:
        cfg = model.eval_config.metrics.mean_std
        return bool(getattr(cfg, "enabled", False))

    def entry_fn(model, batch):
        cfg = model.eval_config.metrics.mean_std
        mean_std_metrics = calculate_mean_std(model=model, cfg=cfg, entry=batch)
        for name, value in mean_std_metrics.items():
            if isinstance(name, str) and name.startswith("_"):
                continue
            model.log(
                f"metric/{name}",
                value,
                on_step=False,
                on_epoch=True,
                sync_dist=True,
            )
        return mean_std_metrics

    def finalize_fn(model, entries: List[Any]) -> None:
        cfg = model.eval_config.metrics.mean_std
        if not entries:
            return
        entries.sort(key=lambda item: item["seed"])
        log_mean_std_grid(
            entries=entries,
            module=model,
            channel_titles=model.eval_config.channel_titles,
            cfg=cfg,
            prefix="metric",
        )

    return DistributedMetric(
        name="mean_std",
        enabled_fn=enabled_fn,
        entry_fn=entry_fn,
        finalize_fn=finalize_fn,
    )
