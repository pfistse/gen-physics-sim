import contextlib
import copy
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, Optional

import torch
import pytorch_lightning as pl
from pytorch_lightning import Callback


class EMA(Callback):
    """Exponential Moving Average (EMA) callback adapted from pde-transformer.

    - Wraps Trainer optimizers with `EMAOptimizer` on fit start.
    - Swaps to EMA weights for validation/test if `validate_original_weights` is False.

    Args:
        decay: EMA decay factor in [0, 1]. Higher → smoother/older average.
        validate_original_weights: If False, use EMA weights during val/test.
        every_n_steps: Update EMA every N optimizer steps.
        cpu_offload: Store EMA tensors on CPU to save GPU memory.
        checkpoint_every_n_epochs: Save EMA checkpoints every N epochs (None disables).
    """

    def __init__(
        self,
        decay: float = 0.999,
        validate_original_weights: bool = False,
        every_n_steps: int = 1,
        cpu_offload: bool = False,
        checkpoint_every_n_epochs: Optional[int] = None,
    ) -> None:
        super().__init__()
        if not (0.0 <= decay <= 1.0):
            raise ValueError("EMA decay value must be between 0 and 1")
        self.decay = decay
        self.validate_original_weights = validate_original_weights
        self.every_n_steps = every_n_steps
        self.cpu_offload = cpu_offload
        if checkpoint_every_n_epochs is not None:
            assert checkpoint_every_n_epochs >= 1, "checkpoint_every_n_epochs >= 1"
        self.checkpoint_every_n_epochs = checkpoint_every_n_epochs

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        device = pl_module.device if not self.cpu_offload else torch.device("cpu")
        wrapped = []
        for optim in trainer.optimizers:
            if isinstance(optim, EMAOptimizer):
                wrapped.append(optim)
            else:
                wrapped.append(
                    EMAOptimizer(
                        optimizer=optim,
                        device=device,
                        decay=self.decay,
                        every_n_steps=self.every_n_steps,
                        current_step=trainer.global_step,
                    )
                )
        trainer.optimizers = wrapped

    # Swap to EMA weights for validation/test if configured
    def on_validation_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.validate_original_weights:
            self._swap(trainer)

    def on_validation_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.validate_original_weights:
            self._swap(trainer)

    def on_test_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.validate_original_weights:
            self._swap(trainer)

    def on_test_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if not self.validate_original_weights:
            self._swap(trainer)

    def _swap(self, trainer: pl.Trainer) -> None:
        for optimizer in trainer.optimizers:
            if isinstance(optimizer, EMAOptimizer):
                optimizer.switch_main_parameter_weights()

    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        """Save lightweight checkpoints with EMA weights alongside main checkpoints."""
        dirpath = self._checkpoint_dir(trainer)
        if dirpath is None:
            return

        self._swap(trainer)
        try:
            trainer.save_checkpoint(str(dirpath / "last-ema.ckpt"), weights_only=True)
            self._maybe_save_periodic_checkpoint(trainer, dirpath)
        finally:
            self._swap(trainer)

    def _checkpoint_dir(self, trainer: pl.Trainer) -> Optional[Path]:
        try:
            ckpt_cb = trainer.checkpoint_callback  # type: ignore[attr-defined]
            dirpath = getattr(ckpt_cb, "dirpath", None)
        except Exception:
            return None

        if not dirpath:
            return None
        return Path(dirpath)

    def _maybe_save_periodic_checkpoint(self, trainer: pl.Trainer, dirpath: Path) -> None:
        if self.checkpoint_every_n_epochs is None:
            return

        epoch = trainer.current_epoch + 1
        if epoch % self.checkpoint_every_n_epochs != 0:
            return
        filename = dirpath / f"epoch{epoch:04d}-ema.ckpt"
        trainer.save_checkpoint(str(filename), weights_only=True)


@torch.no_grad()
def _ema_update_(ema_tensors: Iterable[torch.Tensor], cur_tensors: Iterable[torch.Tensor], decay: float) -> None:
    # in-place: ema = decay * ema + (1 - decay) * cur
    ema_list = list(ema_tensors)
    cur_list = list(cur_tensors)
    torch._foreach_mul_(ema_list, decay)
    torch._foreach_add_(ema_list, cur_list, alpha=(1.0 - decay))


def _run_ema_update_cpu(
    ema_tensors: Iterable[torch.Tensor],
    cur_tensors: Iterable[torch.Tensor],
    decay: float,
    pre_sync_stream: Optional[torch.cuda.Stream] = None,
) -> None:
    if pre_sync_stream is not None:
        pre_sync_stream.synchronize()
    _ema_update_(ema_tensors, cur_tensors, decay)


class EMAOptimizer(torch.optim.Optimizer):
    """Optimizer wrapper that maintains an EMA of registered parameters.

    Notes:
    - Wrap your optimizer via this class; the callback does it automatically.
    - Access EMA weights with the `swap_ema_weights()` context or `switch_main_parameter_weights()`.
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        decay: float = 0.999,
        every_n_steps: int = 1,
        current_step: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.decay = float(decay)
        self.device = device
        self.every_n_steps = int(every_n_steps)
        self.current_step = int(current_step)

        self._first_iter = True
        self._rebuild_ema_params = True
        self._stream: Optional[torch.cuda.Stream] = None
        self._thread: Optional[threading.Thread] = None

        self._ema_params: tuple[torch.Tensor, ...] = ()
        self._in_swap_ctx = False
        self.save_original_optimizer_state = False

    # ---- Helpers ----
    def _all_parameters(self) -> Iterable[torch.Tensor]:
        return (p for g in self.param_groups for p in g["params"])  # type: ignore[attr-defined]

    def _maybe_init_stream(self) -> None:
        if self._first_iter:
            if any(p.is_cuda for p in self._all_parameters()):
                self._stream = torch.cuda.Stream()
            self._first_iter = False

    def _maybe_rebuild_ema_params(self) -> None:
        if self._rebuild_ema_params:
            opt_params = list(self._all_parameters())
            self._ema_params += tuple(
                copy.deepcopy(p.data.detach()).to(self.device) for p in opt_params[len(self._ema_params) :]
            )
            self._rebuild_ema_params = False

    def _should_update(self) -> bool:
        return self.current_step % self.every_n_steps == 0

    def join(self) -> None:
        if self._stream is not None:
            self._stream.synchronize()
        if self._thread is not None:
            self._thread.join()

    # ---- Optimizer API passthrough ----
    def __getattr__(self, name: str) -> Any:  # delegate to wrapped optimizer
        return getattr(self.optimizer, name)

    @property
    def param_groups(self):  # type: ignore[override]
        return self.optimizer.param_groups

    # ---- Core logic ----
    def step(self, closure=None, grad_scaler=None, **kwargs):  # type: ignore[override]
        self.join()
        self._maybe_init_stream()
        self._maybe_rebuild_ema_params()

        if getattr(self.optimizer, "_step_supports_amp_scaling", False) and grad_scaler is not None:
            loss = self.optimizer.step(closure=closure, grad_scaler=grad_scaler)
        else:
            loss = self.optimizer.step(closure)

        if self._should_update():
            self.update()
        self.current_step += 1
        return loss

    @torch.no_grad()
    def update(self) -> None:
        if self._stream is not None:
            self._stream.wait_stream(torch.cuda.current_stream())

        with torch.cuda.stream(self._stream) if self._stream is not None else contextlib.nullcontext():
            current = tuple(p.data.to(self.device, non_blocking=True) for p in self._all_parameters())

            if self.device.type == "cuda":
                _ema_update_(list(self._ema_params), current, self.decay)

        if self.device.type == "cpu":
            self._thread = threading.Thread(
                target=_run_ema_update_cpu,
                args=(self._ema_params, current, self.decay, self._stream),
            )
            self._thread.start()

    def _swap_tensors(self, t1: torch.Tensor, t2: torch.Tensor) -> None:
        tmp = torch.empty_like(t1)
        tmp.copy_(t1)
        t1.copy_(t2)
        t2.copy_(tmp)

    def switch_main_parameter_weights(self, saving_ema_model: bool = False) -> None:
        self.join()
        self._in_swap_ctx = saving_ema_model
        for p, ep in zip(self._all_parameters(), self._ema_params):
            self._swap_tensors(p.data, ep)

    @contextlib.contextmanager
    def swap_ema_weights(self, enabled: bool = True) -> Iterator[None]:
        if enabled:
            self.switch_main_parameter_weights()
        try:
            yield
        finally:
            if enabled:
                self.switch_main_parameter_weights()

    # ---- State dict ----
    def state_dict(self) -> Dict[str, Any]:  # type: ignore[override]
        self.join()
        if self.save_original_optimizer_state:
            return self.optimizer.state_dict()
        # If inside a context where EMA weights are currently in-place on the module,
        # we still want to store the EMA tensors coming from `self._ema_params`.
        return {
            "opt": self.optimizer.state_dict(),
            "ema": self._ema_params,
            "current_step": self.current_step,
            "decay": self.decay,
            "every_n_steps": self.every_n_steps,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:  # type: ignore[override]
        self.join()
        if "opt" not in state_dict:
            # Legacy checkpoint without EMA-specific payload.
            self.optimizer.load_state_dict(state_dict)
            self._ema_params = tuple(copy.deepcopy(p.data.detach()).to(self.device) for p in self._all_parameters())
            self.current_step = int(state_dict.get("current_step", self.current_step))
            self._rebuild_ema_params = False
            return

        self.optimizer.load_state_dict(state_dict["opt"])  # type: ignore[arg-type]
        self._ema_params = tuple(param.to(self.device) for param in copy.deepcopy(state_dict["ema"]))
        self.current_step = int(state_dict.get("current_step", self.current_step))  # type: ignore[assignment]
        self.decay = float(state_dict.get("decay", self.decay))  # type: ignore[assignment]
        self.every_n_steps = int(state_dict.get("every_n_steps", self.every_n_steps))  # type: ignore[assignment]
        self._rebuild_ema_params = False

    def add_param_group(self, param_group):  # type: ignore[override]
        self.optimizer.add_param_group(param_group)
        self._rebuild_ema_params = True
