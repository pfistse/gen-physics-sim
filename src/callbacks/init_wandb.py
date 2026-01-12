# callbacks/wandb_in_worker.py
import os, wandb
from pytorch_lightning import Callback
from pytorch_lightning.loggers import WandbLogger, LoggerCollection
from pytorch_lightning.utilities import rank_zero_only


class InitWandB(Callback):
    """Init a WandB logger inside the Ray worker (rank-0 only)."""

    def __init__(
        self, project: str, entity: str | None = None, name: str | None = None
    ):
        self.kw = {
            k: v
            for k, v in dict(project=project, entity=entity, name=name).items()
            if v is not None
        }

    @rank_zero_only
    def setup(self, trainer, pl_module, stage=None):
        # ensure fresh init in worker, not attach
        os.environ.pop("WANDB_RUN_ID", None)
        os.environ.setdefault("WANDB_START_METHOD", "thread")
        # optional (deprecated but harmless): os.environ.setdefault("WANDB_DISABLE_SERVICE", "true")

        # already have a W&B logger? keep it
        existing = trainer.logger
        has_wandb = isinstance(existing, WandbLogger) or (
            isinstance(existing, LoggerCollection)
            and any(isinstance(lg, WandbLogger) for lg in existing._logger_iterable)
        )
        if has_wandb:
            return

        wb = WandbLogger(
            log_model=False,
            **self.kw,
            # pass through to wandb.init directly (no init_args!)
            resume="allow",
            settings=wandb.Settings(start_method="thread"),
        )

        # install alongside any existing non-W&B logger(s)
        if isinstance(existing, LoggerCollection):
            trainer.logger = LoggerCollection([*existing._logger_iterable, wb])
        elif existing is None or existing is False:
            trainer.logger = wb
        else:
            trainer.logger = LoggerCollection([existing, wb])
