from typing import Optional, List, Tuple

import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities.rank_zero import rank_zero_only

from rich.progress import Progress, TaskID
from rich.console import Console

from metrics.distributed import DistributedMetric, create_mean_std_metric
from metrics.rank0 import (
    RankZeroMetric,
    create_mse_metric,
    create_enstrophy_metric,
    create_variance_metric,
    create_video_metric,
)


class BaseGenerativeModel(pl.LightningModule):
    """Base with shared sampling and metrics."""

    def __init__(self) -> None:
        super().__init__()
        self._distributed_metrics: List[DistributedMetric] = []
        self._rank0_metrics: List[RankZeroMetric] = []
        self.register_distributed_metric(create_mean_std_metric())
        self.register_rank0_metric(create_mse_metric())
        self.register_rank0_metric(create_enstrophy_metric())
        self.register_rank0_metric(create_variance_metric())
        self.register_rank0_metric(create_video_metric())

    def generate_samples(self, cond: torch.Tensor, num_steps: int = 1, **kwargs):
        raise NotImplementedError

    def compute_loss(self, target: torch.Tensor, cond: torch.Tensor):
        raise NotImplementedError

    def prediction_step(self, cond: torch.Tensor, num_steps: Optional[int] = None):
        """Prediction step for inference.

        cond: [B, S, C, H, W]
        """
        steps = num_steps if num_steps is not None else self.num_steps_eval
        return self.generate_samples(cond, num_steps=steps)

    def forward(
        self,
        cond: torch.Tensor,
        target: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ):
        """Training/inference entry.

        cond: [B, S, C, H, W]
        target: [B, 1, C, H, W]
        """
        if target is not None:
            return self.compute_loss(target, cond)
        return self.prediction_step(cond, num_steps=num_steps)

    def generate_sequence(
        self,
        cond: torch.Tensor,
        seq_len: int,
        num_steps: int = 1,
        task: Optional[Tuple[Progress, TaskID]] = None,
        **sample_kwargs,
    ):
        """Generate sequential samples.

        cond: [B, S, C, H, W]
        """
        B, S, C, H, W = cond.shape

        P = len(self.sim_params)

        cond_len = S

        gen_seq = []

        const_params = None
        if P > 0 and S > 0:
            const_params = cond[:, :1, -P:, :, :]

        chunk_size = self.trainer.datamodule.batch_size
        for _ in range(seq_len):
            next_frame = cond.new_empty((B, 1, C, H, W))

            for chunk_from in range(0, B, chunk_size):
                chunk_to = min(chunk_from + chunk_size, B)

                next_frame[chunk_from:chunk_to] = self.generate_samples(
                    cond[chunk_from:chunk_to],
                    num_steps=num_steps,
                    **sample_kwargs,
                )

                if task:
                    task_id, progress = task
                    progress.advance(task_id, chunk_to - chunk_from)

            if const_params is not None:
                next_frame[:, :, -P:, :, :] = const_params

            gen_seq.append(next_frame)

            if cond_len > 1:
                cond = torch.cat([cond[:, 1:], next_frame], dim=1)
            else:
                cond = next_frame

        return torch.cat(gen_seq, dim=1)  # [B, seq_len, C, H, W]

    def training_step(self, batch, batch_idx):
        """Training step using subclass loss."""
        cond, target = batch
        loss = self.compute_loss(target, cond)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        """Test step to enable callback hooks."""
        self.validation_step(batch, batch_idx)
        return None

    def on_validation_epoch_start(self):
        self.eval_setup()

    def on_test_epoch_start(self):
        self.eval_setup()

    def on_validation_epoch_end(self):
        self.eval_finalize()

    def on_test_epoch_end(self):
        self.eval_finalize()

    def validation_step(self, batch, batch_idx):
        """Dispatch validation work to rank-specific hooks."""
        self.eval_step_distributed(batch, batch_idx)
        if batch_idx == 0:
            self.eval_step_rank0()
        return None

    def eval_step_distributed(self, batch, batch_idx):
        """Per-rank mean/std evaluation using cached simulator data."""

        for metric in self._distributed_metrics:
            metric.collect_step(self, batch)

        return None

    @rank_zero_only
    def eval_step_rank0(self):
        """Shared rank0-only logging tasks."""

        with Progress(console=Console()) as progress:
            for metric in self._rank0_metrics:
                metric.run(self, progress)

        return None

    def eval_setup(self):
        for metric in self._distributed_metrics:
            metric.reset()

    def eval_finalize(self):
        for metric in self._distributed_metrics:
            metric.finalize(self)

    def register_distributed_metric(self, metric: DistributedMetric) -> None:
        self._distributed_metrics.append(metric)

    def register_rank0_metric(self, metric: RankZeroMetric) -> None:
        self._rank0_metrics.append(metric)

    def configure_optimizers(self):
        """Configure optimizers."""
        return torch.optim.Adam(
            self.net.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
