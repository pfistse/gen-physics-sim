from typing import Callable, Optional

from metrics.utils import (
    calculate_mse,
    calculate_enstrophy,
    calculate_var_samples,
    calculate_std_steps,
)
from utils.wandb import (
    log_mse,
    log_enstrophy,
    log_samples_video,
    log_comparison_video,
    log_std_grid,
    log_denoising_video,
)


class RankZeroMetric:
    """Callable rank-zero evaluation task."""

    def __init__(
        self,
        name: str,
        enabled_fn: Callable[["BaseGenerativeModel"], bool],
        run_fn: Callable[["BaseGenerativeModel", Optional[object]], None],
    ) -> None:
        self.name = name
        self.enabled_fn = enabled_fn
        self.run_fn = run_fn

    def run(self, model: "BaseGenerativeModel", progress: Optional[object]) -> None:
        if not self.enabled_fn(model):
            return
        self.run_fn(model, progress)


def create_mse_metric() -> RankZeroMetric:
    def enabled_fn(model) -> bool:
        return model.eval_config.metrics.mse.enabled

    def run_fn(model, progress) -> None:
        cfg = model.eval_config.metrics.mse
        mse_metrics = calculate_mse(
            model=model,
            channel_titles=model.eval_config.channel_titles,
            num_steps=model.num_steps_eval,
            cfg=cfg,
            progress=progress,
        )
        log_mse(mse_metrics, model, cfg, prefix="test")

    return RankZeroMetric("mse", enabled_fn, run_fn)


def create_enstrophy_metric() -> RankZeroMetric:
    def enabled_fn(model) -> bool:
        return model.eval_config.metrics.enstrophy.enabled

    def run_fn(model, progress) -> None:
        cfg = model.eval_config.metrics.enstrophy
        metrics = calculate_enstrophy(model=model, cfg=cfg, progress=progress)
        for name, value in metrics.items():
            if isinstance(name, str) and name.startswith("_"):
                continue
            model.log(f"metric/{name}", value, on_step=True, on_epoch=False)
        log_enstrophy(metrics=metrics, module=model, prefix="metric", cfg=cfg)

    return RankZeroMetric("enstrophy", enabled_fn, run_fn)


def create_variance_metric() -> RankZeroMetric:
    def enabled_fn(model) -> bool:
        return model.eval_config.debug.variance.enabled

    def run_fn(model, progress) -> None:
        cfg = model.eval_config.debug.variance
        samples = calculate_var_samples(model=model, cfg=cfg, progress=progress)
        log_samples_video(
            data=samples,
            module=model,
            channel_titles=model.eval_config.channel_titles,
            cfg=cfg,
            prefix="debug",
            pre_title="var",
        )

    return RankZeroMetric("variance", enabled_fn, run_fn)


def create_video_metric() -> RankZeroMetric:
    def enabled_fn(model) -> bool:
        return model.eval_config.video.enabled

    def run_fn(model, progress) -> None:
        cfg = model.eval_config.video
        total_len = model.ctx_len + cfg.num_frames
        seq_solver = model.trainer.datamodule.train_set.generate_sequence(
            start_frame=cfg.warmup,
            len=total_len,
        )

        cond = seq_solver[: model.ctx_len].unsqueeze(0).to(model.device)
        gt = seq_solver[model.ctx_len : model.ctx_len + cfg.num_frames]

        seq_model = (
            model.generate_sequence(
                cond=cond,
                seq_len=cfg.num_frames,
                num_steps=model.num_steps_eval,
            )[0]
            .cpu()
            .numpy()
        )

        log_samples_video(
            data=seq_model,
            channel_titles=model.eval_config.channel_titles,
            module=model,
            cfg=cfg,
        )
        log_comparison_video(
            gen=seq_model,
            gt=gt.cpu().numpy(),
            channel_titles=model.eval_config.channel_titles,
            module=model,
            cfg=cfg,
        )

    return RankZeroMetric("video", enabled_fn, run_fn)


def create_std_steps_metric(
    *,
    name: str = "std_steps",
    pretitle: str = "std_steps",
    cmap: str = "viridis",
    prefix: str = "metric",
) -> RankZeroMetric:
    def enabled_fn(model) -> bool:
        return bool(getattr(model.eval_config.debug.std_steps, "enabled", False))

    def run_fn(model, progress) -> None:
        cfg = model.eval_config.debug.std_steps
        std_steps = calculate_std_steps(model=model, cfg=cfg, progress=progress)
        log_std_grid(
            std_by_step=std_steps,
            steps=cfg.steps,
            module=model,
            channel_titles=model.eval_config.channel_titles,
            prefix=prefix,
            pretitle=pretitle,
            cmap=cmap,
        )

    return RankZeroMetric(name, enabled_fn, run_fn)


def create_denoising_metric() -> RankZeroMetric:
    def enabled_fn(model) -> bool:
        return bool(getattr(model.eval_config.debug.denoising, "enabled", False))

    def run_fn(model, progress) -> None:
        cfg = model.eval_config.debug.denoising
        dataset = model.trainer.datamodule.train_set
        try:
            cond_seq = dataset.load_sequence(len=model.ctx_len)
        except TypeError:
            cond_seq = dataset.load_sequence(start_frame=500, len=model.ctx_len)
        cond = cond_seq.to(model.device).unsqueeze(0)
        log_denoising_video(
            cond=cond,
            channel_titles=model.eval_config.channel_titles,
            module=model,
            cfg=cfg,
        )

    return RankZeroMetric("denoising", enabled_fn, run_fn)
