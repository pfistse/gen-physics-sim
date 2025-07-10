#!/usr/bin/env python3
"""Run the test phase of a trained model."""

from typing import Any, Dict, List, Tuple
from pathlib import Path
import hydra
import rootutils
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig, ListConfig, OmegaConf

# Setup root directory to allow absolute imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.log import get_logger
from utils.evaluation import calculate_temporal_deviation
from utils.wandb import log_multi_temporal_deviation

log = get_logger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiate callbacks from configuration."""
    callbacks: List[Callback] = []

    if not callbacks_cfg:
        log.warning("No callback configs found! Skipping..")
        return callbacks

    for _, cb_conf in callbacks_cfg.items():
        if isinstance(cb_conf, DictConfig) and "_target_" in cb_conf:
            log.info(f"Instantiating callback <{cb_conf._target_}>")
            callbacks.append(hydra.utils.instantiate(cb_conf))

    return callbacks


def instantiate_tracking(tracking_cfg: DictConfig) -> List[Logger]:
    """Instantiate experiment tracking loggers from configuration."""
    trackers: List[Logger] = []

    if not tracking_cfg:
        log.warning("No tracking configs found! Skipping...")
        return trackers

    if isinstance(tracking_cfg, DictConfig) and "_target_" in tracking_cfg:
        log.info(f"Instantiating logger <{tracking_cfg._target_}>")
        trackers.append(hydra.utils.instantiate(tracking_cfg))

    return trackers


def evaluate(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Evaluate a trained model on the test set.

    Parameters
    ----------
    cfg: DictConfig
        Hydra composed configuration specifying model, data and trainer.

    Returns
    -------
    Tuple containing metric dictionary and a dictionary of instantiated objects.
    """

    if cfg.get("seed"):
        log.info(f"Setting random seed to {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = instantiate_callbacks(cfg.get("callbacks"))

    log.info("Instantiating experiment tracking...")
    trackers: List[Logger] = instantiate_tracking(cfg.get("tracking"))

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer, callbacks=callbacks, logger=trackers
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "callbacks": callbacks,
        "logger": trackers,
        "trainer": trainer,
    }

    log.info("Setting up data module...")
    datamodule.setup()

    ckpt_entries = cfg.get("checkpoints")
    if not ckpt_entries:
        log.warning("No checkpoints provided for evaluation")
        return {}, object_dict

    if not isinstance(ckpt_entries, (list, ListConfig)):
        raise TypeError(
            f"cfg.checkpoints must be a list, got {type(ckpt_entries).__name__}"
        )

    combined_metrics = {}

    metric_dicts = {}

    for idx, ckpt in enumerate(ckpt_entries):
        if not isinstance(ckpt, (dict, DictConfig)):
            raise TypeError(
                f"Checkpoint entry #{idx} must be a mapping, got {type(ckpt).__name__}"
            )

        ckpt_path = ckpt.get("path")
        name = ckpt.get("name") or Path(ckpt_path).stem
        model_cfg_path = ckpt.get("model_config")

        log.info(f"Starting testing for checkpoint: {ckpt_path}")
        log.info(f"Loading model config from {model_cfg_path}")

        # Merge model config with the data and evaluation settings so
        # interpolations (e.g. `${evaluation}`) are resolved properly when
        # converting to plain dictionaries.
        base_cfg = OmegaConf.create({"data": cfg.data, "evaluation": cfg.evaluation})
        model_cfg = OmegaConf.merge(base_cfg, OmegaConf.load(model_cfg_path))

        global_override = cfg.get("model")
        if global_override:
            model_cfg = OmegaConf.merge(model_cfg, global_override)
        if ckpt.get("model"):
            model_cfg = OmegaConf.merge(model_cfg, ckpt.model)

        model_cls = hydra.utils.get_class(model_cfg._target_)

        network = hydra.utils.instantiate(model_cfg.network)

        # Convert config to plain dict for load_from_checkpoint overrides
        model_kwargs = OmegaConf.to_container(model_cfg, resolve=True)
        model_kwargs.pop("network", None)
        model_kwargs.pop("_target_", None)
        model_kwargs.pop("eval_config", None)

        model: LightningModule = model_cls.load_from_checkpoint(
            ckpt_path,
            network=network,
            eval_config=cfg.evaluation,
            **model_kwargs,
        )

        trainer.test(model=model, datamodule=datamodule, ckpt_path=None)

        # TODO quick fix
        device = trainer.strategy.root_device
        model.to(device)

        metric_dicts[name] = dict(trainer.callback_metrics)

        temp_cfg = cfg.evaluation.metrics.temp_deviation
        if temp_cfg.enabled:
            deviation_metrics = calculate_temporal_deviation(
                model=model,
                num_simulations=temp_cfg.num_simulations,
                num_time_steps=temp_cfg.num_time_steps,
                start_frame=temp_cfg.get("start_frame"),
                channel_titles=cfg.evaluation.channel_titles,
                num_steps=model.num_steps_eval,
            )
            combined_metrics[name] = deviation_metrics

    if combined_metrics:
        ref_cfg = cfg.get("references")
        ref_lines = (
            OmegaConf.to_container(ref_cfg, resolve=True)
            if ref_cfg is not None
            else None
        )
        log_multi_temporal_deviation(
            combined_metrics,
            prefix="test",
            references=ref_lines,
        )

    if trackers:
        for lg in trackers:
            if hasattr(lg, "experiment") and hasattr(lg.experiment, "finish"):
                lg.experiment.finish()

    return metric_dicts, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Entry point for evaluation script."""
    evaluate(cfg)


if __name__ == "__main__":
    main()
