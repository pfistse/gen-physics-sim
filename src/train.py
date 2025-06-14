#!/usr/bin/env python3

"""Train generative physics models with Hydra and Lightning."""

from typing import Any, Dict, List, Optional, Tuple

import logging
import sys

import hydra
import rootutils
import pytorch_lightning as pl
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer
from pytorch_lightning.loggers import Logger
from omegaconf import DictConfig

rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from utils.log import get_logger

log = get_logger(__name__)


def instantiate_callbacks(callbacks_cfg: DictConfig) -> List[Callback]:
    """Instantiates callbacks from config."""
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
    """Instantiate experiment tracking loggers from config."""
    trackers: List[Logger] = []

    if not tracking_cfg:
        log.warning("No tracking configs found! Skipping...")
        return trackers

    if isinstance(tracking_cfg, DictConfig) and "_target_" in tracking_cfg:
        log.info(f"Instantiating logger <{tracking_cfg._target_}>")
        trackers.append(hydra.utils.instantiate(tracking_cfg))

    return trackers


def train(cfg: DictConfig) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Trains the model. Can additionally evaluate on a testset, using best weights obtained during
    training.

    :param cfg: A DictConfig configuration composed by Hydra.
    :return: A tuple with metrics and dict with all instantiated objects.
    """

    if cfg.get("seed"):
        log.info(f"Setting random seed to {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)

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
        "model": model,
        "callbacks": callbacks,
        "logger": trackers,
        "trainer": trainer,
    }

    log.info("Setting up data module...")
    datamodule.setup()

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics

    metric_dict = {**train_metrics, **test_metrics}

    if trackers:
        for lg in trackers:
            if hasattr(lg, 'experiment') and hasattr(lg.experiment, 'finish'):
                lg.experiment.finish()

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    """
    train(cfg)


if __name__ == "__main__":
    # Allow passing arguments with a leading '--' as produced by wandb sweeps
    for idx, arg in enumerate(sys.argv[1:], start=1):
        if arg.startswith("--") and "=" in arg:
            sys.argv[idx] = arg[2:]
    main()