#!/usr/bin/env python3

"""Evaluate a trained model on the test set using Hydra and Lightning."""

import logging
import hydra
import rootutils
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

# Setup root directory to allow absolute imports
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

# Register resolvers
OmegaConf.register_new_resolver("mul", lambda x, y: x * y)
OmegaConf.register_new_resolver("add", lambda x, y: x + y)
OmegaConf.register_new_resolver(
    "run_name",
    lambda name, ckpt: f"{name}_epoch{torch.load(ckpt, map_location='cpu')['epoch']}"
    if ckpt
    else name,
)

log = logging.getLogger(__name__)


@hydra.main(version_base="1.3", config_path="../configs", config_name="eval")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""

    if cfg.get("seed"):
        log.info(f"Setting random seed to {cfg.seed}")
        pl.seed_everything(cfg.seed, workers=True)

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    log.info("Instantiating callbacks...")
    callbacks = [hydra.utils.instantiate(cb) for cb in cfg.get("callbacks", {}).values()]

    log.info("Instantiating experiment tracking...")
    loggers = [hydra.utils.instantiate(lg) for lg in cfg.get("tracking", {}).values()]

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=loggers)

    log.info(f"Starting testing using checkpoint: {cfg.ckpt_path}")
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)

    # Finish loggers
    for lg in loggers:
        if hasattr(lg, "experiment") and hasattr(lg.experiment, "finish"):
            lg.experiment.finish()


if __name__ == "__main__":
    main()
