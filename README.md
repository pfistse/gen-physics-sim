# Gen Physics Sim

This repository contains code for training generative models on 2D fluid simulations. The code is based on PyTorch Lightning and Hydra for configuration management.

## Setup

Create a conda environment and install the dependencies:

```bash
conda env create -f environment.yml
conda activate gps
```

Example datasets can be downloaded from [https://mediatum.ub.tum.de/1734798](https://mediatum.ub.tum.de/1734798).

## Training

Models are launched via Hydra. The main entry point is `src/train.py`:

```bash
python src/train.py
```

Check `configs/train.yaml` for configuration options.

## Evaluation

After training you can run the test phase with `src/eval.py`:

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

The configuration under `configs/` controls which model and data module are used.
