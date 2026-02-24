![Header](header.gif)

# Setup

Create a conda environment and install the dependencies:

```bash
conda env create -f environment.yml
conda activate gps
```

# Implemented Models

The repository currently includes the following model families (configured via `configs/model/*.yaml` and implemented in `src/models/`):

| Hydra config | Class | Description |
|---|---|---|
| `fm` | `FlowMatchingModel` | Conditional flow matching model for next-frame generation with ODE-based sampling. |
| `si` | `StochasticInterpolation` | Stochastic interpolation model (supports Foellmer-style variants via config options). |
| `dm` | `DiffusionModel` | Diffusion model with DDPM-style training and optional DDIM sampling at evaluation. |
| `cm` | `ConsistencyModel` | Consistency model / consistency training. |
| `add_fm` | `ADDFlowMatching` | Adversarial diffusion distillation for flow matching (teacher-student + discriminator). |

Check `configs/train.yaml` and the `configs/model/` directory for all available configuration options.

# Training

Models are launched via Hydra. The main entry point is `src/train.py`.

Example (Flow Matching):

```bash
python src/train.py model=fm
```

# Evaluation

After training you can run the test phase with `src/eval.py`:

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

You can also override the model or data configuration during evaluation if needed:
```bash
python src/eval.py model=dm ckpt_path=...
```

# Tensor Dimension Notation

Symbols used in tensor shape annotations in docstrings and assertions.

| Symbol | Meaning |
|---|---|
| B | batch size |
| S | context frames |
| C | channels per frame (F+P) |
| F | field channels |
| P | parameter channels |
| H | height (spatial dimension) |
| W | width (spatial dimension) |
| T | noise steps |
| U | unroll steps in training |
| L | trajectory length in evaluation |
