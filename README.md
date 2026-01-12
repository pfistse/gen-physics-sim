# Setup

Create a conda environment and install the dependencies:

```bash
conda env create -f environment.yml
conda activate gps
```

# Training

Models are launched via Hydra. The main entry point is `src/train.py`.

To train the default model (Consistency Model):
```bash
python src/train.py
```

To train a different model (e.g., Flow Matching):
```bash
python src/train.py model=fm
```

To enable Weights & Biases tracking:
```bash
python src/train.py tracking=wandb_train
```

Check `configs/train.yaml` and the `configs/model/` directory for all available configuration options.

# Evaluation

After training you can run the test phase with `src/eval.py`:

```bash
python src/eval.py ckpt_path=/path/to/checkpoint.ckpt
```

You can also override the model or data configuration during evaluation if needed:
```bash
python src/eval.py model=dm ckpt_path=...
```

# Notation

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
