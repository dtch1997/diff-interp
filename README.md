# Diff-Interp

A library for training classifiers to predict features, with the goal of creating metrics for interpretability in diffusion models. 

# Quickstart

## Install system dependencies

We use `pdm` to manage the project. Follow instructions [here](https://github.com/pdm-project/pdm) to install `pdm`. 
We use `git lfs` to store pre-trained checkpoints. Follow instructions [here](https://git-lfs.com/) to install. 

## Install the library

Install from source: 
```bash
git clone git@github.com:dtch1997/diff-interp.git
cd diff-interp
pdm install
```

## Load a pre-trained checkpoint

Download the checkpoints:
```bash
git lfs pull
```

See an example in `notebooks/load_checkpoint.ipynb`

## Train a new model

First, login to WandB. 
```bash
pdm run wandb login
```

You can now run a training run.
```
pdm run python -m diff_interp.train
```