# SAE Heist

SAE Heist is a reinforcement learning environment for improving Sparse Autoencoder (SAE) representations. The goal is to modify SAE encoder and decoder weights to maximize Spurious Correlation Removal (SCR), encouraging disentanglement between target features and spurious features while preserving useful information.

## Overview

The environment provides:

- A Sparse Autoencoder with 4096 latent dimensions
- Encoder and decoder weight matrices
- Synthetic activation data and evaluation metrics

The agent can modify the SAE weights and is rewarded based on how effectively the resulting representation removes spurious correlations while retaining target signal.

## Task

Given:

- `sae_W_enc.npy`
- `sae_W_dec.npy`
- `sae_meta.json`

the agent edits the SAE weights and submits a modified model.

Evaluation is based on:

- Spurious Correlation Removal (SCR)
- Target feature preservation
- Latent disentanglement quality

## Setup

```bash
git lfs install
git lfs pull
uv sync
```

## Training

Run a training rollout:

```bash
cd rl-training
uv run python simple_train.py --steps 1 --group 1 --max-concurrent 1
```

Serve the environment locally:

```bash
hud serve env:env
```

Evaluate a model:

```bash
hud run sae_heist --model <model-id>
```

## Approach

Our approach uses reinforcement learning to search over modifications to SAE encoder and decoder weights.

The objective is to improve representation quality by reducing reliance on spurious features while maintaining information relevant to the target feature. Agents receive rewards derived from SCR and related disentanglement metrics, allowing iterative optimization of the SAE representation.

## Repository Structure

```text
.
├── env.py
├── tasks/
│   └── sae_heist/
├── rl-training/
├── tools/
├── cases/
└── README.md
```

## Team

- Mary Le
- Alor Sahoo

Built during the HUD Frontier/RSI RL Environments Hackathon at Y Combinator.
