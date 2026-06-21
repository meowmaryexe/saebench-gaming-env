# SAE Heist

A reinforcement learning environment focused on improving Sparse Autoencoder (SAE) representations for spurious correlation removal.

## Overview

The agent is given an SAE with encoder and decoder weights and must modify the latent representation to maximize the Spurious Correlation Removal (SCR) score while preserving target feature information.

The environment evaluates:
- SCR (Spurious Correlation Removal)
- Ground-truth disentanglement metrics
- Generalization across synthetic evaluation panels

## Task

Given:
- `sae_W_enc.npy`
- `sae_W_dec.npy`
- SAE metadata

The agent may edit the SAE weights and submit the modified model. Rewards are based on improved disentanglement and bias removal performance.

## Running Locally

```bash
uv sync
hud serve env:env
