# Sparse Autoencoder Implementation in PyTorch

<p align="center">
  <img src="static/header.png" width="500" title="header">
</p>

## 👨🏽‍💻 Overview
This code implements a basic sparse autoencoder (SAE) in PyTorch. The loss is implemented from scratch; it uses MSE plus a penalty using KL divergence. In this case I used a very basic encoder and decoder architecture. This code is designed to be educational and is not focused on performance.

## 🛠️ Parameters
The parsed arguments allow the architecture to be launched from the terminal. Here's a toy example:

```
python3 sae.py \
--in_dims 100
--h_dims 300
--sparsity_lambda 0.0001
--sparsity_target 0.05
```
