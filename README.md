# Sparse Autoencoder Implementation in PyTorch

<p align="center">
  <img src="static/header.png" width="500" title="header">
</p>

## 👨🏽‍💻 Overview
This code implements a basic sparse autoencoder (SAE) in PyTorch. The loss is implemented from scratch; it uses MSE plus a penalty using KL divergence. In this case I used a very basic encoder and decoder architecture. This code is designed to be educational and is not focused on performance.

## 🛠️ Parameters
The parsed arguments allow the architecture to be launched from the terminal.

- ```batch_size``` -> int : sets the batch size for training the model.
- ```n_epochs``` -> int: sets the number of epochs for training.
- ```lr``` -> float: sets the learning rate.
- ```in_dims``` -> int: sets input dimensions of the model.
- ```h_dims``` -> int: sets hidden dimensions of the model.
- ```sparsity_lambda``` -> float: sets the lambda value for the SAE (controls the strength of the sparsity penalty in the loss function. Larger value equals to more sparsity but potential loss in reconstruction quality).
- ```sparsity_target``` -> float: sets the sparsity target value (average activation you want each hidden neuron to have across all input samples. Smaller value allows more sparsity).
- ```xavier_norm_init``` -> bool: sets xavier norm initialization for the weights of the encoder and decoder.
- ```show_summary``` -> bool: show model parameters and summary.
- ```download_mnist``` -> bool: download MNIST dataset for training.
- ```train``` -> bool: launch training.
- ```visualize_activations``` -> bool: visualize neuron activations
- ```save_model``` -> bool: saves the model to a folder called 'files'.

Here's the default values for the code:
```
python3 sae.py \
--batch_size 64 \
--n_epochs 20 \
--lr 0.001 \
--in_dims 784\
--h_dims 1024 \
--sparsity_lambda 1e-4 \
--sparsity_target 0.05 \
--xavier_norm_init True \
--show_summary True \
--download_mnist True \
--train False \
--save_model False \
```

## 🚀 Fast Launch
Use this in your terminal to train the SAE model:
```
git clone https://github.com/IParraMartin/Sparse-Autoencoder.git
cd Sparse-Autoencoder
python3 sae.py --train True --save_model True
```

## 🤝 Contribute
Feel free to reach out to contribute to this repo!😊