Basic Diffusion Model (DDPM) Implementation

This project implements a Denoising Diffusion Probabilistic Model (DDPM) from scratch using PyTorch.
It demonstrates how diffusion models work for synthetic image generation and allows experimenting with different noise schedules (linear, cosine).

Features

Forward diffusion (adding Gaussian noise step by step).
Reverse diffusion (denoising with a neural network).
Simple UNet-like architecture for noise prediction.
Training on MNIST dataset.
Sampling new synthetic images.
Support for multiple noise schedules:
  Linear schedule
  Cosine schedule
