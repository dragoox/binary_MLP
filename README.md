# binary_MLP
A JAX implementation of a fully Binary Neural Network trained from scratch on the MNIST dataset using the Straight-Through Estimator.

## Overview

This project explores an extreme form of model quantization where both the neural network's **weights** and its **activations** are constrained to binary values (`-1` or `1`). Such networks offer massive potential for efficiency, dramatically reducing memory footprint and replacing expensive floating-point multiplications with simple bitwise operations (XNOR).

The main challenge is that the binarization function is non-differentiable, making it incompatible with standard backpropagation. This notebook demonstrates how to overcome this by using the **Straight-Through Estimator (STE)**, a technique that approximates the gradient to enable end-to-end training.

## Key Features

-   **Framework**: Built with [JAX](https://github.com/google/jax) for high-performance, differentiable programming.
-   **Model**: A simple Multi-Layer Perceptron (MLP) with binary weights and activations.
-   **Training Technique**: Implements the Straight-Through Estimator (STE) to train the non-differentiable model.
-   **Dataset**: Trained and evaluated on the classic MNIST handwritten digit dataset.

## Results

This implementation successfully trains a binary MLP and achieves approximately **95% accuracy** on the MNIST test set, demonstrating the viability of this approach for learning meaningful representations.

## How to Run

The easiest way to run this project is to open the notebook in Google Colab. All dependencies are included and can be installed within the notebook.
