# Deep Neural Network Architecture: Early vs. Late Regularization

## Overview

This project explores the impact of regularizer placement within a deep neural network. Specifically, it tests the consequences of applying Dropout at the early, entry-level layers versus applying it at the later, deeper layers of the network when classifying the MNIST dataset.

---

## Methodology

The experiment trains two separate Keras Sequential models using the **NADAM** optimizer for **15 epochs**. Both models utilize three hidden layers with 64 neurons each and Batch Normalization, but they differ fundamentally in their Dropout strategy:

- **Model 1 (Late Regularization):** Places a single `Dropout(0.2)` layer at the very end of the hidden layers, immediately before the output layer.
- **Model 2 (Early Regularization):** Places a single `Dropout(0.2)` layer immediately after the first entry-level hidden layer.

---

## Findings

Based on the training and validation metrics tracked over 15 epochs, the two models exhibited distinct learning and generalization behaviors:

| Model | Training Accuracy | Validation Accuracy |
|-------|-------------------|---------------------|
| Model 1 — Late Dropout | ~99.06% | ~97.36% |
| Model 2 — Early Dropout | ~97.83% | ~97.72% |

---

## Consequences and Analysis

### 1. The Information Bottleneck vs. Generalization (Model 2)

Applying dropout early in the network restricts the flow of foundational, low-level features (like basic edges and lines) to the subsequent layers.

Because 20% of the foundational information is randomly dropped right at the entry level, Model 2 is forced to work much harder to understand the images. While an early bottleneck can sometimes stunt learning, in this specific experiment it acted as a strict penalty that successfully prevented the network from simply memorizing the dataset — leading to slightly better generalization on unseen data.

### 2. Rapid Feature Extraction and Overfitting (Model 1)

Model 1 had an uninterrupted flow of information through its first two layers, allowing it to extract complex features rapidly. However, because MNIST is a relatively simple dataset, this unrestricted learning capacity caused Model 1 to begin overfitting.

The network quickly memorized the specific training images (pushing training accuracy past 99%). The late dropout layer alone was not a strong enough restriction to prevent this memorization phase, resulting in a plateaued and slightly lower validation accuracy compared to Model 2.

---

## Conclusion

**Early Dropout** heavily penalizes the network's initial feature extraction. While this can prevent memorization on simple datasets like MNIST, it risks causing underfitting on highly complex datasets where the network desperately needs all foundational data.

**Late Dropout** allows the network to learn rich, complex features uninterrupted. However, on simpler tasks, it may require additional regularization techniques (such as Early Stopping or heavier dropout rates) to prevent the model from rapidly overfitting the training data.

---

## Usage

```bash
jupyter notebook deep_net.ipynb
```

**Dependencies:** `tensorflow`, `numpy`, `matplotlib`
