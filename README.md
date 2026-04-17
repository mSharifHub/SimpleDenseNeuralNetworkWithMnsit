# Deep Learning: MNIST Digit Classification

A feedforward neural network trained on the MNIST handwritten digit dataset using TensorFlow/Keras.

## Dataset

**MNIST** — 70,000 grayscale images of handwritten digits (0–9), each 28×28 pixels.

| Split      | Samples |
|------------|---------|
| Training   | 60,000  |
| Validation | 10,000  |

**Preprocessing:**
- Images flattened from 28×28 to 784-dimensional vectors
- Pixel values normalized from [0, 255] → [0.0, 1.0]
- Labels one-hot encoded into 10-class vectors

## Model Architecture

```
Input (784,)
    ↓
Dense(64, ReLU)         — 50,240 params
    ↓
BatchNormalization       — 256 params
    ↓
Dense(64, ReLU)         — 4,160 params
    ↓
BatchNormalization       — 256 params
    ↓
Dropout(0.2)
    ↓
Dense(10, Softmax)      — 650 params
    ↓
Output (10 classes)
```

**Total parameters:** 55,562 (217 KB)

### Layer Explanations

| Layer | Purpose |
|-------|---------|
| `Dense(64, relu)` | Learns non-linear feature representations from the flattened image |
| `BatchNormalization` | Normalizes activations across the mini-batch, stabilizing training and allowing higher learning rates |
| `Dropout(0.2)` | Randomly zeros 20% of neurons during training to reduce overfitting |
| `Dense(10, softmax)` | Outputs a probability distribution over the 10 digit classes |

## Optimizer: Nadam

The model uses **Nadam** (Nesterov-accelerated Adaptive Moment Estimation) — a combination of Adam and Nesterov momentum.

### Optimizer Comparison

| Optimizer | Description | Pros | Cons |
|-----------|-------------|------|------|
| **SGD** | Updates weights by a fixed learning rate × gradient | Simple, memory efficient | Slow convergence, sensitive to learning rate |
| **SGD + Momentum** | Accumulates a velocity vector to dampen oscillations | Faster than SGD | Still requires careful LR tuning |
| **RMSProp** | Scales learning rate per-parameter using a moving average of squared gradients | Good for non-stationary problems | No momentum |
| **Adam** | Combines momentum + RMSProp; adapts learning rate per parameter | Fast convergence, robust defaults | Can generalize slightly worse than SGD |
| **Nadam** (used) | Adam + Nesterov momentum; applies gradient using a "look-ahead" gradient | Better convergence than Adam in many cases | Slightly more compute |

**Why Nadam?** Nadam's look-ahead correction gives smoother, more accurate gradient updates compared to vanilla Adam, which benefits this classification task.

## Training

| Hyperparameter | Value |
|----------------|-------|
| Loss function  | Categorical cross-entropy |
| Optimizer      | Nadam |
| Batch size     | 128 |
| Epochs         | 128 |
| Metric         | Accuracy |

## Results

| Metric        | Value  |
|---------------|--------|
| Test accuracy | 77.55% |
| Test loss     | 244.65 |

The high test loss relative to accuracy suggests the model's probability outputs are poorly calibrated (overconfident on wrong predictions), which can happen with extended training without a learning rate schedule or early stopping.

## Usage

Open and run `deep_net.ipynb` in Jupyter:

```bash
jupyter notebook deep_net.ipynb
```

**Dependencies:** `tensorflow`, `numpy`, `matplotlib`
