# Deep Learning: MNIST Digit Classification

A feedforward neural network trained on the MNIST handwritten digit dataset using TensorFlow/Keras.
This project demonstrates a critical architectural lesson: **where you place Dropout determines whether your entire network learns or memorizes.**

---

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

---

## The Core Lesson: Dropout Placement

### What Dropout Does

Dropout randomly sets a fraction of neuron outputs to zero during each training step.
This prevents any single neuron from becoming a specialist that memorizes a specific training pattern — forcing the network to learn **redundant, generalizable features** instead.

---

### Bad Architecture — Dropout Only at the End

```
Input (784,)
    ↓
Dense(64, ReLU)      ← NO regularization — free to memorize
    ↓
BatchNormalization
    ↓
Dense(64, ReLU)      ← receives corrupt input from Layer 1
    ↓
BatchNormalization
    ↓
Dropout(0.2)         ← too late, damage already done
    ↓
Dense(10, Softmax)
```

**Results: Test accuracy = 77.55% | Test loss = 244.65**

#### Why Layer 1 Without Dropout Breaks Everything

Layer 1 is the **feature extractor** — it is the first thing that touches raw pixel data and decides what patterns matter (edges, curves, strokes). It has **50,240 parameters**, by far the largest layer in the network.

Without Dropout, Layer 1 can freely memorize arbitrary pixel combinations specific to the training set. There is no noise, no pressure to generalize, no constraint. It learns to recognize training examples, not digit concepts.

This creates a cascading failure through the rest of the network:

**Layer 1 memorizes → Layer 2 is trained on memorized features → Layer 2 inherits and reinforces that memorization → Output layer receives a corrupt representation and cannot generalize**

Dropout at the end only disrupts the connection between Layer 2 and the output. But the rot starts at Layer 1 — by the time signals reach the dropout layer, they are already built on memorized, non-generalizable representations. The regularization is too shallow to matter.

The **loss of 244.65** (normal is 0.05–0.5) tells the full story. The model had become extremely overconfident in wrong predictions. Accuracy looks "okay" at 77% because `argmax` only checks which class has the highest probability — not how confident the model is. The loss punishes that overconfidence catastrophically: `-log(~0.000001) ≈ 13.8` per wrong sample, summed across 10,000 examples.

---

### Fixed Architecture — Dropout After Every Hidden Layer

```
Input (784,)
    ↓
Dense(64, ReLU)
    ↓
BatchNormalization
    ↓
Dropout(0.3)         ← regularizes Layer 1 immediately
    ↓
Dense(64, ReLU)      ← receives noisy, robust features from Layer 1
    ↓
BatchNormalization
    ↓
Dropout(0.3)         ← regularizes Layer 2
    ↓
Dense(10, Softmax)
```

**Results: Test accuracy = 97.79% | Test loss = 0.08**

#### Why This Works

**Layer 1 with Dropout(0.3):**
30% of Layer 1's neurons are randomly silenced each training step. No single neuron can monopolize a pattern. The layer is forced to distribute feature detection across many neurons redundantly — what one neuron learns, several others partially learn too. These are real digit features (strokes, curves, loops), not training-set artifacts.

**Layer 2 receives cleaner signals:**
Because Layer 1 now passes robust, generalized features forward, Layer 2 starts from a solid foundation. Its job becomes learning higher-level combinations of those features (e.g., "closed loop on top + vertical stroke = 9"). It can do this effectively because its inputs are not poisoned by memorized noise.

**Dropout on Layer 2 compounds the effect:**
Layer 2's dropout forces the output layer to handle incomplete, noisy combinations of higher-level features. The output layer cannot develop brittle dependencies on specific neuron patterns — it must learn a robust decision boundary. This is why the softmax probabilities stay calibrated (loss = 0.08) rather than exploding.

---

## Side-by-Side Comparison

| | Bad Architecture | Fixed Architecture |
|---|---|---|
| Layer 1 regularization | None | Dropout(0.3) |
| Layer 2 regularization | Dropout(0.2) at the end | Dropout(0.3) after layer |
| Layer 1 behavior | Memorizes training pixels | Learns generalizable features |
| Layer 2 input quality | Corrupt, overfitted | Robust, generalizable |
| Test accuracy | 77.55% | 97.79% |
| Test loss | 244.65 | 0.08 |

The network capacity (55,562 parameters) and optimizer (Nadam) were identical in both cases. **The only difference was dropout placement.** This shows the architecture is not about how many parameters you have — it is about whether regularization is applied where memorization is most likely to start.

---

## Optimizer: Nadam

The model uses **Nadam** (Nesterov-accelerated Adaptive Moment Estimation).

| Optimizer | Description | Pros | Cons |
|-----------|-------------|------|------|
| **SGD** | Fixed learning rate × gradient | Simple, memory efficient | Slow, sensitive to LR |
| **SGD + Momentum** | Accumulates velocity to dampen oscillations | Faster than SGD | Careful LR tuning needed |
| **RMSProp** | Scales LR per-parameter via squared gradient average | Good for non-stationary problems | No momentum |
| **Adam** | Momentum + RMSProp combined | Fast convergence, robust defaults | Can generalize slightly worse |
| **Nadam** (used) | Adam + Nesterov look-ahead momentum | Smoother, more accurate updates | Slightly more compute |

---

## Training Config

| Hyperparameter | Value |
|----------------|-------|
| Loss function  | Categorical cross-entropy |
| Optimizer      | Nadam |
| Batch size     | 128 |
| Epochs         | 128 |

---

## Usage

```bash
jupyter notebook deep_net.ipynb
```

**Dependencies:** `tensorflow`, `numpy`, `matplotlib`
