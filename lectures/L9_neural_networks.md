# Lecture 8: Neural Networks
**MSA220/MVE441 Statistical Learning for Big Data** — Rebecka Jörnsten, Mathematical Sciences, 20th April 2026

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- Logistic Regression as a Network Model
- Multi-class Logistic Regression (Softmax)
- Neural Network Architecture
- Backpropagation
- Training Procedure
- Tuning and Regularization
- Autoencoders, kPCA, and PCA
- Neural Network Classifier as Autoencoder + Classification Layer
- Comparison: NN, KRR, and SVMs

---

## 📝 Summary

This lecture introduces neural networks by building up from logistic regression, reframing it as a single-node network. It extends this to multi-class classification via softmax and then stacks multiple layers to form a full neural network. The key training mechanism — backpropagation — is derived from the chain rule and applied backward through the layers. The lecture also covers tuning strategies (early stopping, dropout, weight decay, SGD), connections to autoencoders and PCA, and a concluding comparison of neural networks against kernel ridge regression and SVMs.

---

## 🎯 Learning Goals

- Understand how logistic regression maps onto a single-layer network and how stacking layers generalizes this to a full neural network.
- Derive and interpret backpropagation as an application of the chain rule across layers.
- Know the main hyperparameters and regularization strategies for neural networks.
- Understand the relationship between autoencoders, kPCA, and PCA.
- Be able to compare the practical tradeoffs between NNs, KRR, and SVMs.

---

## 📚 Concepts

### Recap: Logistic Regression as a Network

**Summary:** Logistic regression can be drawn as a one-layer network: input features $x_1, \dots, x_n$ are weighted by coefficients $w_1, \dots, w_n$ (with a bias $w_0$ from a constant input $x_0 = 1$), summed to form the linear predictor $z = \sum_{i=0}^n w_i x_i$, and then passed through the sigmoid activation $\sigma(z) = \frac{1}{1 + e^{-z}}$ to produce an output probability. The gradient descent update for the binary cross-entropy loss simplifies (because the sigmoid derivative cancels) to $\beta_j(t+1) = \beta_j(t) + \eta \sum_i (y_i - f_i) x_{ij}$, which is the residual-weighted update familiar from linear regression.

---

### Multi-class Logistic Regression (Softmax)

**Why it matters:** Many classification problems have more than two classes. Softmax generalizes binary logistic regression to $K$ classes and is used as the output activation in virtually all multi-class neural networks.

**Intuition:** Instead of one output node, there are $K$ output nodes — one per class. Each node computes a class score, and softmax converts these scores into a valid probability distribution by exponentiating and normalizing.

**Prerequisites:**
- Binary logistic regression and the sigmoid function
- Cross-entropy loss

**How it works:**

Each class $k$ has its own weight vector $\beta^k$, forming a linear predictor $z_{ik} = \sum_j \beta^k_j x_{ij}$. The softmax function maps these to class probabilities:

$$
f_{ik} = \frac{e^{z_{ik}}}{\sum_l e^{z_{il}}}
$$

The loss is the cross-entropy:

$$
\mathcal{L} = -\sum_i \sum_k y_{ik} \log(f_{ik}), \quad y_{ik} = \mathbf{1}[\text{obs } i \in \text{class } k]
$$

Taking derivatives requires care because all $f_{il}$ are coupled through the softmax denominator. After accounting for both $\frac{\partial f_{ik}}{\partial z_{ik}} = f_{ik}(1-f_{ik})$ and $\frac{\partial f_{il}}{\partial z_{ik}} = -f_{il}f_{ik}$ for $l \neq k$, and using $\sum_l y_{il} = 1$, terms cancel and the update is:

$$
\beta^k_j(t+1) = \beta^k_j(t) + \eta \sum_i (y_{ik} - f_{ik}) x_{ij}
$$

This has the same intuitive form as the binary case: update proportional to prediction error times input.

**Key Equations:**

$$
f_{ik} = \frac{e^{z_{ik}}}{\sum_l e^{z_{il}}}
$$

$$
\frac{\partial \mathcal{L}}{\partial \beta^k_j} = -\sum_i (y_{ik} - f_{ik}) x_{ij}
$$

Where:
- $z_{ik}$ = linear predictor for observation $i$, class $k$
- $f_{ik}$ = predicted probability for class $k$
- $y_{ik}$ = binary indicator (1 if observation $i$ belongs to class $k$)

**Python Implementation:**

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=500)
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)
```

⚠️ **Theory vs. Practice:** By default, `LogisticRegression` applies L2 regularization (`C=1.0`). This is not the unregularized GD update derived in the lecture. Setting `C` to a very large value approximates the unregularized case, but sklearn does not guarantee convergence for unregularized multinomial problems when classes are linearly separable — the weights will diverge. Always set `max_iter` explicitly; the default of 100 is insufficient for many real datasets.

---

### Neural Network Architecture

**Why it matters:** Stacking multiple layers of sigmoid (or other nonlinear) units creates a model that can approximate arbitrarily complex decision boundaries — something a single logistic layer cannot do.

**Intuition:** Each hidden layer applies a nonlinear transformation to its input, progressively building up more abstract feature representations. The final layer then classifies in this learned feature space. It is useful to think of the hidden layers as learning a data-adaptive kernel, with the last layer performing classification in that kernel space.

**Prerequisites:**
- Logistic regression as a single-layer network
- Matrix multiplication and chain rule

**How it works:**

Denote layers as $Z^l$, $l = 0, 1, \dots, L$, where $Z^0 = X$ (input). Between layers, weights $W^l$ project the previous layer's output $f^{l-1}$ to the next layer's pre-activations $Z^l = W^l f^{l-1}$. An activation function $g^l$ is then applied elementwise: $f^l = g^l(Z^l)$. Common choices are:

- **Sigmoid**: $\sigma(z) = \frac{1}{1+e^{-z}}$ — smooth but prone to vanishing gradients
- **Tanh**: $\tanh(z)$ — zero-centered sigmoid variant
- **ReLU**: $\max(0, z)$ — most widely used; avoids vanishing gradient for positive inputs

If identity activations are used throughout and MSE is the loss, a network with a bottleneck hidden layer is equivalent to PCA.

**Key Equations:**

$$
Z^l = W^l f^{l-1}, \quad f^l = g^l(Z^l)
$$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Universal approximator (sufficiently wide/deep) | Requires large amounts of data |
| Learns hierarchical representations | Many hyperparameters to tune |
| Can leverage structured weight sharing (CNN, GNN) | Sensitive to initialization |
| State of the art for complex tasks (vision, NLP) | High computational cost |

**Python Implementation:**

```python
import torch
import torch.nn as nn

model = nn.Sequential(
    nn.Linear(input_dim, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes)
)
```

⚠️ **Theory vs. Practice:** PyTorch's `nn.Linear` applies no regularization by default. The lecture derives GD updates for the sigmoid activation where the derivative cancels cleanly — this does **not** happen for ReLU (which is non-differentiable at 0) or other activations. PyTorch handles this via subgradients automatically, but the clean cancellation from the lecture derivation is sigmoid-specific. Do not assume update formulas derived for sigmoid transfer to other activations.

---

### Backpropagation

**Why it matters:** Backpropagation is the algorithm that makes training deep networks tractable. Without it, computing gradients across many layers would require recomputing from scratch for each weight.

**Intuition:** Apply the chain rule backward through the network. The gradient at each layer is the gradient from the layer ahead, multiplied by the local derivative. Errors propagate from output to input, with each layer's gradient depending on the layer in front of it.

**Prerequisites:**
- Chain rule of calculus
- Matrix calculus basics

**How it works:**

For a 1-hidden-layer network with weights $W^1$ (input → hidden) and $W^2$ (hidden → output), the gradients are:

$$
\frac{\partial \mathcal{L}}{\partial W^2_{jk}} = -\sum_i (y_{ik} - f^2_{ik}) \cdot (g^2_k(W^2 f^1))' \cdot f^1_{ij}
$$

$$
\frac{\partial \mathcal{L}}{\partial W^1_{lj}} = -\sum_i \sum_k \underbrace{(y_{ik} - f^2_{ik})(g^2_k(W^2 f^1))'}_{\delta_{ik}} \cdot W^2_{jk} \cdot (g^1_j(W^1 X))' \cdot x_{il}
$$

The inner-layer error signal $s_{ij}$ is:

$$
s_{ij} = (g^1_j(W^1 X))' \sum_k \delta_{ik} W^2_{jk}
$$

The key insight: the gradient at $W^1$ requires the errors $\delta_{ik}$ computed at $W^2$. Earlier layers depend on later layers' errors — hence "backward" propagation. This generalizes identically to $L > 1$ layers.

**Key Equations:**

$$
\delta_{ik} = (y_{ik} - f^2_{ik}) \cdot (g^2_k)' \quad \text{(output layer error)}
$$

$$
s_{ij} = (g^1_j)' \cdot \sum_k \delta_{ik} W^2_{jk} \quad \text{(hidden layer error)}
$$

---

### Training Procedure

**Why it matters:** Neural networks are trained iteratively. Understanding the steps clarifies where things can go wrong (divergence, overfitting, local optima).

**Intuition:** Start with small random weights (so the network behaves nearly linearly initially), push data forward through the network to get predictions, compute gradients by pushing errors backward, then update weights. Repeat.

**Prerequisites:**
- Gradient descent
- Backpropagation

**How it works:**

1. **Initialize** weights near 0 with small random values (ensures near-linear initial behavior).
2. **Forward pass**: propagate the input $X$ through all layers to obtain output $f^L_{ik}$.
3. **Backward pass**: compute the loss and propagate errors back through all layers using backpropagation.
4. **Update**: apply the gradient descent step to all weight matrices $W^l$.
5. **Iterate** from step 2 until convergence or a stopping criterion is met.

---

### Tuning and Regularization

**Why it matters:** Neural networks have enormous capacity and will overfit without regularization. Getting the right training configuration is often as important as the architecture itself.

**Intuition:** Most regularization strategies either limit the number of training steps, add noise to the optimization, or penalize the magnitude or complexity of the learned weights.

**Prerequisites:**
- Gradient descent and overfitting
- Ridge/Lasso regularization

**How it works:**

**Learning rate $\eta$:** If too large, gradient descent diverges. If too small, training is slow and may stall in shallow local optima.

**Early stopping:** Monitor validation loss and halt training when it starts increasing. Directly analogous to early stopping in gradient boosting.

**Stochastic Gradient Descent (SGD):** Compute gradients on random mini-batches instead of the full dataset. One full pass through all mini-batches = one *epoch*. Smaller batches inject more noise into training, which can escape local optima and acts as a regularizer.

**Weight decay:** Add an L2 (ridge) or L1 (lasso) penalty on all network weights to the loss function.

**Dropout:** During each training iteration, randomly zero out the output of a subset of nodes. This prevents the network from over-relying on any individual unit and forces redundant representations.

**Smoothness penalties:** Penalize large $\frac{\partial f}{\partial x}$ (sensitivity of outputs to inputs) or large $\frac{\partial f}{\partial W}$ (sensitivity to parameters). Computing these requires *double backpropagation* — two backward passes.

**Advanced optimizers:** Methods like **Adam** and **SGD with momentum** maintain a running average of past gradients, introducing inertia that smooths the optimization trajectory.

**Architecture:** Width (nodes per layer) and depth (number of layers) are both tuning parameters. Wide, shallow networks are easier to interpret and can approximate complex functions; narrow, deep networks build up complex representations through sequential nonlinear transformations.

**Strengths and Weaknesses:**

| Regularization | Effect | Cost |
|----------------|--------|------|
| Early stopping | Prevents overfitting | Requires held-out validation set |
| Dropout | Prevents co-adaptation of nodes | Slower convergence |
| Weight decay | Shrinks weights | Adds hyperparameter |
| Small batch SGD | Escapes local optima, implicit regularization | Noisier training |

---

### Comparison: Autoencoders, kPCA, and PCA

Autoencoders, kernel PCA, and standard PCA all address dimensionality reduction, but differ in their linearity assumptions, computational demands, and flexibility.

| Property | PCA | kPCA | Autoencoder |
|----------|-----|------|-------------|
| Linearity | Linear only | Non-linear (via kernel) | Non-linear |
| Reconstruction | Explicit (projection) | Not always explicit | Explicit (encoder + decoder) |
| Kernel | Fixed (linear) | Fixed (user-specified) | Learned from data |
| Overfitting risk | None | Low | High without regularization |
| Computational cost | Low | Moderate ($O(n^2)$–$O(n^3)$) | High |
| Hyperparameter tuning | Minimal | Kernel choice + bandwidth | Architecture, activations, lr, etc. |

- PCA is efficient and interpretable but cannot capture non-linear structure.
- kPCA captures non-linear patterns but requires choosing a kernel and scales poorly with $n$.
- Autoencoders are the most flexible — learning a data-adaptive representation — but are computationally expensive and require careful regularization to avoid overfitting.
- An autoencoder with linear activations trained with MSE recovers PCA (when the hidden layer is smaller than the input).

---

### Neural Network Classifier as Autoencoder + Classification Layer

**Why it matters:** This architecture unifies unsupervised feature learning (dimensionality reduction via the encoder) with supervised classification, and makes explicit the connection between neural networks and kernel-based methods.

**Intuition:** The encoder compresses the input to a low-dimensional bottleneck. A decoder then reconstructs the input (unsupervised objective), while a separate classification head maps the bottleneck representation to class labels (supervised objective). Training both simultaneously means the representation is shaped by both reconstruction and classification goals.

**Prerequisites:**
- Autoencoders
- Multi-class logistic regression

**How it works:**

- **Encoder**: maps input $X \to$ bottleneck representation $Z$ (lower-dimensional).
- **Decoder**: maps $Z \to \hat{X}$ to reconstruct the input (reconstruction loss).
- **Classification layer**: maps $Z \to \hat{y}$ (cross-entropy loss).
- Both losses are combined and the full network is trained end-to-end.

The encoder's learned transformation plays the role of a data-adaptive kernel: it is analogous to kernel logistic regression, but rather than using a fixed kernel (as in KRR or SVM), the kernel is learned from the data during training.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Learns compact, task-relevant representations | Complex architecture to tune |
| Combines supervised and unsupervised learning | Risk of overfitting if not regularized |
| Data-adaptive kernel (more flexible than KRR/SVM) | Requires more data and compute than kernel methods |

---

### Comparison: NN, KRR, and SVMs

All three methods can model nonlinear decision boundaries, but differ substantially in how they achieve this and their practical trade-offs.

| Property | Neural Network | KRR | SVM |
|----------|---------------|-----|-----|
| Kernel | Learned (data-adaptive) | Fixed (user-specified) | Fixed (user-specified) |
| Loss | Cross-entropy / MSE | Squared loss | Hinge loss |
| Computational cost | High | Moderate | Moderate |
| Data requirements | High | Moderate | Moderate |
| Robustness to mislabeling | Moderate | Moderate | High (hinge loss is robust) |
| Sensitivity to initialization | High | None | None |
| Hyperparameters | Many | Few (kernel + $\lambda$) | Few (kernel + $C$) |
| Interpretability | Low | Moderate | Moderate |

- For small-to-moderate datasets, KRR and SVMs are often comparable to NNs in performance with far lower computational cost and fewer hyperparameters.
- NNs have an edge for complex, large-scale problems when sufficient data and compute are available.
- SVMs are more robust to noisy labels due to the hinge loss, but all three can be regularized to handle label noise.
- **Do not underestimate classical kernel methods.** They remain highly competitive and are far easier to deploy and interpret.

---

## ✅ Key Takeaways

- Logistic regression is a degenerate (single-node) neural network; stacking layers extends it to model arbitrary nonlinear decision boundaries.
- Backpropagation is the chain rule applied backward through the network: each layer's gradient depends on the errors from the layer in front of it.
- Training requires careful management of the learning rate, number of epochs, and regularization (early stopping, dropout, weight decay, SGD batch size).
- A neural network with linear activations and MSE loss trained with a bottleneck layer is equivalent to PCA.
- The hidden layers of a neural network can be thought of as learning a data-adaptive kernel, making the final classification layer analogous to kernel logistic regression.
- Classical kernel methods (KRR, SVMs) are computationally cheaper, require fewer hyperparameters, and remain strong baselines — don't default to NNs when data is limited.