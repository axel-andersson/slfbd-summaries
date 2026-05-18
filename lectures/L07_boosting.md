# Lecture 7: Boosting

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- Ensemble Methods Overview
- Gradient Descent (Recap)
- Functional Gradient Descent
- Kernel Ridge Regression (KRR)
- KRR with Gradient Descent
- Kernel Logistic Regression with Gradient Descent
- Forward Fitting Schemes (Forward Selection & Forward Stagewise Regression)
- AdaBoost
- Gradient Boosting Machines (GBM)

---

## 📝 Summary

This lecture introduces boosting as a third ensemble strategy, contrasted with bagging and Random Forests. Starting from a functional gradient descent perspective, the lecture builds intuition for iterative model-fitting by connecting kernel ridge regression, logistic regression, and forward stagewise methods. It then derives AdaBoost from first principles using an exponential loss, and culminates in the Gradient Boosting Machine (GBM) framework — a flexible algorithm that can use any differentiable loss function and any weak learner (typically shallow trees) to build powerful predictive models.

---

## 🎯 Learning Goals

- Understand how boosting differs from bagging in addressing the bias-variance tradeoff.
- Derive and interpret the AdaBoost algorithm as forward stagewise regression under an exponential loss.
- Understand the connection between functional gradient descent and gradient boosting.
- Apply the GBM algorithm with different loss functions and weak learners.
- Know when to stop adding model components (early stopping, validation loss, model selection criteria).

---

## 📚 Concepts

### Recap: Ensemble Methods — Bagging and Random Forests

**Summary:** The bias-variance tradeoff governs generalization: a biased model makes systematic errors, while a high-variance model overfits the training data (wiggly boundaries, isolated prediction regions). Bagging generates many versions of the same model from bootstrap samples of training data; averaging cancels out the noisy predictions of individual overfitted models. Random Forests apply bagging to deep trees with additional randomness in the allowed splits at each node, further reducing variance. Both methods combine component predictions via a majority vote or average, with equal weighting.

---

### Boosting — Core Idea

**Why it matters:** Boosting offers a fundamentally different approach to reducing error — instead of averaging independently-trained high-variance models, it builds a sequence of low-variance (biased) models that are optimally combined to reduce bias.

**Intuition:** Rather than training many models in parallel and averaging, boosting trains models sequentially. Each new model focuses on what the current ensemble gets wrong. The contributions of the weak learners are optimally weighted (not a simple majority vote), so the ensemble progressively improves.

**Prerequisites:**
- Bias-variance tradeoff
- Gradient descent optimization
- Decision trees / classification

**How it works:**
Each iteration fits a "weak learner" — a simple model that alone suffers from high bias — to the errors of the current model. The weak learner is then added to the ensemble with an optimal weight $\alpha_t$. Over many iterations, the weighted sum of weak learners reduces bias while keeping variance low (controlled by early stopping or the learning rate).

The key distinction from bagging is **optimal weighting**: boosting solves for $\alpha_t$ that minimizes the loss at each step.

---

### Recap: Gradient Descent

**Summary:** Gradient descent minimizes a loss $L(\theta)$ by iteratively updating parameters in the direction of the negative gradient: $\theta_{t+1} = \theta_t - \eta \nabla_\theta L$. The learning rate $\eta$ controls step size — too small means slow convergence, too large risks divergence. This lecture extends gradient descent to the space of functions (functional gradient descent), which is the theoretical backbone of boosting.

---

### Functional Gradient Descent

**Why it matters:** Estimating a function $f$ from noisy data is an infinite-dimensional problem. Applying gradient descent directly in function space reveals the mechanism behind boosting — and explains why naive application leads to overfitting.

**Intuition:** Instead of optimizing over a finite parameter vector, we optimize over the function values $f(x_i)$ at each training point. The gradient at each point tells us how far the current estimate is from the data. This is equivalent to fitting the residuals iteratively.

**Prerequisites:**
- Gradient descent
- L2 (squared error) loss

**How it works:**

Given the L2 loss $L = \sum_i (y_i - f(x_i))^2$, the pointwise gradient with respect to $f$ is:

$$\nabla_f L = -(y_i - f_t(x_i))$$

The update rule is:

$$f_{t+1}(x_i) = f_t(x_i) + \eta\,(y_i - f_t(x_i))$$

Starting from $f_0(x_i) = 0$, each step moves the current estimate toward the observed value $y_i$. After enough iterations, the fit will pass through every training point perfectly — but this means **overfitting**: the function interpolates noise rather than the true signal.

**Two critical problems with naive functional gradient descent:**

1. **No generalization**: predictions are only defined at training points — there is no model that generalizes to new $x$.
2. **No overfitting control**: pointwise estimation cannot be regularized, and the function will eventually memorize the training data.

---

### Kernel Ridge Regression (KRR)

**Why it matters:** KRR resolves both problems of naive functional gradient descent by imposing a smoothness constraint on $f$ through a kernel and an explicit penalty on the function norm.

**Intuition:** Functions that are close in input space should have similar values. The kernel $k(x, x')$ encodes this similarity. Penalizing the RKHS norm of $f$ prevents the function from fitting noise — it enforces smooth interpolation between training points.

**Prerequisites:**
- Kernel PCA
- Ridge regression / dual formulation
- Reproducing Kernel Hilbert Spaces (RKHS)

**How it works:**

The prediction function is parameterized as:

$$f(x) = \sum_{l=1}^n \hat{\alpha}_l \, k(x_l, x)$$

where the dual variables $\hat{\alpha}$ are obtained from:

$$\hat{\alpha} = (\mathbf{K} + \lambda \mathbf{I}_n)^{-1} \mathbf{y}$$

and $\mathbf{K}$ is the $n \times n$ kernel (Gram) matrix with entries $K_{ij} = k(x_i, x_j)$.

The function space is:

$$\mathcal{G} = \left\{ \sum_i \alpha_i k(x_i, \cdot),\; \alpha \in \mathbb{R}^n \right\}$$

The RKHS inner product allows evaluating $f$ at any point $x$ via:

$$f(x) = \langle f, k(x, \cdot) \rangle = \sum_i \alpha_i k(x_i, x)$$

The smoothness penalty (RKHS norm squared) is:

$$\|f\|^2 = \alpha^\top \mathbf{K} \alpha$$

This is the term penalized in KRR, preventing overfitting.

**Key Equations:**

Dual solution:
$$\hat{\alpha} = (\mathbf{K} + \lambda \mathbf{I})^{-1} \mathbf{y}$$

Prediction for new point $x^*$:
$$\hat{f}(x^*) = \sum_{l=1}^n \hat{\alpha}_l \, k(x_l, x^*)$$

Where:
- $\mathbf{K}$ = kernel (Gram) matrix
- $\lambda$ = regularization strength (controls smoothness)
- $\alpha$ = dual coefficients (one per training point)

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Closed-form solution | $O(n^3)$ inversion cost — doesn't scale to large $n$ |
| Flexible via kernel choice | Kernel and $\lambda$ must be chosen/tuned |
| Principled smoothness control | Dense solution (all training points used) |

**Python Implementation:**

```python
from sklearn.kernel_ridge import KernelRidge
import numpy as np

krr = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
krr.fit(X_train, y_train)
y_pred = krr.predict(X_test)
```

⚠️ **Theory vs. Practice:** sklearn's `KernelRidge` uses `alpha` for $\lambda$ (the regularization coefficient), **not** the dual weights $\alpha$ in the lecture notation — these are two different things. The `gamma` parameter controls the RBF kernel width, not a learning rate. The default `alpha=1.0` is often far too strong; always tune it. The `kernel='rbf'` default is not the linear kernel — if you want standard ridge regression, you must set `kernel='linear'`.

---

### KRR with Gradient Descent

**Why it matters:** When the loss is non-quadratic (e.g., logistic) or the penalty is non-standard (e.g., sparse), the closed-form KRR solution doesn't apply. Gradient descent in the dual space generalizes KRR and previews GBM and neural network estimation.

**How it works:**

Minimize $L = \frac{1}{2}\|\mathbf{y} - \mathbf{K}\alpha\|^2 + \lambda \alpha^\top \mathbf{K} \alpha$ via gradient descent on $\alpha$:

1. Initialize $\alpha_t = \mathbf{0}$
2. Compute gradient: $\text{grad} = -\mathbf{K}(\mathbf{y} - \mathbf{K}\alpha) + \lambda \mathbf{K}\alpha$
3. Update: $\alpha_{t+1} = \alpha_t - \eta\, \text{grad}$
4. Update function: $f_{t+1} = \mathbf{K}\alpha_{t+1}$

For out-of-sample prediction at $x^*$: compute the rectangular kernel vector $k(x^*, x_i)$ for all training $x_i$, then $\hat{f}(x^*) = K(x^*, X)\alpha$.

---

### Kernel Logistic Regression with Gradient Descent

**Why it matters:** Demonstrates how functional gradient descent extends beyond regression to classification, and previews the chain rule structure used in GBM.

**How it works:**

The logistic loss (log-likelihood) is:

$$\ell = \sum_i y_i \log(f_i) + (1 - y_i)\log(1 - f_i)$$

where $f_i = \sigma(z_i)$, $z_i = (K\alpha)_i$, and $\sigma(z) = \frac{e^z}{1 + e^z}$.

By the chain rule:

$$\frac{d\ell}{d\alpha} = \frac{d\ell}{df} \cdot \frac{df}{dz} \cdot \frac{dz}{d\alpha}$$

A key simplification: the derivative of the sigmoid $\frac{df}{dz} = f(z)(1-f(z))$ cancels with the denominator in $\frac{d\ell}{df} = \frac{y_i - f_i}{f_i(1-f_i)}$, yielding:

$$\frac{d\ell}{d\alpha} = \mathbf{K}(\mathbf{y} - f)$$

With ridge penalty:

$$\frac{d\ell}{d\alpha} = \mathbf{K}(\mathbf{y} - (\mathbf{K} + \lambda\mathbf{I})\alpha)$$

**Algorithm:**
1. Initialize $\alpha_t = \mathbf{0}$
2. Compute: $\text{grad} = \mathbf{K}(\mathbf{y} - (\mathbf{K} + \lambda\mathbf{I})\alpha)$
3. Update: $\alpha_{t+1} = \alpha_t + \eta\, \text{grad}$ (note: **+** since we are maximizing the likelihood)
4. Update: $f_{t+1} = \sigma(\mathbf{K}\alpha_{t+1})$

Out-of-sample: $\hat{f}(x^*) = \sigma(K(x^*, X)\alpha)$

---

### Forward Fitting Schemes

**Why it matters:** Forward fitting methods provide controlled, interpretable ways to grow a model one component at a time, naturally preventing overfitting through early stopping.

**Intuition:** Rather than fitting all parameters simultaneously, you add one feature (or one model component) at a time, stopping when adding more components no longer improves validation performance.

**Prerequisites:**
- Linear regression
- Residuals and correlation
- Model selection criteria (AIC, BIC, cross-validation)

**How it works — Forward Selection:**

1. Start with an empty model.
2. At each step, add the predictor $j^*$ that most reduces the loss:
$$j^* = \arg\min_{j \in \{1, \ldots, p\}} \text{Loss}(\text{model}_t \cup \{j\})$$
3. Repeat until a stopping criterion is met (validation loss, AIC, etc.).

Note: **all** coefficient estimates are updated at every iteration.

**How it works — Forward Stagewise Regression:**

1. Initialize $\beta^{(0)} = \mathbf{0}$, compute residuals $r = y - X\beta_t$.
2. Find the feature most correlated with residuals:
$$j^* = \arg\max_j |\text{corr}(x_j, r)|$$
3. Update only that coefficient by a small step:
$$\beta_{j^*}(t) = \beta_{j^*}(t) + \eta \cdot \text{sign}(\text{corr}(x_{j^*}, r))$$
4. Repeat until stopping criterion.

This is more conservative than forward selection — only one coefficient changes at a time, by a small amount. With an optimized step size $\eta$ (traveling as far as possible until another feature becomes more correlated), this yields the LARS algorithm, a highly sparse regression method closely related to Lasso.

**Key Equations:**

Residuals: $r = y - X\beta_t$

Correlation-based update: $\beta_{j^*} \leftarrow \beta_{j^*} + \eta \cdot \text{sign}(\text{corr}(x_{j^*}, r))$

---

### AdaBoost

**Why it matters:** AdaBoost is the foundational boosting algorithm. It re-weights training observations to force successive classifiers to focus on examples the current ensemble misclassifies, progressively reducing the overall error.

**Intuition:** After each round, examples that were misclassified receive higher weight — the next classifier is therefore trained harder on those hard cases. The final prediction is a weighted vote over all classifiers. This is equivalent to forward stagewise regression under an **exponential loss**.

**Prerequisites:**
- Binary classification
- Weighted training
- Forward stagewise regression

**How it works:**

For $y \in \{-1, +1\}$ and a sequence of classifiers $f_t(x) \in \{-1, +1\}$:

1. Initialize observation weights $w_i = 1/n$
2. Fit classifier $f_t$ to data using weights $w_i$
3. Compute weighted error rate: $\text{err}_t = \sum_i w_i \cdot \mathbf{1}[y_i \neq f_t(x_i)]$
4. Compute optimal weight: $\alpha_t = \frac{1}{2}\log\!\left(\frac{1 - \text{err}_t}{\text{err}_t}\right)$
5. Update observation weights: $w_{i,t+1} = w_{i,t} \exp(-\alpha_t y_i f_t(x_i)) / Z$ (where $Z$ normalizes weights to sum to 1)
6. Update ensemble: $F_{t+1} = F_t + \alpha_t f_t$
7. Predict: $\hat{y} = \text{sign}(F_{t+1})$

**Key Equations:**

Exponential loss: $L(y, f) = \exp(-y \cdot f)$

Optimal component weight:
$$\alpha_t = \frac{1}{2}\log\!\left(\frac{1 - \text{err}_t}{\text{err}_t}\right)$$

Observation weight update:
$$w_{i,t+1} \propto \exp\!\left(-y_i \sum_{t'=1}^{t} \alpha_{t'} f_{t'}(x_i)\right) = \exp(-y_i F_t(x_i))$$

This shows that **AdaBoost IS forward stagewise regression** under the exponential loss: the observation weights $w_{i,t}$ equal (up to normalization) $\exp(-y_i F_{t-1}(x_i))$, which is exactly the loss gradient weighting.

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Adaptive focus on hard examples | Exponential loss is sensitive to outliers and label noise |
| Optimal weighting of weak learners | Only applicable to classification ($y \in \{-1,1\}$) in basic form |
| Theoretically grounded (exponential loss) | Requires selecting a weak learner and tuning iterations |

**Python Implementation:**

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),  # "decision stump"
    n_estimators=100,
    learning_rate=1.0,
    algorithm='SAMME'
)
ada.fit(X_train, y_train)
y_pred = ada.predict(X_test)
```

⚠️ **Theory vs. Practice:** As of sklearn 1.2+, the default `algorithm='SAMME.R'` (real-valued AdaBoost) was deprecated and `'SAMME'` is required for the discrete version matching the lecture. Using `algorithm='SAMME.R'` gives a fundamentally different update rule — it uses probability estimates rather than discrete class labels and does **not** match the derivation in the lecture. The `learning_rate` parameter shrinks each $\alpha_t$ but the lecture derives $\alpha_t$ analytically — adding a separate learning rate deviates from theory and you must re-tune `n_estimators` accordingly.

---

### Gradient Boosting Machines (GBM)

**Why it matters:** GBM generalizes AdaBoost to any differentiable loss function and any parameterized weak learner, making it one of the most powerful and widely used ML algorithms in practice.

**Intuition:** Instead of using observation weights (as in AdaBoost), GBM computes **pseudo-residuals** — the negative gradient of the loss with respect to the current model predictions. A weak learner (typically a shallow tree) is then fitted to these pseudo-residuals. This is functional gradient descent, but with the gradient approximated by a parameterized model so that predictions generalize to new data.

**Prerequisites:**
- Functional gradient descent
- AdaBoost
- Decision trees

**How it works:**

**Key Equations:**

Pseudo-residuals at step $t$:
$$r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\bigg|_t$$

For L2 loss ($L = \frac{1}{2}(y - F)^2$): $r_i = y_i - F_t(x_i)$ (ordinary residuals)

For logistic loss: $r_i = y_i - \sigma(F_t(x_i))$ (after chain rule cancellation)

Optimal step size at each iteration:
$$\alpha^* = \arg\min_\alpha L(y,\, F_{t-1}(x) + \alpha f_t(x, \theta^*))$$

Model update:
$$F_t(x) = F_{t-1}(x) + \alpha^* f_t(x, \theta^*)$$

**Full Algorithm:**

1. Initialize $F_0 = 0$ everywhere
2. For each iteration $t$:
   - Compute pseudo-residuals $r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\big|_t$
   - Fit a weak learner $f_t(x, \theta)$ to the pseudo-residuals (e.g., a shallow tree)
   - Find optimal step: $\alpha^* = \arg\min_\alpha L(y,\, F_{t-1} + \alpha f_t)$
   - Update: $F_t(x) = F_{t-1}(x) + \alpha^* f_t(x, \theta^*)$
3. Stop based on validation loss, cross-validation, or a fixed $T$

**Why shallow trees?** A shallow tree (depth 2–8) is a biased but low-variance weak learner. It can generalize to unseen $x$ (unlike pointwise functional gradient descent), and fitting it to pseudo-residuals is fast. By using trees, GBM also inherits tree-based advantages: automatic handling of mixed feature types, no feature scaling needed.

**When to stop?** Use a held-out validation set and stop when validation loss stops improving (early stopping). You can also use cross-validation or information criteria. Adding too many trees without early stopping leads to overfitting — the model starts fitting the noise in the pseudo-residuals.

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Works with any differentiable loss | Sequential — cannot be parallelized like bagging |
| Flexible weak learner choice | Sensitive to hyperparameters ($T$, tree depth, $\eta$) |
| State-of-the-art on tabular data | Slower to train than Random Forests |
| Built-in feature importance | Prone to overfitting without careful tuning |

**Python Implementation:**

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

# Regression (L2 loss by default)
gbm = GradientBoostingRegressor(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=3,
    subsample=1.0
)
gbm.fit(X_train, y_train)
y_pred = gbm.predict(X_test)
```

For large datasets, prefer `HistGradientBoostingRegressor` (histogram-based, much faster):

```python
from sklearn.ensemble import HistGradientBoostingRegressor

hgbm = HistGradientBoostingRegressor(
    max_iter=200,
    learning_rate=0.1,
    max_depth=3,
    early_stopping=True,
    validation_fraction=0.1
)
hgbm.fit(X_train, y_train)
```

⚠️ **Theory vs. Practice:** `GradientBoostingClassifier` uses **deviance (log loss) by default**, not exponential loss — this means it is **not** equivalent to AdaBoost even though both are boosting classifiers. The `learning_rate` in sklearn shrinks every tree's contribution by a factor $\eta$ but does **not** optimize $\alpha^*$ analytically — the lecture's step 4 is simplified to a fixed shrinkage. If you use `n_estimators=200` with `learning_rate=1.0`, you will almost certainly overfit: lower `learning_rate` (0.01–0.1) requires more trees but generalizes better. The `subsample < 1.0` parameter introduces **stochastic gradient boosting** (random subsampling of training data per tree) — this is a variance-reduction technique not covered in the lecture, but it often improves performance. Always use early stopping or cross-validation to set `n_estimators`.

---

### Comparison: Ensemble Methods

AdaBoost, GBM, Bagging, and Random Forests all build ensembles, but differ in how components are trained and combined.

| Property | Bagging | Random Forest | AdaBoost | GBM |
|----------|---------|---------------|----------|-----|
| Training order | Parallel | Parallel | Sequential | Sequential |
| Component combination | Equal-weight average/vote | Equal-weight average/vote | Optimal $\alpha_t$ | Optimal $\alpha_t$ |
| Target of each component | Full data (resampled) | Full data (resampled + feature subset) | Weighted data (hard examples) | Pseudo-residuals (loss gradient) |
| Variance reduction | ✓ (main goal) | ✓✓ (extra feature randomness) | Indirect | Via learning rate / early stopping |
| Bias reduction | ✗ | ✗ | ✓ (main goal) | ✓ (main goal) |
| Loss function flexibility | Limited | Limited | Exponential only | Any differentiable loss |
| Sensitivity to noise/outliers | Low | Low | High (exponential loss) | Moderate (depends on loss) |

- Boosting methods (AdaBoost, GBM) primarily reduce **bias** by iteratively correcting errors; bagging methods primarily reduce **variance** by averaging noisy estimates.
- GBM is strictly more general than AdaBoost: AdaBoost is GBM with an exponential loss.
- Random Forests are generally more robust to outliers and less sensitive to hyperparameter choices; GBMs tend to achieve better accuracy with careful tuning.
- For large-scale tabular prediction, GBM implementations like XGBoost and LightGBM dominate empirical benchmarks.

---

## ✅ Key Takeaways

- Boosting builds models sequentially, fitting each new component to the **errors** of the current ensemble — this reduces bias while controlling variance through optimal weighting and early stopping.
- Naive functional gradient descent overfits because there is no model generalization and no smoothness constraint; Kernel Ridge Regression fixes this by working in an RKHS with a smoothness penalty $\lambda \alpha^\top K \alpha$.
- Forward stagewise regression is the linear version of boosting: update one parameter by a small step in the direction of maximum residual correlation.
- AdaBoost is forward stagewise regression under the exponential loss $L(y, f) = \exp(-yf)$; the observation weights $w_{i,t} \propto \exp(-y_i F_{t-1}(x_i))$ arise naturally from this loss.
- GBM replaces observation weights with **pseudo-residuals** (negative loss gradients) and fits a weak learner (typically a shallow tree) to them, enabling generalization to any loss function.
- The choice of when to stop adding trees is critical — always use a validation set or cross-validation for early stopping to prevent overfitting.
- XGBoost and LightGBM are state-of-the-art GBM implementations used in practice; sklearn's `HistGradientBoostingRegressor` is the recommended sklearn entry point for large datasets.