
# Lecture 8: Boosting, SVMs and Flexible Discriminant Analysis

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- Tuning Boosting Machines (Recap)
- XGBoost
- Support Vector Machines: Separating Hyperplanes
- Support Vector Machines: Soft Margin (Slack Variables)
- SVMs and the Kernel Trick
- Discriminant Analysis (Recap)
- Mixture Discriminant Analysis (MDA)
- Fisher's Linear Discriminant Analysis (FLDA)
- Flexible Discriminant Analysis (FDA) via Optimal Scoring
- Kernel Discriminant Analysis

---

## 📝 Summary

This lecture covers three major areas of supervised classification. It opens with practical tuning strategies for gradient boosting machines and introduces XGBoost's second-order Taylor approximation as a faster, more principled boosting step. The bulk of the lecture develops Support Vector Machines from first principles — from maximum-margin separating hyperplanes to soft-margin classifiers and the kernel trick. The final third revisits discriminant analysis, progressing from standard LDA/QDA through mixture models to Fisher's discriminant analysis, and culminating in Flexible Discriminant Analysis (FDA), which reframes DA as a regression problem to enable penalization and nonlinear extensions.

---

## 🎯 Learning Goals

- Understand how XGBoost improves on standard GBM via a second-order loss approximation.
- Derive the SVM optimization problem (primal and dual) and understand the role of support vectors.
- Understand how slack variables and the parameter $K$ control the soft-margin trade-off.
- Explain the kernel trick and why it makes SVMs highly flexible.
- Derive Fisher's LDA as an eigendecomposition of $\Sigma_W^{-1}\Sigma_B$ and understand why it differs from PCA.
- Understand Optimal Scoring and how it connects FLDA to regression, enabling flexible/penalized extensions.

---

## 📚 Concepts

### Recap: Tuning Boosting Machines

**Summary:** Boosting models are tuned via early stopping (monitoring validation performance to halt training before overfitting), choice of weak learner complexity (shallow trees are standard), and the learning rate, which controls how aggressively each step updates the model. Updating with less than the full gradient step — found via a line search — acts as regularization. Subsampling the data stochastically and using dropout (discarding early learners) are further regularization strategies that also help avoid local optima.

---

### XGBoost

**Why it matters:** Standard GBM is slow for large, high-dimensional datasets and uses only a first-order gradient approximation. XGBoost addresses both issues with a second-order expansion and highly optimized tree-building, making it one of the most widely used ML methods in practice.

**Intuition:** In GBM, each new tree tries to approximate the gradient of the loss. XGBoost goes one step further: it also uses the curvature (second derivative) of the loss, so instead of gradient descent it performs a Newton–Raphson step. This gives a better-informed update at each iteration.

**Prerequisites:**
- Gradient Boosting Machines (GBM)
- Taylor series expansion
- Newton–Raphson optimization

**How it works:**

The boosting objective at step $t$ is:

$$\min_{\alpha, f_t} \mathcal{L}(y, F_{t-1} + \alpha f_t)$$

**GBM (first-order):** approximate the loss with a linear (first-order) Taylor expansion around $F_{t-1}$:

$$\mathcal{L}(y, F_{t-1} + \alpha f_t) \approx \mathcal{L}(y, F_{t-1}) + \nabla_{F_{t-1}}(\mathcal{L})\,\alpha f_t$$

The new weak learner $f_t$ is trained to be correlated with the negative gradient — i.e., to point in the steepest descent direction.

**XGBoost (second-order):** uses a quadratic Taylor expansion:

$$\mathcal{L}(y, F_{t-1} + \alpha f_t) \approx \mathcal{L}(y, F_{t-1}) + \nabla_{F_{t-1}}(\mathcal{L})\,\alpha f_t + \frac{1}{2}(\nabla^2_{F_{t-1}}\mathcal{L})(\alpha f_t)^2$$

This is equivalent to a Newton–Raphson update, yielding a more precise step direction and size.

Additional features of XGBoost:
- Fast, optimized tree-building with adaptive pruning (trees can vary in depth).
- Dropout regularization: randomly "forget" early weak learners to avoid being trapped by early bad steps.
- Stochastic subsampling of data per iteration.
- **LightGBM** is a further variant optimized for even greater scalability.

**Key Equations:**

$$\mathcal{L}(y, F_{t-1} + \alpha f_t) \approx \mathcal{L}(y, F_{t-1}) + \nabla_{F_{t-1}}(\mathcal{L})\,\alpha f_t + \frac{1}{2}(\nabla^2_{F_{t-1}^2}\mathcal{L})(\alpha f_t)^2$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Fast implementation for large/high-dimensional data | Many hyperparameters to tune |
| Second-order update is more principled than gradient descent | Can overfit without proper regularization |
| Handles variable-depth trees adaptively | Less interpretable than single trees |
| Widely supported with strong software ecosystems | |

**Python Implementation:**

```python
import xgboost as xgb
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)

model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=3,
    subsample=0.8,
    early_stopping_rounds=20,
    eval_metric="logloss",
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
```

> ⚠️ **Theory vs. Practice:** XGBoost's `learning_rate` (called `eta` in its native API) scales the contribution of each tree — this is the $\alpha$ from the lecture. If you omit `early_stopping_rounds`, XGBoost will use all `n_estimators` trees regardless of overfitting. Without a validation set passed to `eval_set`, early stopping silently does nothing. The default `max_depth=6` is much deeper than the "shallow weak learner" recommendation in the lecture; for a proper regularized boosting model, set `max_depth=3` or `max_depth=4`. The `subsample` parameter controls stochastic subsampling of rows per tree — not to be confused with `colsample_bytree`, which subsamples features.

---

### Support Vector Machines: Separating Hyperplanes

**Why it matters:** The SVM maximum-margin classifier is the principled answer to "which of the infinitely many separating hyperplanes should we pick?" The maximum-margin solution is unique and tends to generalize well to unseen data.

**Intuition:** If two classes are perfectly separable, you could draw many different lines between them. The SVM picks the line that is as far as possible from both classes — maximizing the "buffer zone" (margin) between them. The only training points that determine this line are the ones sitting exactly on the edge of the margin; these are the **support vectors**.

**Prerequisites:**
- Linear classifiers and decision boundaries
- Lagrangian optimization and KKT conditions
- Basic linear algebra (inner products, norms)

**How it works:**

Label classes $y \in \{-1, +1\}$ and define the linear classifier:

$$f(x) = x'\beta + \beta_0, \quad \hat{y} = \text{sign}(f(x))$$

The decision boundary is the hyperplane $\{x : x'\beta + \beta_0 = 0\}$. The vector $\beta$ is orthogonal to this boundary, and the signed distance from any point $x$ to the boundary is:

$$\text{distance} = \frac{f(x)}{\|\beta\|}$$

**Primal problem** (maximize margin $C$, all points correctly classified at distance $\geq C$):

$$\max_{\beta,\beta_0} C \quad \text{s.t.} \quad \frac{1}{\|\beta\|} y_i(x_i'\beta + \beta_0) \geq C, \; \forall i$$

Setting $\|\beta\| = 1/C$ (the scale of $\beta$ is free), this is equivalent to:

$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i'\beta + \beta_0) \geq 1, \; \forall i$$

**Dual problem** (via Lagrangian with multipliers $\alpha_i \geq 0$):

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_k \alpha_i \alpha_k y_i y_k (x_i' x_k)$$
$$\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_i \alpha_i y_i = 0$$

With $\beta = \sum_i \alpha_i y_i x_i$.

**KKT complementary slackness:** $\alpha_i(y_i(x_i'\beta + \beta_0) - 1) = 0$. This means:
- If $\alpha_i > 0$, then $y_i(x_i'\beta + \beta_0) = 1$ — the point lies exactly on the margin (**support vector**).
- If $y_i(x_i'\beta + \beta_0) > 1$, then $\alpha_i = 0$ — interior points have no influence on $\beta$.

**Key Equations:**

Primal:
$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i'\beta + \beta_0) \geq 1$$

Dual:
$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_k \alpha_i \alpha_k y_i y_k (x_i' x_k)$$

---

### Support Vector Machines: Soft Margin

**Why it matters:** Real data is rarely perfectly separable. The soft-margin SVM introduces slack variables to allow some misclassification, controlled by a regularization parameter.

**Intuition:** Each observation gets a slack variable $\xi_i \geq 0$ that measures how much it violates the hard-margin constraint. Points within the margin have $\xi_i > 0$; misclassified points have $\xi_i > 1$. The parameter $K$ balances the desire for a large margin against tolerating violations.

**How it works:**

The constraint is softened to:

$$y_i(x_i'\beta + \beta_0) \geq C(1 - \xi_i), \quad \xi_i \geq 0, \quad \sum_i \xi_i \leq \text{constant}$$

Setting $\|\beta\| = 1/C$, the primal problem becomes:

$$\min_{\beta,\beta_0} \frac{1}{2}\|\beta\|^2 \quad \text{s.t.} \quad y_i(x_i'\beta + \beta_0) \geq 1 - \xi_i, \; \xi_i \geq 0, \; \sum_i \xi_i \leq \text{const}$$

The Lagrangian introduces multipliers $\alpha_i$ and $\mu_i$, leading to the dual:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_k \alpha_i \alpha_k y_i y_k (x_i' x_k) \quad \text{s.t.} \quad \alpha_i \in [0, K], \; \sum_i \alpha_i y_i = 0$$

The only difference from the hard-margin case is that $\alpha_i$ is now bounded above by $K$ (instead of only $\alpha_i \geq 0$). Larger $K$ penalizes violations more heavily (closer to hard margin); smaller $K$ allows more violations (wider effective margin).

**Key Equations:**

$$\beta = \sum_i \alpha_i y_i x_i, \quad \alpha_i \in [0, K], \quad \sum_i \alpha_i y_i = 0$$

KKT conditions:
$$\alpha_i\big(y_i(x_i'\beta + \beta_0) - (1 - \xi_i)\big) = 0, \quad \mu_i \xi_i = 0$$

---

### SVMs and the Kernel Trick

**Why it matters:** The dual SVM only ever uses observations through their pairwise inner products $x_i' x_k$. This means we can implicitly work in a very high-dimensional (even infinite-dimensional) feature space without ever computing the feature vectors explicitly — making SVMs extremely flexible at low computational cost.

**Intuition:** Replace each $x_i' x_k$ with a kernel function $k(x_i, x_k) = \phi(x_i)'\phi(x_k)$. The kernel computes the inner product in a transformed space $\phi(\cdot)$ without needing to know $\phi$ explicitly. Using a radial basis (Gaussian) kernel, for example, implicitly maps data into an infinite-dimensional space, enabling nonlinear decision boundaries in the original feature space.

**How it works:**

Replace $x_i' x_k$ with $k(x_i, x_k)$ everywhere in the dual:

$$\max_\alpha \sum_i \alpha_i - \frac{1}{2}\sum_i\sum_k \alpha_i \alpha_k y_i y_k \, k(x_i, x_k)$$

Predictions become:

$$f(x) = \sum_i \alpha_i y_i k(x, x_i) + \beta_0$$

Common kernels:
- **Linear:** $k(x,x') = x'x'$ (standard SVM)
- **Polynomial:** $k(x,x') = (1 + x'x')^d$
- **Radial basis (Gaussian):** $k(x,x') = \exp(-\gamma\|x-x'\|^2)$

**Hinge loss formulation:** The SVM can equivalently be derived as regularized hinge loss minimization:

$$\min_\beta \sum_i [1 - y_i f(x_i)]_+ + \frac{1}{\lambda}\|\beta\|^2$$

where $[z]_+ = \max(0, z)$ is the hinge function.

**Key Equations:**

$$f(x) = \sum_i \alpha_i y_i k(x, x_i), \quad k(x_i, x_k) = \phi(x_i)'\phi(x_k)$$

**Python Implementation:**

```python
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler

X, y = make_circles(n_samples=300, noise=0.1, random_state=42)
X = StandardScaler().fit_transform(X)

# RBF kernel SVM
clf = SVC(kernel='rbf', C=1.0, gamma='scale')
clf.fit(X, y)
print("Support vectors:", clf.n_support_)
```

> ⚠️ **Theory vs. Practice:** The `C` parameter in sklearn's `SVC` is NOT the margin $C$ from the lecture — it is the inverse of $\lambda$ in the hinge loss formulation. Larger `C` means less regularization (closer to hard margin), which is the opposite of what you might expect from a regularization parameter. `gamma='scale'` computes $\gamma = 1 / (n_\text{features} \cdot \text{Var}(X))$ — do not use `gamma='auto'` (which uses only $n_\text{features}$) without understanding the data scale. Always standardize features before fitting an SVM; the margin geometry is not scale-invariant and unscaled features will give you the wrong model.

---

### Recap: Discriminant Analysis

**Summary:** Discriminant analysis assumes $p(x \mid y = c) \sim \mathcal{N}(\mu_c, \Sigma_c)$ and classifies based on Bayes' theorem. LDA simplifies by sharing one covariance matrix $\Sigma$ across all classes (pooled estimate), reducing parameters at the cost of a less flexible boundary. QDA allows class-specific $\Sigma_c$ giving quadratic boundaries. Variants such as diagonal-covariance (Naive Bayes / DLDA), nearest centroid, and regularized DA (RDA) sit along a spectrum of complexity. The classification boundary between classes $l$ and $c$ is $\{x : \pi_l p(x \mid l) = \pi_c p(x \mid c)\}$, and the class prior $\pi_c$ shifts this boundary toward classes with lower prior probability.

---

### Mixture Discriminant Analysis (MDA)

**Why it matters:** QDA allows each class its own covariance but requires estimating a full $p \times p$ matrix per class, which is costly and unstable for large $p$. MDA achieves flexible, non-elliptical class boundaries using simple building blocks: each class is modeled as a mixture of Gaussians that all share the same covariance $\Sigma$.

**Intuition:** Complex shapes in feature space can be approximated by combining several simple spherical or ellipsoidal distributions. A donut-shaped class can be represented by, say, 5–6 Gaussian components arranged in a ring. Since all components share $\Sigma$ (as in LDA), parameter count stays manageable.

**Prerequisites:**
- LDA / QDA
- Gaussian mixture models
- EM algorithm (for fitting)

**How it works:**

Each class distribution is a mixture of $R_c$ Gaussian components:

$$p(x \mid y = c) = \sum_{r=1}^{R_c} \pi_{cr} \, \mathcal{N}(x; \mu_{cr}, \Sigma)$$

Where:
- $R_c$ = number of components for class $c$ (can differ across classes)
- $\pi_{cr}$ = mixing weight of component $r$ within class $c$, with $\sum_r \pi_{cr} = 1$
- $\Sigma$ = shared covariance matrix across all components and all classes

The classification boundary between classes is no longer a hyperplane or a quadric; it can be any shape achievable by combining Gaussian blobs. The shared $\Sigma$ greatly reduces the number of covariance parameters compared to QDA.

**Key Equations:**

$$p(x \mid y = c) = \sum_{r=1}^{R_c} \pi_{cr} \, \mathcal{N}(x; \mu_{cr}, \Sigma)$$

**Python Implementation:**

```python
# No direct sklearn implementation for MDA.
# The 'mda' package or manual implementation via sklearn's GaussianMixture is needed.
from sklearn.mixture import GaussianMixture
import numpy as np

# Fit per-class Gaussian mixtures (shared covariance requires manual enforcement)
# For a full MDA, use the R package 'mda' or implement EM manually.
```

> ⚠️ **Theory vs. Practice:** sklearn does not implement MDA. `GaussianMixture` fits a mixture without class structure and does not enforce the shared covariance constraint. Using separate `GaussianMixture` objects per class and setting `covariance_type='full'` is QDA with mixtures, not MDA. For genuine MDA use the R `mda` package (by Hastie & Tibshirani) or implement the constrained EM manually.

---

### Fisher's Linear Discriminant Analysis (FLDA)

**Why it matters:** Standard LDA performs classification in the full $p$-dimensional space. FLDA finds the optimal lower-dimensional projection for separating class centroids, which is often far more informative than the leading principal components (PCA directions). This is essential when $p$ is large or when the class-separating signal lies in directions of low marginal variance.

**Intuition:** PCA finds directions of maximum total variance. FLDA asks a different question: find the directions along which the class *means* are as spread out as possible *relative to the within-class spread*. The LDA boundary is perpendicular to these discriminant directions. The key insight (shown in the lecture diagram) is that the best discriminant direction is not the direction of most variance — it is the direction where classes are most separated.

**Prerequisites:**
- LDA and the pooled covariance matrix $\Sigma_W$
- Eigendecomposition and generalized eigenproblems
- Rayleigh quotient

**How it works:**

Define:
- **Within-class covariance:** $\Sigma_W = \sum_c \sum_{y_i=c}(x_i - \hat\mu_c)(x_i - \hat\mu_c)'/(N-C)$
- **Between-class covariance:** $\Sigma_B = \sum_c (\hat\mu_c - \bar{x})(\hat\mu_c - \bar{x})'/(C-1)$

**Fisher's problem:** Find projection directions $a$ maximizing the Rayleigh quotient:

$$\max_a \frac{a'\Sigma_B a}{a'\Sigma_W a}$$

Equivalently, solve the constrained problem:

$$\max_a a'\Sigma_B a \quad \text{s.t.} \quad a'\Sigma_W a = I$$

This leads to the generalized eigenproblem:

$$\Sigma_B a = \lambda \Sigma_W a \quad \Longleftrightarrow \quad (\Sigma_W^{-1}\Sigma_B)\,a = \lambda a$$

Since $\Sigma_W^{-1}\Sigma_B$ is not symmetric, this requires a **generalized eigendecomposition**. It is solved by "sphering" with $\Sigma_W^{1/2}$:

$$(\Sigma_W^{-1/2}\Sigma_B\Sigma_W^{-1/2})\,w = \lambda w, \quad w = \Sigma_W^{1/2} a$$

The symmetric matrix $\Sigma_W^{-1/2}\Sigma_B\Sigma_W^{-1/2}$ has a standard eigendecomposition. The original discriminant directions are then $v = \Sigma_W^{-1/2} w$.

There are at most $C - 1$ non-zero discriminant directions (since $\Sigma_B$ has rank at most $C-1$). **Reduced-rank LDA** uses only the leading $L < C-1$ discriminant directions.

**Key Equations:**

$$\max_a \frac{a'\Sigma_B a}{a'\Sigma_W a}, \qquad (\Sigma_W^{-1}\Sigma_B)\,a = \lambda a$$

$$v = \Sigma_W^{-1/2} w, \quad (\Sigma_W^{-1/2}\Sigma_B\Sigma_W^{-1/2})\,w = \lambda w$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Optimal projection for class separation | At most $C-1$ discriminant directions |
| Dimensionality reduction is classification-aware | Assumes shared $\Sigma_W$ (LDA assumption) |
| More informative than PCA for classification | Sensitive to poorly estimated $\Sigma_W$ when $p > n$ |

**Python Implementation:**

```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
lda = LinearDiscriminantAnalysis(n_components=2)  # reduced-rank: keep 2 discriminants
X_proj = lda.fit_transform(X, y)  # shape: (150, 2)
```

> ⚠️ **Theory vs. Practice:** sklearn's `LinearDiscriminantAnalysis` computes the discriminant directions internally using SVD, not by explicitly forming $\Sigma_W^{-1}\Sigma_B$. When $p \geq n$, $\Sigma_W$ is singular; sklearn silently adds `tol`-based truncation of small singular values — this is implicit regularization that changes your model. If you want explicit control, use `solver='lsqr'` with `shrinkage='auto'` (Ledoit-Wolf). The `n_components` parameter is the number of retained discriminant directions (max $C-1$); setting it to `None` keeps all $C-1$ directions. `transform()` returns the projected coordinates, not the discriminant vectors themselves.

---

### Flexible Discriminant Analysis (FDA) via Optimal Scoring

**Why it matters:** Optimal Scoring shows that FLDA is equivalent to a specific multivariate regression problem. This reformulation immediately suggests powerful extensions: replace ordinary least squares with penalized regression (ridge, lasso) for high-dimensional settings, or with nonlinear/spline regression for flexible nonlinear boundaries.

**Intuition:** Instead of computing discriminant directions via eigendecomposition, encode class membership as a $N \times C$ binary matrix $Y$ and regress $Y$ on $X$. The eigendecomposition of the regression fit recovers exactly the FLDA discriminant vectors. Since it's now just a regression problem, you can swap in any regression method you like.

**Prerequisites:**
- FLDA and the generalized eigenproblem
- Ordinary least squares / hat matrix
- Eigendecomposition

**How it works:**

**Step 1:** Create the $N \times C$ indicator matrix $Y$: $Y_{ic} = 1$ if observation $i$ belongs to class $c$, else $0$.

**Step 2:** Regress $Y$ on $X$ via least squares:
$$\hat{B} = (X'X)^{-1}X'Y, \qquad \hat{Y} = HY, \quad H = X(X'X)^{-1}X'$$

**Step 3:** Perform eigendecomposition of $\hat{Y}'Y = Y'HY$:
$$(Y'HY)\,\theta = \lambda\theta$$

The eigenvectors $\theta$ are proportional to the FLDA discriminant vectors $V$.

**Why they are equivalent:**

$X'X \propto \Sigma_W$ and $X'Y \propto M$ (the centroid matrix), so:

$$\hat{B} = (X'X)^{-1}X'Y = \Sigma_W^{-1}M$$

$$\hat{Y}'\hat{Y} \propto M'\Sigma_W^{-1}M = (\Sigma_W^{-1/2}M)'(\Sigma_W^{-1/2}M) = M^{*'}M^* = \Sigma_B^*$$

which is exactly the between-centroid spread in the sphered coordinate system — the same matrix whose eigenvectors define the FLDA directions.

**Extensions enabled by FDA:**
- **Penalized FDA:** replace OLS with ridge/lasso regression to handle $p \gg n$.
- **Nonlinear FDA:** use spline, polynomial, or kernel regression to produce nonlinear class boundaries.
- Any regression method that produces fitted values $\hat{Y}$ can be plugged in.

**Key Equations:**

$$\hat{B} = (X'X)^{-1}X'Y, \qquad (Y'HY)\,\theta = \lambda\theta, \qquad H = X(X'X)^{-1}X'$$

**Python Implementation:**

```python
# sklearn does not implement FDA directly.
# The R 'mda' package has fda(). In Python, penalized FDA can be approximated via:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

# Nonlinear FDA: expand features then apply LDA
pipe = Pipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('lda', LinearDiscriminantAnalysis()),
])
pipe.fit(X, y)
```

> ⚠️ **Theory vs. Practice:** sklearn has no direct FDA implementation. The pipeline above is an approximation: polynomial feature expansion followed by LDA is not the same as FDA with polynomial regression, because FDA fits the regression and extracts discriminant directions jointly — sklearn fits them sequentially. For genuine penalized FDA, use the R `mda` package. For nonlinear FDA in Python, you must implement the Optimal Scoring regression step manually and replace the OLS fit with your chosen regression method.

---

### Kernel Discriminant Analysis

**Why it matters:** Just as the kernel trick extends SVMs to nonlinear boundaries, it can also make discriminant analysis nonlinear — without relying on the regression reformulation of FDA.

**Intuition:** Map the data to a high-dimensional feature space $\phi(x)$ and compute $\Sigma_B^\phi$ and $\Sigma_W^\phi$ in that space. By the kernel trick, everything can be expressed in terms of the kernel function $k(x_i, x_j) = \phi(x_i)'\phi(x_j)$, so the feature map $\phi$ never needs to be computed explicitly.

**How it works:**

Replace the original FLDA problem:

$$\max_a \frac{a'\Sigma_B a}{a'\Sigma_W a}$$

with the same problem in the feature space $\phi(x)$. The between-class covariance $\Sigma_B^\phi$ is computed from pseudo-means expressed through kernel densities, and the within-class covariance $\Sigma_W^\phi$ is computed from within-class centered kernel distances. The entire generalized eigenproblem can be rewritten in terms of the kernel matrix $K_{ij} = k(x_i, x_j)$.

---

## ✅ Key Takeaways

- **XGBoost** improves on GBM by using a second-order (Newton–Raphson) update instead of gradient descent, and includes fast adaptive tree-building plus multiple regularization mechanisms (dropout, subsampling, pruning).
- **SVM hard margin:** the optimal separating hyperplane maximizes the margin; only the support vectors (points on the margin) determine $\beta$.
- **SVM soft margin:** slack variables $\xi_i$ relax the hard constraints; the parameter $K$ controls the bias-variance tradeoff between large margin and allowing violations.
- **The kernel trick** lets SVMs (and DA methods) work in implicit infinite-dimensional feature spaces by replacing inner products $x_i'x_k$ with a kernel function $k(x_i, x_k)$.
- **Hinge loss** provides a clean loss-function view of SVM: $\sum_i[1 - y_i f(x_i)]_+ + \frac{1}{\lambda}\|\beta\|^2$.
- **FLDA** finds directions maximizing the Rayleigh quotient $a'\Sigma_B a / a'\Sigma_W a$, solved via the generalized eigenproblem $(\Sigma_W^{-1}\Sigma_B)a = \lambda a$. This is fundamentally different from PCA.
- **Optimal Scoring** shows FLDA equals multivariate regression of the class-indicator matrix $Y$ on $X$, enabling flexible (FDA) and penalized extensions by replacing OLS with any regression method.
- **MDA** models each class as a mixture of Gaussians with shared covariance, building complex class shapes from simple components while keeping parameter counts tractable.