# Lecture 6: Kernel Methods

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- Kernels: definition and examples
- Mercer / positive definite kernels
- The kernel trick
- Recap: PCA and its limitations
- Kernel PCA (kPCA)
- Recap: Ridge regression
- Woodbury matrix identity
- Kernel ridge regression
- Functional gradient descent and kernel regression
- Regularisation in kernel ridge regression

---

## 📝 Summary

This lecture introduces kernel methods as a principled way to extend linear algorithms to non-linear settings. A kernel is a similarity function between feature vectors that, via Mercer's theorem, implicitly corresponds to an inner product in a high-dimensional (possibly infinite-dimensional) feature space. The kernel trick allows computations in this space without ever explicitly constructing the feature map. Two classical algorithms are kernelised: PCA is extended to kernel PCA (kPCA), which can uncover non-linear structure; and ridge regression is extended to kernel ridge regression, which enables non-linear prediction. The lecture closes with a gradient-descent perspective on kernel regression and regularisation via early stopping or explicit penalisation.

---

## 🎯 Learning Goals

- Understand what a kernel is and why positive definiteness (Mercer condition) matters.
- Explain the kernel trick and why it is computationally attractive.
- Derive and implement kernel PCA, including centring of the Gram matrix.
- Derive kernel ridge regression via the Woodbury identity and dual variables.
- Make predictions with kernel ridge regression on unseen data.

---

## 📚 Concepts

### Kernels

**Why it matters:** Kernels provide a flexible, unified way to measure similarity between data points that generalises the standard dot product to non-linear settings.

**Intuition:** Think of a kernel as a generalised inner product. Instead of comparing two vectors by their ordinary dot product, you apply a function $k(\mathbf{x}, \mathbf{y})$ that scores how "similar" the two feature vectors are — possibly in a very rich space — without ever computing that space explicitly.

**Prerequisites:**
- Inner products and vector spaces
- Symmetric positive semi-definite matrices

**How it works:**

A kernel is any function

$$k(\mathbf{x}, \mathbf{y}) : \mathbb{R}^p \times \mathbb{R}^p \to \mathbb{R}$$

satisfying symmetry $k(\mathbf{x}, \mathbf{y}) = k(\mathbf{y}, \mathbf{x})$ and non-negativity $k(\mathbf{x}, \mathbf{y}) \geq 0$.

**Key Equations — Common kernels:**

| Kernel | Formula |
|--------|---------|
| Linear | $k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top \mathbf{y}$ |
| Polynomial | $k(\mathbf{x}, \mathbf{y}) = (\gamma \mathbf{x}^\top \mathbf{y} + r)^m$ |
| RBF (Gaussian) | $k(\mathbf{x}, \mathbf{y}) = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{y}\|_2^2\right)$ |
| Laplacian | $k(\mathbf{x}, \mathbf{y}) = \exp\!\left(-\gamma \|\mathbf{x} - \mathbf{y}\|_1\right)$ |
| Sigmoid | $k(\mathbf{x}, \mathbf{y}) = \tanh(\alpha \mathbf{x}^\top \mathbf{y} + c)$ |

---

### Mercer / Positive Definite Kernels

**Why it matters:** Only positive definite kernels guarantee the existence of a valid feature map, which is required for the kernel trick to work.

**Intuition:** Given training data $\mathbf{x}_1, \ldots, \mathbf{x}_n$, a kernel induces an $n \times n$ **Gram matrix** whose $(i,j)$ entry is $k(\mathbf{x}_i, \mathbf{x}_j)$. Positive semi-definiteness of this matrix for *all* possible datasets is the hallmark of a well-behaved kernel.

**How it works:**

The Gram matrix is

$$\mathbf{K} = \begin{pmatrix} k(\mathbf{x}_1, \mathbf{x}_1) & \cdots & k(\mathbf{x}_1, \mathbf{x}_n) \\ \vdots & & \vdots \\ k(\mathbf{x}_n, \mathbf{x}_1) & \cdots & k(\mathbf{x}_n, \mathbf{x}_n) \end{pmatrix}$$

A kernel is called a **Mercer kernel** (or positive definite kernel) if $\mathbf{K}$ is positive semi-definite for all $n$ and all possible feature sets.

**Mercer's theorem** then guarantees the existence of a (possibly infinite-dimensional) feature map $\boldsymbol{\phi}$ such that

$$k(\mathbf{x}, \mathbf{y}) = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{y})$$

Among the kernels listed above, all except the sigmoid kernel are positive definite.

---

### The Kernel Trick

**Why it matters:** Computing $\boldsymbol{\phi}(\mathbf{x})$ explicitly can be costly or impossible (e.g. infinite-dimensional spaces). The kernel trick sidesteps this entirely.

**Intuition:** Many learning algorithms only ever need dot products between feature vectors. If every dot product $\boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{y})$ can be replaced by a kernel evaluation $k(\mathbf{x}, \mathbf{y})$, the algorithm operates implicitly in the high-dimensional space without constructing it.

**How it works:**

Using a positive definite kernel to measure similarity between $m$-dimensional feature vectors is equivalent to:

1. Applying a (potentially non-linear) map $\boldsymbol{\phi} : \mathbb{R}^m \to \mathbb{R}^q$ (with $q$ potentially infinite).
2. Using the Euclidean dot product between transformed vectors $\boldsymbol{\phi}(\mathbf{x})$.

All computations remain manageable because only kernel evaluations — not the mapped vectors — are needed.

**Example (polynomial kernel):**

For $\gamma = r = 1$, $m = 2$, the degree-2 polynomial kernel on $\mathbb{R}^2$ satisfies

$$k(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^\top \mathbf{y} + 1)^2 = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{y})$$

with the six-dimensional map

$$\boldsymbol{\phi}(\mathbf{x}) = \left(1,\ \sqrt{2}\,x_1,\ \sqrt{2}\,x_2,\ x_1^2,\ x_2^2,\ \sqrt{2}\,x_1 x_2\right)^\top$$

Computing the kernel takes one dot product and one squaring; computing $\boldsymbol{\phi}(\mathbf{x})$ explicitly and then taking the dot product requires six multiplications. The advantage grows dramatically with $m$ and $q$.

---

### Recap: PCA

PCA finds the directions of maximum variance in a data matrix $\hat{\mathbf{X}} \in \mathbb{R}^{n \times p}$ by decomposing the sample covariance

$$\hat{\boldsymbol{\Sigma}} = \frac{\hat{\mathbf{X}}^\top \hat{\mathbf{X}}}{n-1} = \mathbf{V} \mathbf{D} \mathbf{V}^\top$$

where $\mathbf{V} \in \mathbb{R}^{p \times p}$ is orthogonal and $\mathbf{D}$ diagonal. The goals are dimension reduction (e.g. for visualisation) and identifying directions relevant to classification or clustering. PCA is a linear method and cannot uncover non-linear structure in the data.

---

### Kernel PCA (kPCA)

**Why it matters:** Standard PCA fails when the meaningful structure in the data is non-linear (e.g. concentric rings, interleaved crescents). kPCA applies PCA in the high-dimensional feature space of $\boldsymbol{\phi}(\mathbf{x})$ via the kernel trick.

**Intuition:** Instead of finding linear directions of variance in the original space, kPCA finds linear directions of variance in the (non-linear) feature space defined by a kernel — all without constructing $\boldsymbol{\phi}$ explicitly.

**Prerequisites:**
- PCA and eigendecomposition
- Mercer kernels and the Gram matrix

**How it works:**

Assume the transformed vectors $\boldsymbol{\phi}(\mathbf{x}_l)$ are centred. PCA in the feature space requires the eigenvectors of the empirical covariance

$$\hat{\boldsymbol{\Sigma}}_\phi = \frac{1}{n} \sum_{l=1}^n \boldsymbol{\phi}(\mathbf{x}_l)\boldsymbol{\phi}(\mathbf{x}_l)^\top$$

The $i$-th eigenvector $\mathbf{v}_i$ (with eigenvalue $d_i$) satisfies

$$\hat{\boldsymbol{\Sigma}}_\phi \mathbf{v}_i = d_i \mathbf{v}_i \implies \mathbf{v}_i = \sum_{l=1}^n a_i^{(l)} \boldsymbol{\phi}(\mathbf{x}_l)$$

Projecting from both sides with $\boldsymbol{\phi}(\mathbf{x}_k)^\top$ shows that the coefficient vectors $\mathbf{a}_i$ solve the eigenvalue problem

$$\mathbf{K} \mathbf{a}_i = d_i n\, \mathbf{a}_i$$

The projection of any (new) mapped point $\boldsymbol{\phi}(\mathbf{x})$ onto the $i$-th principal component axis is then

$$\boldsymbol{\phi}(\mathbf{x})^\top \mathbf{v}_i = \sum_{l=1}^n a_i^{(l)}\, k(\mathbf{x}, \mathbf{x}_l)$$

**Key Equations:**

Eigenvalue problem:
$$\mathbf{K} \mathbf{a}_i = d_i n\, \mathbf{a}_i$$

Normalisation constraint:
$$\mathbf{a}_i^\top \mathbf{K} \mathbf{a}_i = 1$$

Projection of observation $l$ onto component $i$:
$$\eta_l^{(i)} = \mathbf{K}'_{(l,\colon)}\, \mathbf{a}_i$$

**Centring the Gram matrix:**

If the $\boldsymbol{\phi}(\mathbf{x}_l)$ are not centred, centring in the implicit feature space corresponds to transforming the Gram matrix as

$$\mathbf{K}' = \mathbf{J}\mathbf{K}\mathbf{J}, \qquad \mathbf{J} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$$

**Algorithm:**

1. Choose kernel $k(\cdot,\cdot)$ and hyperparameters.
2. Compute Gram matrix $\mathbf{K} \in \mathbb{R}^{n \times n}$.
3. Centre: $\mathbf{K}' = \mathbf{J}\mathbf{K}\mathbf{J}$.
4. Eigendecompose $\mathbf{K}' = \mathbf{A}\boldsymbol{\Lambda}\mathbf{A}^\top$; set $d_i = \lambda_i / n$.
5. Project observations: $\eta_l^{(i)} = \mathbf{K}'_{(l,\colon)}\mathbf{a}_i$.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Captures non-linear structure | Kernel and hyperparameters must be chosen (e.g. $\gamma$ in RBF) |
| No explicit feature map needed | Gram matrix is $n \times n$; scales as $O(n^2)$ in memory |
| Reduces to standard PCA for the linear kernel | Out-of-sample projection requires storing training data |

**Python Implementation:**

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.7)
X_transformed = kpca.fit_transform(X)
```

⚠️ **Theory vs. Practice:** `sklearn`'s `KernelPCA` centres the Gram matrix automatically — you do not need to apply $\mathbf{J}$ by hand. However, the `gamma` parameter controls the RBF bandwidth and has an enormous effect on results; the default (`gamma=1/n_features`) is often wrong for your data. Always tune `gamma` via cross-validation. The `n_components` parameter caps the number of components returned but does not affect the eigendecomposition itself; all $n$ eigenvalues are still computed internally, so memory cost remains $O(n^2)$.

---

### Recap: Ridge Regression

Ridge regression minimises the penalised residual sum of squares:

$$\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2$$

with closed-form solution $\hat{\boldsymbol{\beta}} = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_p)^{-1}\mathbf{X}^\top\mathbf{y}$. Applying the kernel trick requires scalar products between observations (i.e. rows of $\mathbf{X}$), but the formula involves $\mathbf{X}^\top\mathbf{X}$ — a product between feature dimensions. The Woodbury identity resolves this.

---

### Woodbury Matrix Identity

**Why it matters:** It lets us rewrite a $p \times p$ matrix inverse (expensive when $p \gg n$) as an $n \times n$ matrix inverse, which is the key step that exposes the Gram matrix in ridge regression.

**Intuition:** Inverting a large matrix perturbed by a low-rank update is equivalent to a cheaper computation involving only the smaller "update" matrices.

**Key Equations:**

For invertible $\mathbf{A} \in \mathbb{R}^{p \times p}$, invertible $\mathbf{C} \in \mathbb{R}^{n \times n}$, $\mathbf{U} \in \mathbb{R}^{p \times n}$, $\mathbf{V} \in \mathbb{R}^{n \times p}$:

$$(\mathbf{A} + \mathbf{U}\mathbf{C}\mathbf{V})^{-1} = \mathbf{A}^{-1} - \mathbf{A}^{-1}\mathbf{U}(\mathbf{C}^{-1} + \mathbf{V}\mathbf{A}^{-1}\mathbf{U})^{-1}\mathbf{V}\mathbf{A}^{-1}$$

Applied to ridge regression (set $\mathbf{A} = \lambda\mathbf{I}_p$, $\mathbf{U} = \mathbf{X}^\top$, $\mathbf{V} = \mathbf{X}$, $\mathbf{C} = \mathbf{I}_n$):

$$(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_p)^{-1}\mathbf{X}^\top = \mathbf{X}^\top(\lambda\mathbf{I}_n + \mathbf{X}\mathbf{X}^\top)^{-1}$$

This rewriting shifts the inversion from a $p \times p$ system to an $n \times n$ system, and $\mathbf{X}\mathbf{X}^\top$ is exactly the linear Gram matrix.

---

### Kernel Ridge Regression

**Why it matters:** Replacing $\mathbf{X}\mathbf{X}^\top$ with a non-linear Gram matrix $\mathbf{K}$ enables non-linear regression while retaining the closed-form structure of ridge regression.

**Intuition:** The dual formulation expresses the regression solution as a weighted sum of training observations. Those weights — the dual variables — are obtained by inverting an $n \times n$ kernel matrix. Predictions on new points require only kernel evaluations between the new point and training points.

**Prerequisites:**
- Ridge regression
- Woodbury identity
- Mercer kernels

**How it works:**

From the Woodbury result, the ridge estimator becomes

$$\hat{\boldsymbol{\beta}} = \mathbf{X}^\top(\mathbf{X}\mathbf{X}^\top + \lambda\mathbf{I}_n)^{-1}\mathbf{y}$$

Replacing $\mathbf{X}\mathbf{X}^\top$ with a general Gram matrix $\mathbf{K}$ and defining the **dual variables**

$$\hat{\boldsymbol{\alpha}} = (\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\mathbf{y}$$

the primal variables satisfy $\hat{\boldsymbol{\beta}} = \mathbf{X}^\top\hat{\boldsymbol{\alpha}} = \sum_{l=1}^n \hat{\alpha}^{(l)} \mathbf{x}_l$.

**Key Equations:**

Dual variables:
$$\hat{\boldsymbol{\alpha}} = (\mathbf{K} + \lambda\mathbf{I}_n)^{-1}\mathbf{y}$$

Prediction on a new point $\mathbf{x}$:
$$\hat{f}(\mathbf{x}) = \sum_{l=1}^n \hat{\alpha}^{(l)}\, k(\mathbf{x}_l, \mathbf{x})$$

Standard ridge regression is recovered by using the linear kernel $k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top\mathbf{y}$.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Closed-form solution | Inversion of $n \times n$ matrix: $O(n^3)$ cost |
| Any Mercer kernel can be used | Storing the Gram matrix costs $O(n^2)$ memory |
| Prediction is a sum over training kernel evaluations | Hyperparameter $\lambda$ (and kernel parameters) must be tuned |

**Python Implementation:**

```python
from sklearn.kernel_ridge import KernelRidge

model = KernelRidge(alpha=1.0, kernel='rbf', gamma=0.1)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

⚠️ **Theory vs. Practice:** `sklearn`'s `alpha` corresponds to $\lambda$ in the lecture — not to the dual variable vector $\hat{\boldsymbol{\alpha}}$. These are completely different objects. Confusing them is a common and consequential mistake. The `gamma` parameter of the RBF kernel controls the length scale and strongly affects model behaviour; the default value is `1/n_features` which is rarely appropriate. Tune both `alpha` and `gamma` jointly via cross-validation.

---

### Functional Gradient Descent and Kernel Regression

**Why it matters:** Provides an alternative, gradient-based derivation of kernel regression and connects it to boosting methods covered in later lectures.

**How it works:**

Without fixing a parametric model, suppose $y = f(x) + \epsilon$ and minimise the MSE loss $L(y, f) = \frac{1}{2}\|y - f(x)\|^2$. The functional gradient is $\nabla_f L = -(y - f)$, giving the update

$$f_{t+1} = f_t + \eta(y - f_t)$$

starting from $f = 0$. This memorises the training data and cannot generalise.

Assuming the **dual (kernel) formulation** $y = \mathbf{K}\boldsymbol{\alpha} + \epsilon$ (where $\mathbf{K} = \mathbf{X}\mathbf{X}^\top$ for the linear case), the gradient of $L(y, \boldsymbol{\alpha}) = \|y - \mathbf{K}\boldsymbol{\alpha}\|^2$ with respect to $\boldsymbol{\alpha}$ is $-\mathbf{K}(y - \mathbf{K}\boldsymbol{\alpha})$, giving

$$\boldsymbol{\alpha}_{t+1} = \boldsymbol{\alpha}_t + \eta \mathbf{K}(y - \mathbf{K}\boldsymbol{\alpha}_t)$$

Predictions on new data: $\hat{f}(\mathbf{x}^*) = \mathbf{K}(\mathbf{x}^*, \mathbf{X}_\text{train})\hat{\boldsymbol{\alpha}}$.

**Regularisation** in the gradient descent framework can be achieved by **early stopping** (stopping iterations before convergence) or by explicit penalisation via the kernel norm $\|\boldsymbol{\beta}\|^2 = \boldsymbol{\alpha}^\top\mathbf{K}\boldsymbol{\alpha}$, leading to the penalised objective

$$L(y, \boldsymbol{\alpha}, \mathbf{K}) = \frac{1}{2}\|y - \mathbf{K}\boldsymbol{\alpha}\|^2 + \lambda\boldsymbol{\alpha}^\top\mathbf{K}\boldsymbol{\alpha}$$

with gradient $\nabla_{\boldsymbol{\alpha}} L = -\mathbf{K}(y - \mathbf{K}\boldsymbol{\alpha}) + \lambda\mathbf{K}\boldsymbol{\alpha}$.

---

## ✅ Key Takeaways

- A kernel $k(\mathbf{x}, \mathbf{y})$ is a generalised inner product; positive definiteness (Mercer condition) guarantees a valid feature map $\boldsymbol{\phi}$ via Mercer's theorem.
- The **kernel trick** replaces explicit dot products in the high-dimensional feature space with cheap kernel evaluations — enabling non-linear methods without constructing $\boldsymbol{\phi}$.
- **Kernel PCA** performs PCA in the feature space by eigendecomposing the (centred) Gram matrix; the Gram matrix must be centred using $\mathbf{K}' = \mathbf{J}\mathbf{K}\mathbf{J}$.
- The **Woodbury identity** rewrites the ridge regression solution to expose the Gram matrix $\mathbf{X}\mathbf{X}^\top$, enabling kernelisation.
- **Kernel ridge regression** replaces $\mathbf{X}\mathbf{X}^\top$ with any Mercer Gram matrix; predictions are computed as $\hat{f}(\mathbf{x}) = \sum_l \hat{\alpha}^{(l)} k(\mathbf{x}_l, \mathbf{x})$.
- The kernel trick applies broadly: PCA, ridge regression, logistic regression, and SVMs (upcoming) can all be kernelised.
- Gradient descent in the dual (kernel) space connects to gradient boosting, which will be covered in future lectures.