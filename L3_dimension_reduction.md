# Lecture 3: A First Look at Dimension Reduction

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- The Curse of Dimensionality
- High-Dimensional Predictive Modelling
- Projection onto a Subspace
- The Rayleigh Quotient
- Principal Component Analysis (PCA)
- Pre-processing and Standardisation
- Singular Value Decomposition (SVD)
- SVD and PCA Connection
- Regularised Discriminant Analysis (RDA)

---

## 📝 Summary

This lecture introduces dimension reduction as a solution to the curse of dimensionality. It shows how high-dimensional data causes neighbourhood-based methods to break down and standard models to become ill-defined. Two core strategies — feature selection and feature transformation — are presented. The lecture then develops Principal Component Analysis (PCA) rigorously from variance maximisation and the Rayleigh quotient, covering the computational procedure, the role of standardisation, and how to choose the number of components. Singular Value Decomposition (SVD) is introduced as a computational workhorse that connects naturally to PCA. The lecture closes with regularised discriminant analysis (RDA) as a stabilised alternative to LDA in high-dimensional settings.

---

## 🎯 Learning Goals

- Understand why high dimensionality breaks standard methods and what strategies exist to address it.
- Derive PCA from the variance-maximisation perspective and understand its connection to eigendecomposition.
- Apply the PCA computational procedure correctly, including centring and standardisation.
- Understand SVD and its relationship to PCA.
- Know when and how to use regularised discriminant analysis (RDA) for stable classification.

---

## 📚 Concepts

### The Curse of Dimensionality

**Why it matters:** As dimensionality grows, data becomes increasingly sparse in ways that fundamentally break local methods and geometric intuitions.

**Intuition:** In low dimensions, a small region of space contains a reasonable fraction of the data. In high dimensions, you need to expand that region to cover almost the entire space just to capture a handful of points — "local" loses its meaning.

**Prerequisites:**
- Basic probability and uniform distributions
- Notion of Euclidean distance

**How it works:**

There are two distinct manifestations:

**1. Samples drift far from the origin.** Let $\mathbf{x} \in [-1, 1]^p$ be uniformly distributed. The fraction of observations in a centred hypercube $[-t, t]^p$ is $q = t^p$, so to contain a fixed fraction $q$ of data you need:

$$t = q^{1/p}$$

As $p$ grows, $t \to 1$ even for tiny $q$. For example, capturing just 10% of the data requires $t \approx 0.98$ in 100 dimensions — the "local" neighbourhood now spans nearly the full range of every variable.

**2. Pairwise distances concentrate.** For $\mathbf{x}, \mathbf{y} \in [0,1]^p$ uniform, the mean pairwise distance $\|\mathbf{x} - \mathbf{y}\|_2$ grows as $O(\sqrt{p})$, while the standard deviation stays roughly constant. All points become nearly equidistant from each other.

| $p$ | Mean distance | SD |
|-----|-------------|-----|
| 2   | 0.52        | 0.25 |
| 10  | 1.28        | 0.25 |
| 100 | 4.07        | 0.24 |
| 1000 | 12.91      | 0.24 |

The practical consequence: methods that rely on proximity (kNN, kernel methods, LDA) lose discriminative power since all neighbours are roughly equally far away.

---

### High-Dimensional Predictive Modelling

**Why it matters:** Many standard classifiers and regressors fail — either conceptually or numerically — when $p$ is large.

**Intuition:** Predictive methods work by borrowing information from nearby observations. When distance is meaningless, so is neighbourhood. Additionally, fitting a model requires estimating parameters; with more parameters than observations the system is underdetermined.

**How it works:**

When $p$ is large, specific methods break in the following ways:

- **kNN**: the neighbourhood concept collapses — all neighbours are equally distant.
- **Logistic regression**: the system $\mathbf{X}^\top \mathbf{X}$ becomes rank-deficient or ill-conditioned; multicollinearity renders coefficient estimates unstable or undefined.
- **Discriminant analysis (LDA/QDA)**: the covariance matrix $\hat{\boldsymbol{\Sigma}}$ cannot be inverted when $p \geq n$, and estimates are unreliable even when $p$ is merely large relative to $n$.

Two general strategies address this:

1. **Feature selection** — retain a subset of original features (e.g. those with highest variance, or those most correlated with the outcome via t-test, ANOVA, or correlation). Simple and interpretable; discussed in detail later in the course.

2. **Feature transformation** — linearly combine features to create a lower-dimensional representation (e.g. PCA). Does not guarantee uninformative features are removed; transformed features may obscure the original relationship with the outcome.

---

### Projection onto a Subspace

**Why it matters:** Projections are the mathematical foundation of PCA and SVD — understanding them makes everything that follows concrete.

**Intuition:** Given a direction (or set of directions), projecting a point means dropping a perpendicular onto the span of those directions. The projected point is the closest point in the subspace to the original.

**How it works:**

Given orthonormal vectors $\mathbf{b}_1, \ldots, \mathbf{b}_m \in \mathbb{R}^p$ (i.e. $\|\mathbf{b}_j\| = 1$ and $\mathbf{b}_j^\top \mathbf{b}_k = 0$ for $j \neq k$), the projection of $\mathbf{x} \in \mathbb{R}^p$ onto the $m$-dimensional subspace $V_m = \text{span}(\mathbf{b}_1, \ldots, \mathbf{b}_m)$ is:

$$\hat{\mathbf{x}} = \sum_{j=1}^{m} (\mathbf{x}^\top \mathbf{b}_j)\mathbf{b}_j = \underbrace{\left(\sum_{j=1}^{m} \mathbf{b}_j \mathbf{b}_j^\top\right)}_{\text{Projection matrix}} \mathbf{x}$$

The projection is **orthogonal**: the residual $\mathbf{x} - \hat{\mathbf{x}}$ is perpendicular to every basis vector:

$$(\mathbf{x} - \hat{\mathbf{x}})^\top \mathbf{b}_j = 0 \quad \text{for all } j$$

This means the projected point is the unique closest point in $V_m$ to $\mathbf{x}$ under Euclidean distance.

---

### The Rayleigh Quotient

**Why it matters:** It provides the theoretical bridge between variance maximisation and eigendecomposition, showing that the direction of maximum variance is an eigenvector of the covariance matrix.

**How it works:**

For a symmetric matrix $\mathbf{A} \in \mathbb{R}^{k \times k}$ and $\mathbf{0} \neq \mathbf{x} \in \mathbb{R}^k$, the **Rayleigh Quotient** is:

$$J(\mathbf{x}) = \frac{\mathbf{x}^\top \mathbf{A} \mathbf{x}}{\mathbf{x}^\top \mathbf{x}}$$

The maximisation problem

$$\max_{\mathbf{x}} J(\mathbf{x}) \quad \text{subject to } \mathbf{x}^\top \mathbf{x} = 1$$

is solved by the unit eigenvector of $\mathbf{A}$ corresponding to its **largest eigenvalue** $\lambda$. Note that $-\mathbf{x}$ is also a solution (sign indeterminacy).

In the PCA context, $\mathbf{A} = \hat{\boldsymbol{\Sigma}}$ (the empirical covariance matrix) and $\mathbf{x} = \mathbf{r}$ is the projection direction.

---

### Principal Component Analysis (PCA)

**Why it matters:** PCA is the standard technique for unsupervised linear dimension reduction. It finds the directions of greatest variance and projects data onto them, discarding low-variance directions that often correspond to noise.

**Intuition:** Rotate the coordinate axes so that the new axes align with the directions of maximum spread in the data. Keep only the first few of these new axes. The data's "interesting" structure — the directions along which observations differ most — is preserved, while noise (low-variance directions) is discarded.

**Prerequisites:**
- Eigendecomposition of symmetric matrices
- Rayleigh Quotient
- Orthogonal projections

**How it works:**

**Objective:** Find directions $\mathbf{r}$ (unit vectors) that maximise the variance of the projected data.

The variance of $n$ data points $\mathbf{x}_1, \ldots, \mathbf{x}_n$ along direction $\mathbf{r}$ is:

$$S(\mathbf{r}) = \sum_{l=1}^{n} (\mathbf{r}^\top (\mathbf{x}_l - \bar{\mathbf{x}}))^2 = (n-1)\, \mathbf{r}^\top \hat{\boldsymbol{\Sigma}} \mathbf{r}$$

where $\hat{\boldsymbol{\Sigma}}$ is the empirical covariance matrix. Maximising $S(\mathbf{r})$ subject to $\|\mathbf{r}\| = 1$ is exactly the Rayleigh Quotient problem for $\hat{\boldsymbol{\Sigma}}$. The solution is the eigenvector $\mathbf{r}_1$ corresponding to the **largest eigenvalue** $\lambda_1$.

Subsequent directions are found by projecting onto the orthogonal complement of previously found directions and repeating.

**Computational Procedure:**

1. **Centre** the columns of $\mathbf{X} \in \mathbb{R}^{n \times p}$ (subtract column means). Standardise if variables are on different scales.
2. Compute the empirical covariance matrix:

$$\hat{\boldsymbol{\Sigma}} = \frac{1}{n-1} \mathbf{X}^\top \mathbf{X}$$

3. Compute the eigendecomposition: find eigenvalues $\lambda_1 \geq \lambda_2 \geq \cdots \geq \lambda_p \geq 0$ and corresponding orthonormal eigenvectors $\mathbf{r}_1, \ldots, \mathbf{r}_p$ of $\hat{\boldsymbol{\Sigma}}$.
4. The $j$-th **principal component** (PC) of a data point $\mathbf{x}$ is the scalar $\mathbf{r}_j^\top \mathbf{x}$. The variance along the $j$-th PC direction equals $\lambda_j$.

Setting $\mathbf{R} = (\mathbf{r}_1, \ldots, \mathbf{r}_p)$ and $\mathbf{D} = \text{diag}(\lambda_1, \ldots, \lambda_p)$:

$$\hat{\boldsymbol{\Sigma}} = \mathbf{R} \mathbf{D} \mathbf{R}^\top, \quad \mathbf{R}^\top \mathbf{R} = \mathbf{R}\mathbf{R}^\top = \mathbf{I}_p$$

**Choosing the number of components $m$:**

The **explained variance** of the first $m$ components is:

$$\frac{\lambda_1 + \cdots + \lambda_m}{\lambda_1 + \cdots + \lambda_p} \times 100\%$$

For standardised variables, $\text{tr}(\hat{\boldsymbol{\Sigma}}) = p$, and a common selection rule is to retain components with:

$$\lambda_j \geq \frac{1}{p}\,\text{tr}(\hat{\boldsymbol{\Sigma}}) \quad (= 1 \text{ for standardised data})$$

This can also be assessed visually with a **scree plot** (eigenvalue vs. component index), looking for an "elbow".

**Key Equations:**

$$S(\mathbf{r}) = (n-1)\,\mathbf{r}^\top \hat{\boldsymbol{\Sigma}}\, \mathbf{r}$$

$$\hat{\mathbf{x}} = \left(\sum_{j=1}^{m} \mathbf{r}_j \mathbf{r}_j^\top\right)\mathbf{x} \quad \text{(reconstruction using } m \text{ PCs)}$$

$$\frac{\lambda_1 + \cdots + \lambda_m}{\sum_{j=1}^p \lambda_j} \times 100\% \quad \text{(explained variance)}$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Unsupervised — no labels needed | PC directions may not be aligned with the outcome $Y$ |
| Removes noise (low-variance directions) | Transformed features are linear combinations — harder to interpret |
| Reduces collinearity | Cannot guarantee uninformative features are removed |
| Well-established theory | Sensitive to scale (requires standardisation) |
| Exact solution (no iteration needed) | Only captures linear structure |

**Python Implementation:**

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np

# Centre and standardise
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA(n_components=None)  # keep all components first
pca.fit(X_scaled)

# Scree plot data
eigenvalues = pca.explained_variance_         # λ_j values
explained_ratio = pca.explained_variance_ratio_

# Select components with eigenvalue >= 1 (Kaiser rule for standardised data)
m = np.sum(eigenvalues >= 1)
pca_reduced = PCA(n_components=m)
X_reduced = pca_reduced.fit_transform(X_scaled)

# Reconstruct
X_reconstructed = pca_reduced.inverse_transform(X_reduced)
```

⚠️ **Theory vs. Practice**

- `pca.explained_variance_` stores $\lambda_j$ values (using $n-1$ denominator), matching the lecture definition. `pca.components_` stores eigenvectors as *rows*, not columns — do not confuse with $\mathbf{R}$.
- `PCA` uses SVD internally (via `scipy.linalg.svd`), not eigendecomposition of $\hat{\boldsymbol{\Sigma}}$ directly. This is numerically more stable but means the signs of components are arbitrary — `sklearn` will flip signs between runs or versions. This does not affect reconstruction but will break any code that interprets the sign of a loading.
- If you pass unscaled data to `PCA`, it will silently proceed and give you the wrong components — dominated by whichever feature has the largest raw variance. Always standardise first with `StandardScaler` unless you have a specific reason not to.
- `n_components=None` keeps all components. Setting `n_components` before inspecting eigenvalues means you have already thrown away information you needed to make the selection.

---

### Pre-processing and Standardisation

**Why it matters:** The covariance matrix — and therefore PCA — is scale-dependent. Without standardisation, features with large numerical ranges dominate the principal components for purely artifactual reasons.

**How it works:**

Given a data matrix $\mathbf{X} \in \mathbb{R}^{n \times p}$, let $\mathbf{m}_c \in \mathbb{R}^p$ be the vector of column means and $\mathbf{s} \in \mathbb{R}^p$ the vector of column standard deviations.

- **Column-centred matrix**: $\mathbf{X} - \mathbf{1}_n \mathbf{m}_c^\top$ (column means are zero)
- **Standardised matrix**: multiply by $\text{diag}(1/\mathbf{s})$ so each column has unit variance

Scale-dependence of covariance: if $z = c \cdot x$, then $s_{zy} = c \cdot s_{xy}$. Scaling a variable arbitrarily inflates or deflates its covariance with all other variables, directly altering which PC directions are found.

**Standardising removes this subjectivity** while preserving the shape of the data (outliers remain outliers, relative relationships are unchanged). For standardised data $\text{tr}(\hat{\boldsymbol{\Sigma}}) = p$.

**When not to standardise:** If the relative scale of features carries genuine information (e.g. all features are in the same physical unit), standardisation discards that information. This is a modelling decision, not a default.

The UCI Wine example illustrates the consequence starkly: raw PCA is dominated by the *Proline* feature (large numeric range ~500–1500) and produces a PC space where wine origins are barely separated. After standardisation, the three origins are clearly distinguished in the first two PCs.

---

### Singular Value Decomposition (SVD)

**Why it matters:** SVD is the standard numerical method for computing PCA, and it is a universal tool throughout numerical linear algebra, signal processing, and machine learning.

**Intuition:** Any matrix can be written as a product of three simple matrices: a rotation/reflection ($\mathbf{U}$), a scaling ($\mathbf{D}$), and another rotation/reflection ($\mathbf{V}^\top$). The singular values on the diagonal of $\mathbf{D}$ describe how much the matrix stretches each direction.

**Prerequisites:**
- Eigendecomposition
- Orthogonal matrices
- Matrix norms (Frobenius norm)

**How it works:**

For $\mathbf{X} \in \mathbb{R}^{n \times p}$ with $n \geq p$, the SVD is:

$$\mathbf{X} = \mathbf{U} \mathbf{D} \mathbf{V}^\top$$

where:
- $\mathbf{U} \in \mathbb{R}^{n \times p}$: columns are left singular vectors, $\mathbf{U}^\top \mathbf{U} = \mathbf{I}_p$
- $\mathbf{V} \in \mathbb{R}^{p \times p}$: columns are right singular vectors, $\mathbf{V}^\top \mathbf{V} = \mathbf{V}\mathbf{V}^\top = \mathbf{I}_p$
- $\mathbf{D} = \text{diag}(d_{11}, \ldots, d_{pp})$ with $d_{11} \geq d_{22} \geq \cdots \geq d_{pp} \geq 0$ (singular values)

From the orthogonality conditions:

$$\mathbf{X}\mathbf{X}^\top \mathbf{U} = \mathbf{U}\mathbf{D}^2, \qquad \mathbf{X}^\top \mathbf{X}\, \mathbf{V} = \mathbf{V}\mathbf{D}^2$$

**Best rank-$q$ approximation:** Writing $\mathbf{u}_j$, $\mathbf{v}_j$ for the columns of $\mathbf{U}$, $\mathbf{V}$:

$$\mathbf{X} = \sum_{j=1}^{p} d_{jj}\, \mathbf{u}_j \mathbf{v}_j^\top$$

The best rank-$q$ approximation (in Frobenius norm) is obtained by truncating after $q$ terms:

$$\mathbf{X}_q = \sum_{j=1}^{q} d_{jj}\, \mathbf{u}_j \mathbf{v}_j^\top$$

with approximation error:

$$\|\mathbf{X} - \mathbf{X}_q\|_F^2 = \sum_{j=q+1}^{p} d_{jj}^2$$

This is the Eckart–Young theorem, and it is the mathematical justification for using SVD for dimension reduction.

---

### SVD and PCA Connection

**Why it matters:** PCA is almost always computed via SVD in practice rather than by explicitly computing $\hat{\boldsymbol{\Sigma}}$ — SVD is more numerically stable.

**How it works:**

For centred $\mathbf{X}$, the empirical covariance matrix is:

$$\hat{\boldsymbol{\Sigma}} = \frac{\mathbf{X}^\top \mathbf{X}}{n-1} = \frac{\mathbf{V}\mathbf{D}\mathbf{U}^\top \mathbf{U}\mathbf{D}\mathbf{V}^\top}{n-1} = \mathbf{V} \left(\frac{\mathbf{D}^2}{n-1}\right) \mathbf{V}^\top$$

This reveals the direct correspondence:

| PCA quantity | SVD quantity |
|---|---|
| PC directions $\mathbf{r}_j$ | Columns of $\mathbf{V}$ (right singular vectors) |
| Eigenvalues $\lambda_j$ of $\hat{\boldsymbol{\Sigma}}$ | $d_{jj}^2 / (n-1)$ |
| PC scores $\mathbf{X}\mathbf{r}_j$ | $d_{jj}\, \mathbf{u}_j$ (scaled left singular vectors) |

**Python Implementation:**

```python
import numpy as np

# Manually via SVD (matches sklearn PCA exactly)
X_centred = X - X.mean(axis=0)
U, d, Vt = np.linalg.svd(X_centred, full_matrices=False)

eigenvalues = d**2 / (X.shape[0] - 1)   # λ_j
pc_directions = Vt.T                      # columns = r_j
pc_scores = U * d                         # n × p matrix of scores
```

⚠️ **Theory vs. Practice**

- `np.linalg.svd` returns $\mathbf{V}^\top$, not $\mathbf{V}$ — the right singular vectors are the *rows* of `Vt`. Passing `Vt` directly as PC directions will give you the transpose of what you want.
- Sign conventions: SVD implementations choose signs to make the largest absolute value in each column of $\mathbf{U}$ positive. `sklearn.decomposition.PCA` does the same. This means individual loadings (entries of $\mathbf{r}_j$) can flip sign between software versions. Any downstream code that hard-codes the expected sign of a loading will silently produce wrong results.

---

### Regularised Discriminant Analysis (RDA)

**Why it matters:** LDA requires inverting the covariance matrix $\hat{\boldsymbol{\Sigma}}$, which is numerically unstable or impossible when eigenvalues are near zero (common in high dimensions). RDA stabilises the inversion by adding a small constant to the diagonal.

**Intuition:** Nudge all eigenvalues away from zero by adding $\lambda > 0$ to each. Large eigenvalues are barely affected; small ones are rescued from near-zero instability.

**Prerequisites:**
- LDA / discriminant analysis
- PCA / eigendecomposition of $\hat{\boldsymbol{\Sigma}}$

**How it works:**

In LDA, the Mahalanobis distance involves $\hat{\boldsymbol{\Sigma}}^{-1}$. Using the eigendecomposition $\hat{\boldsymbol{\Sigma}} = \mathbf{V}\mathbf{D}\mathbf{V}^\top$:

$$\hat{\boldsymbol{\Sigma}}^{-1} = \mathbf{V}\mathbf{D}^{-1}\mathbf{V}^\top, \quad \text{with scaling } 1/d_{jj}$$

When $d_{jj} \approx 0$, the factor $1/d_{jj}$ explodes. **RDA** replaces $\hat{\boldsymbol{\Sigma}}$ with:

$$\hat{\boldsymbol{\Sigma}}_\lambda := \hat{\boldsymbol{\Sigma}} + \lambda \mathbf{I}_p = \mathbf{V}(\mathbf{D} + \lambda \mathbf{I}_p)\mathbf{V}^\top$$

The inverse scaling factors become $1/(d_{jj} + \lambda)$ — bounded away from infinity for any $\lambda > 0$.

**Effect of $\lambda$:**
- Small $\lambda$: close to standard LDA
- Large $\lambda$: $d_{jj}$ becomes negligible relative to $\lambda$, all directions weighted equally → classifier approaches nearest centroid

RDA also extends to QDA by interpolating between class-specific and pooled covariances:

$$\hat{\boldsymbol{\Sigma}}_{i,\lambda} := \underbrace{\hat{\boldsymbol{\Sigma}}_i}_{\text{QDA}} + \lambda\, \underbrace{\hat{\boldsymbol{\Sigma}}}_{\text{LDA}}$$

The tuning parameter $\lambda$ is typically chosen by cross-validation.

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Applicable when $p \geq n$ (unlike standard LDA) | Introduces a tuning parameter $\lambda$ requiring CV |
| Continuously bridges LDA and nearest-centroid | Still assumes Gaussian classes |
| Numerically stable | Regularisation may suppress informative low-variance directions |

---

## ✅ Key Takeaways

- In high dimensions, distances concentrate, neighbourhoods break down, and covariance matrices become ill-conditioned — standard methods fail or become undefined.
- PCA finds the directions of maximum variance by solving an eigenvalue problem on the empirical covariance matrix $\hat{\boldsymbol{\Sigma}}$; the $j$-th principal component direction is the eigenvector corresponding to eigenvalue $\lambda_j$.
- Always centre — and usually standardise — before PCA. Failure to standardise causes features with large numerical ranges to dominate, regardless of their actual relevance.
- The fraction of variance explained by the first $m$ PCs is $(\lambda_1 + \cdots + \lambda_m)/\text{tr}(\hat{\boldsymbol{\Sigma}})$; retain components above the Kaiser criterion ($\lambda_j \geq 1$ for standardised data) or use a scree plot.
- SVD of the data matrix is equivalent to PCA and is the standard computational approach; $\mathbf{V}$ gives PC directions and $d_{jj}^2/(n-1)$ gives eigenvalues.
- RDA stabilises LDA in high-dimensional settings by adding $\lambda \mathbf{I}_p$ to $\hat{\boldsymbol{\Sigma}}$, preventing numerical blow-up from near-zero eigenvalues.