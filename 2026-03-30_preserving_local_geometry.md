# Lecture 4 – Preserving Local Geometry

**MSA220/MVE441 – Statistical Learning for Big Data**
*Rebecka Jörnsten, Mathematical Sciences — 30th March 2026*

## 📋 Contents

- Distance preservation and the Gram matrix
- Kernels and the kernel trick
- Kernel PCA (kPCA)
- Classical Multidimensional Scaling (MDS)
- The problem with global methods: the Swiss roll
- Data-driven distances and Isomap
- t-distributed Stochastic Neighbour Embedding (tSNE)
- Uniform Manifold Approximation and Projection (UMAP)
- Comparison: global vs. local dimension reduction methods

---

## 📝 Summary

This lecture introduces flexible, geometry-aware approaches to dimensionality reduction that go beyond PCA. It begins by formalising distance preservation via Gram matrices and kernels, then develops Kernel PCA and Classical Multidimensional Scaling (MDS) as natural extensions. The second half of the lecture tackles the harder problem of data with non-linear, locally-flat structure (illustrated by the Swiss roll), and introduces three increasingly sophisticated methods — Isomap, tSNE, and UMAP — that exploit local neighbourhood information rather than global Euclidean distances to find faithful low-dimensional embeddings.

---

## 🎯 Learning Goals

- Understand how pairwise distances relate to the Gram matrix, and why this relationship enables kernel-based dimension reduction.
- Be able to describe the kPCA and MDS algorithms, including when exact embeddings exist and when low-rank approximations are used.
- Explain why global methods (PCA, kPCA, classical scaling) fail on datasets with non-linear manifold structure.
- Understand the key algorithmic steps of Isomap, tSNE, and UMAP, including the role of hyperparameters (number of neighbours, perplexity, min-dist).
- Critically compare tSNE and UMAP in terms of their loss functions, treatment of repulsion, and ability to handle test data.

---

## 📚 Concepts

### Preserving Pairwise Distances

**Why it matters:** Many downstream tasks depend on the relative arrangement of points, not just variance directions. A dimension reduction method that distorts distances will mislead clustering, visualisation, and classification.

**Intuition:** Think of making a flat map of the Earth. You cannot preserve all properties simultaneously (area, angles, distances), but you can choose what to optimise. Similarly, different dimension reduction criteria preserve different aspects of data geometry.

**How it works:**
Given high-dimensional vectors $\mathbf{x}_1, \ldots, \mathbf{x}_n \in \mathbb{R}^p$, the goal is to find low-dimensional embeddings $\mathbf{y}_1, \ldots, \mathbf{y}_n \in \mathbb{R}^q$ with $q < p$ such that

$$\|\mathbf{x}_i - \mathbf{x}_l\|_2 \approx \|\mathbf{y}_i - \mathbf{y}_l\|_2$$

The key insight is that pairwise Euclidean distances are entirely encoded in the inner product matrix $\mathbf{X}\mathbf{X}^\top$.

**Key Equations:**

$$\|\mathbf{x}_l - \mathbf{x}_m\|_2^2 = \mathbf{x}_l^\top \mathbf{x}_l - 2\mathbf{x}_l^\top \mathbf{x}_m + \mathbf{x}_m^\top \mathbf{x}_m$$

Let $\mathbf{D}(l,m) = \|\mathbf{x}_l - \mathbf{x}_m\|_2$ be the distance matrix. Then with element-wise squaring:

$$-\frac{1}{2}\mathbf{D}^2 = \mathbf{X}\mathbf{X}^\top - \frac{1}{2}\mathbf{1}\,\text{diag}(\mathbf{X}\mathbf{X}^\top)^\top - \frac{1}{2}\,\text{diag}(\mathbf{X}\mathbf{X}^\top)\mathbf{1}^\top$$

And with the centring matrix $\mathbf{J} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$:

$$\mathbf{K} = \mathbf{J}\!\left(-\tfrac{1}{2}\mathbf{D}^2\right)\!\mathbf{J}$$

Where:
- $\mathbf{K} = \mathbf{X}\mathbf{X}^\top$ is the **Gram matrix** (linear kernel matrix)
- $\mathbf{J}$ is the centring matrix that removes the row and column means

---

### Kernels and the Kernel Trick

**Why it matters:** Kernels allow similarity to be measured in a potentially much richer, higher-dimensional feature space without ever explicitly computing the feature map.

**Intuition:** Suppose you want to compare two images not by their raw pixels but by some complex non-linear combination of them. Instead of computing that transformation explicitly (which might be computationally infeasible or even infinite-dimensional), a kernel function $k(\mathbf{x}, \mathbf{y})$ computes the inner product in the transformed space directly.

**How it works:**
A positive definite kernel $k(\mathbf{x}, \mathbf{y})$ implicitly defines a mapping $\boldsymbol{\phi}: \mathbb{R}^p \to \mathbb{R}^q$ such that:

$$k(\mathbf{x}, \mathbf{y}) = \boldsymbol{\phi}(\mathbf{x})^\top \boldsymbol{\phi}(\mathbf{y})$$

The **kernel trick** replaces all inner product computations with kernel evaluations, performing the computation implicitly in the high-dimensional space of $\boldsymbol{\phi}(\mathbf{x})$ without constructing $\boldsymbol{\phi}$ explicitly.

**Key Equations — Common Kernels:**

| Kernel | Formula |
|--------|---------|
| Linear | $k(\mathbf{x}, \mathbf{y}) = \mathbf{x}^\top \mathbf{y}$ |
| Polynomial | $k(\mathbf{x}, \mathbf{y}) = (\gamma\mathbf{x}^\top\mathbf{y} + r)^m$ |
| RBF (Gaussian) | $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x}-\mathbf{y}\|_2^2)$ |
| Laplacian | $k(\mathbf{x}, \mathbf{y}) = \exp(-\gamma\|\mathbf{x}-\mathbf{y}\|_1)$ |
| Sigmoid | $k(\mathbf{x}, \mathbf{y}) = \tanh(\alpha\mathbf{x}^\top\mathbf{y} + c)$ |

---

### Kernel PCA (kPCA)

**Why it matters:** Standard PCA finds linear structure. Kernel PCA can uncover non-linear structure by implicitly performing PCA in a higher-dimensional feature space.

**Intuition:** If you apply a non-linear transformation to your data and then run PCA, you get kPCA. The kernel trick means you never have to compute the transformation — you only need to evaluate the kernel between pairs of data points.

**Prerequisites:**
- PCA and eigendecomposition
- Gram/kernel matrices (see above)

**How it works:**

Assume the transformed vectors $\boldsymbol{\phi}(\mathbf{x}_l)$ are centred. The empirical covariance in feature space is:

$$\hat{\boldsymbol{\Sigma}}_\phi = \frac{1}{n}\sum_{l=1}^n \boldsymbol{\phi}(\mathbf{x}_l)\boldsymbol{\phi}(\mathbf{x}_l)^\top = \mathbf{V}\mathbf{D}\mathbf{V}^\top$$

Rather than computing this directly, kPCA works entirely through the $n \times n$ Gram matrix:

**General algorithm for kPCA:**

1. Choose a kernel $k(\cdot,\cdot)$ and its hyperparameters.
2. Compute the Gram matrix $\mathbf{K} \in \mathbb{R}^{n\times n}$ with $\mathbf{K}_{ij} = k(\mathbf{x}_i, \mathbf{x}_j)$.
3. Centre $\mathbf{K}$ using $\mathbf{K}' = \mathbf{J}\mathbf{K}\mathbf{J}$.
4. Eigendecompose $\mathbf{K}' = \mathbf{A}\boldsymbol{\Lambda}\mathbf{A}^\top$.
5. Set $d_i = \lambda_i / n$.
6. The projection of the $l$-th observation onto the $i$-th principal component axis is:

$$\eta_l^{(i)} = \mathbf{K}'(l, :)\mathbf{a}_i \in \mathbb{R}$$

**Key Equations:**

$$\mathbf{K}' = \mathbf{J}\mathbf{K}\mathbf{J}, \qquad \mathbf{J} = \mathbf{I}_n - \frac{1}{n}\mathbf{1}\mathbf{1}^\top$$

**Python Implementation:**

```python
from sklearn.decomposition import KernelPCA

kpca = KernelPCA(n_components=2, kernel='rbf', gamma=1.0)
Y = kpca.fit_transform(X)  # X: (n, p) array
```

⚠️ **Theory vs. Practice:** `sklearn`'s `KernelPCA` centres the kernel matrix internally — do not pre-centre your data thinking you are mimicking Step 3. The `gamma` parameter in the RBF kernel equals $\gamma$ in $\exp(-\gamma\|\mathbf{x}-\mathbf{y}\|_2^2)$, but $\gamma = \frac{1}{2\sigma^2}$, so there is an implicit factor of 2 relative to the $\sigma$ notation used in some references. Choosing `kernel='linear'` gives standard PCA on the kernel matrix — it is not equivalent to `sklearn.decomposition.PCA` unless data is already centred. The `fit_transform` method stores the training kernel matrix, so projecting new test data via `transform` is well-defined, but the projection is done via the weighted-average formula, not a simple linear projection — this cannot be undone as cleanly as standard PCA.

---

### Classical Multidimensional Scaling (MDS)

**Why it matters:** MDS is the bridge between a distance matrix (which can come from any source, not just Euclidean space) and a Euclidean embedding. It is the algorithmic backbone underlying Isomap.

**Intuition:** Given only pairwise distances (e.g., travel times between cities), MDS reconstructs coordinates that approximately reproduce those distances. When the distances are truly Euclidean, the reconstruction is exact.

**How it works:**
Given a distance matrix $\mathbf{D} \in \mathbb{R}^{n\times n}_+$:

1. Compute $\mathbf{K} = \mathbf{J}(-\frac{1}{2}\mathbf{D}^2)\mathbf{J}$. An exact embedding exists in $q = \text{rank}(\mathbf{K})$ dimensions.
2. Eigendecompose $\mathbf{K} = \mathbf{U}\boldsymbol{\Lambda}\mathbf{U}^\top$.
3. The embedding is:

$$\mathbf{Y} = \mathbf{U}_q \boldsymbol{\Lambda}_q^{1/2} = (\sqrt{\lambda_1}\mathbf{u}_1, \ldots, \sqrt{\lambda_q}\mathbf{u}_q) \in \mathbb{R}^{n\times q}$$

This satisfies $\mathbf{Y}\mathbf{Y}^\top = \mathbf{K} = \mathbf{X}\mathbf{X}^\top$, which implies $\|\mathbf{x}_l - \mathbf{x}_m\|_2^2 = \|\mathbf{y}_l - \mathbf{y}_m\|_2^2$.

**Approximate (low-rank) MDS:** Keeping only $m < q$ components gives **classical scaling**, minimising the stress:

$$d(\mathbf{D}, \mathbf{Y}) = \left(\sum_{i \neq j}\left(\mathbf{D}(i,j) - \|\mathbf{y}_i - \mathbf{y}_j\|_2\right)^2\right)^{1/2}$$

**Projecting test data:** The embedding is not a simple linear projection. For a test point $\mathbf{x}^*$, compute its kernel distances to training data $K(\mathbf{x}^*, \mathbf{X})$, centre, then impute the embedding:

$$y_k^* = \sum_i k(*, x_i) y_i^k, \quad k = 1, \ldots, q$$

**Python Implementation:**

```python
from sklearn.manifold import MDS

mds = MDS(n_components=2, dissimilarity='euclidean', normalized_stress='auto')
Y = mds.fit_transform(X)

# From a precomputed distance matrix:
mds_pre = MDS(n_components=2, dissimilarity='precomputed')
Y = mds_pre.fit_transform(D)  # D: (n, n) distance matrix
```

⚠️ **Theory vs. Practice:** `sklearn`'s default `MDS` uses SMACOF optimisation (iterative), not the closed-form eigendecomposition described in the lecture. To get classical scaling (the exact spectral solution), use `from sklearn.manifold import smacof` is not what you want — instead use `MDS(metric=True)` with `dissimilarity='precomputed'`, or better, implement the $\mathbf{K} = \mathbf{J}(-\frac{1}{2}\mathbf{D}^2)\mathbf{J}$ → eigendecomposition pipeline manually. `fit_transform` does not provide a projection for new data; calling `transform` on unseen points will give you the wrong answer because SMACOF-based MDS has no analytic out-of-sample extension.

---

### The Problem with Global Methods: Manifold Structure

**Why it matters:** When data lies on a curved, low-dimensional manifold embedded in high-dimensional space, Euclidean distance in the ambient space is a poor measure of true similarity — it cuts through the manifold rather than following it.

**How it works:**
The Swiss roll is a canonical example: data has a simple 2D structure but is rolled into 3D space. PCA, kPCA (with RBF kernel), and classical scaling all measure distances in the Euclidean norm of the surrounding 3D space. None of them can "unroll" the manifold because:

- **PCA** is a global method — it considers all data simultaneously and finds linear directions of maximum variance. It cannot detect the non-linear rolling.
- **Kernel PCA** introduces a different distance measure, but the Gaussian kernel is not adapted to the manifold geometry.
- **Classical scaling** behaves similarly to PCA since it also relies on Euclidean distances in the ambient space.

The fix is to measure distances *along the manifold* (geodesic distances) rather than through the surrounding space.

---

### Isomap

**Why it matters:** Isomap approximates geodesic (along-manifold) distances using a nearest-neighbour graph, then applies MDS to embed these geodesic distances in low dimensions.

**Intuition:** If you want the distance between two cities on opposite sides of a mountain, you measure the road distance, not the straight-line distance through the mountain. Isomap builds a "road network" between nearby data points and measures distances along it.

**Prerequisites:**
- $k$-nearest neighbours
- Classical MDS / spectral embeddings

**How it works:**
1. For each data point $\mathbf{x}_l$, find its $k$ nearest neighbours.
2. Build a weighted graph connecting each point to its $k$ neighbours, with edge weights equal to Euclidean distance.
3. Compute the **geodesic distance** between all pairs of points as the shortest path through the graph (e.g., via Dijkstra's algorithm).
4. This yields a geodesic distance matrix $\mathbf{D}_G$. Apply classical MDS to $\mathbf{D}_G$.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Recovers true manifold geometry for uniformly sampled, connected manifolds | Sensitive to the choice of $k$ — too small → disconnected graph; too large → shortcuts through the manifold |
| Principled: minimises a well-defined stress function | Disconnected graph components produce infinite geodesic distances; implementations return separate embeddings |
| Exact out-of-sample extension via weighted average of training embeddings | Struggles with datasets of varying density |

**Python Implementation:**

```python
from sklearn.manifold import Isomap

iso = Isomap(n_neighbors=6, n_components=2)
Y = iso.fit_transform(X)

# Out-of-sample projection:
Y_test = iso.transform(X_test)
```

⚠️ **Theory vs. Practice:** If the kNN graph is not connected, `sklearn`'s `Isomap` will by default only use the largest connected component and silently drop the other observations. This will give you the wrong shape without any warning. Always check `iso.nbrs_.kneighbors_graph(X).toarray()` for connectivity before trusting the output. The `n_neighbors` parameter has a dramatic effect on results (see lecture figures: knn=6 vs knn=20 look completely different). There is no principled way to choose it from the data alone — this is a real limitation.

---

### tSNE — t-distributed Stochastic Neighbour Embedding

**Why it matters:** tSNE uses probability distributions to capture local neighbourhood structure in the high-dimensional space and optimises an embedding that preserves these neighbourhood relationships. It produces visually striking and interpretable 2D/3D visualisations.

**Intuition:** Instead of measuring distance directly, tSNE asks: "If I were sitting at point $\mathbf{x}_i$ and randomly picked a neighbour, what's the probability I'd pick $\mathbf{x}_j$?" This probability is high for nearby points and near-zero for distant ones. The embedding tries to reproduce these probabilities using a heavy-tailed distribution in the low-dimensional space.

**Prerequisites:**
- Gaussian/RBF kernels
- KL divergence and entropy
- Gradient descent

**How it works:**

**Step 1 — High-dimensional similarities.** For each point $\mathbf{x}_i$ with bandwidth $\sigma_i$, define conditional probabilities:

$$p_{j|i} = \frac{\exp\!\left(-\|\mathbf{x}_j - \mathbf{x}_i\|_2^2 / (2\sigma_i^2)\right)}{\sum_{l\neq i} \exp\!\left(-\|\mathbf{x}_l - \mathbf{x}_i\|_2^2 / (2\sigma_i^2)\right)}, \quad p_{i|i} = 0$$

Each $\mathbf{x}_i$ gets its own $\sigma_i$, chosen by fixing the perplexity:

$$\text{Perp}(X) = 2^{H(X)}, \qquad H(X) = -\sum_{j\neq i} p_{j|i}\log_2 p_{j|i}$$

Perplexity is interpreted as the effective number of neighbours. Symmetrise:

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2}$$

**Step 2 — Low-dimensional similarities.** Use the $t$-distribution with 1 degree of freedom (Cauchy) — heavier tails than Gaussian, which prevents crowding:

$$q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|_2^2)^{-1}}{\sum_{l\neq r}(1 + \|\mathbf{y}_l - \mathbf{y}_r\|_2^2)^{-1}}, \qquad q_{ii} = 0$$

**Step 3 — Optimisation.** Minimise the KL divergence between distributions $P = (p_{ij})$ and $Q = (q_{ij})$:

$$\text{KL}(P \| Q) = \sum_{i\neq l} p_{il}\log\frac{p_{il}}{q_{il}}$$

via gradient descent with numerical tricks (early exaggeration, momentum).

**Key Equations:**

$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2}, \qquad q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|_2^2)^{-1}}{\sum_{l\neq r}(1 + \|\mathbf{y}_l - \mathbf{y}_r\|_2^2)^{-1}}$$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Excellent at separating clusters in visualisation | Convergence to local minima — repeated runs give different results |
| Adapts neighbourhood size per point via perplexity | Cannot project test data without re-fitting (data leakage risk) |
| Heavy-tailed $q$ avoids the crowding problem | Distances and cluster sizes in the embedding are not meaningful |
| Works well on high-dimensional data (images, genomics) | Perplexity is hard to tune; very sensitive to its value |

**Python Implementation:**

```python
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
Y = tsne.fit_transform(X)
```

⚠️ **Theory vs. Practice:** `sklearn`'s `TSNE` has no `transform` method — you cannot apply a fitted tSNE to new test data. Calling `fit_transform` on the full dataset (train + test combined) leaks test information into the embedding. If you need out-of-sample embedding, use UMAP instead. The `perplexity` parameter should typically be between 5 and 50; values outside this range frequently produce degenerate embeddings (see lecture: perplexity=2 → random scatter; perplexity=100 → compressed line). Because tSNE uses random initialisation and gradient descent, set `random_state` for reproducibility — even then, results vary across `sklearn` versions.

---

### UMAP — Uniform Manifold Approximation and Projection

**Why it matters:** UMAP improves on tSNE by preserving both local and global structure, running faster, and — critically — supporting out-of-sample projection without data leakage.

**Intuition:** UMAP also builds a graph of fuzzy neighbourhood memberships, but it uses a more principled loss function that explicitly balances attraction (pulling similar points together) and repulsion (pushing dissimilar points apart) without the global normalisation that causes tSNE's crowding problem.

**Prerequisites:**
- kNN graphs
- tSNE (the comparison to tSNE is central to understanding UMAP's design choices)
- Stochastic gradient descent

**How it works:**

**Step 1 — Build the kNN graph.** For each observation $i$, find its $k$ nearest neighbours $N_k(i)$. Compute distances $d_{ij}$ and the minimum distance to any neighbour $\rho_i = \min_{j \in N_k(i)} d_{ij}$.

**Step 2 — Compute fuzzy high-dimensional memberships.**

$$p_{ij} = \exp\!\left(-\frac{d_{ij} - \rho_i}{\sigma_i}\right)$$

Where $\rho_i$ anchors the representation so the nearest neighbour is always "strongly connected" (avoids the density-dependent collapsing/exploding problem). $\sigma_i$ is chosen to satisfy:

$$\sum_{j < k} \exp\!\left(-\frac{d_{ij}-\rho_i}{\sigma_i}\right) = \log_2(k)$$

This means effectively $\log_2(k)$ "close" neighbours contribute — dense regions get smaller $\sigma_i$, sparse regions get larger $\sigma_i$.

**Step 3 — Symmetrise.**

$$p_{ij} \leftarrow p_{ij} + p_{ji} - p_{ij}p_{ji}$$

(probabilistic OR, rather than tSNE's arithmetic mean)

**Step 4 — Define low-dimensional similarities.** Instead of a normalised distribution, UMAP uses an unnormalised family:

$$q_{ij} = \frac{1}{1 + a\|\mathbf{y}_i - \mathbf{y}_j\|_2^{2b}}$$

Where $a, b$ are determined by the user-chosen `min_dist` hyperparameter by fitting to a target piecewise curve. `min_dist` controls how tightly points are packed in the embedding.

**Step 5 — Optimise via cross-entropy loss (SGD):**

$$-\sum_{i<j}\left[p_{ij}\log(q_{ij}) + (1-p_{ij})\log(1-q_{ij})\right]$$

The first term is **attraction**: large $p_{ij}$ forces $q_{ij}$ close to 1, pulling $\mathbf{y}_i$ and $\mathbf{y}_j$ together. The second term is **repulsion**: small $p_{ij}$ forces $q_{ij}$ close to 0, pushing the points apart. This explicit repulsion replaces tSNE's implicit repulsion from global normalisation of $q$.

SGD iterates by: (a) sampling a "positive" edge $(i,j)$ with $p_{ij} > 0$ and applying the attraction gradient; (b) sampling a few "negative" edges $(i,k)$ with $p_{ik} = 0$ and applying the repulsion gradient.

**Projecting test data:**
Compute $p_{*j} = \exp\!\left(-\frac{d(*,x_j)-\rho_*}{\sigma_*}\right)$ for $j \in N(*)$, then:

$$\hat{\mathbf{y}}_* = \frac{\sum_{j \in N(*)} p_{*j} \mathbf{y}_j}{\sum_{j \in N(*)} p_{*j}} + \text{local SGD updates (training embeddings fixed)}$$

**Key Equations:**

| Quantity | UMAP | tSNE |
|----------|------|------|
| High-dim $p_{ij}$ | $\exp(-(d_{ij}-\rho_i)/\sigma_i)$ | Gaussian, symmetrised mean |
| Low-dim $q_{ij}$ | $(1+a\|y_i-y_j\|^{2b})^{-1}$, unnormalised | $(1+\|y_i-y_j\|^2)^{-1}$, globally normalised |
| Loss | Cross-entropy (explicit repulsion) | KL divergence (implicit repulsion) |
| Test projection | Yes (interpolation + local SGD) | No |

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Preserves both local and global structure | Cluster sizes and distances are still not fully meaningful |
| Supports out-of-sample projection without re-fitting | $k$, `min_dist` require tuning |
| Faster than tSNE for large datasets | Results still vary across runs (SGD) |
| Explicit repulsion gives more stable cluster separation | Theoretical guarantees rely on manifold assumptions |

**Python Implementation:**

```python
import umap

reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
Y_train = reducer.fit_transform(X_train)

# Out-of-sample projection (no data leakage):
Y_test = reducer.transform(X_test)
```

⚠️ **Theory vs. Practice:** `umap-learn` is not part of `sklearn` — install separately with `pip install umap-learn`. Unlike tSNE, `transform` is available and does not re-fit on the test data, making UMAP safe for use as a pre-processing step in a pipeline. However, `fit_transform` and `fit + transform` can give slightly different embeddings because `transform` uses interpolation rather than full optimisation. The `n_neighbors` parameter controls the local/global balance: small values (e.g., 5–10) → very local, fine-grained clusters; large values (e.g., 50) → more global, topology-preserving but less cluster-separated (see lecture figure). `min_dist=0` packs points tightly (good for cluster identification); larger `min_dist` spreads them out (good for seeing continuous structure).

---

### Comparison: Global vs. Local Dimension Reduction

These five methods differ primarily in what notion of distance or similarity they preserve and whether they account for local manifold structure.

| Property | PCA | kPCA | Classical MDS | Isomap | tSNE | UMAP |
|-----------|-----|------|---------------|--------|------|------|
| Distance type | Euclidean (global) | Kernel (global) | Euclidean or custom (global) | Geodesic (local) | Probabilistic (local) | Probabilistic (local) |
| Linear/Non-linear | Linear | Non-linear | Linear | Non-linear | Non-linear | Non-linear |
| Test data projection | Yes (linear) | Yes (kernel avg.) | No (SMACOF) | Yes (approx.) | No | Yes |
| Preserves global structure | Yes | Partial | Yes | Partial | No | Yes |
| Preserves local structure | No | Partial | No | Yes | Yes | Yes |
| Sensitive to hyperparameters | Low | Medium ($\gamma$, kernel) | Low | High ($k$) | High (perplexity) | Medium ($k$, min-dist) |
| Scalability | High | Medium ($n^2$) | Medium ($n^2$) | Medium | Low–Medium | Medium–High |

Key practical distinctions:
- **PCA and classical MDS** behave similarly on smooth data; neither can unroll a Swiss roll.
- **Isomap** is the right tool when the manifold is globally non-linear but locally Euclidean, and when the data is densely and uniformly sampled. It fails badly on sparse or multi-component data.
- **tSNE** excels at 2D/3D visualisation for exploratory analysis but cannot safely project test data and produces non-reproducible embeddings.
- **UMAP** is the current best-practice for dimensionality reduction that is both visualisation-quality and safe for use in a train/test pipeline.

---

## ✅ Key Takeaways

- The Gram matrix $\mathbf{K} = \mathbf{X}\mathbf{X}^\top$ encodes all pairwise Euclidean distances, and centring $\mathbf{K}$ via $\mathbf{J}$ is equivalent to double-centring the squared distance matrix.
- Kernel PCA performs PCA implicitly in the feature space of $\boldsymbol{\phi}(\mathbf{x})$ — only the Gram matrix is needed, never $\boldsymbol{\phi}$ itself.
- Classical MDS finds the exact low-dimensional embedding from a distance matrix via spectral decomposition; keeping $m < q$ components minimises the stress.
- When data lies on a curved manifold, all methods that use Euclidean ambient distances (PCA, kPCA, classical MDS) will fail — they cannot follow the manifold surface.
- Isomap approximates geodesic distances via shortest paths on a kNN graph; it works well but is fragile to disconnected graphs and varying density.
- tSNE converts distances into neighbourhood probabilities, uses a heavy-tailed $t$-distribution in the embedding space to prevent crowding, and minimises KL divergence — but cannot project test data.
- UMAP's cross-entropy loss provides explicit attraction and repulsion, avoids global normalisation, supports out-of-sample projection, and better preserves global structure than tSNE.
- Neither tSNE nor UMAP should be used blindly: distances and cluster sizes in the embedding are not directly interpretable, and both are sensitive to hyperparameter choices.
- UMAP and Kernel PCA can be used as pre-processing steps without data leakage, since the reduction is learnt from training data alone.