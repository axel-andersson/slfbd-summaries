# Regularised Regression (cont'd)

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- Regularisation in Classification
- Recap: Regularised Discriminant Analysis (RDA)
- Recap: Naive Bayes LDA
- Nearest Shrunken Centroids
- Extensions of the Lasso
- The Elastic Net
- The Group Lasso
- Comparison: Lasso, Elastic Net, and Group Lasso
- Penalisation in GLMs
- Sparse Logistic Regression
- Sparse Multi-Class Logistic Regression

---

## 📝 Summary

This lecture extends regularisation techniques from regression to classification settings. It opens with a recap of Regularised Discriminant Analysis (RDA) and Naive Bayes LDA, then introduces nearest shrunken centroids — a lasso-style method for variable selection in high-dimensional classifiers. The second half covers three extensions of the lasso designed to handle correlated variables: the elastic net (which blends lasso and ridge penalties), the group lasso (which enforces group-level sparsity), and sparse logistic regression (which brings penalisation to generalised linear models). Each method addresses a specific limitation of the standard lasso, making the lecture a tour of structured sparsity.

---

## 🎯 Learning Goals

- Understand how regularisation is applied to classification, not just regression.
- Know when and why nearest shrunken centroids outperform unregularised Naive Bayes LDA in high dimensions.
- Understand the motivation for the elastic net and group lasso as alternatives to the standard lasso when predictors are correlated or grouped.
- Be able to write down and interpret the penalised log-likelihood formulation of sparse logistic regression.
- Recognise how all these methods share a common theme: controlling model complexity through structured penalisation.

---

## 📚 Concepts

### Recap: Regularised Discriminant Analysis (RDA)

**Summary:** In Quadratic Discriminant Analysis (QDA), the class-conditional covariance $\hat{\boldsymbol{\Sigma}}_i$ must be inverted to evaluate the Gaussian density. When $p$ is large relative to $n$, this matrix can be singular or near-singular, causing numerical instability. RDA addresses this by blending the QDA estimate with the pooled LDA estimate: $\hat{\boldsymbol{\Sigma}}_i = \hat{\boldsymbol{\Sigma}}_i^{\text{QDA}} + \lambda \hat{\boldsymbol{\Sigma}}^{\text{LDA}}$ for $\lambda > 0$. Alternatively, under LDA with a shared covariance, one can further regularise as $\hat{\boldsymbol{\Sigma}} = \hat{\boldsymbol{\Sigma}}^{\text{LDA}} + \lambda \boldsymbol{\Delta}$ for a diagonal $\boldsymbol{\Delta}$, which ties directly to Naive Bayes LDA.

---

### Recap: Naive Bayes LDA

**Summary:** Naive Bayes LDA assumes $\hat{\boldsymbol{\Sigma}} = \hat{\boldsymbol{\Delta}}$, a diagonal matrix whose $(j,j)$ entry is the pooled within-class variance of feature $j$:

$$
\hat{\Delta}^{(j,j)} = \frac{1}{n - K} \sum_{i=1}^{K} \sum_{i_l = i} \left( \mathbf{x}_l^{(j)} - \hat{\boldsymbol{\mu}}_i^{(j)} \right)^2
$$

Classification assigns the class with the highest discriminant value:

$$
\delta_i(\mathbf{x}) = -\frac{1}{2} \left\| \hat{\boldsymbol{\Delta}}^{-1/2}(\mathbf{x} - \hat{\boldsymbol{\mu}}_i) \right\|_2^2 + \log(\hat{\pi}_i)
$$

This is the foundation on which nearest shrunken centroids build.

---

### Nearest Shrunken Centroids

**Why it matters:** In high-dimensional settings ($p > n$), raw class centroids are noisy and uninterpretable — all $p$ features are active. Variable selection in classifiers is just as important as in regression, and nearest shrunken centroids provide a lasso-style solution.

**Intuition:** Instead of using the raw sample centroid $\hat{\boldsymbol{\mu}}_i$ for each class, shrink each component of the centroid towards the overall (grand) centroid $\hat{\boldsymbol{\mu}}_T$. Features whose class-specific deviation from the overall mean is small get pulled to exactly zero — meaning the feature is dropped from that class's centroid entirely.

**Prerequisites:**
- Naive Bayes LDA and the pooled within-class covariance $\hat{\boldsymbol{\Delta}}$
- Soft-thresholding operator $\text{ST}(\cdot, \lambda)$ from the lasso

**How it works:**

The shrunken centroid for class $i$ solves:

$$
\boldsymbol{\mu}_i = \arg\min_{\mathbf{v}} \frac{1}{2} \sum_{i_l = i} \left\| (\hat{\boldsymbol{\Delta}} + s_0 \mathbf{I}_p)^{-1/2}(\mathbf{x}_l - \mathbf{v}) \right\|_2^2 + \lambda n_i m_i \|\mathbf{v} - \hat{\boldsymbol{\mu}}_T\|_1
$$

where:
- $s_0 = \text{median}(\hat{\Delta}^{(1,1)}, \ldots, \hat{\Delta}^{(p,p)})$ — a floor on the within-class variance, preventing division by near-zero variances
- $m_i = \sqrt{\frac{1}{n_i} - \frac{1}{n}}$ — scales the penalty for unequal class sizes
- $\hat{\boldsymbol{\mu}}_T = \frac{1}{n} \sum_l \mathbf{x}_l$ — the overall centroid

The closed-form solution for component $j$ is:

$$
\boldsymbol{\mu}_i^{(j)} = \hat{\boldsymbol{\mu}}_T^{(j)} + m_i \left( \hat{\Delta}^{(j,j)} + s_0 \right) \cdot \text{ST}\!\left( t_i^{(j)}, \lambda \right)
$$

where the standardised deviation is:

$$
t_i^{(j)} = \frac{\hat{\boldsymbol{\mu}}_i^{(j)} - \hat{\boldsymbol{\mu}}_T^{(j)}}{m_i(\hat{\Delta}^{(j,j)} + s_0)}
$$

The soft-thresholding operator sets the standardised deviation to zero if its magnitude is below $\lambda$, pulling the class centroid component all the way back to the overall centroid. When all classes agree that a feature's centroid equals $\hat{\boldsymbol{\mu}}_T^{(j)}$, that feature is effectively removed from the classifier.

$\lambda$ is a tuning parameter chosen by cross-validation. As $\lambda$ increases, more components collapse to $\hat{\boldsymbol{\mu}}_T$, reducing the number of active features. Misclassification rate typically improves initially with increasing $\lambda$ (variance reduction) and then worsens (bias takes over).

**Key Equations:**

$$
\boldsymbol{\mu}_i^{(j)} = \hat{\boldsymbol{\mu}}_T^{(j)} + m_i (\hat{\Delta}^{(j,j)} + s_0) \cdot \text{ST}(t_i^{(j)}, \lambda)
$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Performs feature selection and classification simultaneously | Assumes Naive Bayes (diagonal covariance) — ignores correlations between features |
| Interpretable: identifies which features distinguish each class | Requires cross-validation to tune $\lambda$ |
| Scales well to $p \gg n$ settings | Feature selection may be unstable if classes are similar |

**Python Implementation:**

```python
from sklearn.neighbors import NearestCentroid
# Note: sklearn's NearestCentroid does NOT implement shrunken centroids.
# Use the dedicated package instead:
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# For true nearest shrunken centroids, use the 'rpy2' bridge to R's 'pamr' package,
# or the 'nearest_shrunken_centroids' implementation in Python (e.g. via pyShrunkenCentroids):
# pip install pyShrunkenCentroids

from pyShrunkenCentroids import ShrunkenCentroidsClassifier
clf = ShrunkenCentroidsClassifier(threshold=2.0)  # threshold = lambda
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
```

⚠️ **Theory vs. Practice:** `sklearn.neighbors.NearestCentroid` is a plain nearest-centroid classifier — it does not perform shrinkage or variable selection. Using it here will give you the unregularised version, defeating the entire purpose of the method. There is no sklearn implementation of nearest shrunken centroids. The canonical implementation is `pamr` in R. If you need Python, verify whatever package you use actually applies the soft-thresholding step to the standardised deviations $t_i^{(j)}$, not to the raw centroids.

---

### The Elastic Net

**Why it matters:** The lasso performs variable selection by setting coefficients exactly to zero, but it behaves poorly when predictors are highly correlated. It arbitrarily picks one variable from a correlated group and discards the others, producing sparse but unstable solutions that may miss the true underlying structure.

**Intuition:** Mix the lasso's $\ell_1$ penalty (which drives sparsity) with ridge regression's $\ell_2$ penalty (which distributes coefficients evenly across correlated variables). The result selects variables like the lasso but shares credit among correlated predictors like ridge.

**Prerequisites:**
- Lasso ($\ell_1$ penalisation)
- Ridge regression ($\ell_2$ penalisation)
- Coordinate descent

**How it works:**

The elastic net solves:

$$
\arg\min_{\boldsymbol{\beta}} \frac{1}{2} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \left( \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right)
$$

- $\alpha = 1$: reduces to the lasso
- $\alpha = 0$: reduces to ridge regression
- $0 < \alpha < 1$: a blend — sparsity with grouping

The constraint set (in the Lagrange-dual sense) has rounded edges compared to the lasso's diamond, which geometrically explains why correlated variables receive similar non-zero coefficients rather than one being zeroed and the others not.

Both $\lambda$ and $\alpha$ are tuning parameters that must be selected, typically via cross-validation. The solution is found by cyclic coordinate descent.

**Key Equations:**

$$
\arg\min_{\boldsymbol{\beta}} \frac{1}{2} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \left( \frac{1-\alpha}{2} \|\boldsymbol{\beta}\|_2^2 + \alpha \|\boldsymbol{\beta}\|_1 \right)
$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Handles correlated predictors better than lasso | Two tuning parameters ($\lambda$ and $\alpha$) instead of one |
| Retains variable selection property | Does not use explicit group structure — grouping emerges only from correlation |
| Lasso and ridge are special cases | May select more variables than necessary |

**Python Implementation:**

```python
from sklearn.linear_model import ElasticNet, ElasticNetCV

# Fit with cross-validated lambda, fixed alpha
model = ElasticNetCV(l1_ratio=0.2, cv=5)
model.fit(X_train, y_train)
print(model.alpha_)  # optimal lambda
```

⚠️ **Theory vs. Practice:** sklearn's `ElasticNet` uses `l1_ratio` for what the lecture calls $\alpha$, and `alpha` for what the lecture calls $\lambda$. These are swapped relative to lecture notation. Setting `l1_ratio=1` gives lasso; `l1_ratio=0` gives ridge — but sklearn does not actually allow `l1_ratio=0` in `ElasticNet` (use `Ridge` instead). By default, `ElasticNetCV` searches `l1_ratio=[0.5, 0.7, 0.9, 0.95, 0.99, 1.0]` — this heavily biases toward the lasso end. If you want to explore the ridge end, you must supply your own `l1_ratio` grid. Also note sklearn normalises the penalty differently from the lecture formulation; compare coefficient magnitudes carefully.

---

### The Group Lasso

**Why it matters:** Neither the standard lasso nor the elastic net can enforce the constraint that a *known* group of variables should enter or leave the model together. When variables are naturally grouped — by domain knowledge, dummy encoding of categorical variables, or signal structure — group-level sparsity is more meaningful than variable-level sparsity.

**Intuition:** Replace the $\ell_1$ penalty on individual coefficients with an $\ell_2$ norm on each group of coefficients. An $\ell_2$ norm on a group behaves like an $\ell_1$ norm at the group level: it drives the entire group vector to zero or leaves it non-zero, but does not encourage individual coefficients within an active group to be zero.

**Prerequisites:**
- Lasso and $\ell_1$ regularisation
- Grouped structure in predictor variables (e.g. dummy-encoded categorical variables)

**How it works:**

Given $K$ pre-specified groups, the group lasso solves:

$$
\arg\min_{\boldsymbol{\beta}} \frac{1}{2} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{k=1}^{K} \|\mathbf{B}_k\|_2
$$

where $\mathbf{B}_k$ collects all coefficients $\beta_i$ belonging to group $k$. Note that for singleton groups, $\|\beta_i\|_2 = |\beta_i|$, so the standard lasso is a special case.

The $\ell_2$ norm on $\mathbf{B}_k$ creates a rotationally symmetric penalty within each group — all directions within the group are treated equally. This is why, geometrically, the group lasso constraint set has a diamond shape along the group axes but smooth (circular) cross-sections within each group.

**Key Equations:**

$$
\arg\min_{\boldsymbol{\beta}} \frac{1}{2} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{k=1}^{K} \|\mathbf{B}_k\|_2
$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Enforces group-level sparsity explicitly | Requires pre-specified groups — not data-driven |
| Natural fit for dummy-encoded categorical variables | Does not perform within-group variable selection |
| Reduces to standard lasso for singleton groups | Groups must be non-overlapping in the basic formulation |

**Python Implementation:**

```python
# sklearn does not implement the group lasso.
# Use the 'group-lasso' package:
# pip install group-lasso

from group_lasso import GroupLasso

gl = GroupLasso(groups=groups_array, group_reg=0.05, l1_reg=0)
gl.fit(X_train, y_train)
print(gl.coef_)
```

⚠️ **Theory vs. Practice:** There is no group lasso in sklearn. The `group-lasso` package's `groups` argument expects an array of integers where the $j$-th entry is the group index of predictor $j$. Getting this wrong silently fits the wrong model. The parameter `group_reg` corresponds to $\lambda$ in the lecture; `l1_reg` adds an additional within-group $\ell_1$ penalty (the sparse group lasso variant) and should be set to 0 for the pure group lasso. Do not use `GroupLasso` without specifying groups — the default behaviour may not match what you intend.

---

### Comparison: Lasso, Elastic Net, and Group Lasso

All three methods add a penalty to the residual sum of squares, but they differ in the geometry of that penalty and the type of sparsity they induce.

| Property | Lasso | Elastic Net | Group Lasso |
|----------|-------|-------------|-------------|
| Penalty | $\lambda \|\boldsymbol{\beta}\|_1$ | $\lambda\!\left(\frac{1-\alpha}{2}\|\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1\right)$ | $\lambda \sum_k \|\mathbf{B}_k\|_2$ |
| Sparsity level | Variable-level | Variable-level | Group-level |
| Handles correlated predictors | Poorly (picks one, drops rest) | Better (shares credit) | Well (if groups are known) |
| Requires group knowledge | No | No | Yes |
| Constraint set geometry | Diamond (flat edges) | Rounded diamond | Diamond along groups, circular cross-sections |
| Special cases | — | Lasso ($\alpha=1$), Ridge ($\alpha=0$) | Lasso (singleton groups) |

- The lasso sets variables to zero at corners or along edges of its constraint diamond; it has no preference for keeping correlated variables together.
- The elastic net's curved constraint boundary pulls non-zero coefficients closer together, but group structure is emergent (from correlation) not imposed.
- The group lasso is the right tool when you have explicit prior knowledge about variable groups and want all-or-nothing group inclusion.
- In practice, if groups are only weakly correlated, the elastic net may not discover them. Conversely, if domain-specified groups are wrong, the group lasso will impose incorrect structure.

---

### Penalisation in GLMs

**Why it matters:** Many real-world classification and regression problems use non-Gaussian responses (binary outcomes, counts, etc.) via GLMs. Extending penalisation to GLMs enables sparse variable selection in these settings, including high-dimensional logistic regression.

**Intuition:** In ordinary regression, the lasso penalises the RSS. In a GLM, the RSS is replaced by the negative log-likelihood. The penalty term is unchanged — the same $\ell_1$ (or other) penalty is added to the negative log-likelihood.

**Prerequisites:**
- Generalised linear models and the log-likelihood
- Lasso penalty

**How it works:**

For a GLM with likelihood $p(y | \boldsymbol{\beta}, \mathbf{x})$, the log-likelihood is:

$$
\mathcal{L}(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) = \sum_{l=1}^n \log p(y_l | \boldsymbol{\beta}, \mathbf{x}_l)
$$

Penalised estimation minimises the negative log-likelihood plus a regularisation term:

$$
\arg\min_{\boldsymbol{\beta}} -\mathcal{L}(\boldsymbol{\beta} | \mathbf{y}, \mathbf{X}) + \lambda \|\boldsymbol{\beta}\|_1
$$

When $p(y | \boldsymbol{\beta}, \mathbf{x})$ is Gaussian and the linear model $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ holds, this reduces exactly to the lasso.

---

### Sparse Logistic Regression

**Why it matters:** In high-dimensional binary classification, standard logistic regression is unidentified when $p > n$. Penalisation makes the problem well-posed and simultaneously performs variable selection.

**Intuition:** The logistic regression loss is convex. Adding a lasso penalty keeps the problem convex, so it can be solved reliably. The resulting classifier is sparse: most features receive zero weight, making it interpretable.

**How it works:**

For binary labels $i_l \in \{0, 1\}$ and the logistic model:

$$
p(1 | \boldsymbol{\beta}, \mathbf{x}) = \frac{\exp(\mathbf{x}^\top \boldsymbol{\beta})}{1 + \exp(\mathbf{x}^\top \boldsymbol{\beta})}, \quad p(0 | \boldsymbol{\beta}, \mathbf{x}) = \frac{1}{1 + \exp(\mathbf{x}^\top \boldsymbol{\beta})}
$$

The penalised problem is:

$$
\arg\min_{\boldsymbol{\beta}} -\sum_{l=1}^n \left( i_l \mathbf{x}_l^\top \boldsymbol{\beta} - \log\!\left(1 + \exp(\mathbf{x}_l^\top \boldsymbol{\beta})\right) \right) + \lambda \|\boldsymbol{\beta}\|_1
$$

The problem is convex but nonlinear in $\boldsymbol{\beta}$. It is solved using iterative quadratic approximations (IRLS) combined with coordinate descent — at each iteration the log-likelihood is locally approximated by a weighted least-squares problem, which is then solved with coordinate descent.

**Key Equations:**

$$
\arg\min_{\boldsymbol{\beta}} -\sum_{l=1}^n \left( i_l \mathbf{x}_l^\top \boldsymbol{\beta} - \log(1 + \exp(\mathbf{x}_l^\top \boldsymbol{\beta})) \right) + \lambda \|\boldsymbol{\beta}\|_1
$$

**Python Implementation:**

```python
from sklearn.linear_model import LogisticRegressionCV

# l1 penalty with liblinear or saga solver
model = LogisticRegressionCV(
    penalty='l1',
    solver='saga',    # or 'liblinear' for smaller data
    cv=5,
    max_iter=1000
)
model.fit(X_train, y_train)
print(model.coef_)
```

⚠️ **Theory vs. Practice:** sklearn's `LogisticRegression` uses `C = 1/lambda` — larger `C` means *less* regularisation, opposite to the lecture convention. Setting a small `C` corresponds to a large $\lambda$. Not all solvers support the `l1` penalty: `lbfgs` and `newton-cg` do not. Use `saga` (large datasets) or `liblinear` (small datasets). The default `solver='lbfgs'` silently falls back to no penalty or raises an error depending on the version — always specify the solver explicitly when using `l1`. Also, sklearn's `LogisticRegressionCV` searches over `C` values, not $\lambda$ values; if you want to search over a specific $\lambda$ grid, invert it (`C = 1/lambda`).

---

### Sparse Multi-Class Logistic Regression

**Why it matters:** Many real problems have more than two classes. Extending sparse logistic regression to multiple classes retains the benefits of sparsity while handling $K > 2$ outcomes.

**Intuition:** With $K$ classes, there is a coefficient vector $\boldsymbol{\beta}_i$ for each class (one class is the reference). Penalising individual entries of the coefficient matrix $\mathbf{B} \in \mathbb{R}^{p \times (K-1)}$ gives variable-level sparsity. Alternatively, applying a group lasso penalty on all coefficients for a given feature $j$ across all classes gives feature-level sparsity: either feature $j$ is used by all classifiers, or by none.

**How it works:**

For $i_l \in \{1, \ldots, K\}$ and coefficient matrix $\mathbf{B} = [\boldsymbol{\beta}_1, \ldots, \boldsymbol{\beta}_{K-1}]$:

$$
p(i | \mathbf{B}, \mathbf{x}) = \frac{\exp(\mathbf{x}^\top \boldsymbol{\beta}_i)}{1 + \sum_{j=1}^{K-1} \exp(\mathbf{x}^\top \boldsymbol{\beta}_j)}, \quad i = 1,\ldots,K-1
$$

$$
p(K | \mathbf{B}, \mathbf{x}) = \frac{1}{1 + \sum_{j=1}^{K-1} \exp(\mathbf{x}^\top \boldsymbol{\beta}_j)}
$$

Two penalisation strategies:
1. **Entry-wise $\ell_1$:** penalise $|B_{ji}|$ for each predictor $j$ and class $i$ — variable-level sparsity.
2. **Row-wise $\ell_2$ (group lasso on rows):** penalise $\|\mathbf{B}_{j\cdot}\|_2$ for each predictor $j$ — either feature $j$ contributes to all class boundaries, or none.

**Python Implementation:**

```python
from sklearn.linear_model import LogisticRegressionCV

# Multi-class with l1 penalty (entry-wise)
model = LogisticRegressionCV(
    penalty='l1',
    multi_class='multinomial',
    solver='saga',
    cv=10,
    max_iter=1000
)
model.fit(X_train, y_train)
```

⚠️ **Theory vs. Practice:** The `multi_class='multinomial'` option uses a symmetric softmax formulation rather than the $(K-1)$-reference-class formulation shown in the lecture — the parameterisation differs but the decision boundaries are equivalent up to reparameterisation. However, the number of non-zero coefficients will differ: the multinomial formulation fits $K$ coefficient vectors (with a sum-to-zero constraint) rather than $K-1$. The group lasso row-penalty variant is not available in sklearn; you need the `group-lasso` package with each row of $\mathbf{B}$ treated as a group.

---

## ✅ Key Takeaways

- Regularisation extends naturally from regression to classification — the same $\ell_1$/$\ell_2$ logic applies wherever there is an objective function to penalise.
- **Nearest shrunken centroids** apply soft-thresholding to standardised class-centroid deviations, enabling simultaneous classification and variable selection in $p \gg n$ settings.
- **The elastic net** addresses the lasso's failure on correlated predictors by blending $\ell_1$ and $\ell_2$ penalties, introducing a second tuning parameter $\alpha$.
- **The group lasso** enforces group-level sparsity when predictor groups are known in advance; it reduces to the standard lasso for singleton groups.
- **Sparse logistic regression** penalises the negative log-likelihood rather than the RSS — the same penalty term, a different loss function.
- In multi-class problems, the group lasso can enforce feature-level sparsity across all $K-1$ coefficient vectors simultaneously, which is often more interpretable than entry-wise sparsity.
- Sparsity is a key tool for interpretability: when only a few features are active, it is possible to understand *why* a model classifies as it does.