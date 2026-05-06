# Feature Selection and Regularised Regression

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** ???

---

## 📋 Contents

- Goals of modelling
- Feature selection: Filtering
- Feature selection: Wrapping
- Feature selection: Embedding
- Comparison: Feature selection methods
- Constrained and regularised regression
- Ridge regression
- SVD and ridge regression
- Effective degrees of freedom (Ridge)
- Lasso regression
- Intuition for the penalties
- Soft-thresholding and the lasso solution
- Relation to OLS estimates
- Shrinkage and effective degrees of freedom (Lasso)
- Notes and caveats of the lasso

---

## 📝 Summary

This lecture covers two interrelated strategies for handling regression when the number of predictors is large or highly correlated: feature selection and regularisation. Feature selection methods (filtering, wrapping, embedding) aim to identify and retain only the most relevant variables. Regularised regression — particularly ridge and lasso — stabilises or biases the OLS estimator by penalising coefficient magnitude. The lasso's L1 penalty is unique in producing exactly sparse solutions, enabling simultaneous estimation and variable selection. The trade-off between sparsity and convexity is a central theme throughout.

---

## 🎯 Learning Goals

- Understand why OLS fails when predictors are highly correlated or when $p > n$, and how feature selection and regularisation address this.
- Distinguish between filtering, wrapping, and embedding approaches to feature selection and know their practical trade-offs.
- Derive and interpret the ridge regression estimator, including its SVD form and effective degrees of freedom.
- Understand why the lasso produces sparse solutions and how soft-thresholding arises in the orthogonal design case.
- Know the conditions under which the lasso is guaranteed to recover the true model (irrepresentable condition).

---

## 📚 Concepts

### Recap: OLS and its limitations

**Summary:** The OLS estimator solves $\mathbf{y} = \mathbf{X}\boldsymbol{\beta} + \boldsymbol{\varepsilon}$ by minimising residual sum of squares, yielding $\hat{\boldsymbol{\beta}}_\text{OLS} = (\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top\mathbf{y}$. This solution requires $\mathbf{X}^\top\mathbf{X}$ to be invertible. When predictors are highly correlated or when $p > n$, the matrix becomes singular or ill-conditioned, making the solution unstable or impossible to compute. Centring and standardising predictors (and centring the outcome) simplifies interpretation and removes the need to estimate an intercept.

---

### Filtering for Feature Selection

**Why it matters:** Filtering is the fastest way to reduce the number of features before model fitting, making downstream estimation cheaper and potentially more stable.

**Intuition:** Before fitting any model, rank or score each feature independently and discard those deemed uninformative. Think of it as a pre-screening step — you never touch the model itself.

**Prerequisites:**
- Basic understanding of variance, correlation, and information theory
- Familiarity with PCA

**How it works:**
Features are selected or discarded based on summary statistics computed independently of the model:

- **Maximum variance:** Keep features with highest variance; low-variance features carry little signal.
- **PCA components:** Replace all features with the first $k$ principal components, retaining maximum variance in a lower-dimensional space.
- **F-score (univariate correlation):** Rank features by their linear correlation with the response. Select those with highest F-score.
- **Mutual Information:** Quantifies the reduction in uncertainty about $\mathbf{x}$ after observing $y$. Captures non-linear dependencies, unlike the F-score.
- **Random Forest variable importance:** Fit a random forest and use permutation importance or Gini importance to rank features.

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Fast and computationally cheap | Operates on single features; ignores interactions |
| Model-agnostic | Not tailored to the downstream estimator |
| Easy to interpret | Multiple testing corrections needed |
| Scales to very high-dimensional data | Often a pre-processing step, not true feature selection |

**Python Implementation:**
```python
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.decomposition import PCA
import numpy as np

# F-score filtering
selector = SelectKBest(score_func=f_regression, k=5)
X_filtered = selector.fit_transform(X, y)

# PCA-based filtering
pca = PCA(n_components=5)
X_pca = pca.fit_transform(X)
```

> ⚠️ **Theory vs. Practice:** `SelectKBest` with `f_regression` fits one simple linear regression per feature and uses the F-statistic — this will give you the wrong ranking if features are on very different scales and you have not standardised. `PCA` from sklearn centres data by default (`mean_`) but does **not** scale; if features have very different variances, the first component will be dominated by high-variance features. Pass `StandardScaler` first or use `whiten=True` in PCA explicitly.

---

### Wrapping for Feature Selection

**Why it matters:** Wrapping treats the feature subset as a hyperparameter and selects it by directly optimising model performance, giving a more principled criterion than filtering.

**Intuition:** Instead of scoring features individually, try different combinations and keep the one that yields the best model. The model itself acts as the judge.

**Prerequisites:**
- Cross-validation
- Understanding of model complexity and overfitting

**How it works:**

- **Best subset selection:** Evaluate all $2^p$ possible feature subsets, fitting a model for each. Impractical for large $p$ but optimal in principle.
- **Forward selection (greedy):** Start with only an intercept. At each step, add the single variable that most improves fit. Stop when adding any variable no longer helps.
- **Backward selection (greedy):** Start with all $p$ variables. At each step, remove the variable with the smallest impact on fit. Stop when removing any variable would degrade performance.

Because these methods select features discretely (in or out), small perturbations in data can lead to very different selected subsets — they exhibit **high variance**.

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Accounts for feature interactions | Computationally heavy ($O(2^p)$ for best subset) |
| Directly optimises model performance | Greedy algorithms can miss the global optimum |
| Model-specific | High variance — unstable to data perturbations |

**Python Implementation:**
```python
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import SequentialFeatureSelector

model = LinearRegression()
sfs = SequentialFeatureSelector(model, n_features_to_select=5, direction='forward', cv=5)
sfs.fit(X, y)
X_selected = sfs.transform(X)
```

> ⚠️ **Theory vs. Practice:** `SequentialFeatureSelector` uses cross-validated score as the criterion. The default `scoring` is $R^2$; if your problem is high-dimensional, this will overfit and you will select too many features. Always set `cv` to at least 5 and consider using a separate held-out test set to evaluate the final selection. The greedy nature means the subset found is **not guaranteed to be globally optimal** — this is not a limitation of the sklearn implementation; it is inherent to the algorithm.

---

### Embedding for Feature Selection

**Why it matters:** Embedding integrates feature selection directly into the estimation procedure, giving a single coherent optimisation problem rather than a two-stage pipeline.

**Intuition:** Rather than deciding which variables to include before or after fitting the model, penalise the coefficients during fitting. Large penalties push some coefficients to zero, effectively removing those features.

**Prerequisites:**
- Lagrangian duality and constrained optimisation — see `prerequisites/CONSTRAINED_OPTIMIZATION.md`
- Familiarity with norms ($\|\cdot\|_q$)

**How it works:**
The ideal embedding approach penalises the number of non-zero coefficients directly:

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda \sum_{j=1}^p \mathbf{1}(\beta_j \neq 0)
$$

This is an NP-hard discrete optimisation problem (no known algorithm can solve it efficiently for large $p$ — the only guaranteed approach is to check all $2^p$ subsets). Instead, a soft (continuous) penalty is used:

$$
\hat{\boldsymbol{\beta}} = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_q^q
$$

where $\lambda \geq 0$ is a tuning parameter. Choices of $q$ yield different methods:
- $q = 2$: Ridge regression — shrinks all coefficients, no sparsity
- $q = 1$: Lasso — convex and sparse

The constrained form (equivalent via Lagrangian duality for $q \geq 1$):

$$
\arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 \quad \text{subject to} \quad \|\boldsymbol{\beta}\|_q^q \leq t
$$

---

### Comparison: Feature Selection Methods

Filtering, wrapping, and embedding offer different trade-offs between computational cost, statistical rigour, and integration with the learning algorithm.

| Property | Filtering | Wrapping | Embedding |
|----------|-----------|----------|-----------|
| Speed | Fast | Very slow | Moderate |
| Uses model feedback | No | Yes | Yes |
| Handles interactions | No (mostly) | Yes | Depends on penalty |
| Variance of selection | Low | High | Low–moderate |
| Handles $p > n$ | Yes (some) | No (best subset) | Yes (lasso/ridge) |
| Produces sparse model | No | Yes | Lasso only |

- Filtering is best as a **pre-processing step** to reduce dimensionality before applying a more principled method.
- Wrapping is only feasible for moderate $p$ (up to ~30–50 features); greedy variants are unstable.
- Embedding via penalised regression is the preferred approach for high-dimensional data when a linear model is appropriate.
- All three methods require careful use of cross-validation to avoid overfitting the feature selection step itself.

---

### Ridge Regression

**Why it matters:** Ridge regression stabilises the OLS estimator when $\mathbf{X}^\top\mathbf{X}$ is ill-conditioned or singular, at the cost of introducing a small bias. It always has a unique solution.

**Intuition:** Add a small positive value to the diagonal of $\mathbf{X}^\top\mathbf{X}$ before inverting. This prevents the inversion from blowing up when eigenvalues are near zero. The penalty pushes all coefficients toward zero proportionally.

**Prerequisites:**
- OLS and the normal equations
- Eigenvalues and matrix invertibility

**How it works:**
For $q = 2$, the penalised problem is:

$$
\hat{\boldsymbol{\beta}}_\text{ridge}(\lambda) = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_2^2
$$

This has a closed-form solution:

$$
\hat{\boldsymbol{\beta}}_\text{ridge}(\lambda) = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_p)^{-1}\mathbf{X}^\top\mathbf{y}
$$

Adding $\lambda\mathbf{I}_p$ ensures the matrix is invertible for any $\lambda > 0$. In the special case $\mathbf{X}^\top\mathbf{X} = \mathbf{I}_p$:

$$
\hat{\beta}_{\text{ridge},j}(\lambda) = \frac{\hat{\beta}_{\text{OLS},j}}{1 + \lambda}
$$

Each OLS coefficient is uniformly shrunk by a factor of $\frac{1}{1+\lambda}$. The estimator is **biased** but has **lower variance** than OLS.

**Key Equations:**

$$
\hat{\boldsymbol{\beta}}_\text{ridge}(\lambda) = (\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_p)^{-1}\mathbf{X}^\top\mathbf{y}
$$

Where:
- $\lambda > 0$ = regularisation strength; larger $\lambda$ means more shrinkage
- $\mathbf{I}_p$ = $p \times p$ identity matrix

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Always has a unique solution | Does not perform variable selection (no zeros) |
| Closed-form solution | Biased estimator |
| Reduces variance / stabilises OLS | Requires tuning $\lambda$ |
| Handles correlated predictors well | All features remain in the model |

**Python Implementation:**
```python
from sklearn.linear_model import Ridge, RidgeCV
import numpy as np

# Fit with a fixed lambda (alpha in sklearn)
ridge = Ridge(alpha=1.0)
ridge.fit(X, y)

# Tune lambda via cross-validation
ridge_cv = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge_cv.fit(X, y)
print("Best lambda:", ridge_cv.alpha_)
```

> ⚠️ **Theory vs. Practice:** sklearn uses `alpha` where the lecture uses $\lambda$ — these are the same parameter. sklearn's `Ridge` uses a slightly different normalisation of the loss by default in some versions; always check whether the solver is operating on the unscaled or scaled design matrix. If you do not standardise `X` before fitting, coefficients on different scales will be penalised unequally, and you will not be fitting the model described in the lecture. **Always apply `StandardScaler` before `Ridge`** unless you explicitly want scale-dependent penalisation. `RidgeCV` uses efficient LOO cross-validation when `cv=None` — this is very fast but uses an analytical shortcut that is only valid for linear models; switch to `cv=5` for a faithful evaluation.

---

### SVD and Ridge Regression

**Why it matters:** Expressing ridge regression via the SVD (Singular Value Decomposition) reveals exactly which directions in the feature space are most affected by the penalty, giving geometric insight into shrinkage.

**Intuition:** The SVD decomposes $\mathbf{X}$ into principal directions. Ridge regression leaves large-eigenvalue directions nearly unchanged but aggressively shrinks small-eigenvalue directions (those associated with near-collinearity).

**How it works:**
Using $\mathbf{X} = \mathbf{U}\mathbf{D}\mathbf{V}^\top$, the ridge estimator becomes:

$$
\hat{\boldsymbol{\beta}}_\text{ridge}(\lambda) = \mathbf{V}(\mathbf{D}^2 + \lambda\mathbf{I}_p)^{-1}\mathbf{D}\mathbf{U}^\top\mathbf{y} = \sum_{j=1}^p \frac{d_j}{d_j^2 + \lambda} \mathbf{v}_j \mathbf{u}_j^\top \mathbf{y}
$$

Where:
- $d_j$ = $j$-th singular value of $\mathbf{X}$
- $\mathbf{v}_j$, $\mathbf{u}_j$ = right and left singular vectors

Each principal component is scaled by $\frac{d_j}{d_j^2 + \lambda}$. For large $d_j$ (strong directions), the factor $\approx \frac{1}{d_j}$ as in OLS. For small $d_j$ (near-collinear directions), the factor is heavily dampened. **Ridge acts most strongly on the least informative principal components.**

---

### Effective Degrees of Freedom

**Why it matters:** The effective degrees of freedom (EDF) quantifies how many parameters the ridge model is "effectively" using — it generalises the OLS concept of $p$ parameters to the regularised case.

**How it works:**
In OLS, the hat matrix $\mathbf{H} = \mathbf{X}(\mathbf{X}^\top\mathbf{X})^{-1}\mathbf{X}^\top$ has $\text{tr}(\mathbf{H}) = p$, equal to the number of free parameters.

For ridge regression, define the analogous hat matrix:

$$
\mathbf{H}(\lambda) = \mathbf{X}(\mathbf{X}^\top\mathbf{X} + \lambda\mathbf{I}_p)^{-1}\mathbf{X}^\top
$$

The effective degrees of freedom are:

$$
\text{df}(\lambda) = \text{tr}(\mathbf{H}(\lambda)) = \sum_{j=1}^p \frac{d_j^2}{d_j^2 + \lambda}
$$

- At $\lambda = 0$: $\text{df}(0) = p$ (same as OLS)
- As $\lambda \to \infty$: $\text{df}(\lambda) \to 0$ (no free parameters)
- EDF decreases monotonically with $\lambda$, providing an interpretable measure of model complexity.

---

### Lasso Regression

**Why it matters:** The lasso (L1 penalty) is the only convex penalised regression method that produces genuinely sparse solutions — many coefficients are set to exactly zero — simultaneously fitting the model and selecting variables.

**Intuition:** The L1 penalty ball has corners on the coordinate axes. When the OLS contour ellipse meets the constraint region, it tends to hit a corner, driving one or more coefficients to exactly zero. This does not happen with the smooth L2 ball (ridge).

**Prerequisites:**
- Convex optimisation
- Understanding of constraint regions and Lagrangian duality — see `prerequisites/CONSTRAINED_OPTIMIZATION.md`

**How it works:**
For $q = 1$, the penalised problem is:

$$
\hat{\boldsymbol{\beta}}_\text{lasso}(\lambda) = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1
$$

Unlike ridge, **no closed-form solution exists** for the general case. The objective is convex but non-differentiable at $\boldsymbol{\beta} = \mathbf{0}$.

In the special case $\mathbf{X}^\top\mathbf{X} = \mathbf{I}_p$, the problem decouples into $p$ independent scalar problems, each solved by **soft-thresholding**:

$$
\hat{\beta}_{\text{lasso},j} = \text{sign}(\hat{\beta}_{\text{OLS},j})(|\hat{\beta}_{\text{OLS},j}| - \lambda)_+
$$

where $(x)_+ = \max(x, 0)$. This is the **soft-thresholding operator** $\text{ST}(\hat{\beta}_{\text{OLS},j}, \lambda)$.

**Key Equations:**

$$
\hat{\boldsymbol{\beta}}_\text{lasso}(\lambda) = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1
$$

$$
\hat{\beta}_{\text{lasso},j} = \text{sign}(\hat{\beta}_{\text{OLS},j})\cdot(|\hat{\beta}_{\text{OLS},j}| - \lambda)_+ \quad \text{(orthogonal design only)}
$$

Where:
- $\lambda > 0$ = regularisation strength; increasing $\lambda$ drives more coefficients to exactly zero
- $(\cdot)_+$ = positive part function

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Produces sparse models (feature selection) | No closed-form solution in general |
| Convex optimisation problem | Biased estimates |
| Scales well with $p$ (coordinate descent) | Selects at most $n$ variables when $p > n$ |
| Interpretable: selected variables identified | Unstable with correlated predictors |

**Python Implementation:**
```python
from sklearn.linear_model import Lasso, LassoCV
import numpy as np

# Fit with fixed lambda (alpha in sklearn)
lasso = Lasso(alpha=0.1)
lasso.fit(X, y)
print("Non-zero coefficients:", np.sum(lasso.coef_ != 0))

# Tune via cross-validation
lasso_cv = LassoCV(alphas=np.logspace(-4, 1, 100), cv=5)
lasso_cv.fit(X, y)
print("Best lambda:", lasso_cv.alpha_)
```

> ⚠️ **Theory vs. Practice:** sklearn uses `alpha` for $\lambda$, but the loss function in sklearn's `Lasso` is $\frac{1}{2n}\|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \alpha\|\boldsymbol{\beta}\|_1$ — it divides the RSS by $2n$. The lecture formulation does **not** normalise by $n$. This means the same `alpha` value will produce different amounts of shrinkage depending on $n$. If you tune `alpha` on one dataset size and then change $n$, you will get the wrong model. Always re-tune `alpha` whenever your sample size changes. Also, sklearn's `Lasso` uses coordinate descent with a `max_iter` default of 1000 — for high-dimensional problems, this may not converge; set `max_iter=10000` and check `lasso.n_iter_`.

---

### Intuition for the Penalties

**Why it matters:** The geometry of the constraint region explains why $q = 1$ produces sparsity while $q = 2$ does not, and why $q < 1$ would give sparser solutions but at the cost of losing convexity.

**How it works:**
The RSS (Residual Sum of Squares) is an ellipse centred at $\hat{\boldsymbol{\beta}}_\text{OLS}$. Adding a constraint $\|\boldsymbol{\beta}\|_q^q \leq t$ restricts solutions to the interior of a $q$-ball. The constrained solution is where the RSS ellipse first touches the constraint region.

- **$q = 2$ (Ridge):** The constraint region is a sphere — smooth boundary. The RSS ellipse touches it at a non-axis point, so neither coefficient is forced to zero.
- **$q = 1$ (Lasso):** The constraint region is a diamond with corners on the axes. The RSS ellipse is likely to first contact the diamond at a corner, which sets one coordinate to zero. This is the geometric origin of sparsity.
- **$q < 1$:** Constraint region is non-convex (concave along axes), making the optimisation NP-hard but producing even sparser solutions.
- **$q = \infty$:** Constraint region is a cube; no sparsity.

**Key rule:** Sparsity requires $q \leq 1$; convexity (tractable optimisation) requires $q \geq 1$. The lasso ($q = 1$) is the unique method satisfying both simultaneously.

---

### Comparison: Ridge vs. Lasso

Ridge and lasso both penalise regression coefficients but differ fundamentally in the type of shrinkage they produce and their suitability for variable selection.

| Property | Ridge ($q=2$) | Lasso ($q=1$) |
|----------|---------------|---------------|
| Closed-form solution | Yes | No (in general) |
| Sparse solutions | No | Yes |
| Convex | Yes | Yes |
| Handles correlated predictors | Well (spreads weights) | Poorly (selects arbitrarily) |
| Feature selection | No | Yes |
| Degrees of freedom | Continuous: $\sum d_j^2/(d_j^2+\lambda)$ | Number of non-zero coefficients |
| Bias | Yes | Yes |

- If the true model is dense (many small effects), ridge will outperform lasso.
- If the true model is sparse (few large effects), lasso will select the correct variables given enough data and the irrepresentable condition holds.
- When predictors are correlated, lasso tends to arbitrarily pick one from a correlated group; ridge includes all with equal weight.
- Ridge has an analytical solution and is more numerically stable; lasso requires iterative methods (coordinate descent).

---

### Notes and Caveats of the Lasso

**Why it matters:** Knowing when the lasso fails is as important as knowing when it works — using it blindly can produce a confidently wrong sparse model.

**How it works:**

**General case ($\mathbf{X}^\top\mathbf{X} \neq \mathbf{I}_p$):** No explicit solution. Numerical methods are required. The standard approach is **coordinate descent**: update each $\beta_j$ one at a time, holding all others fixed. This converges to the global minimum because the objective is convex.

The degrees of freedom equal the number of non-zero coefficients.

**Sparsity assumption:** The lasso only recovers the true model if the data-generating process is genuinely sparse. A dense process with too few observations is unidentifiable regardless of method.

**Irrepresentable condition:** Split $\mathbf{X}$ into $\mathbf{X}_1$ (relevant variables) and $\mathbf{X}_2$ (irrelevant variables). The lasso is guaranteed to select the true model (with high probability for large $n$) if:

$$
|(\mathbf{X}_2^\top\mathbf{X}_1)(\mathbf{X}_1^\top\mathbf{X}_1)^{-1}| < 1 - \boldsymbol{\eta}
$$

for some $\boldsymbol{\eta} > 0$. This condition requires that irrelevant variables are not too strongly correlated with relevant ones.

**In practice:** Neither the sparsity of the true model nor the irrepresentable condition can be verified from data. Domain knowledge and careful assumptions must be used.

---

## ✅ Key Takeaways

- OLS fails when $p > n$ or predictors are strongly correlated; feature selection and regularisation are the two main remedies.
- Filtering is fast but crude; wrapping is principled but computationally heavy and high-variance; embedding (penalised regression) is the most coherent framework for high-dimensional settings.
- Ridge regression ($q=2$) adds $\lambda\mathbf{I}_p$ to $\mathbf{X}^\top\mathbf{X}$, always yielding a unique solution. It shrinks all coefficients continuously but produces no zeros.
- The SVD decomposition shows ridge acts most strongly on low-variance (near-collinear) principal components. Effective degrees of freedom decrease continuously from $p$ to $0$ as $\lambda$ increases.
- The lasso ($q=1$) is the only convex penalty that produces exact zeros, enabling simultaneous estimation and variable selection via soft-thresholding.
- In the orthogonal case, the lasso solution is $\hat{\beta}_{\text{lasso},j} = \text{sign}(\hat{\beta}_{\text{OLS},j})(|\hat{\beta}_{\text{OLS},j}| - \lambda)_+$; in the general case, coordinate descent is required.
- Sparsity ($q \leq 1$) and convexity ($q \geq 1$) are simultaneously achievable only at $q = 1$ — making the lasso the sweet spot in the penalty landscape.
- The lasso can fail when predictors violate the irrepresentable condition; these conditions cannot be verified in practice.