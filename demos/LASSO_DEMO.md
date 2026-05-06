# Model Selection: Lasso, Adaptive Lasso, and Elastic Net

## 📋 Contents

- The bias-variance tradeoff in regularised regression
- Ridge regression (recap)
- Lasso regression (recap)
- Adaptive Lasso
- Elastic Net
- Simulation study methodology
- Comparison: Ridge vs. Lasso vs. Adaptive Lasso vs. Elastic Net
- OLS on Lasso-selected features

---

## 📝 Summary

Regularised regression methods like ridge and lasso reduce estimation variance at the cost of introducing bias. This lecture examines how to recover sparsity *and* remove bias simultaneously through **adaptive lasso**, which reweights the penalty based on an initial estimate to penalise near-zero coefficients more aggressively than large ones. The lecture then introduces **elastic net**, which combines lasso and ridge penalties to handle highly correlated features. All methods are evaluated through Monte Carlo simulation, tracking predictive accuracy, variable selection performance (FPR, TPR, FDR), and coefficient bias across varying sample sizes and correlation structures.

---

## 🎯 Learning Goals

- Understand why lasso and ridge introduce bias and when that tradeoff is beneficial
- Understand how adaptive lasso achieves sparsity while reducing estimation bias
- Know how to implement adaptive lasso in Python via feature rescaling
- Understand elastic net and why it outperforms lasso under high within-group feature correlation
- Be able to evaluate model selection methods using FPR, TPR, FDR, and test MSE in a simulation study

---

## 📚 Concepts

### Recap: Ridge Regression

**Summary:** Ridge regression shrinks all coefficients toward zero by adding an $\ell_2$ penalty to the loss function. For uncorrelated features, the ridge estimate takes the form $\hat{\beta}_{\text{ridge}} = \frac{\hat{\beta}}{1+\lambda}$, meaning the bias grows with the magnitude of the true coefficient — large effects are shrunk the most. Ridge never produces exact zeros, so it performs no variable selection. It is most useful when many features contribute small effects and when features are highly correlated.

---

### Recap: Lasso Regression

**Summary:** Lasso adds an $\ell_1$ penalty, producing estimates of the form $\hat{\beta}_{\text{lasso}} = (|\hat{\beta}| - \lambda)^+ \operatorname{sgn}(\hat{\beta})$. Unlike ridge, the bias is a constant $\lambda$ regardless of the coefficient magnitude, and coefficients can be shrunk to exactly zero, giving automatic variable selection. The selected $\lambda$ is typically chosen by cross-validation. Lasso struggles when features are highly correlated — it tends to select one variable from a correlated group arbitrarily and discard the rest.

---

### Adaptive Lasso

**Why it matters:** Standard lasso applies the same penalty $\lambda$ to every coefficient, which introduces unnecessary bias on large, truly non-zero effects while still failing to zero out small noise coefficients aggressively enough. Adaptive lasso addresses both problems simultaneously.

**Intuition:** If you already had a rough estimate of which coefficients are large and which are small, you would penalise the small ones heavily (to zero them out) and penalise the large ones lightly (to preserve their magnitude). Adaptive lasso does exactly this using an initial estimate — typically ridge or OLS — to calibrate per-coefficient penalties.

**Prerequisites:**
- Lasso and $\ell_1$ penalisation
- Ridge regression as an initialisation step
- Cross-validation for tuning $\lambda$

**How it works:**

**Step 1 — Obtain an initial estimate.** Fit ridge regression (or OLS if $n > p$) with a small penalty to get a near-unbiased estimate $\hat{\beta}_u$.

**Step 2 — Compute adaptive weights.** For each coefficient $j$, define:

$$
\lambda_j = \frac{\lambda}{|\hat{\beta}_{u,j}|^{\gamma}}
$$

where $\gamma > 0$ controls the aggressiveness of the reweighting. Features with large initial estimates get a small effective penalty; features near zero get a large penalty and are more likely to be zeroed out.

**Step 3 — Fit weighted lasso.** Solve the lasso problem with these per-coefficient penalties.

**Key Equations:**

$$
\hat{\beta}_{\text{adaptive lasso}} = \arg\min_{\beta} \left\{ \|y - X\beta\|_2^2 + \sum_{j=1}^p \lambda_j |\beta_j| \right\}
$$

where

$$
\lambda_j = \frac{\lambda}{|\hat{\beta}_{u,j}|^{\gamma}}
$$

Where:
- $\hat{\beta}_{u,j}$ = initial (near-unbiased) estimate of coefficient $j$
- $\gamma$ = exponent controlling reweighting strength (typically $\gamma = 1$)
- $\lambda$ = global penalty parameter, tuned by cross-validation

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Reduces bias on large true coefficients | Requires a good initial estimate |
| More aggressive zeroing of noise coefficients | Two-step procedure adds complexity |
| Oracle property: consistent variable selection under regularity conditions | Performance degrades if initial estimate is poor (e.g. $n \ll p$) |

**Python Implementation:**

```python
import numpy as np
from sklearn.linear_model import Ridge, LassoCV
from sklearn.preprocessing import StandardScaler

# Step 0: Standardise features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 1: Initial ridge estimate with small penalty
ridge_init = Ridge(alpha=0.01)
ridge_init.fit(X_scaled, y)
beta_init = ridge_init.coef_

# Step 2: Compute adaptive penalty weights
gamma = 1
penalty_weights = 1.0 / (np.abs(beta_init) ** gamma + 1e-6)
penalty_weights /= np.mean(penalty_weights)  # normalise

# Step 3: Rescale features by penalty weights and fit lasso
X_weighted = X_scaled / penalty_weights
alasso_cv = LassoCV(cv=10, max_iter=10000, random_state=42)
alasso_cv.fit(X_weighted, y)

# Step 4: Rescale coefficients back to original space
coef_alasso = alasso_cv.coef_ / penalty_weights
```

> ⚠️ **Theory vs. Practice**
>
> Sklearn's `Lasso` does not support per-coefficient penalty weights directly. The feature-rescaling trick above — dividing each column $X_j$ by $\lambda_j$ before fitting — is mathematically equivalent to adaptive lasso *only if you also divide the fitted coefficients by the same weights afterwards*. Forgetting the back-transformation gives you the wrong coefficients on the original scale.
>
> The `1e-6` stabiliser in the denominator of `penalty_weights` is necessary to prevent division by zero when an initial coefficient is exactly zero, but it also subtly caps the maximum effective penalty. If many initial coefficients are near zero, this cap can meaningfully reduce the aggressiveness of the adaptive step. Consider increasing the stabiliser with care, or replacing it with a quantile-based threshold.
>
> Using `alpha=0.01` for the initial ridge fit is a common default but is not principled. If $n$ is small relative to $p$, this ridge fit is itself highly biased, and the resulting penalty weights will be unreliable. In this regime, adaptive lasso can perform *worse* than standard lasso because it builds on a poor foundation.
>
> The global `lambda` is tuned by `LassoCV` on the rescaled features — but the effective per-coefficient penalties depend on the normalisation of `penalty_weights`. Changing the normalisation (e.g., removing `/ np.mean(penalty_weights)`) changes which lambda values `LassoCV` explores and can silently shift the selected model.

---

### Elastic Net

**Why it matters:** Lasso arbitrarily selects one variable from a group of highly correlated features and discards the rest, even when all group members genuinely contribute to the outcome. Elastic net combines the $\ell_1$ (lasso) and $\ell_2$ (ridge) penalties to get both sparsity and grouping behaviour.

**Intuition:** Think of elastic net as a tunable dial between ridge and lasso. At one extreme ($\ell_1$ ratio = 1) it is pure lasso. At the other ($\ell_1$ ratio = 0) it is pure ridge. In between, it zeroes out noise features like lasso while spreading weight across correlated true-signal features like ridge.

**Prerequisites:**
- Ridge and lasso regression
- Cross-validation for $\lambda$ selection

**How it works:**

Elastic net minimises:

$$
\hat{\beta}_{\text{EN}} = \arg\min_{\beta} \left\{ \frac{1}{2n}\|y - X\beta\|_2^2 + \lambda \left[ \frac{1-\alpha}{2} \|\beta\|_2^2 + \alpha \|\beta\|_1 \right] \right\}
$$

The mixing parameter $\alpha$ (called `l1_ratio` in sklearn) controls the balance. The global penalty $\lambda$ is tuned by cross-validation. With high within-group correlation, $\alpha < 1$ improves TPR substantially at the cost of higher FPR.

**Key Equations:**

$$
\text{Penalty} = \lambda \left[ \alpha \|\beta\|_1 + \frac{1-\alpha}{2}\|\beta\|_2^2 \right]
$$

Where:
- $\alpha$ = mixing parameter (`l1_ratio` in sklearn); $\alpha = 1$ is lasso, $\alpha = 0$ is ridge
- $\lambda$ = overall regularisation strength

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Handles highly correlated feature groups | Higher FPR than lasso when $\alpha$ is small |
| Produces sparser models than ridge | One additional hyperparameter ($\alpha$) to select |
| Ridge component stabilises estimates when $n < p$ | Can be computationally heavier than lasso |

**Python Implementation:**

```python
from sklearn.linear_model import ElasticNetCV
import numpy as np

alphas = np.logspace(-1, 2, 100)

# Elastic net with l1_ratio = 0.5
en_cv = ElasticNetCV(l1_ratio=0.5, alphas=alphas, cv=10,
                     random_state=42, max_iter=10000)
en_cv.fit(X_scaled, y)
coef_en = en_cv.coef_
```

> ⚠️ **Theory vs. Practice**
>
> Sklearn's `l1_ratio` parameter corresponds to $\alpha$ in the elastic net formula above — not to be confused with `alpha`, which is $\lambda$ (the regularisation strength). These naming conventions are exactly backwards from much of the statistical literature. If you pass `l1_ratio=0.5`, you are setting the lasso/ridge mixing, not the penalty strength.
>
> `ElasticNetCV` can accept a list of `l1_ratio` values to search over. However, jointly optimising both `l1_ratio` and `alpha` via CV can be unstable with small sample sizes — the CV surface is noisy and the selected model may vary substantially across runs. Fix `l1_ratio` based on domain knowledge when possible, and use CV only for `alpha`.
>
> The `alphas` range matters critically. Sklearn does not automatically scale `alpha` by $n$, unlike R's `glmnet`. If you use a range designed for one sample size, the path will be truncated at the wrong end for a different $n$, and CV will silently select a boundary value. Always inspect the solution path to confirm the selected `alpha` is interior to the grid.
>
> At high within-group correlation ($\rho \approx 0.95$), a small `l1_ratio` substantially increases TPR but also increases FPR. There is no free lunch — you must decide whether false positives or false negatives are more costly for your application.

---

### Simulation Study Methodology

**Why it matters:** Theoretical guarantees for variable selection methods require regularity conditions that may not hold in practice. Monte Carlo simulation with controlled ground truth is the principled way to evaluate and compare methods empirically.

**How it works:**

A sparse regression dataset is generated as follows:

1. **Features** are drawn from a multivariate normal with a block correlation structure. Features within a group share pairwise correlation $\rho$; features across groups are independent.
2. **True coefficients** $\beta$ are sparse: a fraction $f$ of the $p$ features are assigned non-zero coefficients drawn from $|\mathcal{N}(\mu_\beta, 1)|$ with random signs.
3. **Response** is generated as $y = X\beta + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2 I)$.
4. A separate **test response** $y_{\text{test}}$ is generated with the same $X$ and $\beta$ but independent noise, for unbiased evaluation of predictive error.

Each simulation is repeated $B$ times (varying only the noise), and performance metrics are averaged across replications.

**Key performance metrics:**

$$
\text{FPR} = \frac{FP}{FP + TN}, \quad \text{TPR} = \frac{TP}{TP + FN}, \quad \text{FDR} = \frac{FP}{FP + TP}
$$

$$
\text{Test MSE} = \frac{1}{n}\|y_{\text{test}} - X\hat{\beta}\|_2^2
$$

Where TP, FP, FN, TN count the correctly/incorrectly identified non-zero and zero coefficients respectively.

**Python Implementation:**

```python
import numpy as np
from scipy.linalg import cholesky

def generate_sparse_regression_data(n, p, group_sizes, group_correlations,
                                     f, coef_mean, sigma, random_seed=None):
    if random_seed is not None:
        np.random.seed(random_seed)

    X = np.zeros((n, p))
    col_idx = 0
    for group_size, corr in zip(group_sizes, group_correlations):
        if corr == 0:
            X[:, col_idx:col_idx+group_size] = np.random.normal(0, 1, (n, group_size))
        else:
            corr_matrix = np.ones((group_size, group_size)) * corr
            np.fill_diagonal(corr_matrix, 1)
            L = cholesky(corr_matrix, lower=True)
            Z = np.random.normal(0, 1, (n, group_size))
            X[:, col_idx:col_idx+group_size] = Z @ L.T
        col_idx += group_size

    num_nonzero = max(1, int(np.ceil(f * p)))
    nonzero_indices = np.random.choice(p, num_nonzero, replace=False)
    beta = np.zeros(p)
    for idx in nonzero_indices:
        magnitude = np.abs(np.random.normal(coef_mean, 1))
        sign = np.random.choice([-1, 1])
        beta[idx] = sign * magnitude

    y = X @ beta + np.random.normal(0, sigma, n)
    y_test = X @ beta + np.random.normal(0, sigma, n)
    return X, y, y_test, beta
```

> ⚠️ **Theory vs. Practice**
>
> The same design matrix $X$ is reused across Monte Carlo replications — only the noise $\varepsilon$ is resampled. This is intentional: it isolates the effect of sampling variability in the response from variability in the feature structure. If you resample $X$ as well, variance estimates will be higher and reflect a different (more realistic but harder to interpret) source of randomness.
>
> All models are fit on the standardised $X$ (`StandardScaler`), but test MSE is computed using the *unstandardised* $X$ (`X @ coef`). This is consistent only because the scaler is fit on the training $X$ and the true $\beta$ was defined on the unstandardised scale. If you refit the scaler per simulation, you must be careful to apply the same transformation to both train and test evaluation.

---

### OLS on Lasso-Selected Features

**Why it matters:** Lasso shrinks coefficients toward zero even for truly non-zero features. Refitting OLS on the selected support removes this shrinkage bias while retaining the sparsity discovered by lasso.

**Intuition:** Use lasso as a screening device to identify which features matter, then fit unpenalised OLS on only those features. This is sometimes called the *relaxed lasso* or *post-selection OLS*.

**How it works:**

1. Fit lasso with CV to identify the set $\hat{S} = \{j : |\hat{\beta}^{\text{lasso}}_j| > 0\}$.
2. Fit OLS on $X_{\hat{S}}$ (the columns selected by lasso) to obtain unpenalised estimates.
3. Set all other coefficients to zero.

**Python Implementation:**

```python
import numpy as np

# After fitting lasso_cv and obtaining coef_lasso:
selected = np.where(np.abs(coef_lasso) > 1e-6)[0]

coef_ols_lasso = np.zeros(p)
if len(selected) > 0:
    X_sel = X_scaled[:, selected]
    coef_sel = np.linalg.lstsq(X_sel, y, rcond=None)[0]
    coef_ols_lasso[selected] = coef_sel
```

> ⚠️ **Theory vs. Practice**
>
> OLS on lasso-selected features inherits all false positives from the lasso step — there is no mechanism to correct for them. In the small-$n$, high-FPR regime, the selected model can be severely overfit, and OLS will produce high-variance estimates on the noise features. Test MSE can be *worse* than lasso despite lower bias on the true features.
>
> If $|\hat{S}| > n$, `np.linalg.lstsq` silently returns the minimum-norm solution rather than raising an error. This is not OLS — it is an implicit regularised solution. Check `len(selected) < n` before calling `lstsq`.
>
> Post-selection inference is invalid with standard OLS confidence intervals. The selection event (choosing $\hat{S}$) is data-dependent, so the standard OLS $t$-tests on selected features are anti-conservative. Do not report $p$-values from this procedure without correction.

---

### Comparison: Ridge vs. Lasso vs. Adaptive Lasso vs. Elastic Net

These methods all regularise linear regression but differ in the structure of their penalty, their variable selection behaviour, and their performance under correlation.

| Property | Ridge | Lasso | Adaptive Lasso | Elastic Net |
|----------|-------|-------|----------------|-------------|
| Penalty type | $\ell_2$ | $\ell_1$ | Weighted $\ell_1$ | $\ell_1 + \ell_2$ |
| Exact zeros | No | Yes | Yes | Yes |
| Bias on large coefficients | High (grows with $|\beta|$) | Constant ($\lambda$) | Low (adaptive) | Moderate |
| Handles correlated features | Yes (grouping) | Poorly (arbitrary selection) | Poorly | Yes (grouping) |
| Variable selection consistency | No | Yes (under irrepresentability) | Yes (oracle property) | Yes |
| Hyperparameters | $\lambda$ | $\lambda$ | $\lambda$, $\gamma$ | $\lambda$, $\alpha$ |
| Python class | `RidgeCV` | `LassoCV` | `Ridge` + `LassoCV` (rescaled) | `ElasticNetCV` |

- With small $n$ relative to $p$ ($n=50$, $p=100$), all sparse methods (lasso, adaptive lasso) produce high FDR (~70%) — they select many false positives along with true ones.
- With larger $n$ ($n=200$), adaptive lasso's FDR drops dramatically (to ~20%) while maintaining full TPR — it begins to realise its oracle selection property.
- Under high within-group correlation ($\rho = 0.95$), elastic net substantially improves TPR over lasso at the cost of higher FPR. Lasso arbitrarily picks one variable per correlated group and misses others.
- OLS on lasso has lower bias than lasso on identified true features but higher variance; it does not reliably improve test MSE and can degrade it when FPR is high.

---

## ✅ Key Takeaways

- Ridge and lasso both reduce estimation variance at the cost of bias; lasso additionally performs variable selection but with constant bias $\lambda$ on all non-zero coefficients.
- Adaptive lasso applies larger penalties to near-zero coefficients and smaller penalties to large coefficients, reducing bias on true signals while maintaining sparsity — but only works well when the initial estimate is reliable (larger $n$).
- In Python, adaptive lasso is implemented by rescaling each feature $X_j$ by $1/|\hat{\beta}_{u,j}|^\gamma$ before fitting lasso, then rescaling coefficients back. Both transformations are required.
- Elastic net outperforms lasso when features are highly correlated within groups, combining ridge's grouping effect with lasso's sparsity.
- Sample size matters dramatically for variable selection: at $n=50$, $p=100$, FDR is high for all methods; at $n=200$, adaptive lasso achieves near-perfect selection.
- Solution paths should be compared on a common L1-norm axis, not on the $\lambda$ axis, because the effective penalty scales differently for lasso and adaptive lasso.
- OLS post-selection refitting removes lasso's shrinkage bias but inherits all false positives and can fail when the selected model is larger than $n$.