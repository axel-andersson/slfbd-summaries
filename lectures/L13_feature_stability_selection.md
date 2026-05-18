# High-Dimensional Inference: Feature & Stability Selection

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** 2026-05-07

---

## 📋 Contents

- Lasso as a Selection Mechanism (Recap)
- The Selection Bias Problem
- Sample-Splitting
- Multi Sample-Splitting
- Stability Selection
- De-sparsified (De-biased) Lasso
- Bias-Corrected Ridge Regression

---

## 📝 Summary

This lecture addresses a fundamental tension in high-dimensional statistics: the lasso is excellent at selecting sparse models, but using the same data for both selection and inference (p-values, confidence intervals) produces dangerously optimistic results. The lecture surveys several principled alternatives — sample-splitting, multi sample-splitting, stability selection, and de-biasing methods — each offering a different trade-off between statistical validity, reproducibility, and computational cost.

---

## 🎯 Learning Goals

- Understand why refitting a lasso-selected model with OLS on the same data yields invalid p-values.
- Be able to describe and apply the sample-splitting procedure for valid post-selection inference.
- Understand the "p-value lottery" problem and how multi sample-splitting resolves it.
- Understand stability selection: what stability paths are, how to threshold them, and the formal bound on false discoveries.
- Know the core idea behind de-sparsified (de-biased) lasso and bias-corrected ridge, and when each is used.

---

## 📚 Concepts

### Recap: Lasso

**Summary:** The lasso solves

$$
\hat{\boldsymbol{\beta}}_{\text{lasso}}(\lambda) = \arg\min_{\boldsymbol{\beta}} \|\mathbf{y} - \mathbf{X}\boldsymbol{\beta}\|_2^2 + \lambda\|\boldsymbol{\beta}\|_1
$$

The $\ell_1$ penalty is the smallest-$q$ penalty that keeps the constraint convex while producing exact zeros, giving automatic feature selection. When features are correlated, selection can be unstable and pivot between correlated predictors; elastic net, group lasso, or filtering can help. Under mild conditions on the design and an appropriately chosen $\lambda$, lasso is model-selection consistent. In practice, $\lambda$ is typically chosen by cross-validation ($\lambda_{\min}$ or $\lambda_{1\text{SE}}$).

---

### The Selection Bias Problem

**Why it matters:** Practitioners often want p-values and confidence intervals alongside a selected model. A naive approach — refit the lasso-selected variables using OLS on the same data — is statistically invalid and produces misleading results.

**Intuition:** Once you use your data to decide _which_ variables to include, the data are no longer "fresh" for evaluating _how significant_ those variables are. The selection step already found the most promising variables; evaluating them on the same data double-counts the evidence.

**Prerequisites:**

- Lasso and model selection
- Ordinary least squares and logistic regression
- Basic hypothesis testing and p-values

**How it works:**

When you refit the lasso-selected model using OLS on the full dataset:

1. The model was selected _because_ these variables looked good on this data — they were already optimistically screened.
2. Refitting on the same data does not undo this: the p-values will be systematically too small (over-optimistic).
3. Additionally, the selected model may still have $p_{\text{selected}} > n$, making OLS infeasible or undefined.

This is called **selection bias**.

**Strengths and Weaknesses:**

| Approach                        | Strengths                        | Weaknesses                            |
| ------------------------------- | -------------------------------- | ------------------------------------- |
| Naive OLS refit                 | Simple, familiar output          | Biased p-values, invalid inference    |
| Valid inference methods (below) | Correct coverage, valid p-values | More complex, may sacrifice some data |

**Python Implementation:**

```python
from sklearn.linear_model import LassoCV
from sklearn.linear_model import LogisticRegressionCV
import statsmodels.api as sm
import numpy as np

# Step 1: Lasso selection
lasso = LassoCV(cv=5).fit(X, y)
selected = np.where(lasso.coef_ != 0)[0]

# Step 2 (WRONG): refit on same data — do NOT do this for inference
X_selected = X[:, selected]
model = sm.OLS(y, sm.add_constant(X_selected)).fit()
print(model.summary())  # p-values here are invalid!
```

> ⚠️ **Theory vs. Practice:** The code above is shown to illustrate what **not** to do. The p-values from `statsmodels` after lasso selection on the same data are statistically invalid — they will be far too small and will lead you to overstate evidence. There is no flag or warning from `statsmodels` that tells you this has happened. You must track whether selection and inference used the same data yourself.

---

### Sample-Splitting

**Why it matters:** Sample-splitting is the simplest way to get valid p-values after lasso selection: it uses one part of the data for selection and a completely separate part for inference, so the two steps are statistically independent.

**Intuition:** Splitting the data into two groups ensures the test set has never "seen" the selection process. The p-values computed on the test set are therefore genuinely valid — no double-dipping.

**Prerequisites:**

- Selection bias problem (above)
- Lasso
- Logistic/linear regression and hypothesis testing

**How it works (Wasserman & Roeder, 2009):**

1. Randomly split the data into two halves: set 1 (selection) and set 2 (inference).
2. Run lasso on set 1 to select a set of features $\hat{S}$.
3. Fit an unpenalized model (OLS or GLM) using only features in $\hat{S}$, but fitted _only on set 2_.
4. Read off valid p-values from this model.
5. For features not selected in step 2, assign $p = 1$.
6. Multiple-testing correction only needs to cover $|\hat{S}|$ tests, not all $p$ — a significant advantage.

**Key advantage:** Because selection used set 1 and inference used set 2, the p-values are valid by construction. Multiple testing correction is much cheaper since you only correct over the selected features.

**Strengths and Weaknesses:**

| Strengths                                              | Weaknesses                                                     |
| ------------------------------------------------------ | -------------------------------------------------------------- |
| Statistically valid p-values                           | Selection on half the data → smaller, potentially worse model  |
| Multiple testing correction over $\|\hat{S}\|$ only    | P-values are random: a different split gives different results ("p-value lottery") |
| Conceptually simple                                    | Not reproducible across runs                                   |

**Python Implementation:**

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
import numpy as np

X_sel, X_inf, y_sel, y_inf = train_test_split(X, y, test_size=0.5, random_state=42)

# Selection on set 1
lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
lasso.fit(X_sel, y_sel)
selected = np.where(lasso.coef_[0] != 0)[0]

# Inference on set 2
X_inf_sel = sm.add_constant(X_inf[:, selected])
model = sm.Logit(y_inf, X_inf_sel).fit()
print(model.summary())
```

> ⚠️ **Theory vs. Practice:** The `random_state` in `train_test_split` makes the split reproducible _within a session_, but the choice of seed arbitrarily determines which variables get selected and what p-values come out. Different seeds give genuinely different statistical conclusions — this is the p-value lottery. Do not report results from a single split as if they were stable. Use multi sample-splitting (below) instead.

---

### Multi Sample-Splitting

**Why it matters:** A single split produces irreproducible p-values. Repeating the split many times and aggregating the results gives stable, reproducible inference.

**Intuition:** By averaging over many random splits, each feature's p-value stabilizes. The aggregated p-value reflects what would happen across many possible train/test divisions, not just one lucky or unlucky draw.

**Prerequisites:**

- Sample-splitting (above)
- Basic p-value aggregation / multiple testing

**How it works (Meinshausen et al., 2009):**

1. Repeat $B$ times:
    - Randomly split data into set 1 and set 2.
    - Run lasso on set 1 → selected set $\hat{S}^{(b)}$.
    - Fit unpenalized model on set 2 using $\hat{S}^{(b)}$; correct for multiple testing over $|\hat{S}^{(b)}|$.
    - Record p-value $p_j^{(b)}$ for each feature $j$.
2. Aggregate the $B$ p-values per feature.

**Aggregation:** The $B$ p-values for each feature are **not independent** (the splits overlap), so they cannot be simply averaged. Options:

- Use the **median** p-value across splits.
- Search for the optimal quantile and adjust for the search (to avoid optimistically picking the best quantile).

The `hdi` package in R implements this procedure.

**Strengths and Weaknesses:**

| Strengths                                | Weaknesses                                                |
| ---------------------------------------- | --------------------------------------------------------- |
| Reproducible, stable p-values            | Still sacrifices data for inference                       |
| Valid multiple testing correction        | Computationally expensive ($B$ fits)                      |
| General-purpose, works with any selector | Aggregation requires care due to dependence across splits |

**Python Implementation:**

```python
# No standard sklearn implementation; use R's hdi package or manual loop
# Sketch only:
import numpy as np
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.model_selection import train_test_split

B = 100
pval_matrix = np.ones((B, X.shape[1]))

for b in range(B):
    X_sel, X_inf, y_sel, y_inf = train_test_split(X, y, test_size=0.5)
    lasso = LogisticRegression(penalty='l1', C=0.1, solver='liblinear')
    lasso.fit(X_sel, y_sel)
    selected = np.where(lasso.coef_[0] != 0)[0]
    if len(selected) > 0:
        model = sm.Logit(y_inf, sm.add_constant(X_inf[:, selected])).fit(disp=0)
        for i, j in enumerate(selected):
            pval_matrix[b, j] = model.pvalues[i + 1]

# Aggregate: use median per feature
aggregated_pvals = np.median(pval_matrix, axis=0)
```

> ⚠️ **Theory vs. Practice:** The aggregation step is non-trivial. Taking the raw minimum or mean over splits is **invalid** and will give you falsely small p-values. The correct aggregation accounts for the dependence between splits and for the implicit search over quantiles. Use a validated implementation (e.g. R's `hdi`) rather than rolling your own aggregation without the appropriate corrections.

---

### Stability Selection

**Why it matters:** Rather than targeting p-values, stability selection provides a principled way to decide _which_ features to include, with a formal bound on the expected number of false discoveries. It is widely used in network modeling and genomics.

**Intuition:** If a feature is truly relevant, it should be selected consistently across many resampled versions of the data and across a range of regularisation strengths. Features that only appear in the model for some splits or some $\lambda$ values are likely noise.

**Prerequisites:**

- Lasso and the regularisation path
- Subsampling/bootstrapping

**How it works (Meinshausen & Bühlmann, 2010):**

1. Choose a grid of regularisation values $\Lambda$.
2. Repeat many times: subsample half the data and run lasso across all $\lambda \in \Lambda$.
3. For each feature $j$ and each $\lambda$, record the **selection proportion** $\hat{\Pi}_j^\lambda$ — the fraction of subsamples in which feature $j$ was selected.
4. Plot **stability paths**: $\hat{\Pi}_j^\lambda$ vs. $\lambda$ for each feature.
5. Define the **stable set**:

$$
S^{\text{stable}} = \left\{ j : \max_{\lambda \in \Lambda} \hat{\Pi}_j^\lambda \geq \pi_{\text{thr}} \right\}
$$

for a user-chosen threshold $\pi_{\text{thr}} \in (0.5, 1)$ (typically 0.6–0.9).

**Key Equations:**

The expected number of falsely selected features $V$ satisfies:

$$
\mathbb{E}(V) \leq \frac{1}{2\pi_{\text{thr}} - 1} \cdot \frac{q_\Lambda^2}{p}
$$

Where:

- $\pi_{\text{thr}}$ = selection probability threshold
- $q_\Lambda$ = average model size across subsamples and range $\Lambda$
- $p$ = total number of candidate features

**How to use the bound in practice:**

- Fix $\pi_{\text{thr}}$ (e.g. 0.8) and a tolerable number of false positives (e.g. 1). The bound tells you what $q_\Lambda$ (and thus what range $\Lambda$) is permissible.
- Alternatively, fix $\pi_{\text{thr}}$ and $\Lambda$ (e.g. models up to size $\sim\!\sqrt{p}$) and compute the expected false discovery count directly.

**Strengths and Weaknesses:**

| Strengths                                  | Weaknesses                                         |
| ------------------------------------------ | -------------------------------------------------- |
| Formal bound on false discoveries          | Does not directly produce p-values                 |
| Robust to choice of single $\lambda$       | Requires choosing $\pi_{\text{thr}}$ and $\Lambda$ |
| Works well with network/graphical models   | Computationally intensive (many fits)              |
| Reduces sensitivity to correlated features | Bound is conservative in practice                  |

**Python Implementation:**

```python
from sklearn.linear_model import LassoCV, Lasso
from sklearn.utils import resample
import numpy as np

n, p = X.shape
lambdas = np.logspace(-3, 0, 50)
B = 100
selection_counts = np.zeros((p, len(lambdas)))

for b in range(B):
    X_sub, y_sub = resample(X, y, n_samples=n // 2, replace=False)
    for l_idx, lam in enumerate(lambdas):
        model = Lasso(alpha=lam, max_iter=5000)
        model.fit(X_sub, y_sub)
        selection_counts[:, l_idx] += (model.coef_ != 0).astype(float)

selection_probs = selection_counts / B  # shape: (p, len(lambdas))
max_probs = selection_probs.max(axis=1)  # max over lambda for each feature

pi_thr = 0.8
stable_set = np.where(max_probs >= pi_thr)[0]
print("Stable features:", stable_set)
```

> ⚠️ **Theory vs. Practice:** Several gaps between the theory and a naive sklearn implementation:
>
> - **Subsampling vs. bootstrap.** The formal bound assumes subsamples drawn *without* replacement (exactly $n/2$ observations). Using `resample(..., replace=True)` (bootstrap) changes the overlap distribution between subsamples and invalidates the bound. The code above uses `replace=False` — keep it that way.
>
> - **Exact zeros from coordinate descent.** Sklearn's `Lasso` uses coordinate descent, which produces exact zeros via soft-thresholding *only when fully converged*. With $p > n$, convergence is slower; at the default `max_iter=1000`, a coefficient that should be zero may land at `1e-14`, and `coef_ != 0` counts that as selected. Use a tolerance: `np.abs(coef_) > 1e-6`. Watch for `ConvergenceWarning` — it is easy to suppress accidentally with `warnings.filterwarnings('ignore')`.
>
> - **At most $n$ features can be selected when $p > n$.** This is a structural property of the lasso, not a software bug: with $n$ observations the lasso solution has at most $n$ non-zero coefficients regardless of $p$. On each subsample of size $n/2$, the effective cap is $\sim n/2$. If the true signal involves more features than that, they will be systematically suppressed — stability selection in the $p \gg n$ regime can only recover a sparse subset of the truth, and the Meinshausen–Bühlmann bound implicitly assumes $q_\Lambda \ll n/2$.

---

### De-sparsified (De-biased) Lasso — Optional

**Why it matters:** Sample-splitting sacrifices data. The de-sparsified lasso instead corrects for the lasso's bias analytically, allowing valid p-values for _all_ $p$ coefficients simultaneously without splitting the data.

**Intuition:** In ordinary least squares when $p < n$, the estimate of $\beta_j$ can be written using the residuals $Z_j$ from regressing $X_j$ on all other predictors. This representation isolates the contribution of $X_j$. When $p > n$, OLS fails, but substituting a regularised regression to get approximate residuals $Z_j$ restores a usable (if biased) estimate — and that bias can be corrected.

**Prerequisites:**

- OLS and its matrix form
- Lasso
- Block matrix inversion

**How it works (Zhang & Zhang, 2014; van de Geer et al., 2014):**

**Step 1 — LS representation:** When $p < n$, the OLS estimate for $\beta_j$ equals

$$
\hat{\beta}_j^{\text{LS}} = \frac{\mathbf{y}' Z_j}{X_j' Z_j}
$$

where $Z_j$ is the residual from regressing $X_j$ on all other $X$s. In LS, $Z_j \perp X_k$ for $k \neq j$, so plugging in the true model gives an unbiased estimate.

**Step 2 — When $p > n$:** OLS residuals $Z_j = 0$ identically, so this breaks down. Instead, use a lasso regression of $X_j$ on the other $X$s to get approximate residuals $Z_j$.

**Step 3 — Bias correction:** With approximate $Z_j$, the estimate becomes biased. The bias-corrected estimate is:

$$
\hat{\beta}_j = \frac{\mathbf{y}' Z_j}{X_j' Z_j} - \sum_{k \neq j} \hat{\beta}_k \frac{X_k' Z_j}{X_j' Z_j}
$$

where $\hat{\beta}_k$ are the lasso estimates.

**Step 4 — Asymptotic distribution:** Under sparsity of $\beta^*$ and regularity on $X$:

$$
\sqrt{n}(\hat{\boldsymbol{\beta}} - \boldsymbol{\beta}^*) \xrightarrow{d} \mathcal{N}_p(\mathbf{0}, W)
$$

with covariance

$$
W_{jk} = \sigma_\eta \frac{Z_j' Z_k}{(X_j' Z_j)(X_k' Z_k)}
$$

This allows construction of p-values and confidence intervals for every $\beta_j$.

**Strengths and Weaknesses:**

| Strengths                               | Weaknesses                                            |
| --------------------------------------- | ----------------------------------------------------- |
| Valid p-values for all $p$ coefficients | Requires sparsity conditions on $\beta^*$             |
| No data splitting required              | Computationally expensive ($p$ auxiliary regressions) |
| Asymptotically valid CIs                | Validity depends on structural assumptions on $X$     |

**Python Implementation:**

```python
# No standard sklearn implementation.
# Use the R package `hdi` (function `lasso.proj`) or `SILGGM`.
# In Python, the package `linearmodels` does not cover this;
# manual implementation requires fitting p auxiliary lasso regressions.

# Sketch:
from sklearn.linear_model import Lasso
import numpy as np

def desparsified_lasso(X, y, alpha_main=0.1, alpha_aux=0.1):
    n, p = X.shape
    lasso_main = Lasso(alpha=alpha_main).fit(X, y)
    beta_hat = lasso_main.coef_

    Z = np.zeros_like(X)
    for j in range(p):
        X_j = X[:, j]
        X_rest = np.delete(X, j, axis=1)
        aux = Lasso(alpha=alpha_aux).fit(X_rest, X_j)
        Z[:, j] = X_j - X_rest @ aux.coef_

    beta_debias = np.zeros(p)
    for j in range(p):
        num = y @ Z[:, j]
        denom = X[:, j] @ Z[:, j]
        bias = sum(beta_hat[k] * (X[:, k] @ Z[:, j]) / denom for k in range(p) if k != j)
        beta_debias[j] = num / denom - bias
    return beta_debias
```

> ⚠️ **Theory vs. Practice:** This implementation is a sketch only — it does not compute standard errors or p-values, which require estimating $\sigma_\eta$ and the full covariance matrix $W$. The validity of the asymptotic normal distribution requires that the true $\beta^*$ is sparse and that $X$ satisfies restricted eigenvalue conditions. If these do not hold, the distribution result breaks down silently and your p-values will be invalid. Use a validated implementation such as R's `hdi::lasso.proj`.

---

### Bias-Corrected Ridge Regression

**Why it matters:** An alternative to the de-sparsified lasso that is computationally cheaper, starting from a ridge estimate rather than requiring $p$ auxiliary lasso fits.

**Intuition:** Ridge regression shrinks all coefficients, introducing bias. If you can characterise and subtract that bias — using lasso estimates — you recover approximately unbiased estimates with a known sampling distribution.

**Prerequisites:**

- Ridge regression and its closed-form solution
- Lasso

**How it works (Bühlmann, 2013):**

1. Compute the ridge estimate $\hat{\boldsymbol{\beta}}^{\text{ridge}}$.
2. Use lasso estimates $\hat{\boldsymbol{\beta}}^{\text{lasso}}$ to characterise the ridge bias.
3. Subtract the estimated bias to form a corrected estimate.
4. Derive the sampling distribution of the corrected estimate.
5. Use the distribution to compute p-values for every $\beta_j$.

Tuning parameters (ridge penalty, lasso penalty) are selected by cross-validation or other criteria. Implemented in R's `hdi` package.

**Strengths and Weaknesses:**

| Strengths                                        | Weaknesses                                         |
| ------------------------------------------------ | -------------------------------------------------- |
| Computationally cheaper than de-sparsified lasso | Requires both ridge and lasso tuning               |
| P-values for all $p$ coefficients                | Validity under same sparsity/structure assumptions |
| Closed-form ridge step                           | Less widely cited than de-sparsified lasso         |

**Python Implementation:**

```python
# No standard Python implementation.
# Use R's hdi package: hdi::ridge.proj
# In Python, the closest approach is manual; no production-ready library exists.
```

> ⚠️ **Theory vs. Practice:** The bias correction here depends on the lasso estimator being accurate about the zero/nonzero structure. If lasso misses true signals or includes false positives — which is common in finite samples — the bias correction will itself be biased, and the resulting p-values will be invalid. There is no diagnostic in any current implementation that warns you when this has occurred.

---

## ✅ Key Takeaways

- **Refitting lasso-selected variables with OLS on the same data gives invalid, over-optimistic p-values** due to selection bias. This is a subtle but serious error.
- **Sample-splitting** provides valid p-values by separating the data used for selection from the data used for inference — but at the cost of a smaller selection dataset and irreproducible results across splits.
- **Multi sample-splitting** resolves the p-value lottery by aggregating across many random splits; the median p-value (or an optimised quantile) is the stable summary.
- **Stability selection** reframes the problem: instead of p-values, it asks which features are consistently selected across subsamples and a range of $\lambda$. The Meinshausen–Bühlmann bound provides formal control over the expected number of false discoveries.
- **De-sparsified lasso** and **bias-corrected ridge** enable simultaneous inference on all $p$ coefficients without splitting data, at the cost of stronger structural assumptions and more complex implementation.
- When $p_{\text{selected}} < n$ after selection, a common and practical approach is simply to re-estimate using an unpenalised model on the selected features — but _only if selection and inference use separate data_.
