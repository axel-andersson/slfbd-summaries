# Uncertainty Quantification and Beyond

**Course:** MSA220/MVE441 Statistical Learning for Big Data  
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences (Chalmers / University of Gothenburg)

---

## 📋 Contents

- Conformal Predictions
- Adaptive Prediction Sets
- Cross-Validated (Split) Conformal Predictions
- Class Conditional Conformal Predictions
- Local Explainability & LIME
- Shapley Values (SHAP)
- Counterfactuals
- Explainable Boosting Machines (EBMs)
- Missing Values & Imputation
- MICE

---

## 📝 Summary

This lecture covers two major themes in modern statistical learning: uncertainty quantification and explainability. The first half develops conformal prediction — a principled, distribution-free framework for constructing prediction sets with guaranteed coverage. Variants include adaptive sets, cross-validated calibration, and class-conditional quantiles. The second half surveys local explainability methods (LIME, SHAP, counterfactuals, EBMs) that reveal why a model made a specific prediction. The lecture closes with a practical treatment of missing data: the taxonomy of missingness mechanisms (MCAR, MAR, NMAR) and a range of imputation strategies including MICE.

---

## 🎯 Learning Goals

- Understand conformal prediction as a rigorous framework for producing prediction sets with finite-sample coverage guarantees.
- Know how to construct and calibrate non-conformity scores, and how to build adaptive, cross-validated, and class-conditional variants.
- Understand local explainability methods (LIME, SHAP, counterfactuals, EBMs) and when to apply each.
- Classify missing data mechanisms and select appropriate imputation strategies.
- Recognise the data leakage risks in imputation and implement imputation correctly using training data only.

---

## 📚 Concepts

### Recap: Prediction Confidence Without Conformal Methods

**Summary:** Standard classifiers output a single predicted label. Ensemble methods (e.g. random forests) allow a rough confidence estimate via vote fractions. Model-based methods like multinomial logistic regression output softmax probabilities; kNN can produce class probability estimates from vote fractions (unstable for small $k$); SVMs can be scored by distance to the decision boundary. These confidence estimates are biased upward on training data and lack formal coverage guarantees — motivating the conformal prediction framework.

---

### Conformal Predictions

**Why it matters:** Most classifiers return a single label. Conformal prediction replaces this with a *set* of labels that provably contains the true label with user-specified probability, regardless of the underlying model.

**Intuition:** Instead of asking "what is the most likely class?", ask "which classes are *plausible enough* given calibration data?" Hard-to-classify observations get large prediction sets; easy ones get singleton sets.

**Prerequisites:**
- Train/validation/test data splits
- Class probability outputs from a classifier (e.g. softmax probabilities)

**How it works:**

1. **Split the data** into a training set and a calibration (validation) set.
2. **Train** a classifier on the training data.
3. **Compute non-conformity scores** on the calibration set. For classification, the score for observation $i$ with true label $y_i$ is:

$$
s_i = 1 - \hat{f}(X_i)_{y_i}
$$

where $\hat{f}(X_i)_{y_i}$ is the predicted probability assigned to the correct class. A high score means the model is not confident about the true class.

4. **Compute the $1 - \alpha$ quantile** $q_\alpha$ of the calibration scores (e.g. $\alpha = 0.1$ for 90% coverage).
5. **Construct the prediction set** for a new test observation $X_\text{test}$:

$$
C(X_\text{test}) = \{ c : \hat{f}(X_\text{test})_c \geq 1 - q_\alpha \}
$$

Include every class whose predicted probability exceeds $1 - q_\alpha$.

**Key Equations:**

Coverage guarantee:

$$
1 - \alpha \leq P\!\left(y_\text{test} \in C(X_\text{test})\right) \leq 1 - \alpha + \frac{1}{n+1}
$$

Non-conformity score:

$$
s_i = 1 - \hat{f}(X_i)_{y_i}
$$

Prediction set rule:

$$
C(X_\text{test}) = \left\{ c : \hat{f}(X_\text{test})_c \geq 1 - q_\alpha \right\}
$$

Where:
- $\alpha$ = miscoverage level, chosen by user (typically 0.1)
- $q_\alpha$ = empirical $1 - \alpha$ quantile of calibration non-conformity scores
- $\hat{f}(X_i)_{y_i}$ = predicted probability for the true class of observation $i$

**Example:** Suppose $q_{1-\alpha} = 0.65$, so $1 - q_\alpha = 0.35$.
- A test observation with probabilities $(0.25, 0.70, 0.05)$ → prediction set $\{1\}$ (only class 1 exceeds 0.35).
- A test observation with probabilities $(0.55, 0.35, 0.10)$ → prediction set $\{0, 1\}$.

**Why training scores can't be used:** Class probabilities computed on training data are optimistic (the model was fitted to this data), so non-conformity scores are underestimated. Calibration on held-out data corrects this bias.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Distribution-free finite-sample coverage guarantee | Requires a separate calibration set (reduces training data) |
| Works with any classifier that outputs class probabilities | Coverage is marginal (averaged over observations), not per-observation |
| Variable prediction set size reflects difficulty | Single split can be noisy |

**Python Implementation:**

```python
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Split data
X_train, X_cal, y_train, y_cal = train_test_split(X, y, test_size=0.2)

clf = RandomForestClassifier().fit(X_train, y_train)

# Non-conformity scores on calibration set
probs_cal = clf.predict_proba(X_cal)
scores = 1 - probs_cal[np.arange(len(y_cal)), y_cal]

# Quantile threshold
alpha = 0.1
q_alpha = np.quantile(scores, 1 - alpha)  # e.g. 90th percentile

# Prediction sets on test data
probs_test = clf.predict_proba(X_test)
prediction_sets = [np.where(p >= 1 - q_alpha)[0] for p in probs_test]
```

> ⚠️ **Theory vs. Practice:** `np.quantile` computes a standard quantile; the theoretical guarantee uses a slightly corrected quantile $\lceil (n+1)(1-\alpha) \rceil / n$. For small calibration sets this difference matters — use `np.quantile(scores, np.ceil((len(scores)+1)*(1-alpha))/len(scores))` to match the theory exactly. Also, `predict_proba` in sklearn is calibrated only if you explicitly calibrate it (e.g. with `CalibratedClassifierCV`); uncalibrated probabilities can cause the prediction sets to be too wide or too narrow even with this framework.

---

### Adaptive Prediction Sets (APS)

**Why it matters:** Standard conformal prediction includes a class whenever its probability exceeds a global threshold. This can give sets that are too large for easy observations and too small for genuinely ambiguous ones. Adaptive prediction sets fix this by using a score that is sensitive to the *rank* of the true class.

**Intuition:** Sort the predicted class probabilities from largest to smallest. Keep accumulating probability mass until you "reach" the true class. This accumulated mass is the score. A hard observation has a true class buried deep in the ranking (high accumulated mass before it); an easy one has the true class at the top.

**How it works:**

For each calibration observation $i$:
1. Sort predicted class probabilities in descending order.
2. Accumulate probabilities until you include the true class $y_i$.
3. The sum of all probabilities $\geq \hat{f}(X_i)_{y_i}$ is the non-conformity score $s_i$.
4. Compute $q_\alpha$, the $1-\alpha$ quantile of these scores.

For a new test observation:
- Sort class probabilities descending.
- Include classes in order until the cumulative sum exceeds $q_\alpha$.
- This is the prediction set.

**Key Equations:**

Calibration score (APS):

$$
s_i = \sum_{c:\, \hat{f}(X_i)_c \geq \hat{f}(X_i)_{y_i}} \hat{f}(X_i)_c
$$

Prediction set:

$$
C(X_\text{test}) = \left\{ \text{top classes} : \sum \hat{f}(X_\text{test})_c > q_\alpha \right\}
$$

**Python Implementation:**

```python
# APS calibration scores
def aps_score(probs, true_label):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    rank = np.where(sorted_idx == true_label)[0][0]
    return cumsum[rank]

scores_aps = np.array([aps_score(probs_cal[i], y_cal[i]) for i in range(len(y_cal))])
q_aps = np.quantile(scores_aps, 1 - alpha)

# Prediction set
def aps_predict_set(probs, threshold):
    sorted_idx = np.argsort(probs)[::-1]
    cumsum = np.cumsum(probs[sorted_idx])
    cutoff = np.searchsorted(cumsum, threshold) + 1
    return sorted_idx[:cutoff]
```

> ⚠️ **Theory vs. Practice:** The `nonconformist` and `mapie` Python packages implement APS. The `mapie` package (v0.6+) exposes this as `method="aps"` in `MapieClassifier`. Be aware that MAPIE's default is `method="score"` (standard conformal), not APS — you must explicitly set this. Also, MAPIE's APS uses a randomised tie-breaking step; the deterministic version above may give slightly different set sizes.

---

### Cross-Validated (Split) Conformal Predictions

**Why it matters:** Using a single train/calibration split introduces variability — a lucky or unlucky split can make prediction sets systematically too wide or too narrow. Cross-validation stabilises this.

**How it works:**

1. Perform $K$-fold cross-validation on the training data.
2. For each fold $k$: train classifier $\hat{f}^k$ on the in-fold data; compute non-conformity scores on the hold-out observations.
3. Pool all calibration scores across folds.
4. Compute $q_\alpha$ from the pooled scores.
5. For a test observation, average predictions $\hat{f}^k(X_\text{test})_c$ across all $K$ classifiers, then apply the same prediction set rule as before.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| More stable prediction sets | $K$ models must be retained at inference time |
| Uses all data for both training and calibration | Computationally more expensive |

---

### Class Conditional Conformal Predictions

**Why it matters:** Some classes are inherently harder to predict than others. Using a single global quantile $q_\alpha$ will produce prediction sets that are too wide for easy classes and too narrow for hard ones. Class-conditional conformal predictions calibrate separately per class.

**How it works:**

Compute separate non-conformity score distributions for each class $c$ using only calibration observations with true label $c$. Derive a class-specific quantile $q_c$.

Prediction set:

$$
C(X_\text{test}) = \{ c : s(X_\text{test}, c) \leq q_c \}
$$

For adaptive sets: cycle through each candidate class $c$ and check whether the accumulated sorted probabilities up to and including $c$ exceed $q_c$; include $c$ if so.

**Note:** Requires sufficient calibration observations per class to estimate $q_c$ reliably. Works poorly when some classes have very few calibration examples.

---

### Local Explainability & LIME

**Why it matters:** Complex models (deep networks, boosted trees) are black boxes — they give predictions but no explanation. LIME provides local, human-interpretable explanations for any classifier's decision on a specific observation.

**Intuition:** Around any single observation, even a highly nonlinear decision boundary can be well-approximated by a simple linear model. LIME fits such a local surrogate model and reads off feature importance from its coefficients.

**Prerequisites:**
- A trained complex classifier (any type)
- Ability to generate perturbed instances near the observation of interest

**How it works:**

For a target observation $x$:
1. **Perturb** the feature space around $x$ by adding noise to generate a synthetic dataset $Z$.
2. **Predict** labels/probabilities on $Z$ using the black-box model: $\hat{y}_Z$.
3. **Weight** the perturbed points by proximity to $x$ (e.g. via a kernel function).
4. **Fit a simple, interpretable model** (linear regression, lasso, shallow tree) to predict $\hat{y}_Z$ from $Z$, weighted by closeness to $x$.
5. **Read off coefficients** of the simple model as local feature importances.

The local model gives: which features *pushed* the prediction toward the assigned class (supporting features) and which pushed against it (contradicting features).

**Key distinction from global feature importance:** Global importance captures average impact across all data; local importance is specific to a single prediction. These can differ substantially (see the lecture illustration: $X_2$ is globally important but locally irrelevant near a decision boundary parallel to $X_2$).

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Model-agnostic — works with any classifier | Local approximation may be unstable |
| Interpretable output (linear coefficients) | Sensitive to perturbation kernel and sample size |
| Available in both R and Python | Does not capture global behaviour |

**Python Implementation:**

```python
import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train,
    feature_names=feature_names,
    class_names=class_names,
    mode='classification'
)

exp = explainer.explain_instance(X_test[i], clf.predict_proba, num_features=10)
exp.show_in_notebook()
```

> ⚠️ **Theory vs. Practice:** LIME results are stochastic — re-running `explain_instance` on the same point can give different feature importances because $Z$ is randomly sampled. For stable explanations, increase `num_samples` (default 5000) substantially. The kernel width parameter (`kernel_width`) drastically affects which region is considered "local" — the default is $\sqrt{n\_features} \times 0.75$, which can be far too wide in high dimensions. Do not interpret LIME coefficients as causal effects; they are only a local linear approximation of the model's behaviour.

---

### Shapley Values (SHAP)

**Why it matters:** SHAP provides a theoretically grounded way to attribute a model's prediction for a single observation to each individual feature, based on cooperative game theory.

**Intuition:** Ask: "How much does including feature $j$ change the prediction, averaged over all possible subsets of other features?" A feature that consistently increases (or decreases) the prediction has a high positive (or negative) SHAP value.

**How it works (approximation):**

For each observation $O$ and each feature $j$:
1. Randomly sample a background data point $z$.
2. Randomly permute the feature order.
3. Create two instances:
   - $O_1$: all features *after* $j$ in the permutation replaced with values from $z$; features before $j$ from $O$.
   - $O_2$: feature $j$ and all features after it replaced with values from $z$; features before from $O$.
4. Compute $\hat{f}(O_1) - \hat{f}(O_2)$ as one SHAP estimate for feature $j$.
5. Repeat many times; average to get the SHAP value.

**Key Equations:**

The exact Shapley value for feature $j$:

$$
\phi_j = \sum_{S \subseteq F \setminus \{j\}} \frac{|S|!\,(|F|-|S|-1)!}{|F|!} \left[\hat{f}(S \cup \{j\}) - \hat{f}(S)\right]
$$

Where:
- $F$ = full feature set
- $S$ = a subset of features excluding $j$
- $\hat{f}(S)$ = model prediction using only features in $S$ (others marginalised out)

**Python Implementation:**

```python
import shap

explainer = shap.TreeExplainer(clf)  # Fast exact computation for tree models
shap_values = explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, feature_names=feature_names)
shap.waterfall_plot(shap.Explanation(values=shap_values[0][i], base_values=explainer.expected_value[0]))
```

> ⚠️ **Theory vs. Practice:** `shap.TreeExplainer` computes exact Shapley values efficiently for tree-based models (XGBoost, LightGBM, sklearn trees). For other models, `shap.KernelExplainer` approximates Shapley values but is very slow — do not apply it to large datasets without sampling. The default background dataset used to marginalise absent features is critical: `shap.sample(X_train, 100)` is a common choice but affects all SHAP values. SHAP values sum to the prediction minus the base value (`explainer.expected_value`) — verify this as a sanity check.

---

### Counterfactuals

**Why it matters:** Instead of asking "why was this predicted class $c$?", counterfactuals ask "what is the smallest change to this observation that would change the prediction to class $c'$?" This is directly actionable in domains like credit scoring, clinical decision support, and policy analysis.

**How it works:**

Find a modified observation $x'$ that minimises a multi-objective loss:

$$
L(x, x', c') = d(x, x') + \lambda_1 \|x - x'\|_0 + \lambda_2 \cdot d_\text{data}(x') + \lambda_3 \cdot \ell(c', \hat{f}(x'))
$$

Where:
- $d(x, x')$ = distance between original and counterfactual (keep change small)
- $\|x - x'\|_0$ = sparsity: how many features changed (prefer few changes)
- $d_\text{data}(x')$ = distance to nearest real data point (keep the counterfactual realistic)
- $\ell(c', \hat{f}(x'))$ = classification loss for the desired target class $c'$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Actionable, human-interpretable output | Optimisation can be expensive |
| Can reveal intervention mechanisms | May produce unrealistic instances if not regularised |
| Useful for bias/fairness audits | Multiple valid counterfactuals may exist |

**Python Implementation:**

```python
import dice_ml

data = dice_ml.Data(dataframe=df, continuous_features=cont_feats, outcome_name='target')
model = dice_ml.Model(model=clf, backend='sklearn')
exp = dice_ml.Dice(data, model, method='random')

cf = exp.generate_counterfactuals(X_test.iloc[[i]], total_CFs=3, desired_class='opposite')
cf.visualize_as_dataframe()
```

> ⚠️ **Theory vs. Practice:** `dice_ml` supports multiple backends (`sklearn`, `tensorflow`, `pytorch`) — match this to your model type or predictions will be wrong. The `method='random'` option is fast but produces lower-quality counterfactuals; use `method='genetic'` or `method='kdtree'` for more realistic outputs. The proximity-to-data regularisation term is only active when `proximity_weight > 0` — the default may not sufficiently penalise unrealistic instances.

---

### Explainable Boosting Machines (EBMs)

**Why it matters:** EBMs are "glassbox" models — interpretable by construction rather than explained post-hoc. For tabular data they are often competitive with gradient boosting while providing exact, transparent feature contributions.

**Intuition:** A standard GAM sums up individual feature functions $f_k(x_k)$. An EBM uses boosted shallow trees to learn these functions, and extends GAMs to include pairwise interactions. Because each tree in the boosting procedure is dedicated to a single feature (or feature pair), contributions are cleanly attributable.

**Prerequisites:**
- Generalised Additive Models (GAMs)
- Gradient boosting

**How it works:**

**Step 1 — Main effects.** Iterate through features sequentially, fitting a shallow tree (often a stump) to residuals for each feature in turn. Each pass through all features uses a small learning rate (as in standard gradient boosting). Repeating this many times produces clean, well-estimated main effect functions $f_k$.

**Step 2 — Pairwise interactions.** Repeat the same procedure with stumps that split on two features simultaneously, learning $f_{k,l}(x_k, x_l)$.

**Model:**

$$
E(y_i) = \sigma\!\left(\alpha + \sum_k f_k(x_{ik}) + \sum_{k,l} f_{k,l}(x_{ik}, x_{il})\right)
$$

**Prediction for a new observation:** EBMs build a massive lookup table over binned feature values. For a new $x$, read off $f_k(x_k)$ and $f_{k,l}(x_k, x_l)$ directly — fully transparent.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Fully interpretable, exact feature contributions | Does not scale well with large numbers of features |
| Competitive accuracy with gradient boosting on tabular data | Pairwise interactions grow as $O(p^2)$ |
| Built-in individual-prediction explanations | Higher-order interactions require explicit specification |

**Python Implementation:**

```python
from interpret.glassbox import ExplainableBoostingClassifier

ebm = ExplainableBoostingClassifier(interactions=10)
ebm.fit(X_train, y_train)

from interpret import show
ebm_global = ebm.explain_global()
show(ebm_global)  # View learned f_k functions

ebm_local = ebm.explain_local(X_test[:5], y_test[:5])
show(ebm_local)   # View per-observation contributions
```

> ⚠️ **Theory vs. Practice:** `ExplainableBoostingClassifier` is from Microsoft's `interpret` package (InterpretML), not sklearn. The `interactions` parameter controls the number of pairwise terms included — setting it too high will slow training dramatically and risks overfitting. The main effect functions $f_k$ are visualised as bar charts over binned feature values; these are exact contributions, not approximations. If you have hundreds of features, reduce them first (e.g. with a random forest importance ranking) before fitting an EBM — the documentation recommends staying well under 100 features for interactive use.

---

### Missing Values — Taxonomy

**Why it matters:** Ignoring missingness or handling it incorrectly can bias estimates, reduce effective sample size, and invalidate conclusions. The appropriate strategy depends entirely on the missingness mechanism.

**Three mechanisms:**

**MCAR — Missing Completely At Random**
Missingness is independent of all observed and unobserved values. Equivalent to randomly poking holes in the data. Safe to drop or use mean/median imputation, though you lose power.

**MAR — Missing At Random**
Missingness depends on other *observed* features, but not on the missing value itself. The most practically common case. Dropping these observations biases the dataset. Correct approach: impute using regression on other features.

**NMAR — Not Missing At Random**
Missingness depends on the value of the missing feature itself. Structurally problematic — the study design may be flawed. For missing outcomes, survival/censoring models are appropriate. Requires specialist expertise.

**Train/test imputation rule:** Always fit the imputation model on training data only. Apply it to test data. Never impute using the outcome variable. Never reserve "clean" (non-missing) observations exclusively for test sets in MAR/NMAR situations — this skews the evaluation.

---

### Missing Values — Imputation Methods

**Comparison: Imputation Strategies**

Simple methods work only under MCAR; model-based methods handle MAR. Adding prediction error to regression imputation restores the natural spread of the data.

| Method | Works under | Notes |
|--------|-------------|-------|
| Drop observations | MCAR only | Loses power; biases analysis under MAR/NMAR |
| Mean / median imputation | MCAR only | Reduces variance; inflates correlations |
| Random imputation | Initialisation step | Used as starting point for iterative methods |
| kNN imputation | MAR | Uses similar observations; sensitive to scaling |
| Regression imputation | MAR | Deterministic: collapses spread, inflates feature correlations |
| Regression + prediction error | MAR | Adds noise back; preserves variance structure |
| Matrix completion | MAR (low-rank data) | Exploits feature redundancy; efficient for wide matrices |
| Multiple imputation (MICE) | MAR | Gold standard; imputes multiple times, propagates uncertainty |

- Deterministic regression imputation collapses all imputed values to the regression line, artificially increasing correlations between features. Always add back a prediction error term.
- Multiple imputation (MICE) creates $M$ imputed datasets, analyses each, and pools results. This properly propagates uncertainty from imputation into all downstream estimates.
- Data leakage warning: imputing on the full dataset (train + test together) leaks test information into training. Impute on training data; transform test data with the fitted imputation model.

**Python Implementation:**

```python
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  # MICE-style

# Mean imputation (MCAR only)
imp_mean = SimpleImputer(strategy='mean').fit(X_train)
X_train_imputed = imp_mean.transform(X_train)
X_test_imputed = imp_mean.transform(X_test)  # Use training fit only

# MICE-style iterative imputation (MAR)
imp_mice = IterativeImputer(max_iter=10, random_state=42).fit(X_train)
X_train_imputed = imp_mice.transform(X_train)
X_test_imputed = imp_mice.transform(X_test)
```

> ⚠️ **Theory vs. Practice:** `IterativeImputer` in sklearn is experimental (requires `enable_iterative_imputer`) and implements a single-imputation version of MICE, not true multiple imputation — it does not propagate uncertainty across multiple imputed datasets. For proper multiple imputation with pooled inference, use the `miceforest` or `mice` (R) packages. `SimpleImputer` with `strategy='mean'` will give wrong uncertainty estimates under MAR — this is not "safe to use" outside MCAR.

---

## ✅ Key Takeaways

- Conformal prediction replaces single-label outputs with prediction *sets* that carry a finite-sample coverage guarantee: $P(y_\text{test} \in C(X_\text{test})) \geq 1 - \alpha$.
- Non-conformity scores must be computed on held-out calibration data, not training data — training scores are optimistically biased.
- Adaptive prediction sets produce smaller sets for easy observations and larger sets for hard ones, making the output more informative.
- Cross-validation over the calibration procedure reduces variance; class-conditional quantiles correct for heterogeneous class difficulty.
- LIME builds local linear surrogate models; SHAP attributes predictions using Shapley values from cooperative game theory; counterfactuals ask "what is the minimal change to flip the prediction?". All three are model-agnostic.
- EBMs are interpretable by construction: feature contributions $f_k(x_k)$ and $f_{k,l}(x_k, x_l)$ are exact lookup values, not approximations.
- The missingness mechanism (MCAR / MAR / NMAR) determines the valid imputation strategy — using mean imputation under MAR biases all downstream estimates.
- Imputation must always be fitted on training data and applied to test data. The outcome variable must never be used in imputation.
- Deterministic regression imputation collapses the variance of imputed features; always add prediction noise back.