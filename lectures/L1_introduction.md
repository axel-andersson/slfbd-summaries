# Lecture 1: Introduction — Statistical Learning for Big Data

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** 23 March 2026

---

## 📋 Contents

- The Data Deluge and challenges of Big Data
- Big-$n$ vs. Big-$p$ settings
- Statistical challenges: curse of dimensionality, multiple testing, selection bias
- Statistics terminology recap: random variables, probability rules, expectation
- What is Statistical Learning?
- Statistical decision theory for regression
- $k$-Nearest Neighbours (kNN) for regression and classification
- Statistical decision theory for classification and Bayes' rule

---

## 📝 Summary

This lecture introduces the conceptual and statistical foundations of the course. It opens by framing the "data deluge" — the explosion of data across science and industry — and dissects the risks of both high-dimensional (Big-$p$) and large-sample (Big-$n$) settings. After a terminology recap covering random variables, marginalisation, conditioning, and Bayes' law, the lecture formalises the concept of Statistical Learning as minimising expected prediction error under a loss function. It derives two key theoretical results: that squared-error loss is minimised by the conditional mean (motivating regression), and that 0-1 loss is minimised by the Bayes classifier (motivating majority-vote rules). The $k$-nearest neighbour (kNN) method is introduced as a simple, non-parametric approximation to both.

---

## 🎯 Learning Goals

1. Understand what "Big Data" means statistically — both the Big-$n$ and Big-$p$ dimensions — and what pitfalls each introduces.
2. Recall core probability concepts: pmf/pdf, marginalisation, conditioning, and Bayes' law.
3. Formalise the Statistical Learning framework: model, data, loss function, expected prediction error.
4. Derive that the optimal predictor under squared-error loss is the conditional mean.
5. Derive Bayes' rule as the optimal classifier under 0-1 loss, and understand how kNN approximates it.

---

## 📚 Concepts

### The Data Deluge and Challenges of Big Data

**Why it matters:** The rapid growth of data creates both opportunities and serious methodological hazards that a practitioner must recognise before choosing or applying any model.

**Intuition:** More data sounds like a good thing — but a dataset can be large in two very different ways, and each brings its own set of problems. A large number of observations ($n$) and a large number of variables ($p$) are not the same challenge.

**Prerequisites:**
- Basic familiarity with statistical modelling
- Notion of sample vs. population

**How it works:**

"Big Data" is not just about volume. IBM's framing identifies four dimensions (the "Four Vs"): Volume, Velocity, Variety, and Veracity. In this course, "big" refers primarily to the statistical dimensions:

- **Big-$n$ setting:** many observations. Benefits include more power to detect effects and coverage of rare events. Risks include selection bias (which observations get collected?), self-reporting bias, imbalanced subpopulations, and the paradox that with large $n$ *everything* becomes statistically significant — which does not mean everything is scientifically important.

- **Big-$p$ setting:** many variables. Benefits include a more complete picture of each observation. Risks include the *curse of dimensionality* (distances become meaningless in high dimensions), spurious correlations from multiple testing, and algorithmic failures of classical methods designed for $p \ll n$.

- **Big-$n$/Big-$p$ setting:** both grow together. In practice, $p$ often grows with $n$ (e.g., genomic studies), breaking classical asymptotic theory that assumes fixed $p$.

A crucial caution running through the entire lecture: **Big Data does not remove the need to check assumptions.** Selection bias, mislabelled observations, noisy data, and spurious correlations are *more* common, not less, in large datasets. Before any analysis, invest time understanding how and why data was collected.

**Strengths and Weaknesses:**

| Strengths of Big Data | Weaknesses / Risks |
|---|---|
| More power to detect rare events | Selection and self-reporting bias |
| Better population coverage | Significance ≠ importance (Big-$n$) |
| Enables "data-hungry" methods (NNs) | Curse of dimensionality (Big-$p$) |
| Matrix completion / imputation feasible | Multiple testing → spurious correlations |
| SGD and ensemble methods scale well | Computational burden; hard to visualise |

---

### Recap: Probability Basics — Random Variables, pmf/pdf, and Key Rules

**Summary:** A discrete random variable is characterised by a probability mass function (pmf); a continuous one by a probability density function (pdf). The Bernoulli($\theta$) pmf assigns probability $\theta$ to outcome 0 and $1-\theta$ to outcome 1 — the building block of binary classification. The multivariate Gaussian with mean $\boldsymbol{\mu} \in \mathbb{R}^p$ and covariance $\boldsymbol{\Sigma} \in \mathbb{R}^{p \times p}$ is the central model for continuous features in classification and clustering. Two rules govern probability calculations everywhere in the course: **marginalisation** sums/integrates out a variable from a joint density, and **conditioning** factorises a joint density into a conditional times a marginal. Together these imply **Bayes' law**, which is the theoretical basis for classification rules and for updating prior probabilities with observed features.

$$
p(\mathbf{x}) = |2\pi\boldsymbol{\Sigma}|^{-1/2} \exp\!\left(-\tfrac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^\top \boldsymbol{\Sigma}^{-1}(\mathbf{x}-\boldsymbol{\mu})\right)
$$

$$
p(x|y) = \frac{p(y|x)\,p(x)}{p(y)} \quad \text{(Bayes' law)}
$$

---

### What is Statistical Learning?

**Why it matters:** Statistical Learning provides a unified, principled framework that explains why regression, classification, and more exotic methods all work the same way at their core.

**Intuition:** Every supervised learning problem is, at bottom, the same: find a function $f$ (or rule $c$) that maps features $\mathbf{x}$ to an outcome $y$ as accurately as possible on *new* data. The choice of loss function is what distinguishes regression from classification.

**Prerequisites:**
- Expectation and variance of a random variable
- Concept of a probability distribution over $(y, \mathbf{x})$

**How it works:**

Statistical Learning is defined as: *learn a model from data by minimising expected prediction error, as determined by a loss function.*

The four components are:

1. **Model** $f(\mathbf{x})$ — a function that maps features to a predicted outcome.
2. **Data** — labelled training pairs $(y_i, \mathbf{x}_i)$, $i = 1, \ldots, n$, where $y_i$ is the known outcome.
3. **Loss function** $L(y, \hat{y})$ — quantifies the cost of predicting $\hat{y}$ when the truth is $y$.
4. **Expected prediction error** — the expected loss over the *joint* distribution of new data:

$$
J(f) = \mathbb{E}_{p(\mathbf{x},y)}\!\left[L\!\left(y,\, f(\mathbf{x})\right)\right] = \mathbb{E}_{p(\mathbf{x})}\!\left[\mathbb{E}_{p(y|\mathbf{x})}\!\left[L\!\left(y,\, f(\mathbf{x})\right)\right]\right]
$$

The outer expectation is over the distribution of $\mathbf{x}$; the inner expectation is over the distribution of $y$ given $\mathbf{x}$. The optimal model is

$$
\hat{f} = \arg\min_f J(f).
$$

---

### Statistical Decision Theory for Regression

**Why it matters:** This derivation shows *why* regression methods target the conditional mean — it is not an arbitrary choice but a theorem.

**Intuition:** Minimising expected squared error forces your predictions to sit at the centre of mass of $y$ at each value of $\mathbf{x}$. That centre of mass is the conditional mean.

**Prerequisites:**
- Expected prediction error framework (above)
- Variance-bias decomposition of squared error
- Conditional expectation

**How it works:**

Under squared-error loss $L(y, f(\mathbf{x})) = (y - f(\mathbf{x}))^2$, focus on the inner expectation:

$$
\mathbb{E}_{p(y|\mathbf{x})}\!\left[(y - f(\mathbf{x}))^2\right]
= \operatorname{Var}_{p(y|\mathbf{x})}[y] + \left(\mathbb{E}_{p(y|\mathbf{x})}[y] - f(\mathbf{x})\right)^2
$$

The first term is irreducible noise — no model can remove it. The second term is minimised by setting

$$
\hat{f}(\mathbf{x}) = \mathbb{E}_{p(y|\mathbf{x})}[y]
$$

the **conditional mean** of $y$ given $\mathbf{x}$. All regression methods are, in principle, approximations to this quantity. Linear regression assumes the conditional mean is linear: $\mathbb{E}_{p(y|\mathbf{x})}[y] = \mathbf{x}^\top\boldsymbol{\beta}$.

**Key Equations:**

$$
J(f) = \mathbb{E}_{p(\mathbf{x},y)}\!\left[(y - f(\mathbf{x}))^2\right]
$$

$$
\hat{f}(\mathbf{x}) = \mathbb{E}_{p(y|\mathbf{x})}[y] \quad \text{(optimal predictor)}
$$

For the linear regression model $y_i = \mathbf{x}_i^\top \boldsymbol{\beta} + \varepsilon_i$ with $\varepsilon_i \sim N(0,\sigma^2)$:

$$
\mathbb{E}_{p(y|\mathbf{x})}[y] = \mathbf{x}^\top\boldsymbol{\beta}
$$

**Python Implementation:**

```python
from sklearn.linear_model import LinearRegression
import numpy as np

# n observations, p features
X = np.random.randn(100, 5)
y = X @ np.array([1, -2, 0.5, 3, -1]) + np.random.randn(100)

model = LinearRegression()
model.fit(X, y)
print(model.coef_)      # estimated beta
print(model.intercept_) # intercept (sklearn always fits one by default)
```

⚠️ **Theory vs. Practice**

- `LinearRegression` fits by OLS (minimises training RSS), which is equivalent to MLE under Gaussian noise. This matches theory *when the model is correctly specified*. If $p > n$, the normal equations are singular and sklearn uses the Moore–Penrose pseudoinverse — you will silently get a minimum-norm solution, not a unique OLS estimate. Check `np.linalg.matrix_rank(X)` before trusting results.
- `fit_intercept=True` by default. If your theory/slides centre the data, set it to `False` — otherwise sklearn adds an intercept that shifts $\hat{\boldsymbol{\beta}}$ and makes coefficient comparisons with lecture notation wrong.
- The squared-error loss in sklearn is **not** the expected prediction error $J(f)$ — it is the *training* MSE. $J(f)$ requires an independent test set or cross-validation.

---

### $k$-Nearest Neighbours (kNN) for Regression

**Why it matters:** kNN makes the connection between the abstract optimal predictor $\hat{f}(\mathbf{x}) = \mathbb{E}_{p(y|\mathbf{x})}[y]$ and a concrete, implementable algorithm explicit and intuitive.

**Intuition:** If you had infinitely many observations with $\mathbf{x}_i = \mathbf{x}$ exactly, you could average their $y$ values to estimate the conditional mean. In practice, you average over the $k$ training points *closest* to $\mathbf{x}$.

**Prerequisites:**
- Conditional mean interpretation of regression
- Euclidean distance / norms

**How it works:**

Given a query point $\mathbf{x}$, define the $k$-neighbourhood as the $k$ training predictors closest in Euclidean norm:

$$
\mathcal{N}_k(\mathbf{x}) = \{\mathbf{x}_{i_1}, \ldots, \mathbf{x}_{i_k}\}
$$

The kNN regression estimate is:

$$
\hat{f}(\mathbf{x}) = \frac{1}{k} \sum_{\mathbf{x}_{i_l} \in \mathcal{N}_k(\mathbf{x})} y_{i_l}
$$

This is a direct sample approximation of $\mathbb{E}_{p(y|\mathbf{x})}[y]$ using a local neighbourhood. Smaller $k$ gives more flexible (lower-bias, higher-variance) fits; larger $k$ gives smoother (higher-bias, lower-variance) fits.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|---|---|
| No model assumptions — fully non-parametric | Breaks down in high dimensions (curse of dimensionality) |
| Directly approximates the optimal predictor | Computationally expensive at prediction time ($O(np)$ per query) |
| Intuitive and easy to implement | Sensitive to the choice of distance metric and feature scaling |
| Flexible — $k$ controls the bias-variance trade-off | Poor performance when $p$ is large relative to $n$ |

**Python Implementation:**

```python
from sklearn.neighbors import KNeighborsRegressor
import numpy as np

X_train = np.random.randn(100, 2)
y_train = np.sin(X_train[:, 0]) + np.random.randn(100) * 0.1

model = KNeighborsRegressor(n_neighbors=5, metric='euclidean')
model.fit(X_train, y_train)

X_test = np.array([[0.5, -0.3]])
print(model.predict(X_test))
```

⚠️ **Theory vs. Practice**

- sklearn's `KNeighborsRegressor` uses Euclidean distance by default, which is theoretically justified — but only when all features are on comparable scales. If your features have different variances (e.g., age in years vs. income in thousands), the distance metric is dominated by the high-variance feature and the neighbourhood becomes meaningless. **Always standardise features before applying kNN.** sklearn does not do this automatically.
- The `metric` parameter defaults to `'minkowski'` with `p=2` (i.e., Euclidean). Setting `p=1` gives Manhattan distance. This is a first-class modelling choice that changes which points are considered "close" — it is not a tuning detail.
- In Big-$p$ settings, kNN degenerates: all points become approximately equidistant and the neighbourhood loses locality. This is the curse of dimensionality in action, not a bug in sklearn.

---

### Statistical Decision Theory for Classification and Bayes' Rule

**Why it matters:** Just as decision theory pins down the conditional mean as the optimal regression target, it pins down the class-conditional probability as the optimal classification target — and identifies the Bayes classifier as the gold standard against which all classifiers should be judged.

**Intuition:** When you classify, you want to minimise the probability of being wrong. The best strategy is always to predict whichever class has the highest posterior probability given $\mathbf{x}$.

**Prerequisites:**
- Expected prediction error framework
- Conditional probability and Bayes' law

**How it works:**

Under 0-1 loss, the loss is 0 for a correct classification and 1 for an error:

$$
L(i, c(\mathbf{x})) = \mathbf{1}(i \neq c(\mathbf{x}))
$$

The expected prediction error is:

$$
J(c) = \mathbb{E}_{p(\mathbf{x})}\!\left[\mathbb{E}_{p(i|\mathbf{x})}\!\left[\mathbf{1}(i \neq c(\mathbf{x}))\right]\right]
$$

Focusing on the inner expectation:

$$
\mathbb{E}_{p(i|\mathbf{x})}\!\left[\mathbf{1}(i \neq c(\mathbf{x}))\right] = \sum_{i \neq c(\mathbf{x})} p(i|\mathbf{x}) = 1 - p(c(\mathbf{x})|\mathbf{x})
$$

This is minimised by assigning $\mathbf{x}$ to the class with the highest posterior probability:

$$
\hat{c}(\mathbf{x}) = \arg\max_{1 \leq i \leq K} p(i|\mathbf{x})
$$

This is **Bayes' rule** (also called the Bayes classifier or Bayes optimal classifier). It achieves the lowest possible expected misclassification rate — the **Bayes error rate** — which is the irreducible floor for any classifier on a given problem.

**Key Equations:**

$$
\hat{c}(\mathbf{x}) = \arg\max_{1 \leq i \leq K} p(i|\mathbf{x})
$$

---

### $k$-Nearest Neighbours (kNN) for Classification

**Why it matters:** kNN classification is a direct, assumption-free approximation to the Bayes classifier, estimating $p(i|\mathbf{x})$ from local class frequencies.

**Intuition:** The Bayes rule requires knowing $p(i|\mathbf{x})$, which is unknown in practice. kNN estimates it by counting how many of the $k$ nearest training neighbours belong to each class, then predicting the plurality class.

**How it works:**

Given $k$ and a distance metric, find the $k$ training points $\mathcal{N}_k(\mathbf{x})$ closest to $\mathbf{x}$. Assign $\mathbf{x}$ to the most frequent class in the neighbourhood (majority vote):

$$
\hat{c}(\mathbf{x}) = \arg\max_{1 \leq i \leq K} \frac{1}{k} \sum_{\mathbf{x}_{i_l} \in \mathcal{N}_k(\mathbf{x})} \mathbf{1}(i_l = i)
$$

Two design choices govern the method's behaviour:

1. **The distance metric** — changes the shape of the neighbourhood (Euclidean, Manhattan, max-norm, etc.)
2. **The number of neighbours $k$** — controls the bias-variance trade-off. Small $k$: jagged decision boundaries, low bias, high variance. Large $k$: smooth boundaries, high bias, low variance.

**Python Implementation:**

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

# Always scale before kNN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

clf = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
clf.fit(X_train_scaled, y_train_labels)
print(clf.predict(X_test_scaled))
print(clf.predict_proba(X_test_scaled))  # estimated p(i|x)
```

⚠️ **Theory vs. Practice**

- `predict_proba` returns the fraction of neighbours in each class — this is the kNN estimate of $p(i|\mathbf{x})$. For small $k$ these estimates are extremely noisy. Do not treat them as calibrated probabilities without further work.
- sklearn breaks ties in majority vote by picking the class with the smaller index. If your problem has balanced classes and small $k$, tie-breaking can silently bias predictions toward lower-indexed classes. Use odd $k$ for binary problems to avoid ties.
- As in regression: **standardise your features.** Without scaling, the distance metric is dominated by high-variance features and the Bayes approximation fails entirely.

---

### Comparison: Regression vs. Classification in the Statistical Learning Framework

Both regression and classification are instances of the same Statistical Learning problem — they differ only in the loss function and the nature of the outcome.

| Property | Regression | Classification |
|---|---|---|
| Outcome $y$ | Continuous ($y \in \mathbb{R}$) | Discrete (class $\in \{1,\ldots,K\}$) |
| Loss function | Squared error: $(y - f(\mathbf{x}))^2$ | 0-1 loss: $\mathbf{1}(i \neq c(\mathbf{x}))$ |
| Optimal predictor | Conditional mean $\mathbb{E}_{p(y|\mathbf{x})}[y]$ | Bayes rule $\arg\max_i p(i|\mathbf{x})$ |
| kNN approximation | Average $y$ over $k$ neighbours | Majority class over $k$ neighbours |
| Linear/parametric baseline | Linear regression | Logistic regression / LDA |

- Both frameworks reveal that the central object of interest is always the **conditional distribution** $p(y|\mathbf{x})$.
- MSE loss is sensitive to outliers (squared deviations blow up); 0-1 loss ignores *how wrong* you are — a near-miss costs the same as a catastrophic error. Both have known failure modes.
- kNN works for both regression and classification with minimal modification; the only difference is the aggregation rule (mean vs. majority vote).
- In Big-$p$ settings, both regression and kNN face the curse of dimensionality — but for different reasons: regression requires more data to estimate $\boldsymbol{\beta}$ reliably; kNN requires that local neighbourhoods remain local.

---

## ✅ Key Takeaways

- "Big Data" introduces challenges in two directions: Big-$n$ (selection bias, significance vs. importance, computational burden) and Big-$p$ (curse of dimensionality, multiple testing, breakdown of classical theory). More data does not automatically mean better or more trustworthy results.
- Statistical Learning is a unified framework: every supervised method is an attempt to minimise expected prediction error $J(f)$ under a chosen loss function.
- Under squared-error loss, the optimal predictor is the **conditional mean** $\hat{f}(\mathbf{x}) = \mathbb{E}_{p(y|\mathbf{x})}[y]$. All regression methods are approximations to this.
- Under 0-1 loss, the optimal classifier is the **Bayes rule** $\hat{c}(\mathbf{x}) = \arg\max_i p(i|\mathbf{x})$. The Bayes error rate is the irreducible lower bound for any classifier.
- kNN approximates both the conditional mean (regression) and $p(i|\mathbf{x})$ (classification) using local averaging. Its performance degrades badly in high dimensions — always standardise features and be cautious when $p$ is large.
- The choice of $k$ controls the bias-variance trade-off: small $k$ overfits; large $k$ oversmooths.
- Underlying principles — conditional expectation, loss minimisation — are shared across linear regression, logistic regression, random forests, SVMs, and neural networks. The methods differ in how they parameterise or approximate the conditional mean/class probabilities, not in the goal.