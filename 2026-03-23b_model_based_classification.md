# Lecture 2a: Model-based Classification

## 📋 Contents

- Recap: Statistical Learning and Classification
- A Model for Binary Classification (Bernoulli/Logistic)
- Logistic and Probit Models
- Estimating Logistic Regression Coefficients
- Multi-class Logistic Regression (Softmax)
- Notes and Warnings on Logistic Regression
- Discriminant Analysis: A Change of Viewpoint
- Linear Discriminant Analysis (LDA) and Quadratic Discriminant Analysis (QDA)
- Comparison: DA Variants and Their Simplifications
- Summary and Big Data Concerns

---

## 📝 Summary

This lecture covers model-based approaches to classification. It begins by recalling the optimal Bayes classifier and how logistic/probit regression approximates it via a transformed linear predictor. The softmax function extends this to $K > 2$ classes. The lecture then pivots to Discriminant Analysis (DA), which instead models the class-conditional feature densities $p(\mathbf{x}|c)$ using Gaussians and applies Bayes' law. A spectrum of DA variants — from QDA to LDA to Nearest Centroids — are derived by progressively simplifying the assumed covariance structure.

---

## 🎯 Learning Goals

- Understand how logistic and probit regression model $p(c|\mathbf{x})$ and why the decision boundary is linear.
- Be able to derive the log-likelihood for logistic regression and understand how it is maximized.
- Extend logistic regression to $K > 2$ classes using the softmax function.
- Understand the generative viewpoint of Discriminant Analysis and how Bayes' law connects $p(\mathbf{x}|c)$ to $p(c|\mathbf{x})$.
- Know the parameter counts and trade-offs for QDA, LDA, and simpler DA variants.

---

## 📚 Concepts

### Recap: Statistical Learning and Classification

The theoretically optimal classification rule under 0-1 loss is the Bayes classifier:

$$\hat{c}(\mathbf{x}) = \arg\max_{1 \leq c \leq K} p(y_i = c \mid \mathbf{x})$$

This requires knowing the posterior $p(c|\mathbf{x})$, which is unavailable in practice. Two broad strategies exist: (1) approximate it directly from data (e.g. $k$-NN), or (2) impose a parametric model. The goal of this lecture is to explore option (2) rigorously.

---

### Binary Classification: The Bernoulli Model

**Why it matters:** Before adding predictors, understanding the simplest probabilistic model for binary outcomes is the foundation for logistic regression.

**Intuition:** For a binary outcome $y_i \in \{0, 1\}$, we only need to model one probability since $p(0|\mathbf{x}) + p(1|\mathbf{x}) = 1$. A Bernoulli model sets $p(1|\mathbf{x}) = \theta$ and estimates $\theta$ by maximum likelihood. The Bayes rule then classifies as class 1 if $\theta > 1/2$. The challenge is incorporating predictors $\mathbf{x}$.

**Prerequisites:**
- Bernoulli distribution and maximum likelihood estimation
- Bayes' rule for classification

**How it works:**
We model $p(1|\mathbf{x}) = \theta \in (0, 1)$. The Bayes rule gives:

$$c(\mathbf{x}) = \begin{cases} 0 & \theta \leq \tfrac{1}{2} \\ 1 & \text{otherwise} \end{cases}$$

This is a constant classifier — it ignores $\mathbf{x}$. To make the probability depend on the predictors, we need a function that maps a linear combination $\mathbf{x}^\top \boldsymbol{\beta} \in \mathbb{R}$ into $(0,1)$.

---

### Logistic and Probit Models

**Why it matters:** These are the standard models for binary classification with a linear predictor. They introduce the link function concept that generalises linear regression to bounded outputs.

**Intuition:** Both the logistic (sigmoid) function and the standard normal CDF are S-shaped curves mapping $\mathbb{R} \to (0,1)$. Composing either with a linear predictor $\mathbf{x}^\top \boldsymbol{\beta}$ gives a flexible model for $p(1|\mathbf{x})$ that retains a linear decision boundary.

**Prerequisites:**
- Logistic/sigmoid function
- Standard normal CDF $\Phi$
- Linear predictor models

**How it works:**

**Logistic model:**
$$p(1|\mathbf{x}) \approx \sigma(\mathbf{x}^\top \boldsymbol{\beta}), \quad \sigma(x) = \frac{e^x}{1 + e^x}$$

**Probit model:**
$$p(1|\mathbf{x}) \approx \Phi(\mathbf{x}^\top \boldsymbol{\beta}), \quad \Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} e^{-z^2/2}\, dz$$

Both give the same Bayes decision rule, since $\sigma$ and $\Phi$ both equal $1/2$ at zero:

$$c(\mathbf{x}) = \begin{cases} 0 & \mathbf{x}^\top \boldsymbol{\beta} \leq 0 \\ 1 & \text{otherwise} \end{cases}$$

The decision boundary is the hyperplane $\mathbf{x}^\top \boldsymbol{\beta} = 0$ — a linear classifier in feature space.

**Key Equations:**

$$\sigma(x) = \frac{\exp(x)}{1 + \exp(x)}, \qquad \Phi(x) = \int_{-\infty}^{x} \frac{1}{\sqrt{2\pi}} \exp\!\left(-\frac{z^2}{2}\right) dz$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Produces calibrated probability estimates | Linear decision boundary may underfit |
| Interpretable log-odds relationship | Sensitive to perfectly separable classes |
| Well-studied, fast estimation | Struggles in high dimensions without regularization |

**Python Implementation:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)  # columns: [P(y=0|x), P(y=1|x)]
preds = model.predict(X_test)
```

⚠️ **Theory vs. Practice:** `sklearn`'s `LogisticRegression` applies **L2 regularization by default** (`C=1.0`). This is not the unpenalized MLE described in the lecture. If you want the unpenalized estimator, set `penalty=None`. Using the default `C=1.0` will give you a regularized model, which silently biases your coefficient estimates — this will give you the wrong model if you intend to match the theoretical derivation. Also, `sklearn` uses `C = 1/λ` (inverse of penalty strength), not the conventional $\lambda$ notation from statistics courses.

---

### Estimating Logistic Regression Coefficients

**Why it matters:** Maximum likelihood estimation of $\boldsymbol{\beta}$ does not have a closed form — understanding the optimization landscape is essential for knowing when and why fitting may fail.

**How it works:**

The log-likelihood for logistic regression is:

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \mathbf{x}_i^\top \boldsymbol{\beta} - \log\!\left(1 + \exp(\mathbf{x}_i^\top \boldsymbol{\beta})\right) \right]$$

This is a **concave** function of $\boldsymbol{\beta}$, so any local maximum is a global maximum. The gradient is:

$$\nabla_{\boldsymbol{\beta}}\, \ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \mathbf{x}_i \left( y_i - \sigma(\mathbf{x}_i^\top \boldsymbol{\beta}) \right)$$

This gradient can be used in:
- **Gradient ascent**: direct use of the gradient.
- **Newton-Raphson / IRLS**: uses the Hessian to yield an iteratively reweighted least squares (IRLS) problem, which is faster and used in most implementations. (Details in ESL Ch. 4.4.1.)

**Key Equations:**

$$\ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \left[ y_i \mathbf{x}_i^\top \boldsymbol{\beta} - \log(1 + e^{\mathbf{x}_i^\top \boldsymbol{\beta}}) \right]$$

$$\nabla_{\boldsymbol{\beta}}\, \ell(\boldsymbol{\beta}) = \sum_{i=1}^{n} \mathbf{x}_i \bigl(y_i - \sigma(\mathbf{x}_i^\top \boldsymbol{\beta})\bigr)$$

---

### Multi-class Logistic Regression (Softmax Regression)

**Why it matters:** Real classification problems often have $K > 2$ classes. The softmax function is the canonical generalization of the logistic model and underlies modern neural network output layers.

**Prerequisites:**
- Binary logistic regression
- Multinomial/categorical distributions
- Notion of log-odds

**How it works:**

For $K$ classes, we need $p(c|\mathbf{x}) \in (0,1)$ for each class with $\sum_c p(c|\mathbf{x}) = 1$. Use a categorical model and the **softmax function** $\boldsymbol{\sigma}: \mathbb{R}^K \to [0,1]^K$.

Only $K-1$ parameters are needed (class $K$ serves as the reference class, with $z_K = 0$). For $c = 1, \ldots, K-1$:

$$p(y_i = c \mid \mathbf{x}) = \frac{e^{\mathbf{x}^\top \boldsymbol{\beta}_c}}{1 + \sum_{r=1}^{K-1} e^{\mathbf{x}^\top \boldsymbol{\beta}_r}}, \qquad p(K|\mathbf{x}) = \frac{1}{1 + \sum_{r=1}^{K-1} e^{\mathbf{x}^\top \boldsymbol{\beta}_r}}$$

The log-odds of class $c$ vs. the reference class $K$ are linear:

$$\log \frac{p(y_i = c|\mathbf{x})}{p(K|\mathbf{x})} = \mathbf{x}^\top \boldsymbol{\beta}_c$$

**Bayes rule:** Classify to class $K$ if $\mathbf{x}^\top \boldsymbol{\beta}_c < 0$ for all $c \in \{1,\ldots,K-1\}$; otherwise pick $\arg\max_c\, \mathbf{x}^\top \boldsymbol{\beta}_c$.

**Key Equations:**

$$[\boldsymbol{\sigma}(\mathbf{z})]^{(j)} = \frac{e^{z_j}}{\sum_{r=1}^{K} e^{z_r}} = \frac{e^{z_j - z_K}}{1 + \sum_{r=1}^{K-1} e^{z_r - z_K}}$$

Decision boundaries occur where $\mathbf{x}^\top \boldsymbol{\beta}_c = \mathbf{x}^\top \boldsymbol{\beta}_{c'}$ or $\mathbf{x}^\top \boldsymbol{\beta}_c = 0$, so all boundaries are linear.

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Produces class probabilities | $K-1$ coefficient vectors; harder to interpret |
| Principled generalization of binary logistic | High-dimensional: non-identifiability |
| Used widely in neural networks (output layer) | Sensitive to choice of reference class |

**Python Implementation:**
```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(multi_class='multinomial', solver='lbfgs')
model.fit(X_train, y_train)
probs = model.predict_proba(X_test)  # shape: (n_samples, K)
```

⚠️ **Theory vs. Practice:** `sklearn` defaults to `multi_class='auto'`, which selects `'ovr'` (one-vs-rest) for many solvers — **this is not multinomial logistic regression**. OvR fits $K$ separate binary classifiers and does not produce a valid joint probability model. To match the lecture's formulation, you must explicitly set `multi_class='multinomial'` and choose a compatible solver such as `'lbfgs'`, `'newton-cg'`, or `'sag'`. As before, L2 regularization is applied by default; set `penalty=None` for the unpenalized MLE.

---

### Notes and Warnings on Logistic Regression

**Why it matters:** Linear separability causes a fundamental failure mode that practitioners must recognize.

**How it works:**

**Linear separability:** If the two classes can be perfectly separated by a hyperplane, the MLE does not exist. Logistic regression tries to fit a step function, driving $\|\boldsymbol{\beta}\| \to \infty$. Coefficients diverge to $\pm\infty$ and the algorithm fails to converge.

**Big Data concerns:**
- **High-dimensional settings** ($p \gg n$): Coefficients are non-identifiable. Regularization (ridge, lasso, elastic net) is required.
- **Interpretability in multiclass**: Different features may dominate in different class models.
- **Over-dispersion**: More variance than the model assumes — be cautious about $p$-values and standard errors.

---

### Discriminant Analysis: Modelling the Feature Space

**Why it matters:** Instead of directly modelling $p(c|\mathbf{x})$, we can model the distribution of features $\mathbf{x}$ within each class and use Bayes' law to recover the posterior. This generative approach has strong probabilistic grounding and connects to classical statistical methods.

**Intuition:** If classes form distinct clusters in feature space (as in the Iris dataset), modelling a Gaussian distribution per class is natural. Once we have $p(\mathbf{x}|c)$ and the class priors $p(c)$, Bayes' law gives $p(c|\mathbf{x})$ for free.

**Prerequisites:**
- Multivariate Gaussian distribution
- Bayes' law
- Maximum likelihood estimation for Gaussians

**How it works:**

Apply Bayes' law:

$$p(c|\mathbf{x}) = \frac{p(\mathbf{x}|c)\, p(c)}{\sum_{j=1}^{K} p(\mathbf{x}|j)\, p(j)}$$

The core assumption of DA is:

$$p(\mathbf{x}|c) \sim \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$$

The MLE estimates from data $(y_i, \mathbf{x}_i)$ are:

$$\hat{\pi}_c = \frac{n_c}{n}, \qquad \hat{\boldsymbol{\mu}}_c = \frac{1}{n_c}\sum_{y_i = c} \mathbf{x}_i, \qquad \hat{\boldsymbol{\Sigma}}_c = \frac{1}{n_c - 1}\sum_{y_i = c}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_c)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_c)^\top$$

The discriminant score for class $j$ is:

$$\delta_j(\mathbf{x}) = \log \pi_j - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_j)^\top \boldsymbol{\Sigma}_j^{-1}(\mathbf{x} - \boldsymbol{\mu}_j) - \frac{1}{2}\log|\boldsymbol{\Sigma}_j|$$

Classification: $c(\mathbf{x}) = \arg\max_j \delta_j(\mathbf{x})$.

Since $\delta_j(\mathbf{x})$ is **quadratic** in $\mathbf{x}$, this is called **Quadratic Discriminant Analysis (QDA)**.

**Key Equations:**

$$\delta_j(\mathbf{x}) = \log \pi_j - \frac{1}{2}(\mathbf{x} - \boldsymbol{\mu}_j)^\top \boldsymbol{\Sigma}_j^{-1}(\mathbf{x} - \boldsymbol{\mu}_j) - \frac{1}{2}\log|\boldsymbol{\Sigma}_j|$$

---

### Linear Discriminant Analysis (LDA)

**Why it matters:** QDA requires estimating $K$ separate covariance matrices, which is expensive and unstable in high dimensions. Assuming a shared covariance drastically reduces parameter count and yields a linear classifier.

**How it works:**

Replace all class-specific $\boldsymbol{\Sigma}_j$ with a single **pooled** estimate:

$$\hat{\boldsymbol{\Sigma}} = \sum_{j=1}^{K} \hat{\boldsymbol{\Sigma}}_j \frac{n_j - 1}{n - K} = \frac{1}{n-K}\sum_{j=1}^{K}\sum_{y_i=j}(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)(\mathbf{x}_i - \hat{\boldsymbol{\mu}}_j)^\top$$

The discriminant score simplifies to a **linear** function of $\mathbf{x}$:

$$\delta_j(\mathbf{x}) = \log \pi_j + \mathbf{x}^\top \boldsymbol{\Sigma}^{-1} \boldsymbol{\mu}_j - \frac{1}{2}\boldsymbol{\mu}_j^\top \boldsymbol{\Sigma}^{-1}\boldsymbol{\mu}_j$$

All decision boundaries between classes are hyperplanes.

**Key Equations:**

$$\delta_j(\mathbf{x}) = \log \pi_j + \mathbf{x}^\top \hat{\boldsymbol{\Sigma}}^{-1} \hat{\boldsymbol{\mu}}_j - \frac{1}{2}\hat{\boldsymbol{\mu}}_j^\top \hat{\boldsymbol{\Sigma}}^{-1} \hat{\boldsymbol{\mu}}_j$$

**Strengths and Weaknesses:**
| Strengths | Weaknesses |
|-----------|------------|
| Far fewer parameters than QDA | Assumes equal covariance across classes |
| Linear boundaries are interpretable | Can underfit when classes have very different shapes |
| Numerically stable for moderate $p$ | Fails when $p > n - K$ (singular pooled covariance) |

**Python Implementation:**
```python
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

lda = LinearDiscriminantAnalysis()
lda.fit(X_train, y_train)
preds = lda.predict(X_test)
probs = lda.predict_proba(X_test)
```

⚠️ **Theory vs. Practice:** `sklearn`'s LDA uses an SVD-based solver by default, which handles near-singular pooled covariances gracefully — but silently so. If your pooled covariance is near-singular (common when $p$ is large), the SVD solver regularizes implicitly. This will give you the wrong model relative to the theoretical derivation. Use `solver='lsqr'` with `shrinkage='auto'` (Ledoit-Wolf) explicitly if you want controlled regularization, and be aware that the default `solver='svd'` does not support shrinkage. Also, LDA in sklearn can be used for dimensionality reduction via `lda.transform(X)` — this is a separate use case from classification.

---

### Comparison: DA Variants and Their Simplifications

The key axis of comparison is how much structure is assumed for the covariance matrices $\boldsymbol{\Sigma}_j$. Simpler assumptions mean fewer parameters and more stability, but also less flexibility.

| Property | QDA | LDA | Diagonal LDA | Naive Bayes | Nearest Centroids |
|---|---|---|---|---|---|
| Covariance assumption | $\boldsymbol{\Sigma}_j$ per class | Shared $\boldsymbol{\Sigma}$ | Shared diagonal $\boldsymbol{\Lambda}$ | Diagonal $\boldsymbol{\Lambda}_j$ | $\sigma^2 \mathbf{I}$ |
| Decision boundary | Quadratic | Linear | Linear | Quadratic | Linear (Voronoi) |
| Parameters for covariance | $K \cdot p(p+1)/2$ | $p(p+1)/2$ | $p$ | $K \cdot p$ | 1 |
| Correlations modelled | Yes | Yes | No | No | No |
| Class-specific variance | Yes | No | No | Yes | No |

- QDA is most flexible but requires large $n$ relative to $p^2$ to estimate covariance reliably.
- LDA is the standard default; it assumes all classes share the same correlation structure.
- Diagonal LDA and Naive Bayes ignore all feature correlations — surprisingly effective in high dimensions.
- Nearest Centroids is the simplest: classify to the closest class mean under Euclidean distance (adjusted for class frequency $\pi_j$).
- In high-dimensional settings, always simplify or regularize (e.g. Regularized DA / RDA, block-diagonal covariance).

**Python Implementation:**
```python
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import NearestCentroid

qda = QuadraticDiscriminantAnalysis()
gnb = GaussianNB()        # Diagonal QDA / Naive Bayes
nc  = NearestCentroid()   # Nearest centroids
```

⚠️ **Theory vs. Practice:** `GaussianNB` in sklearn estimates per-class variances (diagonal QDA), not a shared diagonal — it does not correspond to Diagonal LDA. For Diagonal LDA specifically (shared diagonal covariance), there is no direct sklearn implementation; you would need to implement it manually or use `LinearDiscriminantAnalysis` with diagonal-constrained shrinkage. Mistaking `GaussianNB` for Diagonal LDA is a common error.

---

## ✅ Key Takeaways

- The theoretically optimal classifier maximizes $p(c|\mathbf{x})$. Logistic regression approximates this directly via a transformed linear predictor; DA approximates it indirectly via a generative model of the features.
- Logistic regression has a linear decision boundary ($\mathbf{x}^\top \boldsymbol{\beta} = 0$); multi-class logistic regression (softmax) has piecewise-linear boundaries.
- MLE for logistic regression is solved iteratively (gradient ascent / Newton-Raphson / IRLS) — there is no closed form.
- Perfect linear separability causes logistic regression MLE to diverge — regularization is the remedy.
- DA assumes $p(\mathbf{x}|c) \sim \mathcal{N}(\boldsymbol{\mu}_c, \boldsymbol{\Sigma}_c)$ and applies Bayes' law. QDA gives quadratic boundaries; LDA (shared covariance) gives linear boundaries.
- The choice of DA variant is a bias-variance trade-off: more structural assumptions → fewer parameters → more stable estimates, but potentially higher bias.
- In high-dimensional settings ($p$ large): simplify the covariance structure (diagonal, spherical) or regularize explicitly (RDA). Naive Bayes and LDA are surprisingly competitive in such regimes.