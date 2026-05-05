# Lecture 2b: Model Evaluation and Bias-Variance Tradeoff

## 📋 Contents

- Model selection vs. model assessment
- Expected prediction error (conditional and total)
- Training error and test error
- Holdout method and cross-validation (CV)
- Leave-one-out cross-validation (LOOCV)
- Stratification
- Tuning parameter selection via CV
- Bias-Variance Tradeoff
- Evaluation metrics for classification (confusion matrix, accuracy, sensitivity, specificity, precision, F1, MCC, ROC/AUC)

---

## 📝 Summary

This lecture covers how to evaluate and compare statistical learning methods. It distinguishes between model selection (choosing hyperparameters or method structures) and model assessment (measuring how well a chosen model performs). The core challenge is that training error is a biased, over-optimistic estimate of true generalization error, so the lecture presents principled alternatives: the holdout method and $c$-fold cross-validation. The second half introduces the bias-variance tradeoff — the fundamental tension between model flexibility and stability — and then surveys a range of classification evaluation metrics beyond simple misclassification rate, including sensitivity, specificity, precision, the F1 score, MCC, and the ROC/AUC framework.

---

## 🎯 Learning Goals

- Distinguish between model selection and model assessment, and know when each is the goal.
- Understand why training error is an unreliable proxy for generalization error.
- Implement and interpret holdout validation and $c$-fold cross-validation, including LOOCV.
- Decompose expected prediction error into irreducible error, bias², and variance, and explain the tradeoff.
- Select and interpret appropriate classification metrics depending on the problem (class imbalance, asymmetric costs, etc.).

---

## 📚 Concepts

### Model Selection vs. Model Assessment

**Why it matters:** Before evaluating performance, it is essential to know what question you are asking — are you trying to choose between models, or measure how good the chosen model is?

**Intuition:** Think of model selection as picking the best runner from a team, and model assessment as timing how fast that runner actually is. These are distinct tasks that may require different data.

**How it works:**

- **Model selection**: Choose a hyperparameter or model structure. Examples: select $k$ in kNN, or decide between logistic regression, LDA, and kNN.
- **Model assessment**: After a model has been selected, estimate how well it performs on unseen data.

Both goals require careful handling of data to avoid overly optimistic estimates.

---

### Expected Prediction Error

**Why it matters:** The fundamental goal of supervised learning is to minimize prediction error on new data. Formalizing this precisely is necessary to understand what estimators like cross-validation are actually approximating.

**Intuition:** Any estimate $\hat{f}$ is trained on one particular dataset. If we imagine re-running the data collection many times, we get many different $\hat{f}$'s. The expected prediction error averages over this randomness.

**How it works:**

Given a loss function $L(y, \hat{f}(\mathbf{x}))$, the **total expected prediction error** averages over both the randomness in the training data $\mathcal{T}$ and new test points $(\mathbf{x}, y)$:

$$
R = \mathbb{E}_{p(\mathcal{T})} \left[ \mathbb{E}_{p(\mathbf{x}, y)} \left[ L(y, \hat{f}(\mathbf{x}|\mathcal{T})) \right] \right]
$$

The **conditional expected prediction error** fixes the training set $\mathcal{T}$ and only averages over new data:

$$
R(\mathcal{T}) = \mathbb{E}_{p(\mathbf{x}, y)} \left[ L(y, \hat{f}(\mathbf{x}|\mathcal{T})) \right]
$$

**Key Equations:**

$$
R = \mathbb{E}_{p(\mathcal{T})}[R(\mathcal{T})]
$$

Where:
- $\mathcal{T} = \{(y_l, \mathbf{x}_l) : l = 1, \ldots, n\}$ = training data (itself random, drawn i.i.d. from $p(\mathbf{x}, y)$)
- $\hat{f}(\mathbf{x}|\mathcal{T})$ = model estimated from $\mathcal{T}$

---

### Training Error and Test Error

**Why it matters:** Training error is what you compute on the data you used to fit the model. It systematically underestimates generalization error — this is called the *optimism* of the training error.

**Intuition:** A model that memorizes its training data perfectly has zero training error but may fail completely on new data (overfitting). This is why we need separate test data.

**How it works:**

**Training error** (evaluated on the same data used to fit $\hat{f}$):

$$
R^{tr} = \frac{1}{n} \sum_{l=1}^{n} L(y_l, \hat{f}(\mathbf{x}_l | \mathcal{T}))
$$

**Test error** (evaluated on fresh samples $(y'_l, \mathbf{x}'_l)$ not used during training):

$$
R^{te} = \frac{1}{m} \sum_{l=1}^{m} L(y'_l, \hat{f}(\mathbf{x}'_l | \mathcal{T}))
$$

Key problem: More complex models tend to have lower training error (they overfit to available data), making training error unreliable for model selection.

---

### Holdout Method and $c$-Fold Cross-Validation

**Why it matters:** When you cannot assume access to a separate test population, you must manufacture held-out data from your existing sample. These are the two standard approaches.

**Intuition:** The holdout method is simple — split your data once and never touch the test portion during training. Cross-validation is more data-efficient: it rotates which portion is held out, so every observation gets tested exactly once.

**How it works:**

**Holdout method** (use when data is plentiful):
- Randomly partition data into training (e.g. 75%) and test (e.g. 25%) sets.
- Fit on training, evaluate on test. Use $R^{te}$ as estimate of generalization error.

**$c$-fold cross-validation** (use when data is scarce):
1. Randomly partition data into $c$ equally sized folds $\mathcal{F}_1, \ldots, \mathcal{F}_c$.
2. For each fold $j$:
   - Train on $\mathcal{F}_{-j} = \bigcup_{i \neq j} \mathcal{F}_i$
   - Evaluate on $\mathcal{F}_j$
3. Average the $c$ test errors:

$$
R^{cv} = \frac{1}{c} \sum_{j=1}^{c} \frac{1}{|\mathcal{F}_j|} \sum_{(y_l, \mathbf{x}_l) \in \mathcal{F}_j} L\!\left(y_l, \hat{f}(\mathbf{x}_l | \mathcal{F}_{-j})\right)
$$

> ⚠️ No training must be done on the test fold, or outside of the CV loop. Preprocessing steps (e.g. feature selection, scaling) that use test data leak information and invalidate the estimate.

The spread across folds contains additional useful information — inspect boxplots of fold-level errors, not just the mean.

For the estimates to be valid, test and training sets must be identically distributed.

**Python Implementation:**

```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.neighbors import KNeighborsClassifier

kf = KFold(n_splits=10, shuffle=True, random_state=42)
model = KNeighborsClassifier(n_neighbors=5)

# cross_val_score returns per-fold scores
scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"Mean CV accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
```

⚠️ **Theory vs. Practice:** `cross_val_score` does NOT refit a final model — it only estimates error. If you use it to select a hyperparameter, you must then refit on all data with the chosen hyperparameter manually. Failing to do this means you are evaluating a model that was never actually trained on your full dataset. Also, `KFold` with `shuffle=False` (the default) will respect the original row order, which can introduce bias if the data has temporal or group structure.

---

### Leave-One-Out Cross-Validation (LOOCV)

**Why it matters:** LOOCV is the extreme case of $c$-fold CV where $c = n$. It uses the maximum amount of data for each training fold and has well-known closed-form solutions for some models.

**Intuition:** Each observation is held out once, tested on a model trained on all remaining $n-1$ points.

**How it works:**

- Special case of $c$-fold CV with $c = n$.
- Each training set has $n-1$ observations; each test set has exactly 1.
- Advantages: maximizes training data, closed-form approximations available for many methods (e.g. linear models).
- Disadvantages: the $n$ training sets overlap almost entirely, making the $n$ estimates highly correlated. This leads to **higher variance** of the LOOCV estimate compared to, say, 5- or 10-fold CV.

In practice, try multiple values of $c$ and be cautious if results change drastically.

---

### Stratification

**Why it matters:** When class proportions or outcome distributions are unequal, random splitting can create folds with very different compositions, invalidating comparisons across folds.

**Intuition:** If 90% of samples are benign, a random fold might end up with 100% benign cases — a useless test set for detecting malignant cases.

**How it works:**

**Class imbalance (classification):** Sample each fold so that class proportions match the original data.

**Localized continuous outcome (regression):** Divide the outcome into intervals (strata), then sample each fold so that the relative frequency from each interval matches the full dataset.

**Python Implementation:**

```python
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
# y must be the class labels for stratification to work
for train_idx, test_idx in skf.split(X, y):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
```

⚠️ **Theory vs. Practice:** Standard `KFold` does not stratify. If you use `KFold` on an imbalanced dataset, folds will not preserve class ratios, and your CV error estimate will be noisy and potentially biased. Always use `StratifiedKFold` for classification problems unless you have verified that your classes are balanced.

---

### Tuning Parameter Selection via Cross-Validation

**Why it matters:** The same CV framework used for model assessment can be used to select hyperparameters (e.g. $k$ in kNN, regularization strength $\lambda$).

**How it works:**

For a sequence of hyperparameter values $\lambda_1, \ldots, \lambda_S$:

1. For each $\lambda_s$, compute $c$-fold CV error:

$$
R^{cv}(\lambda_s) = \frac{1}{c} \sum_{j=1}^{c} \frac{1}{|\mathcal{F}_j|} \sum_{(y_l, \mathbf{x}_l) \in \mathcal{F}_j} L\!\left(y_l, \hat{f}(\mathbf{x}_l | \lambda_s, \mathcal{F}_{-j})\right)
$$

2. Select:

$$
\hat{\lambda} = \arg\min_{\lambda_s} R^{cv}(\lambda_s)
$$

This also works for selecting among entirely different model classes (e.g. comparing kNN, QDA, logistic regression).

---

### Bias-Variance Tradeoff

**Why it matters:** This decomposition explains *why* model complexity must be balanced — and why no single model is universally best.

**Intuition:** Imagine estimating $f$ from many different training sets. Bias measures how far off the average estimate is from the truth; variance measures how much estimates fluctuate across datasets. Simple models are stable but may miss the truth (high bias, low variance). Complex models track the data closely on average but are erratic (low bias, high variance).

**Prerequisites:**
- Expected prediction error
- Regression setup: $y = f(\mathbf{x}) + \varepsilon$, $\varepsilon \sim \mathcal{N}(0, \sigma^2)$

**How it works:**

For squared error loss, the total expected prediction error decomposes as:

$$
R = \sigma^2 + \mathbb{E}_{p(\mathbf{x})}\!\left[\left(f(\mathbf{x}) - \mathbb{E}_{p(\mathcal{T})}[\hat{f}(\mathbf{x})]\right)^2\right] + \mathbb{E}_{p(\mathbf{x})}\!\left[\mathrm{Var}_{p(\mathcal{T})}[\hat{f}(\mathbf{x})]\right]
$$

$$
R = \underbrace{\sigma^2}_{\text{Irreducible error}} + \underbrace{\text{Bias}^2}_{\text{(avg. over } \mathbf{x}\text{)}} + \underbrace{\text{Variance}}_{\text{(avg. over } \mathbf{x}\text{)}}
$$

Where:
- $\sigma^2$ = irreducible noise — cannot be reduced regardless of model choice
- $\text{Bias}^2$ = systematic deviation of the average model from the truth
- $\text{Variance}$ = sensitivity of $\hat{f}$ to the particular training sample drawn

**Key Equations:**

$$
R = \sigma^2 + \mathbb{E}_{p(\mathbf{x})}\left[(f(\mathbf{x}) - \mathbb{E}_{p(\mathcal{T})}[\hat{f}(\mathbf{x})])^2\right] + \mathbb{E}_{p(\mathbf{x})}\left[\mathrm{Var}_{p(\mathcal{T})}[\hat{f}(\mathbf{x})]\right]
$$

**Strengths and Weaknesses:**

| Property | Simple / Global models (e.g. LDA) | Complex / Local models (e.g. kNN, $k$ small) |
|---|---|---|
| Bias | High if true boundary is complex | Low — can adapt locally |
| Variance | Low — stable across training sets | High — sensitive to data |
| Best for | Simple true boundaries, small $n$ | Complex true boundaries, large $n$ |

**Observations:**
- Irreducible error cannot be changed — it is a property of the data generating process.
- For consistent estimators, bias and variance both shrink as $n \to \infty$.
- These guarantees typically assume $p$ (number of features) is fixed. In high-dimensional settings this may not hold.

---

### Comparison: Global vs. Local Rules

The bias-variance behavior of a method depends on whether it uses all data (global) or local neighborhoods (local) to make predictions.

| Property | Global rules (e.g. LDA) | Local rules (e.g. kNN, small $k$) |
|---|---|---|
| Decision boundary | Global linear/parametric | Locally adaptive |
| Variance | Low | High |
| Bias (simple boundary) | Low | Low |
| Bias (complex boundary) | High | Low |
| Best use case | Simple boundary, small $n$ | Complex boundary, large $n$ |

- LDA with a simple true boundary: low bias and low variance — the ideal case.
- LDA with a complex true boundary: low variance but large bias — LDA cannot capture the true shape.
- kNN ($k=3$) with either boundary: high variance (wiggly decision boundary across samples) but low bias on average.
- Increasing $k$ in kNN trades variance for bias (smoother boundary, less local adaptation).

---

### Evaluation Metrics for Classification

**Why it matters:** Misclassification rate treats all errors equally. In practice, false negatives and false positives often have very different costs, and class imbalance can make accuracy misleading.

**Intuition:** A cancer screening test that classifies everyone as healthy achieves high accuracy on a healthy population — but misses every true case. Metrics like sensitivity and specificity capture this failure mode.

**How it works:**

All metrics derive from the **confusion matrix**:

|  | Predicted Positive | Predicted Negative |
|---|---|---|
| **Actual Positive** | True Positive (TP) | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN) |

**Accuracy:**

$$
\text{Accuracy} = \frac{TP + TN}{T}
$$

Symmetric; useful when class proportions are balanced and costs of FP and FN are equal. Misleading for imbalanced datasets.

**Sensitivity / Recall / True Positive Rate (TPR):**

$$
\text{Sensitivity} = \frac{TP}{TP + FN}
$$

Proportion of actual positives correctly identified. Use when false negatives are costly (e.g. failing to detect a disease). A model optimized purely for sensitivity will tend to overpredict positives.

**Specificity / True Negative Rate:**

$$
\text{Specificity} = \frac{TN}{TN + FP}
$$

Proportion of actual negatives correctly identified. Use to ensure the classifier also recognizes negatives. False Positive Rate $= 1 - \text{Specificity}$.

**Precision:**

$$
\text{Precision} = \frac{TP}{TP + FP}
$$

Proportion of predicted positives that are truly positive. Use when false positives are costly (e.g. spam filter). A model optimized purely for precision will tend to overpredict negatives.

**F1 Score:**

$$
F_1 = \frac{2 \cdot \text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}} \in [0, 1]
$$

Harmonic mean of precision and recall; useful when both FP and FN costs matter and classes are imbalanced.

**Matthews Correlation Coefficient (MCC):**

$$
\text{MCC} = \frac{TP \cdot TN - FP \cdot FN}{\sqrt{(TP+FP)(TP+FN)(TN+FP)(TN+FN)}} \in (-1, 1)
$$

- $\text{MCC} = 0$: random classifier
- $\text{MCC} > 0$: better than random; $\text{MCC} < 0$: worse than random
- Takes all four cells of the confusion matrix into account; robust to class imbalance.

**ROC Curve and AUC:**

The **Receiver Operating Characteristic (ROC) curve** plots TPR (sensitivity) vs. FPR (= 1 − specificity) as the classification threshold varies. Key reference points:
- Diagonal line (0,0) → (1,1): random classifier (AUC = 0.5)
- TPR > FPR: better than random; the curve bows toward the top-left corner

The **Area Under the ROC Curve (AUC)** summarizes the full ROC curve:

$$
\text{AUC} \in [0, 1], \quad \text{AUC} = 0.5 \text{ for random, } > 0.5 \text{ for better classifiers}
$$

AUC measures the probability that the classifier ranks a random positive instance above a random negative one.

**Python Implementation:**

```python
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    confusion_matrix, RocCurveDisplay
)

# Assuming y_true (labels) and y_pred (predicted labels), y_prob (predicted probabilities)
print("Accuracy:", accuracy_score(y_true, y_pred))
print("Sensitivity/Recall:", recall_score(y_true, y_pred))
print("Specificity:", recall_score(y_true, y_pred, pos_label=0))
print("Precision:", precision_score(y_true, y_pred))
print("F1:", f1_score(y_true, y_pred))
print("MCC:", matthews_corrcoef(y_true, y_pred))
print("AUC:", roc_auc_score(y_true, y_prob))

RocCurveDisplay.from_predictions(y_true, y_prob)
```

⚠️ **Theory vs. Practice:** `recall_score` computes sensitivity for `pos_label=1` (default). To compute specificity, you must pass `pos_label=0` — sklearn does not have a dedicated `specificity_score`. The `roc_auc_score` requires predicted *probabilities*, not class labels; passing hard predictions (0/1) will not give a meaningful ROC curve. For multiclass problems, these binary metrics require explicit `average` parameter choices (`'macro'`, `'micro'`, `'weighted'`) — the default behavior differs between functions and may silently give misleading results.

---

## ✅ Key Takeaways

- Training error is systematically too optimistic — never use it alone to assess or compare models.
- Cross-validation (typically 5- or 10-fold) is the standard solution when data is limited; the holdout method works when data is abundant.
- For unbalanced data, always stratify your folds to maintain representative class proportions.
- The total expected prediction error decomposes into irreducible error, bias², and variance — reducing one of bias or variance typically increases the other.
- Local methods (small $k$ kNN) have low bias but high variance; global methods (LDA) have low variance but high bias when the true boundary is complex.
- No single classification metric is universally best: accuracy fails on imbalanced data, sensitivity ignores false positives, precision ignores false negatives. Use multiple metrics together.
- The ROC curve and AUC summarize the full sensitivity–specificity tradeoff across all decision thresholds.