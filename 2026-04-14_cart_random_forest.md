# Lecture 5: Rule-based Classification and Regression (CART & Random Forests)

## 📋 Contents

- Classification and Partitions
- Classification and Regression Trees (CART)
- Measures of Node Purity
- Stopping Criteria and Pruning
- Recap: The Bootstrap
- Bootstrap Aggregation (Bagging)
- Random Forests
- Variable Importance

---

## 📝 Summary

This lecture introduces tree-based methods for classification and regression, starting from the core idea of explicitly partitioning feature space into rectangular regions. CART is presented as a practical algorithm for constructing such partitions through greedy binary splitting, with node impurity measures guiding each split. The lecture then covers how overfitting is controlled via pruning and ensemble methods. After a brief recap of the bootstrap, bagging is introduced as a variance-reduction technique, leading naturally to Random Forests, which further decorrelate the ensemble by restricting which features each tree considers. Variable importance measures are discussed as a practical by-product of the random forest procedure.

---

## 🎯 Learning Goals

- Understand how CART constructs a partition of feature space through recursive binary splitting and how node purity measures guide this process.
- Know the three impurity criteria (Gini, entropy, misclassification error) and when each is preferred.
- Understand cost-complexity pruning and how cross-validation is used to select the optimal subtree.
- Understand how bagging reduces variance and why it helps tree-based methods in particular.
- Explain how Random Forests decorrelate bagged trees and why this leads to further variance reduction.
- Interpret variable importance measures derived from random forests.

---

## 📚 Concepts

### Classification as Partitioning

**Why it matters:** Framing classification and regression as a partitioning problem unifies several algorithms (kNN, logistic regression, discriminant analysis, trees) under one conceptual umbrella and motivates tree methods as an explicit alternative.

**Intuition:** Any classifier implicitly divides feature space into regions — one per predicted class. The key question is *how* this division is constructed: implicitly (by modelling class probabilities) or explicitly (by directly carving up the space).

**How it works:**
Given non-overlapping regions $R_m$, $m = 1, \ldots, M$, a **classification rule** assigns to each new point $\mathbf{x}$ the majority class in whichever region it falls into:

$$
\hat{c}(\mathbf{x}) = \arg\max_{1 \le i \le K} \sum_{m=1}^{M} \mathbf{1}(\mathbf{x} \in R_m) \left( \frac{1}{|R_m|} \sum_{\mathbf{x}_l \in R_m} \mathbf{1}(i_l = i) \right)
$$

A **regression function** assigns the region mean:

$$
\hat{f}(\mathbf{x}) = \sum_{m=1}^{M} \left( \frac{1}{|R_m|} \sum_{\mathbf{x}_l \in R_m} y_l \right) \mathbf{1}(\mathbf{x} \in R_m)
$$

The methods differ only in *how* the regions $R_m$ are constructed. CART constructs them via a sequence of binary axis-parallel splits, which is far more tractable than searching over arbitrary partitions.

---

### Classification and Regression Trees (CART)

**Why it matters:** CART produces interpretable, rule-based models that can handle both regression and classification, require no feature scaling, and naturally accommodate missing data.

**Intuition:** Think of CART as a flowchart of yes/no questions about the features. Each question splits the data into two groups, and you keep asking questions until each group is "pure enough" — i.e., mostly one class or close to a single value.

**Prerequisites:**
- Feature space partitioning (see above)
- Concepts of training error and overfitting

**How it works:**

CART grows a tree by **greedy recursive binary splitting**:

1. Start with all training data in a single root node.
2. At each node, consider every feature $x_{\cdot j}$, $j = 1, \ldots, p$:
   - For continuous features: try all thresholds $t_j$ and split into $\{i_l : x_{lj} > t_j\}$ and $\{i_l : x_{lj} \le t_j\}$.
   - For categorical features: try all binary partitions of the categories.
   - Select the $(j, t_j)$ pair that gives the greatest *decrease in node impurity*.
3. Split the node and recurse on both child nodes.
4. Stop when a stopping criterion is met (see below).

The leaf nodes define the regions $R_m$, and each region is assigned the majority class (classification) or the mean response value (regression).

**Key Equations:**

Define the empirical class probability in region $R_m$:

$$
\hat{\pi}_{im} := \frac{1}{|R_m|} \sum_{\mathbf{x}_l \in R_m} \mathbf{1}(i_l = i)
$$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Highly interpretable (flowchart structure) | Prone to overfitting |
| Handles missing data naturally | High variance — small data changes can produce very different trees |
| No feature scaling required | Only axis-parallel decision boundaries |
| Works for both classification and regression | Features with more possible split points have a systematic advantage in being selected |

**Python Implementation:**

```python
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

# Classification
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, min_samples_leaf=5)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

# Regression
reg = DecisionTreeRegressor(criterion='squared_error', max_depth=5)
reg.fit(X_train, y_train)
```

⚠️ **Theory vs. Practice:** sklearn's `DecisionTreeClassifier` uses `criterion='gini'` by default — not entropy and not misclassification error. Misclassification error is not available as a splitting criterion in sklearn at all because it is less sensitive to class probability changes and performs poorly for growing trees. Additionally, `max_features=None` by default means all features are considered at each split — this is standard CART, not Random Forest. Setting `max_features` to a subset is what converts a single tree into a Random Forest tree.

---

### Measures of Node Purity

**Why it matters:** The choice of impurity measure determines how greedily the tree exploits small class imbalances. Gini and entropy are preferred over misclassification error for tree *growing* because they are more sensitive to changes in class probabilities.

**Intuition:** A pure node contains observations of only one class — the impurity of such a node is zero. The more mixed the classes in a node, the higher the impurity. The splitting algorithm picks whatever split best reduces total impurity across the two child nodes.

**Key Equations:**

All three measures are defined per region $R_m$ using $\hat{\pi}_{im}$:

$$
\text{Misclassification error:} \quad 1 - \max_i \hat{\pi}_{im}
$$

$$
\text{Gini impurity:} \quad \sum_{i=1}^{K} \hat{\pi}_{im}(1 - \hat{\pi}_{im})
$$

$$
\text{Entropy/Deviance:} \quad -\sum_{i=1}^{K} \hat{\pi}_{im} \log \hat{\pi}_{im}
$$

All three measures equal zero when only one class is present and reach their maximum when all classes are equally frequent.

For **regression trees**, the decrease in mean squared error after a split is used:

$$
\Delta\text{MSE} = \text{MSE}(\text{parent}) - \left(\frac{|R_L|}{|R|}\text{MSE}(R_L) + \frac{|R_R|}{|R|}\text{MSE}(R_R)\right)
$$

In the two-class case ($i \in \{0, 1\}$), entropy is the most sensitive measure (especially near $\hat{\pi}_{0m} = 0.5$), Gini is intermediate, and misclassification error is the least sensitive near purity boundaries. This is why entropy and Gini are preferred during tree *growing*, while misclassification error is commonly used to evaluate final tree *performance*.

---

### Stopping Criteria

**Why it matters:** Without stopping criteria, CART grows until every leaf is pure, which guarantees overfitting on training data.

**How it works:**

The tree stops splitting when any of the following are met:
- **Minimum leaf size**: e.g., each node must contain at least 5 samples.
- **Minimum impurity decrease**: splits that reduce impurity by less than some threshold (e.g., 1%) are rejected.
- **Maximum depth**: the number of splits from the root is capped (e.g., at 30).
- **Maximum number of leaf nodes**: the tree cannot create more than a fixed number of leaves.

Growing until one of these criteria is satisfied produces a **max tree**. Note: early stopping is risky because a weak early split might enable a much stronger split later.

---

### Cost-Complexity Pruning

**Why it matters:** Pruning is a principled alternative to stopping criteria — it builds the full max tree first (avoiding the early-stopping problem) and then removes nodes that do not justify their complexity.

**Intuition:** Rather than halting growth early, let the tree grow fully and then "prune back" nodes that cost more (in complexity) than they gain (in accuracy). A penalty $\alpha$ controls the cost-complexity trade-off.

**How it works:**

For a given $\alpha > 0$, the **cost-complexity** of a tree $T$ is:

$$
C_\alpha(T) = \underbrace{\sum_{R_m \in T} \frac{1}{|R_m|} \sum_{\mathbf{x}_l \in R_m} \mathbf{1}(i_l \ne \hat{c}(\mathbf{x}))}_{\text{Cost}} + \alpha |T|
$$

where $|T|$ is the number of leaf nodes.

It can be shown that a sequence of nested subtrees $T_0 \supset T_1 \supset \cdots \supset T_J$ exists (where $T_0 = T_{\max}$) such that each $T_k$ minimizes $C_{\alpha_k}(T)$ for some $0 = \alpha_0 < \alpha_1 < \cdots < \alpha_J$. As $\alpha$ increases, the optimal tree shrinks.

**Selecting the best subtree via cross-validation:**

1. Fit the max tree on the full training set. Run cost-complexity pruning to obtain the sequence $\alpha_0 < \alpha_1 < \cdots < \alpha_J$.
2. Split training data into $K$ folds. For each fold held out: fit a max tree on the remaining folds, run pruning to get subtrees.
3. For each $\alpha_k$ from Step 1, compute test error on each held-out fold.
4. Select the subtree from Step 1 whose $\alpha_k$ gives the lowest cross-validated test error.

The full dataset is only used in Step 1 to generate candidate $\alpha$ values. CV trees are separate.

**Python Implementation:**

```python
from sklearn.tree import DecisionTreeClassifier

# Get cost-complexity pruning path
clf = DecisionTreeClassifier()
path = clf.cost_complexity_pruning_path(X_train, y_train)
alphas = path.ccp_alphas

# Fit trees for each alpha and evaluate via CV
from sklearn.model_selection import cross_val_score
import numpy as np

best_alpha = alphas[np.argmax([
    cross_val_score(DecisionTreeClassifier(ccp_alpha=a), X_train, y_train, cv=5).mean()
    for a in alphas
])]

final_clf = DecisionTreeClassifier(ccp_alpha=best_alpha).fit(X_train, y_train)
```

⚠️ **Theory vs. Practice:** `cost_complexity_pruning_path` returns impurity-based alphas computed on the training set. These are used as candidates, not directly as the selected alpha — you must evaluate them via cross-validation. Using the alpha with the lowest *training* cost is equivalent to just using the max tree and will overfit.

---

### Recap: The Bootstrap

The nonparametric bootstrap estimates the sampling distribution of a statistic $\hat{\theta}$ by repeatedly resampling the data with replacement. For $b = 1, \ldots, B$:

1. Sample $\tilde{x}_1, \ldots, \tilde{x}_n$ with replacement from the original sample.
2. Compute $\hat{\theta}_b(\tilde{x}_1, \ldots, \tilde{x}_n)$.

The distribution of $\hat{\theta}_b$ approximates the sampling distribution of $\hat{\theta}$. Two common CI methods are the **normal approximation** (assumes $\hat{\theta} \approx N(\theta, \hat{\sigma}_{se})$) and the **percentile method** (uses empirical quantiles of the bootstrap estimates). The bootstrap works well for smooth statistics like the mean but poorly for extremes (quantiles, min/max), which may not appear in bootstrap samples.

---

### Bootstrap Aggregation (Bagging)

**Why it matters:** Single CART trees are highly unstable — small data changes produce very different trees. Averaging many trees trained on bootstrap samples dramatically reduces this variance.

**Intuition:** Averaging noisy-but-unbiased estimates gives a less noisy estimate. By training many trees on different bootstrap resamples and averaging their predictions, the variance cancels out while the bias stays low.

**How it works:**

1. For $b = 1, \ldots, B$, draw a bootstrap sample from $(y_l, \mathbf{x}_l)$ and fit a model $\hat{f}_b(\mathbf{x})$.
2. Aggregate:

$$
\hat{f}_{\text{bag}}(\mathbf{x}) = \frac{1}{B} \sum_{b=1}^{B} \hat{f}_b(\mathbf{x})
$$

For classification, majority vote is used instead of averaging.

**Why bagging reduces variance:**

Bagging approximates the oracle average $f_{\text{ag}}(\mathbf{x}) = \mathbb{E}_{p(\mathcal{T})}[\hat{f}(\mathbf{x})]$. Under squared error loss:

$$
\mathbb{E}_{p(\mathcal{T}, y|\mathbf{x})}[(y - \hat{f}(\mathbf{x}))^2] \ge \mathbb{E}_{p(\mathcal{T}, y|\mathbf{x})}[(y - f_{\text{ag}}(\mathbf{x}))^2]
$$

Bagging has no effect on linear models (averaging linear models gives the same linear model), but dramatically helps high-variance models like deep trees.

**The correlation problem:**

For i.d. random variables $x_i$ with variance $\sigma^2$ and pairwise correlation $\rho$:

$$
\text{Var}\left(\frac{1}{n}\sum_{i=1}^n x_i\right) = \frac{1-\rho}{n}\sigma^2 + \rho\sigma^2
$$

Even with $B \to \infty$, the second term $\rho\sigma^2$ persists. Since bootstrap samples overlap heavily, the resulting trees are correlated, so bagging can only reduce variance to $\rho\sigma^2$. Reducing $\rho$ would reduce the variance further — this is the key motivation for Random Forests.

---

### Random Forests

**Why it matters:** Random Forests decorrelate bagged trees by restricting the features each tree can use at each split. This reduces the $\rho\sigma^2$ floor from bagging and typically achieves lower test error.

**Intuition:** If all trees in a bagged ensemble use the same dominant feature at the root, they will be very similar (highly correlated). By randomly restricting each split to a subset of $m < p$ features, different trees end up using different features, making them more diverse and less correlated.

**Prerequisites:**
- Bagging and the correlation formula above
- CART tree growing

**How it works:**

1. For $b = 1, \ldots, B$:
   - Draw a bootstrap sample of size $n$ from training data.
   - Grow a tree $T_b$ until nodes reach minimum size $n_{\min}$. At **each split**:
     - Randomly select $m$ features from the $p$ available.
     - Find the best split among those $m$ features only.
     - Split the node.
2. Predict for new $\mathbf{x}$:
   - **Regression:** $\hat{f}_{rf}(\mathbf{x}) = \frac{1}{B}\sum_{b=1}^B T_b(\mathbf{x})$
   - **Classification:** majority vote across trees.

**Key parameter:** $m$, the number of features considered at each split. Common defaults are $m = \sqrt{p}$ for classification and $m = p/3$ for regression. Smaller $m$ → less correlated trees → lower variance, but potentially higher bias.

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Lower variance than bagging due to decorrelation | Less interpretable than a single tree |
| Provides out-of-bag error estimate for free | Computationally expensive (many trees) |
| Measures variable importance | Still limited to axis-parallel splits |
| Robust to hyperparameter choices | Can be slow to predict with very large forests |

**Python Implementation:**

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# Classification (default m = sqrt(p))
rf_clf = RandomForestClassifier(n_estimators=200, max_features='sqrt',
                                 min_samples_leaf=1, oob_score=True)
rf_clf.fit(X_train, y_train)
print(rf_clf.oob_score_)  # out-of-bag accuracy

# Regression (default m = p/3 ≈ max_features='sqrt' for regression too in sklearn)
rf_reg = RandomForestRegressor(n_estimators=200, max_features=1.0/3,
                                oob_score=True)
rf_reg.fit(X_train, y_train)
```

⚠️ **Theory vs. Practice:** sklearn's `RandomForestRegressor` uses `max_features=1.0` by default (all features), which is **bagging, not a random forest**. The standard recommendation is $m = p/3$ for regression. If you use the default, you are not getting the decorrelation benefit. Always set `max_features` explicitly. Also note: sklearn grows trees to full depth by default (`max_depth=None`, `min_samples_leaf=1`), which is correct for random forests — do not prune individual trees in an ensemble.

---

### Variable Importance

**Why it matters:** Random forests sacrifice interpretability of individual trees but recover it at the ensemble level via variable importance scores.

**How it works:**

Two measures are provided:

**1. Impurity-based importance:** Each split on a feature reduces node impurity by some amount. Summing these reductions across all splits and all trees, normalized by tree count, gives a feature's importance:

$$
\text{Importance}(j) = \frac{1}{B} \sum_{b=1}^{B} \sum_{\text{splits on } j \text{ in } T_b} \Delta\text{Impurity}
$$

**2. Out-of-bag (OOB) permutation importance:** Each bootstrap sample leaves out roughly 37% of the data (out-of-bag samples). These can be used as a test set for each tree:

- Compute OOB error $E_0$ for tree $T_b$ on its OOB samples.
- Randomly permute feature $j$ in the OOB samples and recompute error $E_1^{(j)}$.
- The importance of feature $j$ is $E_1^{(j)} - E_0 \ge 0$.

A large increase in error when feature $j$ is permuted indicates that feature $j$ is important.

**Caution:** Impurity-based importance is biased toward features with many possible split values (continuous or high-cardinality categorical features). Permutation importance is more reliable but more expensive to compute.

---

### Comparison: CART vs. Bagging vs. Random Forests

Illustrated on the toy problem $y = x_1^2 + \varepsilon$, $\varepsilon \sim N(0,1)$, with $\mathbf{x} \in \mathbb{R}^5$ and highly correlated predictors ($\Sigma_{lk} = 0.98$ for $l \ne k$):

| Property | Single CART | Bagging | Random Forest |
|---|---|---|---|
| Variance | High | Reduced | Further reduced |
| Bias | Low | Low | Slightly higher (due to feature restriction) |
| Interpretability | High | Low | Low |
| Correlation between trees | N/A | High (all features available) | Lower (feature subset) |
| OOB error estimate | No | Yes | Yes |
| Variable importance | Weak | Weak | Strong |
| Effect of more trees | None | Decreases variance | Decreases variance |

- With highly correlated predictors, bagging still produces correlated trees (they all tend to split on the same dominant feature). Random forests break this by forcing variety.
- A single CART test error stabilizes at a fixed (high) value regardless of number of trees; bagging and RF errors decrease and plateau as more trees are added.
- RF achieves lower final test error than bagging in the high-correlation setting because it forces different features to be explored.

---

## ✅ Key Takeaways

- CART constructs a partition of feature space by greedy recursive binary splitting, guided by a node impurity measure (Gini, entropy, or misclassification error).
- CART is interpretable and flexible but highly unstable — small data changes can produce very different trees, making single trees poor predictors.
- Pruning via cost-complexity regularization (with $\alpha$ selected by cross-validation) reduces overfitting while avoiding the pitfall of early stopping.
- Bagging averages many bootstrap-trained trees, reducing variance. However, correlated predictors lead to correlated trees, leaving a residual $\rho\sigma^2$ variance floor.
- Random forests address this by randomly restricting the features available at each split, decorrelating trees and further reducing ensemble variance.
- Two variable importance measures emerge naturally from the random forest procedure: impurity reduction (fast, but biased toward high-cardinality features) and OOB permutation importance (more reliable).
- As the number of trees $B \to \infty$, random forest variance decreases monotonically — unlike CART, there is no overfitting from using more trees.