# Analysis Pipeline — Method Comparisons

## 📋 Contents

- Importance of data splitting
- Imbalanced data
- Data leakage
- Reporting and comparing methods
- A full analysis pipeline (3 levels of splits)
- Non-parametric tests: Wilcoxon and Friedman
- Parametric tests: corrected paired t-tests
- Bayesian corrected paired t-tests and ROPE
- Equivalence classes
- Hierarchical models

---

## 📝 Summary

This lecture covers best practices for constructing a rigorous statistical analysis pipeline when comparing multiple machine learning models on a single dataset. Starting from foundational concerns — data splitting, class imbalance, and data leakage — it moves through a full three-level cross-validation framework and then addresses the core statistical challenge: how do you validly test whether two models perform differently when your performance scores are correlated due to shared training data? Both frequentist (Wilcoxon, Friedman, corrected paired t-test) and Bayesian (Benavoli et al., ROPE) approaches are covered, along with practical guidance on summarising results via equivalence classes.

---

## 🎯 Learning Goals

- Understand why performance estimates from cross-validation are correlated and what consequences this has for hypothesis testing.
- Know when to use non-parametric vs. parametric tests to compare classifiers, and how to correct for the dependencies induced by repeated CV.
- Apply the Bouckaert–Frank corrected t-test and interpret the Bayesian ROPE framework as alternatives to naive significance testing.
- Design a full three-level analysis pipeline (inner tuning CV, K-fold performance CV, outer stability loop) without data leakage.
- Summarise multi-method comparisons using equivalence classes, CD-diagrams, and probability heatmaps.

---

## 📚 Concepts

### Data Splitting and Unbiased Error Estimation

**Why it matters:** Using the same data for both training and evaluation produces optimistically biased error estimates that do not generalise to new data.

**Intuition:** Imagine studying for a test using exactly the questions that will appear on the exam — your score won't reflect how you'd do on new questions. A held-out test set plays the role of genuinely new questions.

**Prerequisites:**
- Cross-validation basics
- Bias–variance tradeoff

**How it works:**
A hold-out test set must never be touched during training or tuning. Two practical tensions arise: holding out too much data handicaps model training, while holding out too little yields a noisy error estimate. Stratified sampling addresses class imbalance by ensuring the class proportions in training and test sets mirror those of the full dataset.

**Key Equations:** None specific to this section; the principle is structural.

---

### Imbalanced Data

**Why it matters:** Standard accuracy is a misleading metric when classes are heavily imbalanced — a classifier that always predicts the majority class can score very high accuracy while being completely useless.

**Intuition:** If 95% of your data belongs to class A, predicting "A" for everything yields 95% accuracy. Metrics that account for class-level performance expose this failure.

**Prerequisites:**
- Confusion matrices
- Precision, recall, F1

**How it works:**
Several strategies address imbalance at the *metric* level and at the *data* level.

At the metric level, alternatives to raw accuracy include:

- **Weighted accuracy** — a weighted average of class-specific error rates with weights $w_c = \frac{n}{C \cdot n_c}$, upweighting rare classes.
- **AUC** (area under the ROC curve) — trades off sensitivity against false positive rate without fixing a threshold.
- **F1** — harmonic mean of precision and recall; generalises to multi-class via macro/micro averaging.
- **Matthew's Correlation Coefficient (MCC)** — correlation between predicted and true labels, accounts for all cells of the confusion matrix. Generalisable to multi-class.
- **Kappa** — normalises accuracy by what would be expected purely by chance given the observed class proportions.

At the *data* level, the training set itself can be rebalanced:

- **Downsample** the majority class.
- **Upsample** the minority class (e.g., with replacement).
- **SMOTE** — generate synthetic minority-class observations as linear combinations of a sample and its nearest neighbours within that class.

**Key Equations:**

$$
w_c = \frac{n}{C \cdot n_c}
$$

Where $C$ is the number of classes and $n_c$ is the count of class $c$.

**Strengths and Weaknesses:**
| Approach | Strengths | Weaknesses |
|---|---|---|
| Weighted metrics | No data modification needed | May not align with business cost of errors |
| SMOTE | Can improve minority-class recall substantially | Risk of overfitting to synthetic points; do not apply to test data |
| Weighted loss | Easy to integrate; adjusts training directly | Requires careful weight calibration |

**⚠️ Rebalancing is for training only.** Never rebalance the test set. You can and should use balanced metrics on the (unmodified) test set.

---

### Data Leakage

**Why it matters:** Any information from the test set that influences model training, tuning, or feature selection will produce optimistically biased performance estimates — sometimes dramatically so.

**Intuition:** If you standardise your features using the mean and variance of *all* data (including test samples), your test set is no longer truly unseen — the model's preprocessing has already "peeked" at it.

**Prerequisites:**
- Cross-validation
- PCA, standardisation, feature selection

**How it works:**
All data-dependent choices — feature transforms, standardisation, PCA projections, feature filtering thresholds — must be *learned on the training data only* and then *applied* (not re-fit) to the test data. Concretely:

- Fit a `StandardScaler` on training data, `transform` both train and test.
- Fit PCA on training data, `transform` test data using the same projection.
- Feature selection thresholds (e.g., filter by t-statistic) must be computed inside the CV loop on training folds, not on the full dataset.

Anything that is tuned by a performance metric belongs inside the CV loop. When you repeat train–test splits for stability assessment, even exploratory choices made on the full dataset introduce a mild form of leakage that can cause slight optimism in repeated estimates.

---

### A Full Analysis Pipeline — 3 Levels of Splits

**Why it matters:** A single train/test split yields one performance number per method — no sense of variability, no tuning, no stability assessment. Three nested loops address all three goals simultaneously.

**Intuition:** Think of three concentric loops: the innermost selects tuning parameters, the middle one estimates performance, and the outermost checks whether that estimate is stable across different random splits of the data.

**Prerequisites:**
- K-fold cross-validation
- Hyperparameter tuning

**How it works:**

**Level 1 — Inner CV loop (tuning):** For each training fold in the K-fold loop, run a nested CV to select tuning parameters, features, transforms, etc. Refit the model with the selected parameters on the *full* training fold before evaluating on the held-out test fold.

**Level 2 — K-fold CV (performance):** Use K-fold CV so that every observation appears in a test set exactly once. This is the main performance estimation loop. Common choice: $K = 10$.

**Level 3 — Outer loop (stability):** Repeat the K-fold CV $R$ times with different random permutations of the data. This gives $R \times K$ performance scores per method and reveals instability. Common choice: $R = 10$, yielding 100 scores per method.

The three levels give you tuning (inner loop), performance (K-fold), and stability (outer loop) — all without leakage.

---

### Non-Parametric Tests: Wilcoxon and Friedman

**Why it matters:** Rank-based tests are more robust than t-tests when performance scores are skewed or saturated (e.g., accuracy near 1.0). However, they assume independent rows in the score matrix — an assumption that is violated in repeated CV.

**Intuition:** Instead of asking "is the mean difference zero?", rank tests ask "is one method consistently ranked higher than another?" This is less sensitive to extreme scores but equally sensitive to ordering.

**Prerequisites:**
- Hypothesis testing, p-values
- Multiple comparisons correction (Holm, Bonferroni)

**How it works:**

**Wilcoxon signed-rank test** is a paired test that tests whether the median of the pairwise score differences is zero. When comparing $M$ methods, this requires $\frac{M(M-1)}{2}$ tests — multiple comparisons correction (e.g., Holm) is mandatory.

**Friedman test** is an omnibus test comparing all $M$ methods simultaneously. For an $N \times M$ score matrix (rows = datasets/splits, columns = methods), compute ranks row by row, then:

$$
Q = \frac{12N}{M(M+1)} \sum_{j=1}^{M} \left(\bar{r}_{.j} - \frac{M+1}{2}\right)^2
$$

Where $\bar{r}_{.j}$ is the mean rank of method $j$ across the $N$ datasets. $Q$ is compared against $\chi^2_{M-1}$. Rejection tells you *some* method differs, but not which.

**Post-hoc — Conover:** Compares pairs of methods using ranks from the Friedman test. The paired rank difference $|\bar{r}_i - \bar{r}_j|$ is compared against a pooled variance estimate derived from the $Q$-statistic. The resulting $t$-statistic is compared to $t_{(N-1)(M-1)}$. Apply Holm correction.

**Post-hoc — Nemenyi:** Compares paired average ranks to a critical distance:

$$
CD = q_{\alpha, M, \infty} \sqrt{\frac{M(M+1)}{6N}}
$$

Where $q$ is from the studentized range distribution. Nemenyi is very conservative but produces the popular **CD-diagram**: methods whose average ranks differ by less than $CD$ are connected by a horizontal bar, indicating non-significant difference.

**Strengths and Weaknesses:**
| | Strengths | Weaknesses |
|---|---|---|
| Wilcoxon | Paired, robust to outliers | Requires many corrections for $M > 2$ |
| Friedman | Single omnibus test, uses all comparisons jointly | Requires post-hoc; assumes independent rows |
| Nemenyi | Visualisable via CD-diagram | Very conservative |
| Conover | More powerful than Nemenyi; pools all ranks | Less commonly known |

**⚠️ Independence assumption:** All rank tests assume the $N$ rows of the score matrix are independent. In repeated CV they are not — scores from the same split are correlated. This means variance is underestimated, yielding anti-conservative p-values (apparent significance where none exists).

---

### Parametric Tests: Corrected Paired t-Test

**Why it matters:** The naive paired t-test applied to K-fold CV scores treats $K$ scores as $K$ independent observations, but they are correlated because the training folds overlap. This underestimates the true variance and produces anti-conservative inference.

**Intuition:** If you only effectively have ~5 independent observations when you think you have 10 (due to correlation), your t-test is overconfident. The Bouckaert–Frank correction inflates the variance estimate to account for this.

**Prerequisites:**
- Paired t-test
- Variance of a sum with covariance terms

**How it works:**

For K-fold CV comparing two methods, compute pairwise score differences $\delta_k$, $k = 1, \ldots, K$. The true variance of $\bar{\delta}$ is:

$$
V(\bar{\delta}) = \frac{\sigma^2}{K} + \frac{K-1}{K}\rho\sigma^2
$$

Since the sample variance $S^2$ estimates $\sigma^2(1 - \rho)$, substituting gives:

$$
V(\bar{\delta}) = E(S^2)\left(\frac{1}{K} + \frac{\rho}{1-\rho}\right)
$$

**Nadeau–Bengio (2003)** approximated $\frac{\rho}{1-\rho} \approx \frac{n_{te}}{n_{tr}}$, where $n_{te}$ and $n_{tr}$ are the test and training set sizes. For K-fold CV this becomes $\frac{1}{K-1}$, so:

$$
V(\bar{\delta}) \approx S^2\left(\frac{1}{K} + \frac{1}{K-1}\right)
$$

The effective sample size for 10-fold CV is therefore only $n_{eff} \approx 5$.

**Bouckaert–Frank (2004)** extended this to repeated K-fold CV ($R$ repeats):

$$
V(\bar{\delta}) = S^2\left(\frac{1}{RK} + \frac{n_{te}}{n_{tr}}\right)
$$

Where $\frac{n_{te}}{n_{tr}} = \frac{1}{K-1}$ for K-fold. The corrected t-statistic is:

$$
t = \frac{\bar{\delta}}{\sqrt{S^2\left(\frac{1}{RK} + \frac{n_{te}}{n_{tr}}\right)}}
$$

Compared against $t_{KR-1}$. Effective sample size for 10-fold, 10 repeats: $n_{eff} \approx 8$.

**Key observation:** Increasing $R$ has diminishing returns because $\frac{n_{te}}{n_{tr}}$ dominates the variance expression.

---

### Bayesian Corrected Paired t-Test and ROPE

**Why it matters:** The frequentist corrected t-test answers "can we reject equal performance?" — a binary decision. The Bayesian version answers "how probable is it that model A outperforms model B?" and "how probable is it that they are practically equivalent?" — a richer set of questions.

**Intuition:** Instead of a p-value, you get a full posterior distribution over the mean performance difference $\mu$. You can then integrate over any region of interest.

**Prerequisites:**
- Bayesian inference, posterior distributions
- Student-t distribution
- Normal-Inverse Gamma conjugate prior

**How it works:**

**Benavoli et al. (2017)** model the pairwise score differences as multivariate normal with mean $\mu$ and covariance $\Sigma_{ij} = \rho\sigma^2$ (same structure as the frequentist version). With Normal-Inverse Gamma conjugate priors (using specific hyperparameters that yield a non-informative prior), the posterior for $\mu$ is:

$$
p(\mu \mid \delta, \text{priors}) = St\left(\mu;\, RK-1,\, \bar{\delta},\, \left(\frac{1}{RK} + \frac{\rho}{1-\rho}\right)\hat{\sigma}^2\right)
$$

using the Nadeau–Bengio approximation $\frac{\rho}{1-\rho} = \frac{n_{te}}{n_{tr}}$.

This is nearly identical to the frequentist corrected t-test, but is parameterised as a distribution over $\mu$, enabling direct probability statements:

$$
P(\text{model A} > \text{model B}) = \int_0^\infty p(\mu \mid \delta) \, d\mu
$$

$$
P(\text{model A} \approx \text{model B}) = P(|\mu| \leq d)
$$

**ROPE (Region of Practical Equivalence):** Define a threshold $d$ such that any performance difference smaller than $d$ is practically irrelevant (e.g., $d = 0.01$ for accuracy). The ROPE probability is the posterior mass within $[-d, d]$:

$$
P(\text{model A} \approx \text{model B}) = P(|\mu| \leq d)
$$

This shifts the question from *statistical significance* to *practical importance* — a model that is "significantly" better by 0.001% accuracy may not warrant preferring it in practice.

**Strengths and Weaknesses:**
| | Strengths | Weaknesses |
|---|---|---|
| Bayesian corrected t | Direct probability statements; ROPE interpretability | Requires specifying $d$; still relies on the Nadeau–Bengio approximation |
| Frequentist corrected t | Familiar p-value framework | Binary reject/fail-to-reject; no practical importance built in |

---

### Equivalence Classes

**Why it matters:** With $M$ methods, pairwise tests produce an $M \times M$ matrix of p-values or probabilities. Equivalence classes distil this into groups of models that are statistically indistinguishable.

**Intuition:** Build a graph where each method is a node. Connect two methods with an edge if their pairwise test shows no significant difference (or if $P(\text{ROPE})$ is high). Connected components are equivalence classes — models within a class cannot be reliably distinguished in terms of performance.

**How it works:**

From the frequentist corrected t-test, threshold the $M \times M$ p-value matrix at significance level $\alpha$ and build the graph. From the Bayesian framework, threshold the $P(\text{ROPE})$ matrix.

Results can be visualised as:
- **Barcharts** grouped by equivalence class, showing mean score per method with colour encoding the class.
- **Heatmaps** of $P(\text{model i} > \text{model j})$ — red for high probability, blue for low.
- **Normalised score heatmaps** across multiple metrics simultaneously (4 metrics × 5 models).

**Caveat:** Pairwise comparisons can produce cyclic contradictions ($A > B$, $B > C$, $C > A$). This is a fundamental limitation of pairwise testing that hierarchical models partially address.

---

### Hierarchical Models

**Why it matters:** As the number of models $M$ grows, pairwise comparisons become unwieldy and risk conflicting orderings. Hierarchical Bayesian models generalise the corrected t-test framework to handle multiple models jointly, including random effects for different datasets.

**Intuition:** A mixed-effects model introduces a "dataset effect" — acknowledging that some datasets are simply harder, and that performance variability is partly due to the dataset rather than the method.

**Prerequisites:**
- Bayesian hierarchical models
- Mixed effects / random effects models
- MCMC

**How it works:**

Packages such as `tidyposterior` (R), `bambi`, and `PyMC` implement hierarchical Bayesian models for classifier comparison. These model a random intercept per dataset and random slopes per method, with MCMC used to compute posterior samples.

When comparing classifiers across *independent* datasets (not repeated CV), packages like `autorank` provide a full pipeline: Friedman test → post-hoc Nemenyi → CD-diagram.

**⚠️ `autorank` assumes independent datasets.** It is not designed for repeated CV scores. Applying it to correlated scores from repeated CV will give anti-conservative results.

---

## ✅ Key Takeaways

- Never use the same data for training and performance evaluation. Use stratified splits and hold out test data strictly.
- Imbalanced classes require balanced metrics (MCC, F1, balanced accuracy) and potentially rebalanced training data (SMOTE, downsampling) — but never rebalance the test set.
- Data leakage occurs whenever any fit (scaler, PCA, feature selector) is computed on data that includes test observations. All such fits belong inside the CV loop.
- A three-level pipeline (inner tuning CV → K-fold performance CV → outer stability loop) is the gold standard for a single-dataset analysis.
- K-fold CV scores are correlated due to overlapping training sets. The Bouckaert–Frank correction inflates the variance of the paired t-test to account for this; effective sample size for 10-fold, 10-repeat CV is only ~8.
- The Bayesian corrected t-test (Benavoli et al., 2017) provides richer inference: instead of a binary reject/fail-to-reject decision, it yields posterior probabilities such as $P(A > B)$ and $P(|μ| \leq d)$ via ROPE.
- Equivalence classes — built from pairwise test results — are the recommended way to summarise comparisons among many methods: group indistinguishable methods together rather than selecting a single "winner".
- Read package documentation carefully. LDA, Logistic Regression, and Random Forest all have default behaviours (implicit regularisation, default penalty terms, default tree-depth limits) that can severely distort comparisons if left unchecked.