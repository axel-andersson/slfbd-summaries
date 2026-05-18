# Analysis Pipeline — Code Review Checklist

Work through every item before submitting. If you cannot tick a box, fix the issue first.

---

## 1. Data Splitting

- [ ] Training and test sets are split **before** any fitting, transformation, or exploration is done on the data.
- [ ] The test set is **never used** for any decision — not for tuning, not for feature selection, not for choosing transforms.
- [ ] Splitting uses **stratified sampling** to preserve class proportions in both sets.
- [ ] You have verified (e.g. by printing class counts) that the class proportions in train and test are approximately equal.
- [ ] If you explored the full dataset (e.g. inspected distributions, chose to log-transform a feature), you have acknowledged this introduces mild leakage and noted it explicitly.

---

## 2. Imbalanced Data

- [ ] You have checked whether your dataset is class-imbalanced (e.g. by printing `value_counts()`).
- [ ] If imbalanced: you are **not** using raw accuracy as your primary metric.
- [ ] You are using at least one of: balanced accuracy, F1, MCC, AUC, Kappa, or weighted accuracy.
- [ ] If you rebalanced the training data (SMOTE, upsampling, downsampling): this was applied **only to training folds**, never to validation or test data.
- [ ] If you used a weighted loss function: the weights are derived from training data only.

---

## 3. Data Leakage

- [ ] **Standardisation / normalisation:** `fit` on training data only; `transform` applied to both train and test using the training-fit parameters.
- [ ] **PCA / dimensionality reduction:** fitted on training data only; test data projected using the training-fit projection.
- [ ] **Feature selection / filtering** (e.g. variance threshold, t-test filter): computed inside the CV loop on training folds only, never on the full dataset.
- [ ] **Box-Cox or other distribution transforms:** parameters estimated on training data only.
- [ ] **Any other learned transform:** ask yourself — "did I fit this on data that includes test observations?" If yes, fix it.
- [ ] All of the above are implemented **inside** the CV loop if they are tunable or data-dependent.

---

## 4. Cross-Validation Structure

- [ ] You are using **K-fold CV** (not a single train/test split) for performance estimation.
- [ ] The outer K-fold CV is **repeated R times** for stability assessment.
- [ ] Tuning of hyperparameters happens in an **inner CV loop** on each training fold — not on the same fold used for performance evaluation.
- [ ] Models are **refit** on the full training fold with the selected tuning parameters before being evaluated on the held-out fold.
- [ ] You are collecting **R × K performance scores** per method (e.g. 100 scores for R=10, K=10).
- [ ] The **same data splits** are used for all methods being compared (paired comparison requires this).
- [ ] Tuning parameter search grids are explicitly defined — you have not relied solely on package defaults without checking what they are.

---

## 5. Package and Method Defaults

- [ ] You have read the documentation for every method you use.
- [ ] **LDA:** if $n < p$, confirm the package has not silently switched to a regularised variant without warning.
- [ ] **Logistic Regression (sklearn):** `penalty='l2'` and `C=1.0` are the defaults — confirm this is intentional or override explicitly.
- [ ] **Multinomial / Logistic Regression:** confirm whether elastic net is being applied by default.
- [ ] **Random Forest:** check `max_features`, `min_samples_leaf` and other defaults that can hamper training on weak-signal data.
- [ ] **KNN:** confirm `k` is being tuned, not left at an arbitrary default.
- [ ] **Neural networks:** confirm the grid search over architecture/learning rate is explicitly defined.
- [ ] For every method: you know what the default tuning parameter settings are and have made a conscious choice to use or override them.

---

## 6. Performance Metrics

- [ ] You are computing **multiple metrics** (not just accuracy) to compare methods.
- [ ] You are inspecting **confusion matrices** for every method — not just aggregate scores.
- [ ] No method is being disqualified or selected based solely on overall accuracy.
- [ ] You have checked whether any method **completely fails to detect a minority class**.
- [ ] If using proper scoring rules (cross-entropy, Brier): you are comparing predicted *probabilities* to true labels, not just predicted classes.

---

## 7. Statistical Comparison of Methods

- [ ] You are using **paired** comparisons (same splits for all methods), not unpaired.
- [ ] You are **not** using a naive paired t-test directly on K-fold scores without correction.
- [ ] You are applying one of:
  - [ ] **Corrected paired t-test** (Bouckaert–Frank): variance inflated by $\frac{1}{RK} + \frac{n_{te}}{n_{tr}}$
  - [ ] **Bayesian corrected t-test** (Benavoli et al.): posterior over $\mu$ with the same correction factor
  - [ ] **Friedman test** + post-hoc (Conover or Nemenyi) — acknowledging independence assumption is violated
- [ ] If using Wilcoxon or Friedman: you have applied **multiple comparisons correction** (e.g. Holm).
- [ ] You are not treating $RK = 100$ scores as 100 independent observations in any test.
- [ ] If using `autorank` or similar: you have confirmed it is designed for independent datasets, and are **not** applying it directly to repeated CV scores.

---

## 8. Bayesian Analysis (if applicable)

- [ ] You are using the Nadeau–Bengio correction factor $\frac{\rho}{1-\rho} = \frac{n_{te}}{n_{tr}} = \frac{1}{K-1}$ in the variance inflation.
- [ ] You have defined a **ROPE threshold** $d$ that is meaningful for your metric (e.g. $d = 0.01$ for accuracy).
- [ ] You are reporting $P(\text{model A} > \text{model B})$ and $P(|\mu| \leq d)$, not just whether a null hypothesis was rejected.

---

## 9. Reporting and Summarising Results

- [ ] Boxplots show **paired differences** (Method A − Method B) rather than raw scores per method, wherever possible.
- [ ] Results are summarised using **equivalence classes** — groups of models that cannot be distinguished statistically.
- [ ] If reporting a single "best" model: you have checked that it is not in the same equivalence class as simpler or cheaper alternatives.
- [ ] Stability of performance is reported — a method with high variance in scores is flagged even if its median is good.
- [ ] Plots and tables cover **all** methods, not only the best-performing ones.
- [ ] Performance across **multiple metrics** is reported (not a single number per method).

---

## 10. Final Sanity Checks

- [ ] You can trace every number in your results table back to a specific code block and confirm no test data was involved in producing it except at evaluation time.
- [ ] Running your pipeline from scratch (fresh kernel / clear state) produces the same results (no hidden state).
- [ ] Random seeds are fixed and documented for all stochastic steps (train/test splits, SMOTE, model initialisation).
- [ ] No step in your pipeline uses a variable derived from the full dataset after the train/test split is made (except the final held-out test evaluation).