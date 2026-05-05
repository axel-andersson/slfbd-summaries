# Python Code Review Instructions for AI Agents

## Overview

This project collects implementations of statistical learning methods and documents pitfalls where Python implementations diverge from theory. When reviewing Python code, assume the user is implementing the theory **unless explicitly documented in comments or docstrings**. Be critical and lean toward identifying discrepancies.

## Review Workflow

### Step 1: Locate the Relevant Theory

1. Check the **CONTENTS.md** file to identify which lecture(s) are relevant
2. Read the corresponding lecture notes in the `lectures/` directory
3. Look immediately for **⚠️ "Theory vs. Practice"** subsections — these document exactly what goes wrong in implementations
4. Understand the "How it works" sections and "Key Equations"
5. Note any "Notes and Warnings" sections for edge cases (e.g., linear separability in logistic regression)

### Step 2: Understand the Documented Pitfalls

1. Search the lecture notes for explicit warnings about implementation details
2. Look for sections titled "Notes," "Warnings," "Caveats," or "Big Data Concerns"
3. Check for discussion of:
   - Numerical stability issues
   - Assumptions about data properties (scaling, centering, distribution)
   - Edge cases or degenerate scenarios
   - Computational complexity constraints in big data settings

### Step 3: Independent Verification

1. **Look up the documentation** of relevant libraries (NumPy, SciPy, scikit-learn, etc.)
   - Check default parameter settings
   - Understand what pre-processing is automatic vs. manual
   - Verify expected input/output formats
2. **Think through the mathematics yourself**
   - Work through small examples by hand
   - Identify where rounding, tolerance, or numerical precision matters
   - Check whether the code matches the order of operations in the theory

### Step 4: Critical Review Points

When reviewing code, specifically check for:

#### Data Preprocessing
- **Standardization/Centering**: Is data centered (mean = 0) or scaled (std = 1) where required? Many methods assume standardized input.
- **Scaling differences**: Does the code match the theory's implicit or explicit scaling assumptions?
- **Missing preprocessing**: Are users expected to standardize first, but the code doesn't check or warn?

#### Matrix Operations & Numerical Stability
- **Singular/near-singular matrices**: Does the code handle near-singular matrices (e.g., when $n < p$)? Does it use SVD or QR instead of direct inversion?
- **Regularization**: Are regularization terms (ridge, elastic net, etc.) correctly implemented?
- **Matrix dimensions**: Do shapes match the mathematical notation? (e.g., is $X$ $n \times p$ or $p \times n$?)

#### Algorithm Implementation
- **Convergence criteria**: Are tolerances and stopping rules matching the theory?
- **Initialization**: Do random initializations match the theory's assumptions? (e.g., do they need centering?)
- **Order of operations**: Does the code perform steps in the same order as the theory?
- **Kernel computations**: If using kernels, is the Gram matrix computed correctly?

#### Hyperparameter Defaults
- **Cross-validation defaults**: Does CV use stratification where needed (classification)? Is the random seed set?
- **Regularization parameters**: Are defaults theoretically justified?
- **Iteration limits**: Do they match typical convergence expectations?

#### Output & Interpretation
- **Scaling of outputs**: Are results in the same scale as the theory expects? (e.g., probabilities vs. log-odds)
- **Variable importance**: If provided, is the computation method documented and theoretically sound?
- **Transformations**: For dimensionality reduction, are transformed outputs correctly scaled?

#### Big Data Concerns
- **Memory efficiency**: Does the code avoid materializing large matrices unnecessarily?
- **Approximations**: Are any approximations used (e.g., kernel approximations in kPCA)? Are they documented?
- **Numerical precision**: In high dimensions, are floating-point errors accumulating?

## Red Flags — Always Investigate

- **Silent parameter changes**: If sklearn (or any library) applies defaults that differ from theory (e.g., L2 regularization), is the code explicitly configured, or does it rely on defaults?
- **Theory vs. Practice mismatches in lecture notes**: Every lecture has an **⚠️ "Theory vs. Practice"** section. If the code doesn't address documented pitfalls, flag it.
- **Undocumented assumptions**: Are assumptions about data (e.g., no missing values, finite variance, standardization) enforced or checked?
- **No validation warnings**: If users are likely to misuse the code (e.g., forget to scale, pass linearly separable data), are there checks or warnings?
- **Inconsistent with lecture examples**: If the code doesn't match worked examples in the lectures, investigate why. The lecture is the source of truth.
- **Silent regularization or solver changes**: Methods like LDA may silently regularize near-singular covariances; this needs to be explicit.

## Review Template

When reviewing code, document findings as:

```
**Issue**: [Brief title]
**Location**: [File and line reference]
**Theory**: [What the lecture notes say]
**Code**: [What the implementation does]
**Impact**: [Why this matters — wrong answer, silent failure, efficiency, etc.]
**Recommendation**: [Specific fix or alternative approach]
```

## Examples of Common Pitfalls

### Example 1: Silent L2 Regularization in Logistic Regression

From Lecture 2a (Model-based Classification):

- **Theory**: The lecture derives unpenalized MLE for logistic regression coefficients
- **Pitfall**: `sklearn.linear_model.LogisticRegression` applies L2 regularization **by default** with `C=1.0`
- **Impact**: Your fitted model is silently regularized, giving you the wrong coefficients if you intended to match the theory. The model works, but it's not the one you derived on paper
- **Fix**: Set `penalty=None` to match the unpenalized MLE, or document that you're using regularization and adjust theory accordingly

### Example 2: Multi-class Logistic Regression vs. One-vs-Rest

From Lecture 2a:

- **Theory**: Softmax regression produces valid joint class probabilities with linear log-odds
- **Pitfall**: `sklearn.linear_model.LogisticRegression` defaults to `multi_class='auto'`, which selects `'ovr'` (one-vs-rest) — a different model entirely
- **Impact**: Your probabilities don't sum to 1 in the way the theory prescribes; you fit $K$ separate classifiers instead of one joint model
- **Fix**: Explicitly set `multi_class='multinomial'` with a compatible solver like `'lbfgs'`

### Example 3: Missing Standardization
- **Theory**: PCA assumes centered data
- **Pitfall**: Code doesn't center before computing principal components
- **Impact**: Results don't match the math; explained variance is meaningless

### Example 2: Wrong Matrix Dimensions
- **Theory**: Gram matrix $K$ should be $n \times n$
- **Pitfall**: Code computes $K$ as $p \times p$ due to transposing in the wrong place
- **Impact**: Silent wrong answer; dimensions hide the error

### Example 3: Regularization Not Applied
- **Theory**: Ridge regression adds $\lambda I$ to $X^T X$
- **Pitfall**: Code forgets to add regularization, or adds it to the wrong term
- **Impact**: Unstable estimates in high-$p$ settings; overfitting

### Example 4: Premature Convergence
- **Theory**: Algorithm should iterate until convergence tolerance is met
- **Pitfall**: Default iteration limit is too small; algorithm stops before converging
- **Impact**: Suboptimal solution; results depend on undocumented default

## Severity Levels

When documenting pitfalls, classify as:

- **Critical**: Code produces mathematically incorrect results or silently fails
- **High**: Violates theoretical assumptions; results are unreliable without user awareness
- **Medium**: Inefficient or non-standard; works but differs from theory without documentation
- **Low**: Edge cases or minor deviations; unlikely to affect typical usage

---

**Remember**: This project's value is in documenting the gap between theory and practice. Be thorough and specific about *why* an implementation differs from theory, not just *that* it does.
