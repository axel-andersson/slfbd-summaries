# Statistical Learning for Big Data — Complete Contents

**Course:** MSA220/MVE441 Statistical Learning for Big Data  
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences  
**Period:** March–May 2026

---

## Lecture 1: Introduction — Statistical Learning for Big Data

**Topics:**
- The Data Deluge and challenges of Big Data
- Big-$n$ vs. Big-$p$ settings
- Statistical challenges: curse of dimensionality, multiple testing, selection bias
- Statistics terminology recap: random variables, probability rules, expectation
- What is Statistical Learning?
- Statistical decision theory for regression
- $k$-Nearest Neighbours (kNN) for regression and classification
- Statistical decision theory for classification and Bayes' rule

---

## Lecture 2a: Model-based Classification

**Topics:**
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

## Lecture 2b: Model Evaluation and Bias-Variance Tradeoff

**Topics:**
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

## Lecture 3: A First Look at Dimension Reduction

**Topics:**
- The Curse of Dimensionality
- High-Dimensional Predictive Modelling
- Projection onto a Subspace
- The Rayleigh Quotient
- Principal Component Analysis (PCA)
- Pre-processing and Standardisation
- Singular Value Decomposition (SVD)
- SVD and PCA Connection
- Regularised Discriminant Analysis (RDA)

---

## Lecture 4: Preserving Local Geometry

**Topics:**
- Distance preservation and the Gram matrix
- Kernels and the kernel trick
- Kernel PCA (kPCA)
- Classical Multidimensional Scaling (MDS)
- The problem with global methods: the Swiss roll
- Data-driven distances and Isomap
- t-distributed Stochastic Neighbour Embedding (tSNE)
- Uniform Manifold Approximation and Projection (UMAP)
- Comparison: global vs. local dimension reduction methods

---

## Lecture 5: Rule-based Classification and Regression (CART & Random Forests)

**Topics:**
- Classification as Partitions
- Classification and Regression Trees (CART)
- Measures of Node Purity
- Stopping Criteria and Pruning
- Recap: The Bootstrap
- Bootstrap Aggregation (Bagging)
- Random Forests
- Variable Importance

---

## Lecture 6: Kernel Methods

**Topics:**
- Kernels: definition and examples
- Mercer / positive definite kernels
- The kernel trick
- Recap: PCA and its limitations
- Kernel PCA (kPCA)
- Recap: Ridge regression
- Woodbury matrix identity
- Kernel ridge regression
- Functional gradient descent and kernel regression
- Regularisation in kernel ridge regression

---

## Lecture 7: Boosting

**Topics:**
- Ensemble Methods Overview
- Gradient Descent (Recap)
- Functional Gradient Descent
- Kernel Ridge Regression (KRR)
- KRR with Gradient Descent
- Kernel Logistic Regression with Gradient Descent
- Forward Fitting Schemes (Forward Selection & Forward Stagewise Regression)
- AdaBoost
- Gradient Boosting Machines (GBM)

---

## Lecture 8: Boosting, SVMs and Flexible Discriminant Analysis

**Topics:**
- Tuning Boosting Machines (Recap)
- XGBoost
- Support Vector Machines: Separating Hyperplanes
- Support Vector Machines: Soft Margin (Slack Variables)
- SVMs and the Kernel Trick
- Discriminant Analysis (Recap)
- Mixture Discriminant Analysis (MDA)
- Fisher's Linear Discriminant Analysis (FLDA)
- Flexible Discriminant Analysis (FDA) via Optimal Scoring
- Kernel Discriminant Analysis

---

## Lecture 9: Neural Networks

**Topics:**
- Logistic Regression as a Network Model
- Multi-class Logistic Regression (Softmax)
- Neural Network Architecture
- Backpropagation
- Training Procedure
- Tuning and Regularization
- Autoencoders, kPCA, and PCA
- Neural Network Classifier as Autoencoder + Classification Layer
- Comparison: NN, KRR, and SVMs

---

## Lecture 10: Feature Selection and Regularised Regression

**Topics:**
- Goals of modelling
- Feature selection: Filtering
- Feature selection: Wrapping
- Feature selection: Embedding
- Comparison: Feature selection methods
- Constrained and regularised regression
- Ridge regression
- SVD and ridge regression
- Effective degrees of freedom (Ridge)
- Lasso regression
- Intuition for the penalties
- Soft-thresholding and the lasso solution
- Relation to OLS estimates
- Shrinkage and effective degrees of freedom (Lasso)
- Notes and caveats of the lasso

---

## Lecture 11: Regularised Regression (cont'd)

**Topics:**
- Regularisation in Classification
- Recap: Regularised Discriminant Analysis (RDA)
- Recap: Naive Bayes LDA
- Nearest Shrunken Centroids
- Extensions of the Lasso
- The Elastic Net
- The Group Lasso
- Comparison: Lasso, Elastic Net, and Group Lasso
- Penalisation in GLMs
- Sparse Logistic Regression
- Sparse Multi-Class Logistic Regression

---

## Lecture 12: Multiple Testing

**Topics:**
- Statistical Testing recap (p-values, null distributions, Type I/II errors)
- The Multiple Testing Problem
- Family-Wise Error Rate (FWER) and Bonferroni Correction
- False Discovery Rate (FDR) and the Benjamini-Hochberg Procedure
- Adjusted p-values
- Holm-Bonferroni: a middle ground

---

## Lecture 13: High-Dimensional Inference — Feature & Stability Selection

**Topics:**
- Lasso as a Selection Mechanism (Recap)
- The Selection Bias Problem
- Sample-Splitting
- Multi Sample-Splitting
- Stability Selection
- De-sparsified (De-biased) Lasso
- Bias-Corrected Ridge Regression

---

## Course Structure

The course progresses through four main areas:

1. **Foundations & Classification** (Lectures 1–2b): Statistical learning principles, Bayes' rule, model-based classification, and model evaluation

2. **Dimension Reduction & Non-linear Methods** (Lectures 3–4): Handling high-dimensional data through PCA, kernel methods, and manifold learning

3. **Complex Methods & Applications** (Lectures 5–9): Tree-based methods, boosting, SVMs, neural networks, and kernel methods

4. **High-Dimensional Inference** (Lectures 10–13): Feature selection, regularised regression, and valid statistical inference in high dimensions
