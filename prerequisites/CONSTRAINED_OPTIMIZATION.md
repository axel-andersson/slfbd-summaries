# Constrained Optimisation and Lagrangian Duality

**Needed for:** `lectures/L10_feature_selection.md` (embedding / penalised regression)

---

## Why it matters

Sometimes you want to minimise something — a loss, an error, a cost — but you are not free to do so without restriction. Constrained optimisation is the framework for finding the best solution when you are limited by a rule your solution must obey. Lagrangian duality is a technique for converting a problem with such a restriction into a simpler, unrestricted problem by folding the constraint into the objective. This matters because it gives you two different but equivalent ways to write the same problem, and in statistics, one of those forms is often much easier to solve or to reason about.

---

## Intuition

Imagine you are a cyclist in a race. You can take energy gels with you — each gel delays fatigue and makes you faster over the course. But gels have weight, and extra weight costs you speed. You want to finish as fast as possible, so you are balancing two competing forces: more gels help, but the weight of carrying them hurts. You are allowed to carry at most $t$ kg of gels.

Now consider a different version of the same race: there is no weight limit, but the race organiser charges a time penalty for every kilogram you carry. A small penalty and you load up on gels; a large penalty and you travel light. Somewhere there is a penalty rate that leads you to pack exactly the same $t$ kg as in the weight-limited version — giving you the exact same finishing time.

The two races have the same solution. The penalty rate is the **Lagrange multiplier** and the trick of converting the weight limit into a time penalty is called forming the **Lagrangian**.

The key insight: a larger penalty rate corresponds to a tighter weight limit. Dial the penalty to infinity and you carry nothing; set it to zero and the weight limit is irrelevant.

---

## How it works

### The constrained problem

$$
\min_{\boldsymbol{\beta}} \; f(\boldsymbol{\beta}) \quad \text{subject to} \quad g(\boldsymbol{\beta}) \leq t
$$

$f$ is the thing you want to minimise (e.g. prediction error). $g$ is the restriction (e.g. total size of the coefficients). $t$ is the budget — how much of $g$ you are allowed.

### The Lagrangian (penalised) form

Fold the constraint into the objective with a multiplier $\lambda \geq 0$:

$$
\min_{\boldsymbol{\beta}} \; f(\boldsymbol{\beta}) + \lambda \, g(\boldsymbol{\beta})
$$

$\lambda$ is the price of violating the constraint. A larger $\lambda$ makes exceeding the budget more costly, so the solution stays closer to $g(\boldsymbol{\beta}) = 0$ — equivalent to a tighter budget $t$.

### When are the two forms equivalent?

When both $f$ and $g$ are **convex** functions, the two problems are fully equivalent: for every budget $t$ there is a multiplier $\lambda$ that produces the same solution, and vice versa. This is called **strong duality**.

Convexity is the condition that matters. When it fails — for example if $g$ is a non-convex penalty like $\|\boldsymbol{\beta}\|_q^q$ with $q < 1$ — the two forms can give different answers and the equivalence breaks down.

---

## Key Equations

| Expression | Meaning |
|---|---|
| $\min f(\boldsymbol{\beta})$ s.t. $g(\boldsymbol{\beta}) \leq t$ | Constrained form: find the best $\boldsymbol{\beta}$ within a budget |
| $\min f(\boldsymbol{\beta}) + \lambda\, g(\boldsymbol{\beta})$ | Penalised form: fold the budget into the objective |
| $\lambda \geq 0$ | Lagrange multiplier; larger $\lambda$ = tighter effective budget |

---

## Tie-in to the lecture notes

In L10, penalised regression is introduced in both forms. The constrained form imposes $\|\boldsymbol{\beta}\|_q^q \leq t$ directly on the coefficients. The penalised form adds $\lambda\|\boldsymbol{\beta}\|_q^q$ to the RSS and is what is actually optimised in practice. The lecture notes state these are "equivalent via Lagrangian duality" — that equivalence holds because the RSS and the $q \geq 1$ norms are both convex. Tuning $\lambda$ via cross-validation is therefore the same as searching over feasible budgets $t$; you never need to solve the constrained form directly.

---

## Further reading

- Boyd & Vandenberghe, *Convex Optimization*, Chapter 5 (Lagrangian duality) — freely available online.
- Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning*, Section 3.4 (the constrained view of ridge and lasso).
