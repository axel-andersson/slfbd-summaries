# Multiple Testing

**Course:** MSA220/MVE441 Statistical Learning for Big Data
**Lecturer:** Rebecka Jörnsten, Mathematical Sciences
**Date:** 4th May 2026

---

## 📋 Contents

- Statistical Testing recap (p-values, null distributions, Type I/II errors)
- The Multiple Testing Problem
- Family-Wise Error Rate (FWER) and Bonferroni Correction
- False Discovery Rate (FDR) and the Benjamini-Hochberg Procedure
- Adjusted p-values
- Holm-Bonferroni: a middle ground

---

## 📝 Summary

This lecture addresses the problem that arises when many statistical tests are performed simultaneously. It begins with a recap of single-hypothesis testing: test statistics, p-values, and their distributions under the null. It then shows that performing even a modest number of tests virtually guarantees at least one false positive. Two main correction frameworks are developed: FWER control (via Bonferroni and Holm-Bonferroni, which bound the probability of any false rejection) and FDR control (via the Benjamini-Hochberg procedure, which bounds the expected proportion of false rejections among all rejections). Practical tradeoffs between these approaches are discussed with worked examples.

---

## 🎯 Learning Goals

- Understand why running multiple tests inflates the probability of false positives.
- Know the definition of FWER and FDR, and be able to distinguish between them.
- Apply the Bonferroni correction and understand its conservatism.
- Apply the Benjamini-Hochberg procedure and interpret adjusted p-values.
- Recognise when each correction method is appropriate.

---

## 📚 Concepts

### Recap: Statistical Testing

**Summary:** A statistical test uses a test statistic $T(X)$ computed from data $X$. Under the null hypothesis $H_0$, the distribution of $T$ is known (e.g. a $t$-distribution) or can be simulated (e.g. via permutation). The **p-value** is the probability of observing $T$ as extreme or more extreme than what was actually observed, under $H_0$:

$$
\text{p-value} = P(T \geq t_{\text{obs}} \mid H_0)
$$

A key fact: when $H_0$ is true, p-values are uniformly distributed on $[0,1]$. We reject $H_0$ when the p-value falls below a chosen significance level $\alpha$. We never "accept" the alternative $H_1$ — we only fail to reject $H_0$.

---

### The Multiple Testing Problem

**Why it matters:** Every time an additional hypothesis is tested, the risk of at least one false positive grows. In high-dimensional data settings (genomics, neuroimaging, network security), thousands of tests are run simultaneously, making spurious findings nearly certain without correction.

**Intuition:** Suppose each individual test has a 5% chance of a false positive. Running 9 independent tests, the probability of at least one false positive is not 5% — it is:

$$
P(\text{at least one false positive}) = 1 - (1 - \alpha)^n
$$

For $n = 9$, $\alpha = 0.05$: this is already ~37%. By $n = 100$ it exceeds 99%.

**Prerequisites:**
- Single-hypothesis testing, p-values
- Probability of complementary events

**How it works:**

When $n$ true null hypotheses are tested simultaneously at level $\alpha$, the expected number of false rejections is $\alpha \times n$. This creates a systematic problem: the more tests run, the more false positives accumulate purely by chance.

The **error types** in a table of $n$ tests:

|               | $H_0$ true | $H_0$ false | Total     |
|---------------|-----------|-------------|-----------|
| Reject $H_0$  | $V$       | $S$         | $R$       |
| Accept $H_0$  | $U$       | $T$         | $n - R$   |
|               | $n_0$     | $n - n_0$   | $n$       |

- $V$ = false positives (Type I errors)
- $T$ = false negatives (Type II errors)
- $R$ = total rejections (our "findings")

**Key Equations:**

$$
P(\text{at least one false positive}) = 1 - (1-\alpha)^n
$$

---

### Family-Wise Error Rate (FWER)

**Why it matters:** FWER is the most stringent multiple testing criterion. It is appropriate when any false positive is unacceptable — e.g. in confirmatory clinical trials.

**Intuition:** Rather than controlling the false positive rate test by test, FWER controls the probability that even a single false rejection occurs across the entire family of tests.

**Prerequisites:**
- Type I/II error definitions
- Union bound (Boole's inequality)

**How it works:**

$$
\text{FWER} = P(V \geq 1)
$$

FWER is what the exponential false-positive growth curve measures. To control it, we tighten the per-test significance threshold.

**Key Equations:**

$$
\text{FWER} = P(V \geq 1) \leq \alpha
$$

---

### Bonferroni Correction

**Why it matters:** The Bonferroni correction is the simplest and most widely known FWER control procedure. It applies without any assumptions about the dependence structure between tests.

**Intuition:** If you want the overall family-wise error rate to be at most $\alpha$, and you're running $n$ tests, then you should only reject a single test if it would survive even if all $n$ tests were given a share of the error budget: $\alpha / n$.

**Prerequisites:**
- FWER definition
- Union bound

**How it works:**

Perform each test at level $\alpha/n$ instead of $\alpha$. By the union bound (Boole's inequality):

$$
P(V \geq 1) = P\left(\bigcup_{j=1}^{n_0} \text{reject } H_0^j\right) \leq \sum_{j=1}^{n_0} P(\text{reject } H_0^j) = n_0 \cdot \frac{\alpha}{n} \leq \alpha
$$

This holds regardless of how many hypotheses are actually null ($n_0 \leq n$).

**Key Equations:**

$$
\text{Adjusted level: } \alpha_n = \frac{\alpha}{n}
$$

$$
\text{Adjusted p-value: } p_{\text{adj}}^B = \min(1,\; n \cdot p_{\text{raw}})
$$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Assumption-free (works for any dependence structure) | Very conservative — especially when $n$ is large |
| Provably controls FWER at level $\alpha$ | Low statistical power; many true signals are missed |
| Simple to apply | Requires all $n$ raw p-values before adjustment |

**Python Implementation:**

```python
from statsmodels.stats.multitest import multipletests

pvals = [0.001, 0.02, 0.3, 0.0005, 0.4]  # raw p-values

reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='bonferroni')
print(reject)         # Boolean array: which hypotheses are rejected
print(pvals_corrected)  # Bonferroni-adjusted p-values
```

⚠️ **Theory vs. Practice:** `multipletests` uses `method='bonferroni'` to multiply each p-value by $n$ and cap at 1. This matches the formula exactly. However, the function takes the *number of tests* as `len(pvals)` — if you pre-filter your p-values before calling (e.g. excluding NaNs), your $n$ will be wrong and the correction will be too lenient. Always pass the complete, unfiltered set of p-values.

---

### False Discovery Rate (FDR) and the Benjamini-Hochberg Procedure

**Why it matters:** Bonferroni is often too conservative in large-scale testing (e.g. genomics with 10,000+ genes). FDR control offers a more powerful alternative that still keeps false positives in check — instead of banning any false positive, it bounds their *proportion* among findings.

**Intuition:** FDR asks: "Among all the hypotheses I rejected, what fraction are actually nulls?" rather than "Did I make even one mistake?" This is a more pragmatic goal when running thousands of tests, because a small proportion of false positives among many true discoveries is acceptable — especially if findings will be followed up with experiments.

**Prerequisites:**
- p-value distributions under null and alternative
- Expected value; proportion vs. probability

**How it works:**

Define the **False Discovery Proportion (FDP)**:

$$
\text{FDP} = \frac{V}{R} \cdot \mathbf{1}[R \geq 1]
$$

The **False Discovery Rate** is its expectation:

$$
\text{FDR} = E(\text{FDP})
$$

The **Benjamini-Hochberg (BH) procedure** at level $q$:

1. Sort p-values: $P_{(1)} \leq P_{(2)} \leq \cdots \leq P_{(n)}$
2. Find the largest rank $r$ such that:

$$
P_{(r)} \leq q \cdot \frac{r}{n}
$$

3. Reject all null hypotheses $H_{(1)}, \ldots, H_{(r)}$.

This is equivalent to choosing $\alpha$ in a data-dependent way to reject as many hypotheses as possible while keeping an estimated FDP below $q$. The estimated FDP at any threshold $\alpha$ is:

$$
\widehat{\text{FDP}} = \frac{n \cdot \alpha}{M(\alpha)}
$$

where $M(\alpha)$ is the number of p-values $\leq \alpha$. Setting $\alpha = P_{(r)}$ and requiring $\widehat{\text{FDP}} \leq q$ yields exactly the BH cutoff.

**Theorem (Benjamini & Hochberg 1995):** If test statistics are independent, the BH procedure at level $q$ satisfies:

$$
\text{FDR} \leq \frac{n_0}{n} \cdot q \leq q
$$

> ⚠️ FDR control is **not guaranteed** when test statistics are dependent.

**Key Equations:**

$$
P_{(r)} \leq q \cdot \frac{r}{n} \quad \text{(BH rejection criterion)}
$$

$$
\text{FDR} \leq \frac{n_0 \cdot q}{n} \leq q \quad \text{(BH guarantee under independence)}
$$

$$
p_{\text{adj}}^{BH}(j) = p_{\text{raw}(j)} \cdot \frac{n}{j} \quad \text{(adjusted p-value for rank } j\text{)}
$$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Much more powerful than Bonferroni in large-scale testing | Only controls FDR *in expectation*, not the FDP in each run |
| Controls at a meaningful, interpretable rate | Requires independence of test statistics (or positive dependence) |
| Works well in exploratory settings with follow-up experiments | Less appropriate when any false positive is intolerable |

**Python Implementation:**

```python
from statsmodels.stats.multitest import multipletests

pvals = [0.001, 0.02, 0.3, 0.0005, 0.4]

reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='fdr_bh')
print(reject)
print(pvals_corrected)
```

⚠️ **Theory vs. Practice:** The adjusted p-values returned by `multipletests` for BH are computed as $\min(1, p_{\text{raw}(j)} \cdot n/j)$ with an additional cumulative minimum applied from the largest rank downward, ensuring monotonicity. This is correct but different from the raw formula — do not manually compute $p_{\text{raw}} \cdot n / j$ and compare to $q$; use the library. Also note: `method='fdr_bh'` assumes independence. If your tests are correlated (e.g. nearby SNPs in GWAS), you must use `method='fdr_by'` (Benjamini-Yekutieli), which is valid under arbitrary dependence but more conservative.

---

### Comparison: Bonferroni vs. BH vs. Unadjusted (on a 10,000-test example)

A simulation with $n = 10{,}000$ tests, $n_0 = 9{,}900$ true nulls and $n - n_0 = 100$ true alternatives at $\alpha = q = 0.05$:

| Method | False rejections $V$ | True rejections $S$ | Total rejections $R$ |
|--------|---------------------|---------------------|----------------------|
| Unadjusted | 477 | 100 | 577 |
| Bonferroni | 1 | 16 | 17 |
| BH | 3 | 82 | 85 |

- Unadjusted testing finds all true signals but produces hundreds of false positives.
- Bonferroni eliminates nearly all false positives but misses 84 of 100 true signals.
- BH strikes a practical balance: 82 true signals found, only 3 false positives, FDP = 0.035.
- The BH cutoff is visualised as a diagonal line (slope $q$) on a sorted p-value plot; the Bonferroni cutoff is a flat horizontal line much lower.

---

### Adjusted p-values

**Why it matters:** Rather than fixing $\alpha$ and re-thresholding every time, adjusted p-values let you apply any desired $\alpha$ later without re-running the correction procedure.

**How it works:**

- **Bonferroni:** $p_{\text{adj}}^B = \min(1,\; n \cdot p_{\text{raw}})$. Reject if $p_{\text{adj}} \leq \alpha$.

- **Benjamini-Hochberg:** Sort raw p-values as $p_{\text{raw}(1)} \leq \cdots \leq p_{\text{raw}(n)}$. The BH rule is to reject rank $j$ if $p_{\text{raw}(j)} \leq \alpha \cdot (j/n)$, equivalently if $(n \cdot p_{\text{raw}(j)})/j \leq \alpha$. So:

$$
p_{\text{adj}}^{BH}(j) = p_{\text{raw}(j)} \cdot \frac{n}{j}
$$

Reject if $p_{\text{adj}}^{BH}(j) \leq \alpha$ (with monotonicity enforced from largest to smallest rank).

Reporting adjusted p-values alongside raw p-values lets readers apply their own preferred $\alpha$ without re-deriving the correction.

---

### Holm-Bonferroni Correction

**Why it matters:** Bonferroni controls FWER but is uniformly conservative. The Holm-Bonferroni procedure controls FWER at the same $\alpha$ but is always at least as powerful — it never rejects fewer hypotheses than Bonferroni.

**Intuition:** Instead of dividing $\alpha$ by $n$ for every test, the threshold is made progressively less strict as smaller p-values are confirmed. The smallest p-value must pass the strictest test ($\alpha/n$), the next must pass $\alpha/(n-1)$, and so on. As soon as one test fails, we stop rejecting.

**Prerequisites:**
- Bonferroni correction and FWER
- Sequential testing concepts

**How it works:**

1. Compute raw p-values $p_j$, sort them: $p_{(1)} \leq \cdots \leq p_{(n)}$.
2. For rank $j$ (starting from 1), test: $p_{(j)} \leq \frac{\alpha}{n - j + 1}$
3. Reject $H_{(j)}$ and continue as long as the condition holds. Stop at the first failure; all remaining hypotheses are not rejected.

The Holm-Bonferroni adjusted p-value at rank $j$ is:

$$
p_{\text{adj}}^{HB}(j) = p_{(j)} \cdot (n - j + 1)
$$

**Why this controls FWER:** The argument focuses on $p_{\min} = \min_{i \in I_0} p_i$, the smallest p-value among true nulls. Its worst-case rank is $k = n - |I_0| + 1$ (i.e. all non-null p-values are smaller). At that rank, the threshold is $\alpha / (n - k + 1) = \alpha / |I_0|$, giving:

$$
P(p_{\min} \text{ rejected}) \leq |I_0| \cdot \frac{\alpha}{|I_0|} = \alpha
$$

**Strengths and Weaknesses:**

| Strengths | Weaknesses |
|-----------|------------|
| Uniformly more powerful than Bonferroni | Still conservative relative to FDR-based methods |
| Exact FWER control; no independence assumption required | Sequential nature makes it harder to report a single "adjusted" threshold |
| Controls FWER regardless of $n_0$ | Power gain over Bonferroni is modest in practice for large $n$ |

**Python Implementation:**

```python
from statsmodels.stats.multitest import multipletests

pvals = [0.001, 0.02, 0.3, 0.0005, 0.4]

reject, pvals_corrected, _, _ = multipletests(pvals, alpha=0.05, method='holm')
print(reject)
print(pvals_corrected)
```

⚠️ **Theory vs. Practice:** The Holm procedure is sequential — once rejection stops, all remaining hypotheses are retained. `multipletests` handles this correctly internally, but if you sort the adjusted p-values yourself and apply a flat threshold, you will get the wrong answer. The adjusted p-values from Holm are not monotone in the same simple way as Bonferroni's; `multipletests` applies a cumulative maximum to enforce monotonicity, which is correct. Do not reimplement this manually.

---

## ✅ Key Takeaways

- With $n$ independent tests at level $\alpha$, the probability of at least one false positive is $1 - (1-\alpha)^n$ — this reaches 99% by 100 tests.
- **FWER** ($P(V \geq 1) \leq \alpha$) is the right goal when false positives are intolerable. The Bonferroni correction achieves this by testing each hypothesis at $\alpha/n$.
- Bonferroni is simple and assumption-free, but very conservative — it sacrifices power, especially when $n$ is large.
- **FDR** ($ E[\text{FDP}] \leq q$) is a less stringent but more practically useful criterion in exploratory settings. The BH procedure controls FDR under independence of test statistics.
- BH dramatically outperforms Bonferroni in power while still providing a meaningful false-positive guarantee.
- **Holm-Bonferroni** is a strictly more powerful FWER procedure than Bonferroni, achieved by applying tighter thresholds sequentially.
- Always consider: what test are you running? Are the p-values from valid, independent tests? For correlated tests, use Benjamini-Yekutieli (FDR) or permutation-based FWER corrections.
- Large sample sizes create their own issues for testing — upcoming lectures will address this.