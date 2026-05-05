# Project Instructions

## Repository Layout

```
lectures/        Lecture summary files (L01_*.md, L02_*.md, …)
prerequisites/   Self-contained explainers for background concepts
```

---

## Creating a Prerequisites File

Prerequisites files are brief, self-contained explainers for background concepts that a reader of the lecture notes is expected to know but may not. They live in `prerequisites/` and are linked from the relevant lecture notes.

### When to create one

Create a prerequisites file when:
- A lecture note lists a concept under a **Prerequisites** bullet and it is non-trivial (e.g. "Lagrangian duality", "SVD", "coordinate descent").
- A user explicitly says they are unfamiliar with a concept needed to understand a lecture.
- A concept is referenced repeatedly across lectures without ever being derived.

If it is not clear which prerequisite the user is missing or in which context, **ask before writing**.

---

### Step-by-step workflow

**1. Identify the concept and context**

Read the relevant lecture file (e.g. `lectures/L10_feature_selection.md`). Understand:
- Where exactly the concept is used and why it matters in that context.
- What level of depth is needed — just enough to follow the lecture, not a full textbook treatment.

**2. Write the prerequisites file**

Create `prerequisites/<CONCEPT_NAME>.md` using ALL CAPS with underscores (e.g. `CONSTRAINED_OPTIMIZATION.md`, `SVD.md`, `COORDINATE_DESCENT.md`).

Follow this structure and style:

```markdown
# <Concept Name>

**Needed for:** <lecture file name(s) where this concept appears>

---

## Why it matters

One short paragraph explaining the role this concept plays in the course material. Ground it in the specific lecture context — do not write a generic textbook introduction.

---

## Intuition

Explain the core idea in plain language before any formalism. Use an analogy or geometric picture where possible. Aim for 3–6 sentences.

---

## How it works

Present the formal definition and key results. Use equations where necessary. Keep derivations minimal — state results and explain what they mean rather than proving everything from scratch.

Include subsections (###) if there are distinct sub-concepts worth separating.

---

## Key Equations

List the 1–4 most important equations with a one-line gloss for each symbol or term that is not self-evident.

---

## Tie-in to the lecture notes

One short paragraph (3–5 sentences) connecting the concept directly back to how it appears in the lecture. Be specific — name the estimator, formula, or algorithm that relies on it.

---

## Further reading

- Optional: 1–2 pointers (textbook section, Wikipedia article title) for readers who want more depth. Do not include URLs unless the user supplies them.
```

**Style rules** (match the lecture notes):
- Write in clear, precise English. No padding or filler sentences.
- Lead every section with the "why" before the "what".
- Use $\LaTeX$ for all mathematical notation (inline `$…$`, display `$$…$$`).
- Prefer tables for comparisons; prefer bullet lists over prose for enumerable items.
- Do not add emojis unless the user asks.
- Do not write multi-paragraph comment blocks or unnecessary meta-commentary.
- Depth calibration: a reader who has seen the concept once but forgotten it should be able to follow the lecture after reading the file. A reader who has never seen it will need further reading.

**Section-specific rules:**

*Why it matters:* The reader is here precisely because they do not yet understand the lecture context. Do not assume they have grasped where or how the concept is used in the lecture — that understanding is what the whole file is building toward. Explain why the concept is worth knowing in its own right, using plain language. A brief, loose connection to the lecture is fine, but do not lean on lecture-specific notation or ideas that themselves require the concept to understand.

*Intuition:* Language must be natural and conversational — write the way you would explain it to someone in person, not the way a textbook would. Avoid incidentally formal phrases like "the constraint limits your solution" or "the objective is subject to a budget". Instead, describe what actually happens physically or concretely: what changes, what gets worse, what you are trading off. Analogies should be grounded in everyday experience and carried through consistently. Before finalising an analogy, verify that it contains a genuine trade-off: the reader must be able to see why both sides of the tension exist. If an analogy has no reason to do the thing being constrained (e.g. carry weight when weight only hurts you), it is broken — rewrite it.

*Extra sub-concepts (e.g. KKT conditions, dual variables):* Only include a sub-concept if a reader who does not know it would be unable to follow the main idea or the tie-in to the lecture. If it adds a new thing to learn without meaningfully helping the reader understand the core concept, leave it out. When in doubt, omit it — a shorter file that builds one idea clearly is better than a longer file that introduces confusion.

**3. Link back from the lecture notes**

In the lecture file, find the **Prerequisites** bullet list inside the relevant concept section. On the same line as the prerequisite that now has an explainer, append:

```
— see `prerequisites/<CONCEPT_NAME>.md`
```

Example — before:
```markdown
- Lagrangian duality and constrained optimisation
```
After:
```markdown
- Lagrangian duality and constrained optimisation — see `prerequisites/CONSTRAINED_OPTIMIZATION.md`
```

Do not add a new bullet; edit the existing one in-place.

---

### What NOT to do

- Do not reproduce the full lecture content in the prerequisites file — just the background.
- Do not add prerequisites files for trivial concepts (e.g. "linear algebra basics", "what a matrix is").
- Do not create a prerequisites file without updating the corresponding lecture note, and vice versa.
- Do not invent URLs or external links.
