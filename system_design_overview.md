  # System design: generalized campaign / profit optimization

This doc reframes the Ingosstrakh case as a domain-agnostic system, so "insurance"
becomes one instance of a general pattern rather than the whole spec. Just enough
detail to align on architecture before writing code.

## 1. The actual problem, stripped of insurance vocabulary

Given a large set of **subjects**, each with several mutually-exclusive **options**,
and each option worth some **value**, pick at most one option per subject to
maximize total value, subject to a pile of **capacity constraints** shared across
subjects. The constraints are declared over an unknown, variable number of
**decision dimensions**, which may nest inside each other (a tree).

Insurance instance: subject = client, options = (product, channel, segment),
value = EV. Farm instance: subject = field, options = (crop, method), value =
profit. Same shape, different labels. The system must not hardcode the labels.

## 2. Core abstractions

| Concept | Meaning | Insurance example | Farm example |
|---|---|---|---|
| **Subject** | The thing being allocated to; gets ≤1 option | client (`SUBJISN`) | field |
| **Dimension** | A named axis an option varies along | product, channel, segment | crop, irrigation method |
| **Dimension tree** | Parent/child nesting among a dimension's values — can nest *within* a single dimension at arbitrary depth, not only across separate dimension columns | segment `KSK_OSG_AA_A` under product `KSK_OSG`; **or**, entirely within `channel` alone: `email` → `personal` / `subscription` → (under `personal`) `automated` / `hand-written` → (under `automated`) `expensive` / `cheap` | variety under crop family |
| **Value component** | A numeric input that composes into the objective | Premium, Margin, Response, Cost | Price, Yield, Margin, Cost |
| **Objective / value** | Per-(subject, option) number to maximize the sum of | `EV = Margin×Premium×Response − Cost` | `Profit = Margin×Price×YieldProb − Cost` |
| **Constraint** | A min/max bound on an aggregate (count or cost) over a **scope** | "≤4000 KASKO/SMS offers total" | "≤500 ha of wheat total" |
| **Scope** | The subset of dimension values (at any tree depth) a constraint applies to | `{product: KSK_OSG}` → covers every segment/channel under it | `{crop: wheat}` → covers every variety |

Nothing here is insurance-specific. That's the point: the same five abstractions
should describe any "allocate scarce capacity across subjects to maximize value"
problem handed to this system.

## 3. Multi-level, simultaneous, and conflicting constraints

The channel example above (`email → personal/subscription → automated/hand-written
→ expensive/cheap`) matters because it shows the tree isn't only a *cross-dimension*
nesting (product containing segments) — a *single* dimension can itself be an
arbitrarily deep taxonomy. Two consequences follow directly, and a third risk
does too.

**Constraints bind at whatever depth they name, and several can be active on
overlapping scopes at once.** Nothing requires constraints to all target the
same tree level, or even the same dimension. All of the following can be true
of one campaign simultaneously: a cap on all of `email` (root), a separate cap
on `hand-written email` specifically (a mid-level node three levels down), and
an unrelated cap on `expensive SMS` (a different dimension's subtree
entirely). Because scope is ancestor-inclusive, a single offer row can sit
inside several constraints' scopes at once — its own leaf value, plus every
ancestor above it, in however many dimensions have a tree. This is not an edge
case to special-case; it's the normal shape of the problem, and the scope
representation has to support it natively: each constraint resolves its own
mask independently, with no knowledge of any other constraint, and enforcement
(dual multipliers for global constraints, the demote/fill rounds for
per-subject ones) already operates over an arbitrary *list* of independently-scoped
constraints rather than assuming a partition. The generalization work is
making sure a scope can name a node at *any* depth of *any* dimension's tree —
not adding new solver machinery.

**Overlapping multi-level constraints can be jointly contradictory.** A cap on
`email` (max 100 total) and separate minimums on `automated email` (min 60)
and `hand-written email` (min 60) cannot all hold at once — the children's
minimums alone exceed the parent's maximum. This isn't hypothetical: the task
spec itself warns constraints "may be duplicated or incorrect." A system that
only checks constraints one at a time (as `verify.py` does today, correctly,
for reporting) will happily report multiple individual violations without ever
saying *why* — that the constraint set itself was unsatisfiable before any
solver ran. The system needs:

- **A conflict-detection pass**, run once on the parsed constraint set (before
  or alongside solving): for every ancestor→descendant chain in every
  dimension's tree, check whether an ancestor's max is smaller than the sum of
  its (known, resolvable) descendants' mins, and flag it. Cheap — this only
  ever touches the small constraint table, never subject-level rows.
- **A documented precedence rule** for what the solver should actually do when
  full joint satisfaction is impossible, rather than leaving it to whatever
  the optimizer's internals happen to produce. Default proposal: a more
  specific (deeper-scoped) constraint takes precedence over a broader one it
  conflicts with, on the reasoning that an explicit narrow rule was probably
  written intentionally and shouldn't be silently overridden by a coarser
  default. **This default is an assumption, not a certainty — worth
  confirming with the contractor once a real conflicting example shows up**,
  since the "correct" tie-break is a business call, not a technical one.
- **A conflict report** surfaced alongside the normal verification report,
  distinguishing "these constraints were jointly infeasible and had to be
  traded off" (expected, tolerated, explained) from "the solver is broken"
  (not tolerated) — the same spirit as `repair.py`'s existing
  `UNRESOLVED violation` log lines, but stated as a first-class, pre-solve
  finding instead of an implicit side effect discovered only after the fact.

## 4. Canonical internal schemas

Everything upstream (raw files) is domain-specific and messy. Everything
downstream of these two schemas is domain-agnostic and must never see a raw
column name again.

**Option table** (long format, one row per subject×option):

```
subject_id | dim_1 ... dim_k | value_component_1 ... value_component_m | value
```

`k` and `m` are discovered per dataset, not fixed. `value` is either read
directly (already-computed score) or composed from the value components by a
declared formula.

**Constraint table** (already close to what `constraint_med/hard.csv` look like):

```
constraint_id | scope: {dim_name -> value_or_subtree} | measure: count|cost | min | max | per_subject: bool
```

A constraint whose scope names a non-leaf node (e.g. a product, not a segment
— or, per Section 3, a mid-level node *within* one dimension's own taxonomy,
like `personal` under `email`) binds the aggregate over **every descendant**
of that node — this is the "inheritance" the contractor keeps describing.
Flat exact-match scoping (what a naive parser does) is a special case where
every dimension's tree happens to be one level deep.

## 5. Pipeline

```
raw files (arbitrary schema, arbitrary vocabulary)
        │
        ▼
[1] schema resolution        — LLM-assisted, one-time, small data
[2] dimension hierarchy build — LLM-assisted, one-time, small data
[3] constraint parsing        — LLM-assisted, one-time, small data
        │
        ▼           canonical option table + canonical constraint set
        ▼
[4] objective construction    — deterministic, vectorized
[5] optimization / solve      — deterministic, vectorized/GPU, the hot path
[6] verifier code generation   — LLM-assisted, one-time
[7] verifier execution         — deterministic, vectorized
        │
        ▼
assignment + verification report (+ conflict report, per Section 3)
```

**Step 1 — schema resolution.** Figure out which raw column is the subject id,
which are dimensions, which are value components, without hardcoding names
like `SUBJISN`/`PRODUCT`. Cheap heuristics (cardinality, dtype) get you most of
the way; an LLM resolves ambiguous cases and produces a mapping to the
canonical schema.

**Step 2 — hierarchy build.** Infer parent/child nesting among each dimension's
distinct values (string-prefix convention if present, semantic grouping via LLM
otherwise, e.g. knowing "durum wheat" nests under "wheat" with no shared
substring) — including nesting *within* a single dimension's own values at
whatever depth it actually goes (the `email`/`personal`/`automated`/`expensive`
example), not just across adjacent dimension columns. Output: a tree per
dimension, plus (Section 3) a conflict-detection pass over how constraints'
scopes sit relative to one another in that tree.

**Step 3 — constraint parsing.** Map each raw constraint row (however it's
worded) onto the canonical `(scope, measure, min, max, per_subject)` tuple,
using the dimension trees from step 2 to resolve scope at whatever depth the
row names. This is where an LLM earns its keep on genuinely novel vocabulary
the parser wasn't written against.

**Step 4 — objective.** Compute `value` per row from the value components. Pure
arithmetic, no LLM.

**Step 5 — solve.** The actual assignment problem: choose ≤1 option per subject
maximizing Σvalue subject to every constraint. This is where GPU/vectorization
and whatever OR technique (LP relaxation, Lagrangian relaxation, greedy +
repair) belongs. No LLM anywhere near this loop — it runs over the full
subject count (millions of rows) and is what gets benchmarked.

**Step 6 — verifier generation.** Given the parsed constraint set, generate the
checking code (e.g. templated "group by scope, sum measure, check bounds" per
constraint) rather than hand-writing a checker per constraint type in advance.

**Step 7 — verify.** Run the generated code against the chosen assignment.
Deterministic, vectorized, no LLM.

## 6. The LLM boundary (why this isn't wasted compute)

LLM calls only ever touch **small, one-time, symbolic** data: the distinct
column names, the distinct constraint-row strings (tens to low-thousands), the
distinct dimension values — never the subject-level rows (millions). Cache the
resolved schema/hierarchy/constraint-set/verifier code once per input dataset;
the expensive part (step 5, and step 7 at scale) is pure vectorized code and
never calls an LLM. This keeps the benchmarked hot path exactly as fast as a
fully hardcoded solution, while the "generality" the grading rubric rewards
comes from steps 1–3 and 6 tolerating vocabulary the developer never saw.

## 7. What's reusable from the current `offer_opt` code

- `schema.py` / `scope.py`: the right shape of idea (a canonical `ConstraintSpec`
  with a `scope` dict, a generic `ScopeIndex` for matching), but `SCOPE_DIMS =
  ("channel", "product", "segment")` is a hardcoded, insurance-specific,
  flat (non-tree) dimension list. Generalizing this to (a) an arbitrary,
  discovered dimension list and (b) ancestor-aware scope matching (not just
  equality, and not just across dimensions but *within* one) is most of the
  delta between "this repo" and "what the contractor described."
- `solver/` (Lagrangian, dual, local search, repair): already domain-agnostic —
  it operates on `base_ev` + a constraint set and doesn't know or care what a
  "product" is, and already treats the constraint set as an arbitrary
  independently-scoped list rather than a partition (Section 3's "several
  constraints active on overlapping scopes at once" already fits its
  structure). Should survive the generalization mostly unchanged.
- `io/raw_constraints.py`: today's parser is the regex/keyword version of step
  3 above — good as a fast-path/fallback, but not what makes the "generic"
  grading tier.

## 8. Confirmed requirements (from the contractor)

- **Hierarchy is never given explicitly.** No parent-map file ever arrives —
  the tree must always be inferred (naming convention where available,
  semantic/LLM inference otherwise). This makes step 2 a mandatory runtime
  component for every dataset, not a one-off design-time convenience.
- **The LLM runs at eval time, per new dataset**, not just at design time. The
  pipeline must actually invoke an LLM as part of the executable system when it
  sees a dataset it wasn't written against — not just "an LLM helped write the
  parser once." Practical implications:
  - Latency/cost of the steps 1–3 + 6 LLM calls become part of the runtime
    budget (small, but real — needs a request budget and a fallback path if a
    call fails or times out, since the rest of the pipeline can't stall on it
    indefinitely).
  - Non-determinism becomes a concern for repeatability: pin temperature/seed
    where possible, and log the resolved schema/hierarchy/constraint-set/
    generated verifier code as an artifact, so a given run's decisions are
    auditable even though the LLM call itself isn't perfectly deterministic.
- **The held-out dataset may introduce new constraint types and deeper
  hierarchies — both confirmed as 100% possible.** This directly rules out any
  fixed enum of constraint-type strings (`offers_per_*`/`cost_per_*`/...) and
  any assumption that a tree is exactly 2 or 3 levels deep (business line →
  product → segment). Step 3's scope resolution and step 2's tree builder both
  need to recurse to arbitrary depth and tolerate constraint-type vocabulary
  never seen before.
- **Extra dimensions are possible but unconfirmed.** Concretely: today every
  case varies along product × channel (× segment). A future dataset could add
  a wholly new axis — time-of-day, device, branch/agent, region, whatever —
  that no one anticipated. This is cheap to hedge against as long as step 1
  treats "which columns are dimensions" as *discovered from the dataset*
  (everything besides the subject id and value components) rather than a
  hardcoded list like `["product", "channel", "segment"]`. Since that
  discovery logic has to exist anyway for the confirmed requirements above,
  supporting an arbitrary dimension count costs effectively nothing extra —
  worth building in by default rather than treating as a separate risk to
  resolve later.
- **A single dimension can itself be a deep taxonomy, multiple constraints can
  bind at different depths/dimensions simultaneously, and those constraints
  can be jointly contradictory.** See Section 3 in full — this is the reason
  hierarchy-building can't stop at "match product to its segments" and the
  reason the system needs an explicit conflict-detection pass and a documented
  (confirmable, not fixed) precedence rule, rather than assuming the parsed
  constraint set is always jointly satisfiable.

## 9. Model choice note

A ~32B open-weight model (e.g. Qwen2.5-32B-Instruct, or a Coder variant for the
verifier-codegen step) is a reasonable fit for the LLM steps (1–3, 6). None of
them require deep open-ended reasoning — they're bounded structured extraction
(constraint string → `(scope, measure, min, max)`) and templated code
generation (aggregation-check functions), tasks a well-prompted 32B model
handles reliably. Two things matter more than model size:

- **Self-hosting is likely a hard requirement anyway**, not just a capability
  tradeoff — this pipeline touches real client-like data, so shipping it to an
  external API probably isn't an option. A self-hosted 32B on GPU also counts
  toward the grading rubric's GPU bonus, independent of whatever GPU the
  optimizer itself uses.
- **Reliability comes from the harness around the model, not the model
  alone.** Constrain LLM output to a JSON schema; validate the parsed
  constraint set against sanity checks (every raw row mapped to something, no
  scope referencing a dimension value that doesn't exist); for generated
  verifier code specifically, execute it and sanity-check it runs cleanly
  before trusting its verdict. Keep the symbolic/regex parser as the first
  pass for known patterns, falling back to the LLM only for constraint types
  it doesn't recognize — this both reduces call volume and bounds the blast
  radius of an occasional bad LLM output.
