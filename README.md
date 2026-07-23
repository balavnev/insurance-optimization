# Insurance marketing-campaign optimization

Given a large set of clients (or, more generally, *subjects*) and a set of
candidate offers for each of them, pick at most one offer per subject to
maximize total expected value, subject to a pile of business constraints
(budgets, contact-count caps, per-segment limits, ...) whose exact shape
isn't known in advance. `offer_opt/` parses the constraints generically,
solves the resulting large-scale assignment problem on GPU, and verifies its
own answer — read `system_design_overview.md` for the full reasoning behind
*why* it's built this way.

## Requirements

- Python 3.13 — a working `.venv/` is already checked into the repo with
  everything installed (torch, pandas, numpy, jsonschema, requests, pytest).
- Nothing else is required to run the 3 example cases or the test suite.
- Optional: a reachable OpenAI-compatible LLM endpoint (e.g. a vLLM server)
  for the parts of the pipeline that generalize to genuinely new/unfamiliar
  datasets — see "Using a real LLM" below. Everything else works without one.

All commands below assume you're running from the repo root.

## Quickstart

```bash
.venv/bin/python3 -m pytest -q      # run the whole test suite (~10 minutes)
```

```python
from offer_opt.pipeline import run_case
from offer_opt.device import get_device

result = run_case("hard", get_device(prefer_gpu=True))
print(result.verification)          # PASS/FAIL, total EV, any violations
```

## Running the test suite

- `pytest` — everything (fast checks + the full-iteration-budget solution-quality
  checks together; no marker filtering is set up to separate them by default).
- `pytest -m slow` — only the tight, full-iteration-budget solution-quality checks.
- `pytest -m llm_integration` — only the tests that need a real LLM endpoint;
  these self-skip automatically unless `LLM_BASE_URL` is set in the environment.

## Solving one of the 3 example cases (low / med / hard)

```python
from offer_opt.pipeline import run_case
from offer_opt.device import get_device

device = get_device(prefer_gpu=True)              # CUDA > MPS > CPU, whichever exists
result = run_case("med", device, max_iters=400, repair_every=20)

print(result.verification)   # PASS/FAIL, total_ev, any violations
print(result.reference_ev)   # the vendor's own reference solution's EV, for comparison
```

`result` is a `CaseResult`: `offer_table`, `constraint_set`, `solve_result`,
`verification`, `reference_ev`.

## Solving an arbitrary new dataset

This is the actual generalization entrypoint — it has no notion of "low/med/hard"
at all, only raw offers/constraints file paths:

```python
from offer_opt.pipeline import run_dataset
from offer_opt.device import get_device

result = run_dataset(
    "path/to/offers.csv",
    "path/to/constraints.csv",
    get_device(prefer_gpu=True),
    llm_client=None,       # NullClient by default -- see "Using a real LLM" below
    max_iters=300,
)

print(result.dims)           # discovered dimension names, e.g. ("product", "channel", "segment")
print(result.trees)          # inferred hierarchy (DimensionTree) per dimension
print(result.conflicts)      # detected ancestor/descendant constraint contradictions, if any
print(result.verification)
print(result.codegen_agrees) # whether the generated verifier code agreed with verify.py
```

`result` is a `DatasetResult`: `offer_table`, `constraint_set`, `dims`, `trees`,
`conflicts`, `solve_result`, `verification`, `generated_checks`, `codegen_agrees`.

Without an `llm_client`, anything the symbolic parser/heuristics can't
confidently resolve on their own — a genuinely novel constraint-type string, an
ambiguous column, a dimension hierarchy with no naming-convention signal at
all — raises loudly (`UnresolvedConstraintError` or similar) instead of
silently guessing. That's what a real LLM client resolves.

## Using a real LLM

Point at any OpenAI-compatible chat-completions endpoint (e.g. a vLLM server
serving Qwen or similar):

```bash
export LLM_BASE_URL="http://your-server:8000"
export LLM_API_KEY="..."          # only if the endpoint requires one
```

```python
from offer_opt.llm.client import VLLMOpenAIClient

client = VLLMOpenAIClient()        # reads LLM_BASE_URL / LLM_API_KEY from the environment
assert client.health_check()       # quick connectivity check (GET /v1/models)

result = run_dataset(offers_path, constraints_path, device, llm_client=client)
```

Sanity-check the endpoint on its own first: `pytest -m llm_integration -v`.

For development/testing with no live endpoint, use
`offer_opt.llm.client.NullClient` (forces the fully-symbolic path — the
default) or `FakeLLMClient` (a scripted test double that returns canned
responses; see any test file under `tests/` for examples, e.g.
`tests/test_generalization.py`).

There's also a small evaluation harness (`offer_opt.llm.evaluate`) for
measuring a real client's accuracy/latency on the constraint-classification
task against a fixed case set (`tests/fixtures/constraint_classification_cases.json`),
in the same style as `vendor-examples/examples/prompt_lab.py`.

## Benchmarking

```python
from offer_opt import metrics
from offer_opt.device import get_device

report = metrics.benchmark("hard", get_device(prefer_gpu=True))
print(report)
```

`baselines/phase0_baseline.md` records this repo's own solver's reference
numbers (CPU and MPS, all 3 cases) captured *before* the generalization work
began — the comparison point for "did this regress?".

## Project layout

```
offer_opt/                    the actual package
  schema.py, scope.py           canonical data model + ancestor-aware scope matching
  constraints.py                 raw constraint row -> ConstraintSpec / ParameterSpec
  features.py                     canonical offer-table construction (per-case + generic)
  discovery/                       schema resolution, dimension-hierarchy inference, conflict detection
  llm/                              swappable LLM client (Null/Fake/real vLLM), prompts, cache, budget
  codegen/                          generates + sandboxes per-constraint verifier code
  solver/                           the optimizer (Lagrangian relaxation + local search + repair)
  verify.py                        the ground-truth constraint checker
  pipeline.py                      run_case() / run_dataset() -- the entrypoints above
  metrics.py                       benchmark harness + reference-solution reconstruction
tests/                        ~150 tests; fixtures/ holds the synthetic generalization datasets
case_1_low/, case_2_med/, case_3_hard/   the 3 vendor-provided example cases
vendor-examples/              the AI summer school's own onboarding material (LLM serving conventions)
baselines/                    pre-generalization performance snapshot
notebooks/solution.ipynb      the original exploratory notebook
system_design_overview.md     the target architecture this package implements, in plain language
Ingosstrakh_task_v20260428.pdf   the original task specification
```

## Where to read more

`system_design_overview.md` explains, in plain language, what problem this
actually solves and why it's built the way it is — read that first if the
code under `offer_opt/` isn't self-explanatory enough on its own.
