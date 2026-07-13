# Architecture

A tour of the design decisions, aimed at a reader familiar with Python and
LLMs but new to Burr / structured tool use / model routing.

## The problem

Given:

- a **recipe** with N ingredients (each free-text: "Peanut Butter and Jelly Jam"),
- a **catalog** of M products (name, description, price),

produce a shopping plan that assigns exactly one product to each ingredient,
preferring lower cost when semantic matches are equally good, and choosing
the closest reasonable substitute when no exact match exists.

## The pipeline (Burr state machine)

```
  load_recipe ─▶ load_products ─▶ preselect_model ─▶ select_products ─▶ check_escalation ─▶ escalate_if_needed ─▶ build_plan
```

Six actions. Each is a Python function decorated with `@action(reads=[...], writes=[...])`.
The framework:

- **Enforces the read/write contract** — an action that writes a field it didn't declare is a bug caught at import time.
- **Persists every transition** — the LocalTrackingClient writes to a SQLite DB, viewable in the Burr UI.
- **Lets you replay from any state** — crucial when LLM calls are non-deterministic.

Why a state machine? Because LLM apps look like functions but behave like
distributed systems: multiple side-effecting steps, need for observability,
need for retry from checkpoint. State machines make all that free.

## Structured output via tool use

The main selector prompt doesn't ask Claude to "return JSON in this format" —
that's how you get parse errors at 2am. Instead we register a tool called
`submit_plan` whose input schema *is* the output shape. Claude has to call
the tool, which means the SDK validates the arguments against the schema
before we ever see them.

```python
{
  "name": "submit_plan",
  "input_schema": {
    "selections": [{
      "line_no": int,
      "product_id": int,
      "confidence": float,        # 0.0 .. 1.0
      "reasoning": str,
    }],
    "total_cost": float,
  }
}
```

Per-item `confidence` is what the cascade router keys off. It also lets
downstream code sanity-check that the LLM's own uncertainty is calibrated.

## The router

Both routers implement one protocol:

```python
class Router(Protocol):
    def preselect_model(self, recipe, products) -> PreselectResult:
        """Called BEFORE the main selector. Picks the model."""

    def should_escalate(self, initial_result) -> EscalationDecision:
        """Called AFTER the main selector. Decides if any ingredients
        need a re-run with a stronger model."""
```

The Burr flow calls both hooks. Which router is active is a config knob.

### Cascade (default)

**Simple, cheap, good for most cases.**

1. `preselect_model` returns `haiku`. No pre-scoring.
2. Selector runs with Haiku.
3. `should_escalate` scans `selections[*].confidence`. Any below the
   threshold (0.80) → escalate.
4. `escalate_if_needed` re-runs *only those ingredients* through Sonnet,
   with the same catalog. Merges into the plan.

You always pay for one Haiku call; you only pay for Sonnet on the hard
subset. Latency = 1 Haiku call + (maybe) 1 Sonnet call.

### Three-phase (opt-in)

**Predicts complexity before the main call. Nicer trace, more interesting eval.**

**Phase A — deterministic signals.** Zero LLM cost, milliseconds:

- `ingredient_count` — more decisions = more places to fail
- `mean_max_similarity` — for each ingredient, max Jaccard on character trigrams against any product's name+description. Averaged. Low mean = surface features can't tell us, need semantic reasoning.
- `count_below_0.3` — how many ingredients have no obvious surface match?

Jaccard on character trigrams is a fast proxy for "how confident can a
non-semantic system be that this ingredient has an obvious product match?"
It's basically a computationally-cheap ROUGE-N. It won't tell you *which*
match is best (that's the LLM's job), but it will tell you when the
problem is easy.

**Phase B — meta-cognitive classifier.** One cheap Haiku call:

```
"You are an LLM systems expert triaging a grocery-matching task before it runs.
 You will NOT perform the matching yourself — you only assess difficulty.
 From the perspective of someone with deep knowledge of what LLMs find easy
 vs. hard, give a REASONABLE estimate of how a general matching LLM would
 perform..."
```

Two things about this prompt:

1. **The classifier only sees the catalog SUMMARY** (categories + counts),
   not individual products. This prevents it from just doing the matching
   itself and returning a self-fulfilling assessment.
2. **The "LLM systems expert" frame** creates psychological distance
   between the model and its self-model. Empirically, asking a model "how
   confident are YOU?" invites overconfidence bias; asking "how well would
   a *general* matching LLM do?" pulls from different training data
   (docs, papers, forums about LLM behavior) and calibrates better.

Returns via tool use:

```json
{
  "match_confidence_1_to_10": 4,
  "cost_complexity_1_to_10":  3,
  "ambiguous_ingredients": [{"name": "White Chocolate", "why": "no direct match"}],
  "confidence_in_own_estimate_1_to_10": 8,
  "reasoning": "..."
}
```

The `confidence_in_own_estimate` is a second-order calibration signal —
we can plot classifier-confidence vs. actual selector-accuracy on the
eval set to see if the classifier itself is well-calibrated.

**Phase C — weighted-sum threshold.** Pure Python:

```python
complexity = (
    0.10 * ingredient_count_norm  +
    0.25 * similarity_gap         +
    0.25 * ambiguous_fraction     +
    0.30 * (1 - match_confidence) +
    0.10 * cost_complexity
)

if complexity < 0.30: return HAIKU
if complexity < 0.65: return SONNET
return SONNET_WITH_THINKING
```

Weights are hyperparameters. Tune them by sweeping against the eval set.

## Model choice

Default is **Haiku 4.5**. Empirically it hits >95% precision on this class
of matching task; going bigger buys a few points of accuracy at 10× cost.
Both routers escalate to **Sonnet 4.6** (thinking off by default).

The router files show two more slots for `SONNET_WITH_THINKING` in case
you want a third tier — the three-phase Phase C already picks it for the
highest-complexity band. Cascade keeps it simple: Haiku → Sonnet only.

## Tracing

Every Burr action writes structured metadata to the tracking DB. LLM
actions additionally record:

- model name
- input_tokens, output_tokens
- estimated cost_usd (computed from the current model's rate card)
- latency_ms
- retry_count

Open `burr` (the CLI) to browse it. This is the single most useful
observability tool for LLM apps — the same "click each action to see
inputs, outputs, timing" experience you'd get from a debugger.

## Evals

`evals/datasets/pairs.jsonl` — 30 hand-labeled (recipe, expected_plan) cases.
Scoring is precision-at-N per ingredient (any correct-keyword hit AND no
wrong-keyword hit → correct). We commit reports to `evals/reports/` so
the numbers are visible without running anything.

The router comparison is the most interesting eval artifact:

```
                 | 3-phase model | 3-phase cost | 3-phase acc | Cascade path      | Cascade cost | Cascade acc
easy PB&J        | haiku         | $0.0005      | 1.00        | haiku             | $0.0002      | 1.00
mixed            | sonnet        | $0.0080      | 0.95        | haiku → 1×sonnet | $0.0035      | 0.95
hard             | sonnet+think  | $0.0250      | 0.98        | haiku → 3×sonnet | $0.0180      | 0.95
```

Cheap when they agree; different profiles on hard cases. Which to choose
depends on whether you care more about latency (cascade wins on easy
tasks) or predictability (three_phase pre-classifies so you know cost
before the main call).
