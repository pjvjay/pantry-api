# pantry-api

*(formerly `pantry-planner` — renamed as part of the
[pantry-platform](https://github.com/pjvjay/pantry-platform) polyrepo split)*

A small demo project: given a recipe and a store catalog, an LLM picks the
best product for each ingredient — optimizing for cost when semantic matches
are tied. The interesting bit is the **model router**: two swappable
strategies (cascade vs. three-phase) decide *which* Claude model to call
based on the shape of the problem.

Built as an interview / portfolio artifact. Non-proprietary, MIT-licensed,
runs on your laptop with one command — and deploys to Kubernetes the GitOps
way (see [How it deploys](#how-it-deploys)).

## What's inside

- **Burr state machine** — the pipeline is a 6-action graph. Every state
  transition is traced automatically; open the Burr UI to inspect any run.
- **Structured LLM output** — Anthropic tool use returns typed results
  with per-item confidence scores. No JSON parsing gymnastics.
- **Two routing strategies, toggleable**:
  - `cascade` (default): start with Haiku, escalate the individual
    low-confidence selections to Sonnet.
  - `three_phase`: deterministic pre-scoring (Phase A) + a
    meta-cognitive Haiku classifier (Phase B) + weighted threshold
    (Phase C) picks the model before the main call.
- **Eval harness** — golden set with precision-at-k; comparison report
  across models and routing strategies committed to the repo.

## Quickstart

```bash
# 1. Install (use python3 on macOS; venv aliases `python` inside)
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# 2. Set your API key
export ANTHROPIC_API_KEY=sk-ant-...

# 3. Seed the SQLite DB from the JSON fixtures
python -m pantry_planner.db seed

# 4. Run the demo pipeline on a sample recipe
python -m pantry_planner.demo pbj_sandwich

# 5. Serve the API
uvicorn pantry_planner.api:app --reload
# → http://localhost:8000/docs

# 6. Inspect any run in the Burr UI
burr
# → http://localhost:7241
```

> **macOS note:** step 1 uses `python3` because Apple doesn't ship a `python` alias.
> Once the venv is activated (you'll see `(.venv)` in your prompt), `python`, `pip`,
> `uvicorn`, `burr` all work — they're the venv-provided binaries.

Or with Docker:

```bash
docker compose up
```

## Configuration

Everything's env-var driven. Defaults in `pantry_planner/config.py`.

| Variable                     | Default                          | What it does                                 |
|------------------------------|----------------------------------|----------------------------------------------|
| `ANTHROPIC_API_KEY`          | *(required)*                     | Auth for Claude calls                        |
| `ROUTING_STRATEGY`           | `cascade`                        | `cascade` or `three_phase`                   |
| `SELECTOR_MODEL_DEFAULT`     | `claude-haiku-4-5-20251001`      | Main selector model                          |
| `SELECTOR_MODEL_ESCALATION`  | `claude-sonnet-4-6`              | Model to escalate to (both strategies)       |
| `CLASSIFIER_MODEL`           | `claude-haiku-4-5-20251001`      | Phase B classifier (three_phase only)        |
| `CONFIDENCE_THRESHOLD`       | `0.80`                           | Below this → escalate (cascade only)         |
| `DB_URL`                     | `sqlite:///./pantry.db`          | SQLAlchemy URL (wins if set)                 |
| `DB_HOST` (+ `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) | *(unset)* | Composed into a Postgres URL when `DB_URL` is unset — the Kubernetes path, parts injected from the CNPG credential secret |

## Try both routers side-by-side

```bash
ROUTING_STRATEGY=cascade      make eval
ROUTING_STRATEGY=three_phase  make eval

# Report:
cat evals/reports/router_comparison.md
```

## Design deep-dive

See [`ARCHITECTURE.md`](ARCHITECTURE.md) for:

- Why Burr for LLM apps
- The router protocol
- The three-phase router's math (Jaccard signal, weighted-sum thresholding)
- The cascade router's confidence-triggered escalation
- The meta-cognitive classifier prompt (and why "you are an LLM expert"
  helps calibration)

## Repo layout

```
pantry-planner/
├── pantry_planner/
│   ├── config.py          # env vars, model choices, router factory
│   ├── models.py          # pydantic models
│   ├── prompts.py         # system prompts + tool schemas
│   ├── db.py              # SQLite + seed loader
│   ├── selector.py        # main LLM call (structured output)
│   ├── tracing.py         # Burr tracking + LLM span metadata
│   ├── flow.py            # Burr state machine (6 actions)
│   ├── api.py             # FastAPI wrapper
│   ├── demo.py            # CLI entrypoint
│   └── router/
│       ├── base.py        # Router protocol + dataclasses
│       ├── deterministic.py   # Phase A: Jaccard + category density
│       ├── classifier.py      # Phase B: meta-cognitive Haiku call
│       ├── decision.py        # Phase C: weighted-sum thresholding
│       ├── three_phase.py     # ThreePhaseRouter
│       └── cascade.py         # CascadeRouter
├── seeds/                 # recipes + products JSON
├── tests/                 # pytest, LLM mocked
├── evals/                 # golden set + comparison harness
└── ARCHITECTURE.md
```

## NL2SQL search (`POST /plan/nl`)

Paste an **entire recipe** (ingredients with quantities, plus inline notes like
"under $30, no dairy") and the pipeline plans your shopping. The design is
**constrained NL2SQL** — the LLM produces a validated semantic parse; Python
compiles it against a fixed pattern menu; the model never writes SQL. Structured
per the [NL2SQL Handbook](https://github.com/HKUSTDial/NL2SQL_handbook) taxonomy:

- **Pre-processing** — unit normalization (`"2 cups shredded mozzarella"` →
  `mozzarella`, 500 ml, prep=shredded), *schema linking* (live
  category/subcategory vocabulary in the extractor prompt), question
  decomposition (RecipeSpec + Constraints in one call)
- **Translation** — multi-shot semantic parse → composable predicate patterns
  (price ceiling, dietary tags, category/subcategory scope, token-AND ingredient
  match with purchase-form tokens, quantity-aware size-fit ranking)
- **Post-processing** — vocabulary clamping with cross-level correction,
  *execution-guided refinement* (zero-hit pools widen form→subcategory→category
  in one batched retry), forced LIMIT, bound parameters only

Retrieval emits one `UNION ALL` query labeled by ingredient (one round trip for
the whole recipe); per-ingredient candidate pools feed **three retrieval-aware
router signals** (`mean_pool_size`, `zero_hit_ingredients`,
`value_disagreement` — cheapest ≠ best `price/unit_qty`) so the model router
selects the model based on the size of the data and evaluated complexity. The
response carries the interpretation chips and the executed SQL for transparency.
Ingredient-form handling distinguishes *purchase forms* ("canned tomatoes" must
match a canned product) from *cook's prep* ("mashed potatoes" buys fresh
potatoes). Related: LifeGuide housing's `market_context.py` shows the
LLM-written-SQL variant with a validation harness — deliberately not used here.

## How it deploys

This repo is one of six in the **pantry-platform** GitOps demo:

```
git push here
  → GitHub Actions: pytest → ghcr.io/pjvjay/pantry-api:dev-<sha> (amd64+arm64)
  → CI bumps the image tag in pantry-gitops
  → ArgoCD reconciles the Deployment on AKS
```

| Repo | Role |
|---|---|
| [pantry-api](https://github.com/pjvjay/pantry-api) | this repo — FastAPI + LLM pipeline |
| [pantry-frontend](https://github.com/pjvjay/pantry-frontend) | React SPA |
| [pantry-db](https://github.com/pjvjay/pantry-db) | schema migrations + seeds (PreSync Job) |
| [pantry-gitops](https://github.com/pjvjay/pantry-gitops) | ArgoCD app-of-apps + Kustomize manifests |
| [pantry-infra](https://github.com/pjvjay/pantry-infra) | Terraform bootstrap (ArgoCD project + root app) |
| [pantry-platform](https://github.com/pjvjay/pantry-platform) | umbrella — architecture docs + local compose |

In Kubernetes the API never runs DDL — schema belongs to pantry-db's
migration Job; this app just reads `DB_HOST`/`DB_USER`/`DB_PASSWORD` from
the CNPG-generated secret.

## License

MIT.
