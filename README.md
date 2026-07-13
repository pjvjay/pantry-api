# pantry-planner

A small demo project: given a recipe and a store catalog, an LLM picks the
best product for each ingredient — optimizing for cost when semantic matches
are tied. The interesting bit is the **model router**: two swappable
strategies (cascade vs. three-phase) decide *which* Claude model to call
based on the shape of the problem.

Built as an interview / portfolio artifact. Non-proprietary, MIT-licensed,
runs on your laptop with one command.

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
| `DB_URL`                     | `sqlite:///./pantry.db`          | SQLAlchemy URL                               |

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

## License

MIT.
