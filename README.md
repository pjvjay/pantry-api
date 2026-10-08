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
| `DEMO_MODE`                  | `false`                          | Deterministic stand-ins replace both LLM calls (public demo: no key, no cost) |
| `TRAVEL_COST_PER_KM`         | `0.50`                           | Split-trip optimizer: $ value of a km of driving |
| `DEFAULT_LAT` / `DEFAULT_LON`| `49.28` / `-123.12`              | Shopping location when the request sends none |
| `ORIGIN_MIN_COVERAGE`        | `0.6`                            | Spend-weighted origin coverage below which a basket is labelled UNVERIFIED |
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
│   ├── storeseed.py       # synthetic store/brand/review/terms generation
│   ├── selector.py        # main LLM call (structured output)
│   ├── tracing.py         # Burr tracking + LLM span metadata
│   ├── flow.py            # Burr state machine (6 actions)
│   ├── api.py             # FastAPI wrapper (+ /mcp mount)
│   ├── mcp_server.py      # MCP server: stdio script + HTTP app
│   ├── origins.py         # provenance: evidence → resolve → rank
│   ├── ingest.py          # loads claude-chrome-container output
│   ├── metrics.py         # /metrics: LLM spend, gates, origin coverage
│   ├── demo.py            # CLI entrypoint
│   ├── nlsearch/          # constrained NL2SQL: parse → query plan → gates
│   │   ├── plan.py        # QueryPlan/StepResult/PlanAlert formalism
│   │   ├── planner.py     # build_plan + execute_plan (t1..t4, abort gates)
│   │   ├── sql_builder.py # named templates (single-pass retrieval, stats)
│   │   ├── query_parser.py# multi-shot semantic parse (forced tool use)
│   │   ├── units.py       # unit normalization + tokenizer
│   │   └── vocab.py       # live schema-linking vocabulary
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

## NL2SQL search (`POST /plan/nl`) — query-plan execution

Paste an **entire recipe** (ingredients with quantities, plus inline notes like
"under $30, no dairy, only stores within 10km") and the pipeline plans your
shopping. The design is **constrained NL2SQL** — the LLM produces a validated
semantic parse; deterministic Python composes a **query plan**: a staged
sequence of named, parameterized SQL templates with hard abort gates. The
model never writes SQL and never composes the plan. Structured per the
[NL2SQL Handbook](https://github.com/HKUSTDial/NL2SQL_handbook) taxonomy:

- **Pre-processing** — unit normalization (`"2 cups shredded mozzarella"` →
  `mozzarella`, 500 ml, prep=shredded), *schema linking* (live
  category/subcategory vocabulary in the extractor prompt), question
  decomposition (RecipeSpec + Constraints in one call)
- **Translation** — multi-shot semantic parse → composable predicate patterns
  (price ceiling, dietary tags, category/subcategory scope, token-AND ingredient
  match with purchase-form tokens, quantity-aware size-fit ranking, distance
  within N km)
- **Post-processing** — vocabulary clamping with cross-level correction,
  *execution-guided refinement* (t1 falls back from strict to form-relaxed
  matching inside one probe), forced LIMIT, bound parameters only

### The plan's stages and gates

| Stage | Template | What it does | Abort gate |
|---|---|---|---|
| t1 | `existence_probe` | One batched probe: is every ingredient stocked at all? (strict tokens, then form-relaxed) | `missing_ingredients` → 409 listing what's missing + same-category suggestions |
| t2 | `options_single_pass` | All ingredients resolved in ONE query: per-product cheapest in-range store offer, ranked per ingredient by size-fit then price, under budget/distance constraints | `unavailable_within_constraints` (attribution re-probe names the out-of-range offer); `budget_infeasible` (cheapest-basket floor vs budget) |
| t3 | `brand_stats` | Per-brand price/rating/review aggregates over the retrieved pools — context the selector uses to break ties | — |
| t4 | `substitute_lookup` | Data-driven: same-subcategory alternatives for thin pools, labeled `substitute`, never silently swapped in | — |
| t5 | `store_price_matrix` | **Split-trip optimizer** (zero LLM): full store×product matrix for the chosen basket → exhaustive store-subset enumeration with exact home→stores→home loops → stops-vs-cost frontier (`trip_options`, best flagged recommended) | — |

Every step's SQL, row count, timing, and outcome are returned as `plan_trace`
(the UI renders it as an expandable timeline); a gate abort returns **409**
with the alert + the trace up to the failed step.

### Retrieval efficiency

Ingredient matching never scans: seeds precompute a **`product_terms`
inverted index** (tokenized name+description, same stemmer as the parser), so
t2 is indexed key joins — the recipe's tokens enter as one `VALUES` rowset,
token-AND via `HAVING COUNT(DISTINCT term) = ntokens`, one window picks each
product's best store, a second applies the per-ingredient LIMIT. 20
ingredients cost the same single round trip as 4. Deliberately deferred at
this catalog size: materialized stats views, a pool/result cache keyed on
grocery sets, `pg_trgm` fuzzy indexing.

Store model: 4 seeded stores with lat/lon (one at ~14 km to demo the distance
gate), per-store prices (±15% deterministic variance), and per-product reviews
powering the brand stats. Per-ingredient pools feed **three retrieval-aware
router signals** (`mean_pool_size`, `zero_hit_ingredients` — needed form
relaxation, `value_disagreement` — cheapest ≠ best `price/unit_qty`) so the
model router selects the model based on the size of the data and evaluated
complexity. Ingredient-form handling distinguishes *purchase forms* ("canned
tomatoes" must match a canned product) from *cook's prep* ("mashed potatoes"
buys fresh potatoes). An LLM-written-SQL variant (guarded generation behind a
keyword filter + read-only execution) is a known alternative — deliberately
not used here; the constrained parse + templated plan is the injection-safe
hot path.

## Weekly menu optimizer (`POST /plan/week`)

Plans N dinners from the recipe library under an optional budget — with no
LLM composing anything. ONE single-pass query prices every ingredient of
every library recipe (37 ingredients, one round trip); a head-noun fallback
rescues strict token-AND misses; then a **marginal-cost greedy** picks the
menu — an ingredient whose cheapest product is already in the basket adds
$0 marginal cost, so ingredient overlap is rewarded exactly, not
heuristically. The `budget_infeasible` gate fires on the cheapest-basket
floor **before any LLM spend**. The existing selector maps each dinner
(one call per day), the shopping list merges shared products once (with
`used_by` per recipe), and the merged basket runs through the split-trip
optimizer.

## Demo mode

**Try it live:**
[pantry-planner-demo.onrender.com/pantry/](https://pantry-planner-demo.onrender.com/pantry/)
(free tier — allow ~a minute to wake if idle).

`DEMO_MODE=1` swaps the two Claude call sites — recipe parse and product
selection — for deterministic stand-ins (`demomode.py`, labeled
`model_used: "demo-deterministic"`; `/health` reports `demo_mode`). The
public demo runs keyless, free, and abuse-proof while the query-plan
machinery, gates, and optimizers run unchanged.
`pantry_planner/demo_server.py` serves the built SPA (`SPA_DIST`) and the
API from one port for single-container hosting — see
[pantry-platform/demo](https://github.com/pjvjay/pantry-platform/tree/main/demo)
for the Hugging Face Space image.

## MCP server

The pipeline is also exposed over the [Model Context
Protocol](https://modelcontextprotocol.io) so any MCP client — Claude
Desktop, Claude Code, or another agent — can use it. One server
definition (`pantry_planner/mcp_server.py`), two transports (stdio and
Streamable HTTP), three primitives: **tools** the agent calls,
**resources** it reads once per session, **prompts** the operator
hands it for multi-step protocols.

### Tools

"Token" means a bearer token from `MCP_AUTH_TOKENS` (below). When no
token is configured the endpoint is anonymous: every read and plan tool
still works, and the two write tools refuse with a message naming
`MCP_AUTH_TOKENS`. Over stdio the operator's own process is trusted and
nothing needs a token.

| Tool | Cost | Needs a token? | Returns |
| --- | --- | --- | --- |
| `list_recipes`, `get_recipe` | free | only when configured | the recipe library; one `Recipe` |
| `list_products` | free | only when configured | `ProductPage {items, total, next_offset}` — `search` substring, exact `category`, `limit`/`offset` |
| `find_product` | free | only when configured | `ProductSearch` — planner-parity retrieval: `match` is `direct`, `relaxed` (same-aisle alternatives the planner would only offer) or `none`; each hit priced at its cheapest in-range store |
| `get_product` | free | only when configured | `ProductDetail` — every store offer, the resolved origin (`status` first), the evidence rows behind it, pending submissions, reviews |
| `get_product_origins` | free | only when configured | `OriginPage {items, total, by_status, next_offset}` — `by_status` counts the whole selection before paging |
| `rank_products_by_origin` | free | only when configured | `OriginRanking` — `ranked` / `excluded` / `unranked` kept separate |
| `origin_triage` | free | only when configured | products worth reading a label for (hints, never origins) |
| `plan_recipe` | 1–3 Claude calls | only when configured | `PlanResult {summary, full}` for a seeded slug |
| `plan_from_text` | 2–4 Claude calls | only when configured | `PlanResult` for pasted recipe text (NL2SQL path) |
| `plan_week` | ~1 selector call per day | only when configured | `WeekResult {summary, full}` |
| `submit_origin_evidence` | free | **always over HTTP** | `Submission` — a PENDING label reading, deduplicated |
| `list_origin_submissions` | free | only when configured | `SubmissionPage` — the review queue, oldest first |
| `review_origin_submission` | free | **always over HTTP** | `Submission` — approved (now evidence) or rejected with a note |
| `pipeline_status` | free | only when configured | strategy, models, DB, `mcp_auth` (`required`/`anonymous`), `write_tools` (`enabled`/`disabled` for this caller) |

Every tool carries `ToolAnnotations`: reads are `read_only_hint=true`,
plan tools additionally `idempotent_hint=false` (the selector may choose
differently), the two write tools are `read_only_hint=false,
destructive_hint=false` (nothing is ever deleted; `submit` is idempotent
because the queue dedupes). `open_world_hint` is false everywhere —
nothing reaches outside the seeded catalog.

**Token-lean results.** Plan tools return a `summary` — the lines
(ingredient → product, brand, size, store, price, origin), `total_cost`,
`origin_status`, `coverage {spend_fraction, count_fraction, meets_floor,
…}`, the recommended `trip` and `notes` (interpretation, substitutions,
the coverage-floor warning). `total_cost` prices every line at its
cheapest in-range store; `trip` is one realistic shopping trip priced at
the stores it visits, with its own per-item prices in `trip.items` and
its own `basket_cost` / `travel_cost` / `total_cost` — two different
baskets, so report the one you mean. Pass `verbose=true` to also get
`full`, the complete `ShoppingPlan` / `WeekPlan` with the retrieval
trace, per-line reasoning and every trip option. On the demo seed the
summary is 1–7k chars of compact JSON and `full` 4–25k; the text copy of
the result that a model reads is about 1.5× those figures. The REST API
is unchanged and always returns the full shape.

**Input bounds** mirror the REST API, so a public `/mcp` is capped the
same way: `recipe_text` ≤ 8000 chars; `exclude_origin`, `preference`,
`exclude`, `exclude_tags` ≤ 50 entries; `search` ≤ 200 chars;
`product_ids` ≤ 200; `plan_week.days` 1–14; page `limit` 1–200. An
over-long call fails validation before anything runs. Unknown country
names are an error with did-you-mean suggestions, never a silent filter
that matches nothing.

### Resources

| URI | Content |
| --- | --- |
| `pantry://recipes` | the recipe library (slug, name, servings, ingredient count) |
| `pantry://recipes/{slug}` | one recipe with its ordered ingredients; an unknown slug is a not-found error naming `list_recipes` |
| `pantry://catalog/categories` | `{category: {subcategory: product count}}` — the vocabulary `list_products(category=…)` accepts |
| `pantry://countries` | `{canonical: [...], aliases: {canonical: [forms]}, ambiguous: {input: [guidance]}}` — every spelling the server accepts for `exclude_origin`, `preference` and `country`; "korea" and "congo" are listed as ambiguous with the real choices |
| `pantry://origins/coverage` | `{products, by_status, evidence_rows, submissions: {pending, approved, rejected}, floor}` — how thin the provenance data is, and the coverage floor a basket must reach to be called verified |

All resources are `application/json` and wrap the same code the tools
use, so they cannot disagree with them.

### Prompts

| Prompt | Arguments | What it tells the agent |
| --- | --- | --- |
| `plan_dinner` | `recipe`, `exclude_origin?`, `budget?` | look up slugs/products first, call the plan tool once, report `origin_status` and `coverage.spend_fraction` verbatim, never call a basket clean below the floor, name every `notes` entry |
| `read_label` | `product` | the vision protocol: `find_product` for the id; transcribe the declaration EXACTLY; classify the claim type from the wording; "Imported by / Distributed by" ⇒ `importer_only`; the high/medium/low confidence rubric; submit and say it is pending |
| `review_submissions` | — | list pending, `get_product` each, check verbatim vs claim type vs country, approve/reject with a note a later reader can audit |

### The write path: from a label photo to a plan

```
submit_origin_evidence ──► origin_submissions (status pending)
                               │  nothing reads it: not the resolver, not a plan
review_origin_submission ──► approve: COPIED into product_origin_evidence
                               │       (source "agent-label"), evidence_id recorded
                               │  reject: kept with the reviewer's note; the same
                               │          wording resubmitted gets that verdict back
origins.resolve_origin ──────► product's origin is "resolved" from the new row
plan_recipe / plan_week ─────► exclude_origin now drops it; coverage counts it
```

A pending submission changes nothing. Approval copies rather than moves,
so the audit trail (who submitted, who approved, when, why) survives;
a transcribed label (`label-photo` or `agent-label`) outranks a
crowd-sourced record at equal claim and confidence. The same path is
available from the CLI: `python -m pantry_planner.ingest submissions
[pending|approved|rejected]` and `python -m pantry_planner.ingest review
<id> approve|reject [--note TEXT] [--by NAME]`.

### Authentication

`MCP_AUTH_TOKENS` is a comma-separated list of `label:secret` entries
(an entry without a colon is labelled `token-<n>`; whitespace is
stripped; empty entries are ignored). Each secret must be at least 16
characters or the process refuses to start — a guessable token reads as
protection and is worse than none. The label is what appears as
`submitted_by` / `reviewed_by` on origin submissions, so name a person
or a client, never the secret.

* **Tokens configured** — every `/mcp` request must carry
  `Authorization: Bearer <secret>`. A missing, wrong or malformed header
  gets `401` with `WWW-Authenticate: Bearer realm="pantry-mcp",
  error="invalid_token"` and a JSON body
  `{"error": "invalid_token", "error_description": …}`. Secrets are
  compared with `hmac.compare_digest`; they are never logged or returned.
* **Anonymous mode** (`MCP_AUTH_TOKENS` unset) — every read and plan
  tool works without a header; `submit_origin_evidence` and
  `review_origin_submission` refuse with a message naming
  `MCP_AUTH_TOKENS`, and `pipeline_status` reports `mcp_auth:
  "anonymous"`, `write_tools: "disabled"`.
* **stdio is trusted** — the server runs inside the operator's own
  process, so the caller is `local` and the write tools work without a
  token.

The deployed instance reads the token list from the
`pantry-mcp-credentials` secret (see pantry-gitops / pantry-infra).

### Client configuration

**Streamable HTTP (remote).** The FastAPI app mounts the same server at
`/mcp` (public: `https://<host>/pantry/api/mcp`) — stateless, plain
JSON responses, so it works unchanged behind nginx and the K8s ingress.
With a bearer token:

```bash
claude mcp add --transport http pantry-remote https://<host>/pantry/api/mcp \
  --header "Authorization: Bearer <secret>"
```

Locally, against `uvicorn pantry_planner.api:app`:

```bash
claude mcp add --transport http pantry-remote http://localhost:8000/mcp
```

Set `MCP_HTTP_ENABLED=false` to turn the mount off (stdio is
unaffected). In `DEMO_MODE=1` the plan tools run keyless and
deterministic, same as the REST endpoints.

**stdio (local clients).** `pip install -e .` provides the
`pantry-mcp` console script. Claude Desktop config (absolute paths —
desktop clients launch servers with no PATH and cwd `/`):

```json
{
  "mcpServers": {
    "pantry-planner": {
      "command": "/path/to/pantry-api/.venv/bin/pantry-mcp",
      "env": {
        "ANTHROPIC_API_KEY": "sk-ant-...",
        "DB_URL": "sqlite:////absolute/path/to/pantry.db"
      }
    }
  }
}
```

Claude Code:

```bash
claude mcp add pantry-planner --env ANTHROPIC_API_KEY=sk-ant-... --env DB_URL=sqlite:////absolute/path/to/pantry.db -- /path/to/pantry-api/.venv/bin/pantry-mcp
```

Burr traces from stdio runs land in `~/.pantry-planner/burr`
(override with `BURR_TRACKING_DIR`).

## Where things come from (provenance)

Products carry country-of-origin **evidence**, and the catalog can be ranked
against a country preference the caller supplies — a buy-local or boycott
filter. Nothing here infers an origin.

```bash
curl -X POST localhost:8000/origins/rank -H 'content-type: application/json' \
  -d '{"preference":["Canada"],"exclude":["United States"]}'
```

Three things this gets right that a single `country` column cannot:

**Ingredient origin and manufacturing origin are separate.** "Made in Canada"
legally permits imported ingredients — peanut butter made in Canada from
American peanuts is the canonical case. Exclusion checks **both** fields, so
that product is caught by an exclusion of the United States. A filter reading
only the manufacturing country passes exactly the items a provenance-conscious
shopper is trying to avoid.

**Claim wording is preserved and ranked.** Under CFIA rules "Product of Canada"
(≥98% Canadian content) outranks "Made in Canada" (processed here, ingredients
may be imported). The verbatim source wording is stored and never paraphrased.

**Absence is never a verdict.** Results come back in separate buckets —
`ranked`, `excluded`, `unranked` — and products with no published origin are
held out with a reason and a count. Coverage is thin and biased *against*
Canadian goods (Open Food Facts began in France), so folding unverified items
into the ranking would silently turn "nobody published this" into a finding.

Sources, cheapest first: the DB cache, then evidence ingested from the
companion `claude-chrome-container` tooling. The old name/brand heuristics
survive only as `origin_triage` — a work queue of what to photograph next,
never as provenance. They were wrong often enough to matter: that tooling's
reconciliation pass found Lindt Excellence 70% is made in **New Hampshire**,
not the Switzerland or France a brand guess produces.

**Origin is a planning constraint, not just a view.** `POST /plan/{slug}`,
`/plan/nl` and `/plan/week` all accept `exclude_origin` and `preference`;
excluded candidates are removed before the selector ever sees them, every
plan line carries a provenance receipt, and the plan reports
`origin_coverage` — count- **and** spend-weighted. Read `spend_fraction`
and `meets_floor` before calling a basket clean: below the floor
(`ORIGIN_MIN_COVERAGE`, default 0.6) the basket is labelled UNVERIFIED,
because unchecked lines are not verified-clean lines.

### Loading evidence

The container is a supervised, hard rate-limited browser agent — it was blocked
by one retailer after three searches ~18s apart — so it is a batch producer of
files, never a live dependency of this API. Run it, then ingest what it wrote:

```bash
# in the container checkout
./grocery --items "butter,rice" > prices.md      # scraped prices + branch
./origin --json < prices.md    > origin.json     # Open Food Facts + reconcile
./label  --json photos/*.jpg   > labels.json     # package photos (vision)

# here
python -m pantry_planner.ingest grocery prices.md --run-id 2026-08-20
python -m pantry_planner.ingest origin  origin.json
python -m pantry_planner.ingest label   labels.json
python -m pantry_planner.ingest refresh          # recompute summaries
```

Rows that match no catalog product are reported rather than dropped; scraped
prices keep a null product reference so the miss stays visible. Evidence is
append-only — a later lookup disagreeing with an earlier one is a fact about
the sources, and surfaces as `status: conflicting` rather than overwriting.

Label photographs are the highest-value input by a distance. For imported
seafood and fresh produce the origin is not published online at all: retailer
pages carry no country anywhere in the DOM and explicitly tell customers to
read the package, and Open Food Facts holds those records with empty origin
fields. Only the printed label answers the question.

## How it deploys

This repo is one of six in the **pantry-platform** GitOps demo:

```
merge a labelled PR to main
  → GitHub Actions (build.yml): plan vX.Y.Z from the release labels
  → pytest → ghcr.io/pjvjay/pantry-api:dev-<sha> (amd64+arm64), version baked in
  → git tag vX.Y.Z → the same digest retagged X.Y.Z, X.Y, latest → GitHub Release
  → CI sets X.Y.Z in pantry-gitops
  → ArgoCD reconciles the Deployment on AKS
```

Every PR carries one release label (`release:major`, `minor`, `patch` or
`none`), checked by `labels.yml`; `.github/versioning.json` says which paths
ship. The process, the 0.x policy, rollback (`promote_version`) and the
platform release train are in
[RELEASING.md](https://github.com/pjvjay/pantry-platform/blob/main/RELEASING.md).
The running release is on `/health` (`version`, `revision`, `build`), the MCP
`serverInfo` and every response's `X-Pantry-Version` header; a build that is
not a release says `unknown`. `pyproject.toml`'s `0.0.0` is a placeholder.

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
