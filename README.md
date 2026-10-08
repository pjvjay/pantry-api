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
| `NL2SQL_MODEL`               | `claude-sonnet-4-6`              | Recipe-text parser (`/plan/nl`)              |
| `GEMINI_API_KEY`             | *(unset)*                        | Auth for any `gemini:<model>` setting; only an actual Gemini call without it is an error |
| `GEMINI_REASONING_EFFORT`    | `low`                            | Sent as `reasoning_effort` to Gemini; empty = omit (model default) |
| `GEMINI_BASE_URL`            | Google's OpenAI-compatible URL   | Override the Gemini endpoint (`…/v1beta/openai`) |
| `RUNTIME_SETTINGS_ENABLED`   | `false`                          | Allow `POST /settings/runtime` to switch demo mode and models live |
| `CONFIDENCE_THRESHOLD`       | `0.80`                           | Below this → escalate (cascade only)         |
| `DB_URL`                     | `sqlite:///./pantry.db`          | SQLAlchemy URL (wins if set)                 |
| `DEMO_MODE`                  | `false`                          | Deterministic stand-ins replace both LLM calls (public demo: no key, no cost) |
| `TRAVEL_COST_PER_KM`         | `0.50`                           | Split-trip optimizer: $ value of a km of driving |
| `DEFAULT_LAT` / `DEFAULT_LON`| `49.28` / `-123.12`              | Shopping location when the request sends none |
| `ORIGIN_MIN_COVERAGE`        | `0.6`                            | Spend-weighted origin coverage below which a basket is labelled UNVERIFIED |
| `DB_HOST` (+ `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD`) | *(unset)* | Composed into a Postgres URL when `DB_URL` is unset — the Kubernetes path, parts injected from the CNPG credential secret |
| `TRUSTED_PROXY_HOPS`         | `0`                              | Proxies in front of the API that append to `X-Forwarded-For`; 0 = the header is ignored and the rate limit keys on the TCP peer |
| `LLM_DAILY_COST_CAP_USD`     | *(unset: no ceiling)*            | Estimated LLM spend per replica per UTC day above which LLM-calling endpoints answer 503 |
| `OFFERS_SYNTHETIC`           | `true`                           | Store prices, stock and reviews are the seeded demo data; the alternatives ranking labels them so (`data_note`, "(demo)" ratings). Set false only for real offers |

The four model settings are **specs**: `gemini:<model>` (e.g.
`gemini:gemini-flash-latest`) routes that call to Google Gemini through its
OpenAI-compatible endpoint; a bare name or `anthropic:<model>` routes to
Anthropic. Mix freely — e.g. a Gemini default selector escalating to
Sonnet. Gemini calls are costed at $0 (free tier); `model_used` and the
metrics labels carry the spec. All three call sites go through one
function, `pantry_planner/llm.py::forced_tool_call`.

`GET /settings/runtime` shows demo mode, the effective model specs and
which keys are configured. With `RUNTIME_SETTINGS_ENABLED=1`,
`POST /settings/runtime` with `{"demo_mode": true}` and/or
`{"models": {"selector_default": "gemini:…", "selector_escalation": …,
"classifier": …, "nl2sql": …}}` applies to the next request — no restart;
`/health` and the MCP `pipeline_status` tool reflect it. Leave it off on a
public deployment: it lets any caller turn real LLM spend on.

### Deployments and costs

| Deployment | LLM | Why |
| --- | --- | --- |
| AKS (`pantry-gitops` `apps/pantry-api`) | **live** | `ANTHROPIC_API_KEY` is mounted and `DEMO_MODE` is not set |
| Hugging Face Space / Render demo | none | `demo/Dockerfile` sets `DEMO_MODE=1`: deterministic stand-ins, $0 |
| Local stack | as configured | whatever `.env` says |

Every public endpoint is bounded (`pantry_planner/limits.py`), per replica and in process:

- **A token bucket per client IP and endpoint**: `/plan/nl` and `/plan/spec` 10 a minute,
  `/recipes/parse-lines`, `/plan/alternatives` and `/plan/reprice` 60 a minute (more endpoints
  join as they land). Over the limit is a
  429 `{"error": "rate_limited", "detail": "Too many requests to ...; retry in N s."}` with
  `Retry-After`. The client is the TCP peer unless `TRUSTED_PROXY_HOPS` says how many
  proxies append to `X-Forwarded-For`; a client-written header is never trusted.
- **A daily LLM cost ceiling**: every LLM call adds its estimated cost (`forced_tool_call`),
  reset at UTC midnight. The estimate prices the models in `config.COST_PER_MTOK` only: a
  Gemini call counts $0 (the free tier), and so does any Anthropic model not listed there,
  whose first call logs a warning while a ceiling is set. Add a model's rates before
  pointing a capped deployment at it. Above `LLM_DAILY_COST_CAP_USD`, `/plan/nl`, `/plan/spec`,
  `/plan/{slug}`, `/plan/week` and the MCP plan tools answer 503 / a tool error, "Live planning
  is paused for today; the demo planner still works" (REST: `{"error":
  "llm_budget_exhausted", "detail": "<that sentence>"}`). Endpoints that call no LLM, and demo
  mode, keep working. `/health` reports `llm_budget {spent, cap}`.
- Both refusals have the body an LLM failure already has: `error` is a code and `detail` the
  sentence to show, so the console, which shows a string `detail` as it is, needs no change.

The MCP endpoint's bearer tokens are unchanged; the buckets apply to REST only.

**Behind a proxy, set `TRUSTED_PROXY_HOPS` before or with this version.** Without it the TCP
peer is the proxy, so every visitor shares one bucket per endpoint: ten plans a minute for
everyone together, not each. uvicorn rewrites the peer from `X-Forwarded-For` only for a proxy on
loopback (its `FORWARDED_ALLOW_IPS` default), and neither deployment runs one there. The first
forwarded request with no trusted proxies logs a warning from `pantry_planner.limits`.

| Deployment | Proxy in front | Where `TRUSTED_PROXY_HOPS` is set |
| --- | --- | --- |
| AKS | ingress-nginx | `pantry-gitops` `apps/pantry-api/deployment.yaml` env, set to the ingress depth |
| Hugging Face Space / Render demo | the platform's front end | `demo/Dockerfile` or the Space's settings, once the platform's hop count is known |
| Local stack | none, or the frontend's nginx | not needed: every request is the same client |

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
│   ├── limits.py          # per-client token bucket + daily LLM cost ceiling
│   ├── packs.py           # pack_count: packs a purchase takes (shared rule)
│   ├── recipe_doc.py      # RecipeDoc -> RecipeSpec (no parse) / display text
│   ├── alternatives.py    # rank a plan line's alternatives; pin checks for reprice
│   ├── demo.py            # CLI entrypoint
│   ├── nlsearch/          # constrained NL2SQL: parse → query plan → gates
│   │   ├── plan.py        # QueryPlan/StepResult/PlanAlert formalism
│   │   ├── planner.py     # build_plan + execute_plan (t1..t4, abort gates)
│   │   ├── sql_builder.py # named templates (single-pass retrieval, stats)
│   │   ├── query_parser.py# multi-shot semantic parse (forced tool use)
│   │   ├── lineparse.py   # deterministic ingredient-line reading (no LLM)
│   │   ├── units.py       # unit normalization + tokenizer
│   │   └── vocab.py       # live schema-linking vocabulary
│   └── router/
│       ├── base.py        # Router protocol + dataclasses
│       ├── deterministic.py   # Phase A: Jaccard + category density
│       ├── classifier.py      # Phase B: meta-cognitive Haiku call
│       ├── decision.py        # Phase C: weighted-sum thresholding
│       ├── three_phase.py     # ThreePhaseRouter
│       └── cascade.py         # CascadeRouter
├── seeds/                 # recipes + products JSON (copies of pantry-db's)
├── skills/recipe-shopper/ # Agent Skill: recipe link → cheapest basket nearby
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
| t1 | `existence_probe` | One batched probe: is every ingredient stocked at all? Four match levels, each tried only when the one before finds nothing: exact (every token, purchase form included), equivalent form (powder/ground swapped — "cumin powder" → Cumin Ground; reported as `form`), form (form dropped), generic (descriptor words such as light/dark/toasted/ground/boneless dropped too — "light brown sugar" → Brown Sugar). Water and ice never reach it: they are skipped, never priced | `missing_ingredients` → 409 listing what's missing + related products (shared words first, then the category hint's cheapest) |
| t2 | `options_single_pass` | All ingredients resolved in ONE query: per-product cheapest in-range store offer, ranked per ingredient by size-fit then price, under budget/distance constraints. Packs over 6× the need are dropped as catering packs while a pack within 6× exists; when none does (a tablespoon of oil against 250 ml-1.4 L bottles), packs within 6× the smallest stay and price decides | `unavailable_within_constraints` (attribution re-probe names the nearest out-of-range offer and the limit it breaks); `budget_infeasible` (cheapest-basket floor vs budget) |
| t3 | `brand_stats` | Per-brand price/rating/review aggregates over the retrieved pools — context the selector uses to break ties | — |
| t4 | `substitute_lookup` | Data-driven: same-subcategory alternatives for thin pools, labeled `substitute`, never silently swapped in | — |
| t5 | `store_price_matrix` | **Split-trip optimizer** (zero LLM): full store×product matrix for the chosen basket → exhaustive store-subset enumeration with exact home→stores→home loops → stops-vs-cost frontier (`trip_options`, best flagged recommended) | — |

Every step's SQL, row count, timing, and outcome are returned as `plan_trace`
(the UI renders it as an expandable timeline); a gate abort returns **409**
with the alert + the trace up to the failed step.

**Partial plans.** A recipe from the wild names far more than any one
catalog stocks. With `"allow_partial": true` the two per-ingredient gates
drop instead of abort: t1 misses go to `not_stocked`, t2 misses (stocked,
but no offer within the distance/price/diet constraints) to `out_of_range`,
each as `{ingredient, reason, suggestions}`, and the plan prices what
remains. `total_cost`, `origin_coverage` and `trip_options` then cover the
planned lines only; `ingredient_count` is what the recipe asked for. If
nothing remains the gate aborts as before; `budget_infeasible` is a
basket-level verdict and always aborts. `"max_km"` (0.5–100) sets the
distance limit and overrides any distance written in the text. Each line
carries `match`: `exact`, `form` or `generic`. Both fields default to the
original behaviour. With or without `allow_partial`, `skipped` lists what
was never planned: water and ice (never bought), ingredients past the
40-ingredient cap, and lines the selector returned no product for. Every
ingredient is in exactly one place: a line, `not_stocked`, `out_of_range`
or `skipped`. Lines that choose the same product are one purchase
(`also_lines`), priced once — or `packs` times when their summed quantity,
known in the pack's unit, needs more than one pack.

### Retrieval efficiency

Ingredient matching never scans: seeds precompute a **`product_terms`
inverted index** (tokenized name+description, same stemmer as the parser;
words the description negates — "no salt added" — are left out, so "salt"
never matches a can of tomatoes), so
t2 is indexed key joins — the recipe's tokens enter as one `VALUES` rowset,
token-AND via `HAVING COUNT(DISTINCT term) = ntokens`, one window picks each
product's best store, a second applies the per-ingredient LIMIT. 20
ingredients cost the same single round trip as 4. Deliberately deferred at
this catalog size: materialized stats views, a pool/result cache keyed on
grocery sets, `pg_trgm` fuzzy indexing.

Catalog: 169 products (`seeds/products.json`) — staples plus the Sichuan,
Indian, Mexican and baking pantry that real recipe links ask for. Products 166-169 (frozen
mango, sliced pepperoni, pizza dough, instant yeast) are **synthetic demo products**, invented
for the meal plan's demo starter recipes; `seeds/demo_products.json` holds the same four rows
with that label, because a product row has no field for it. The file is
a byte-identical copy of [pantry-db](https://github.com/pjvjay/pantry-db)'s
`seeds/products.json`, which also generates the Postgres `seed.sql`; change it
there first, copy it here (`cmp` the two files), and the derived rows (terms,
store prices, brands, reviews) come out the same on both sides because
`storeseed.py` and pantry-db's `gen-seed-sql.py` share one algorithm.

Recipes: 7 library recipes, 37 lines (`seeds/recipes.json`, also a byte-identical copy of
pantry-db's). Every line has a `quantity`, `unit` and `note`, and these are **demo house
amounts**: synthetic gram and millilitre amounts written for this demo ("700 g Chicken
Thighs" for a curry that serves 4), not taken from any cookbook or site. They are labelled
that way wherever they are shown (`amount_basis: "demo_house_amounts"` on every line of
`GET /recipes/{slug}/doc`). A line whose amount is not stated has a null `quantity` and a
`note` saying why. They live in their own table, `recipe_line_amounts` (pantry-db migration
0007), so the classic `/plan/{slug}` path is unchanged and an API running against a database
without the table still works: the doc's lines are then unquantified, with a warning.

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

## Library recipes at nearby stores (`POST /plan/{slug}?lat=&lon=&max_km=`)

The classic path's selector sees the whole catalog and has no location, so a library recipe
used to come back with catalog prices and no stores. Give it a location (any of `lat`, `lon`,
`max_km`; lat/lon default to the server's point, no `max_km` means any distance) and, after the
products are chosen exactly as before:

- each line is priced at its product's cheapest store within range — the convention
  `plan_from_text`'s lines use — and `total_cost` sums those prices;
- the split-trip optimizer (`t5_trip_optimizer`) runs over the basket, as on the NL path, so
  `trip_options` / `summary.trip` recommend which stores to visit;
- a chosen product that no store in range sells is never priced from the catalog: it moves to
  `out_of_range` with the nearest offer anywhere ("Fresh Garlic has no offer within 1 km of the
  shopping location; the nearest is <store>, <distance> km away, at $<price>"), and the total
  and trip cover the rest. When no chosen product is sold in range the plan is a 409
  `unavailable_within_constraints` naming every product's nearest offer.

The selector itself still ignores distance, so it can choose a product only sold farther away
while a nearer alternative exists; that product is then reported, not silently swapped.
`tests/test_recipe_location.py` covers the unchanged default, the prices and trip, a product
taken off the only nearby store's shelf, the no-store gate, and the REST and MCP parameters.

## Reviewed recipes (`POST /plan/spec`, MCP `plan_from_lines`)

A recipe the shopper has already reviewed line by line (a `RecipeDoc`: an imported or pasted
ingredient list, a library recipe with its amounts, a dish the assistant wrote) is planned
exactly as reviewed. `recipe_doc.to_spec` hands each line's name, quantity and unit to
`flow.run_spec`, which runs the same retrieval, gates, selector and trip optimizer as
`/plan/nl` but no parser of either kind: the trace's first step reads "skipped: reviewed
lines". What the shopper reviewed is what gets planned, and `basis.lines` on the plan shows
it, byte for byte (`tests/test_plan_spec.py`). Only the products are still chosen by the
selector.

- `POST /plan/spec` takes `{doc, lat?, lon?, max_km?, exclude_origin?, preference?,
  allow_partial?}` and returns the same `ShoppingPlan` as `/plan/nl`. A line not confirmed yet
  (a video transcription the shopper has not ticked) is a 422 `unconfirmed_lines` naming it;
  an unknown country is a 422 and a gate abort a 409, as on `/plan/nl`. A line's `quantity`
  is a finite number from 0 to 1,000,000 in its own unit (`models.MAX_LINE_QUANTITY`);
  anything else, including JSON's `1e309`, which Python reads as infinity, is a 422 naming
  the line.
- The 40-line planning cap still applies: lines past it, and water or ice, are named on
  `skipped`, never dropped silently.
- `recipe_doc.to_recipe_text` renders a doc in the pasted format for traces and transcripts;
  it is never a planning input.
- `POST /recipes/parse-lines` reads up to 60 ingredient lines (`{title?, yield_text?, lines,
  origin?}`) into the fields a shopper reviews (`line_no, text, name, quantity, unit, note,
  amount_basis`), with `servings` null unless the title or yield states it. It uses the demo
  parser's line reading (`nlsearch/lineparse.py`): no URL, no LLM, no database. Its output,
  `warnings` included, is a valid RecipeDoc as it stands: a servings count or an amount
  over the bounds comes back unstated, and the warnings come one per kind of problem,
  naming its lines.
- `GET /recipes/{slug}/doc` is a library recipe as a RecipeDoc, with its demo house amounts
  (see the catalog notes above), to show line by line. Plan a library recipe by slug
  (`POST /plan/{slug}`): that path gives the selector each line's category, while
  `/plan/spec` first checks by name alone that each line is stocked, as `/plan/nl` does.
  Six of the seven library docs plan through `/plan/spec` as well; `pbj_sandwich` is a 409
  there, because "Peanut Butter and Jelly Jam" and "White Chocolate" find nothing by name.

Every plan carries `basis` (what it was made from: the planned lines, the product chosen for
each, the constraints and location), `servings` (None when the recipe does not say) and, per
purchase, `need_qty`/`need_uom` when every line it covers states an amount in one unit. The MCP
plan tools attach `summary.basis` only with `basis=true`; without it the summary is unchanged.

## Alternatives and re-pricing (`POST /plan/alternatives`, `POST /plan/reprice`)

A finished plan's `basis` is enough to rank the other products for one of its lines and to
price it again with the shopper's choices, with no LLM call, no parse and no write. The MCP
tools `rank_alternatives` and `reprice_plan` do the same; the demo hub uses them for the chat
cart and keeps them away from its model.

- `POST /plan/alternatives` takes `{basis, line_no, limit? 1..25 = 12}` and returns an
  `AlternativeRanking` (`alternatives.py`). Candidates are the line at the exact, equivalent
  form, form and generic levels plus its head word (one options query), and same-aisle
  substitutes from stores in range when the pool is thin (the planner's t4 lookup; price ties
  go to the lower id). Order (`ORDER`): same ingredient first, then
  `alternatives.closeness` (the demo selector's `units.semantic_key`, with its head test read
  from the line's ingredient word, so "cumin powder" is about cumin, not powder; for a line
  that ends in its ingredient word it only breaks semantic_key's ties), pack fit for the
  recipe's amount, origin preference only when the plan had one, the trip total after the
  swap, the cost of the recipe's amount, rating only on exact cent ties, catalog id. Each
  row's `trip` is computed by the same code as a re-price, so choosing it costs exactly that;
  `trip.buys_at` is the store that trip buys the product at and its price a pack there, which
  the re-priced cart charges and `cost_for_need` and `unit_price` are worked out at (`offer` is
  the lowest price in range, which the trip skips when the stop costs more than it saves);
  `rank_reason` says why it is below the row above. Products the origin exclusion drops are
  in `held_back` with their evidence and never ranked; products matching the line's words
  with no offer in range are counted in `unavailable`. Unknowns stay unknown ("Origin not
  checked", no rating, pack fit `unknown`), and `data_note` labels the demo offers and reviews
  while `OFFERS_SYNTHETIC` is true.
- `POST /plan/reprice` takes `{basis, pins? ≤ 40}` (`pins`: `{line_no, product_id}`) and
  returns a `ShoppingPlan` like `/plan/nl`'s. With no pins it is the plan's own lines, trip,
  total and coverage. Each pin is checked first: a planned line, a candidate for it, not held
  back by the origin exclusion, an offer within `max_km` and under the price cap. A pin equal to
  the plan's pick is dropped, so it undoes a swap. `basis.pins` on the result are the merged
  pins.
- Errors on both: 422 `{error: line_not_planned | invalid_pin | invalid_basis, detail}`, 422
  for an unknown country, 409 `stale_basis` when a product left the catalog or its range (plan
  again).

`tests/test_reprice.py` and `tests/test_alternatives.py` cover the round trip, the trip
invariant to the cent, the order, the match levels, the honesty cases, exclusion parity, the
query count and a 300 ms guard.

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

`DEMO_MODE=1` (or `POST /settings/runtime {"demo_mode": true}`) swaps the
LLM call sites — recipe parse and product selection — for deterministic
stand-ins (`demomode.py`, labeled
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
| `find_product` | free | only when configured | `ProductSearch` — planner-parity retrieval: `match` is `direct`, `generic` (matched once descriptor words such as "light" were dropped — the planner selects these too), `relaxed` (same-aisle alternatives the planner would only offer) or `none`; each hit priced at its cheapest in-range store |
| `get_product` | free | only when configured | `ProductDetail` — every store offer, the resolved origin (`status` first), the evidence rows behind it, pending submissions, reviews |
| `get_product_origins` | free | only when configured | `OriginPage {items, total, by_status, next_offset}` — `by_status` counts the whole selection before paging |
| `rank_products_by_origin` | free | only when configured | `OriginRanking` — `ranked` / `excluded` / `unranked` kept separate |
| `origin_triage` | free | only when configured | products worth reading a label for (hints, never origins) |
| `plan_recipe` | 1–3 LLM calls | only when configured | `PlanResult {summary, full}` for a seeded slug. Any of `lat`/`lon`/`max_km` makes it store-aware: each line at its cheapest store in range, `summary.trip` the recommended split, and a chosen product no store in range sells in `summary.out_of_range` (naming the nearest offer); without them, catalog prices and no stores |
| `plan_from_text` | 2–4 Claude calls | only when configured | `PlanResult` for pasted recipe text (NL2SQL path). `lat`/`lon`/`max_km` define "nearby"; `allow_partial=true` plans what is stocked and in range and lists the rest in `summary.not_stocked` / `summary.out_of_range` |
| `plan_from_lines` | 1–3 LLM calls (no parse) | only when configured | `PlanResult` for reviewed lines, planned exactly as given (no parse). The model names `doc_key`; the hub fills the reviewed `lines`, `title` and `servings`, and any other client may pass up to 60 `lines` itself |
| `plan_week` | ~1 selector call per day | only when configured | `WeekResult {summary, full}` |
| `rank_alternatives` | free (no LLM) | only when configured | `AlternativeRanking` for one planned line of a plan's `basis` (`basis=true` on a plan tool): every other product that could fill it in the planner's order, the cart's pick flagged `current`, facts from the catalog or stated as unknown, each row's trip effect and why it sits where it does; `held_back` for what the origin exclusion drops. The demo hub calls it for the cart and hides it from its model |
| `reprice_plan` | free (no LLM) | only when configured | `PlanResult` for a plan's `basis` priced again with the shopper's `pins` (`{line_no, product_id}`, at most 40): with no pins the summary is the plan's own; each pin is checked (a candidate, not held back by origin, an offer in range) and named in `notes`. Hidden from the hub's model too |
| `submit_origin_evidence` | free | **always over HTTP** | `Submission` — a PENDING label reading, deduplicated |
| `list_origin_submissions` | free | only when configured | `SubmissionPage` — the review queue, oldest first |
| `review_origin_submission` | free | **always over HTTP** | `Submission` — approved (now evidence) or rejected with a note |
| `pipeline_status` | free | only when configured | strategy, models, DB, `mcp_auth` (`required`/`anonymous`), `write_tools` (`enabled`/`disabled` for this caller) |

Every tool carries `ToolAnnotations`: reads are `read_only_hint=true`,
plan tools additionally `idempotent_hint=false` (the selector may choose
differently) and `rank_alternatives` / `reprice_plan` `idempotent_hint=true`
(no LLM: the same basis gives the same answer), the two write tools are `read_only_hint=false,
destructive_hint=false` (nothing is ever deleted; `submit` is idempotent
because the queue dedupes). `open_world_hint` is false everywhere —
nothing reaches outside the seeded catalog.

**Token-lean results.** Plan tools return a `summary` — the lines
(ingredient → product, brand, size, store, price, origin, and `match`:
`exact`, `form` or `generic`), `total_cost`, `origin_status`, `coverage
{spend_fraction, count_fraction, meets_floor, …}`, the recommended
`trip`, `not_stocked` / `out_of_range` (what a partial plan left out —
always present, empty unless `allow_partial` dropped something),
`skipped` (water, ice, lines past the cap or without a selection) and
`notes` (interpretation, "planned K of N ingredients" when anything was
left out, every shared purchase and generic match by name, substitutions,
the coverage-floor warning). `total_cost` and `coverage` cover the planned lines only.
`total_cost` prices every line at its cheapest in-range store; `trip` is one realistic shopping trip priced at
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

## Agent Skill: recipe-shopper

`skills/recipe-shopper/` is an Agent Skill (a `SKILL.md` Claude loads when
a request matches its description, plus a script) that turns a recipe link
into the cheapest basket at nearby stores, using the MCP server above. Claude fetches the page on the user's machine; the
server only ever receives ingredient text and never fetches a URL.

1. `scripts/extract_recipe.py <url | file | ->` (standard library only)
   fetches the page and reads its schema.org Recipe: every
   `application/ld+json` block (`@graph`, arrays, `@type` lists), then
   microdata. It prints `{source, name, yield, servings, ingredients,
   recipe_text, omitted}` and exits 2 when the page has no structured
   recipe (the skill then reads the page with WebFetch and copies the lines
   verbatim) or 1 when the fetch fails. It never runs page scripts.
   A YouTube link has no structured recipe. The skill reads the video's
   description, through the user's own YouTube Data API key
   (`YOUTUBE_API_KEY`) or pasted by the user. Failing that, it reads the
   one recipe page the description links, and failing that it asks for a
   paste. It never reads the watch page, captions or a transcript.
2. Claude calls `plan_from_text` once with `allow_partial: true`, plus
   `lat`/`lon` and `max_km` when the user gave a place or a distance.
3. It reports every line (generic matches flagged as substitutions, a
   shared purchase once), `total_cost` as returned, the recommended trip,
   every `not_stocked`, `out_of_range` and `skipped` entry, and
   `origin_status` / `coverage.spend_fraction` verbatim under the
   `plan_dinner` prompt's rules. It offers a wider `max_km` only when an
   offer is beyond the distance limit, and never invents a product, price
   or store.

**Install** — copy the directory into your personal or a project's skills:

```bash
cp -r skills/recipe-shopper ~/.claude/skills/          # every project
cp -r skills/recipe-shopper <project>/.claude/skills/  # one project
```

**Prerequisites** — the pantry MCP server connected (see [Client
configuration](#client-configuration)), e.g. `claude mcp add --transport
http pantry <url>/mcp --header "Authorization: Bearer <token>"`, and
`python3`. Through the ContextForge gateway the tool is
`pantry-plan-from-text`; the skill uses whichever name the session lists.
Each plan costs Claude credits on the server and takes 10-60 s.

**Deploying a server change behind a gateway** — ContextForge keeps its own
copy of each federated tool's input schema, taken when the gateway was
registered or last refreshed. After deploying a version of this server whose
tool parameters changed (this one added `max_km` and `allow_partial` to
`plan_from_text`), refresh the pantry gateway's tools in ContextForge, or
the gateway keeps offering the old schema. SKILL.md checks for the two
parameters and says so when they are missing.

**What a session looks like:**

```
you    What would https://omnivorescookbook.com/mala-chicken/ cost within 5 km?
claude [runs extract_recipe.py] "La Zi Ji (Sichuan Mala Chicken)", serves 4,
       17 ingredient lines. Planning them calls Claude on the pantry server
       (about a minute).
claude [plan_from_text: recipe_text, allow_partial=true, max_km=5]
       | ingredient | product | store | price |   (generic matches flagged)
       total_cost (planned lines only: K of 17), the recommended trip
       (stores, basket + travel), not stocked + suggestions, out of range +
       the nearest offer, origin_status, every note
```

`tests/test_recipe_extractor.py` runs the extractor on synthetic pages and
checks that every tool, parameter, prompt and result field SKILL.md names
exists in `mcp_server.py`, which is why the skill lives next to the server.

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

When an exclusion removes every candidate for an ingredient, the plan stops
with `excluded_by_origin` naming it, the products it removed and the
same-aisle alternatives still available (Red Onion, Mexico, for an American
yellow onion): the trade is stated, never made silently. With
`allow_partial` (`POST /plan/{slug}?allow_partial=true`, or the MCP tools'
argument) the rest of the recipe is planned and the ingredient goes to
`out_of_range` with the same options; the MCP error says to retry that way.

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

### Checking the catalog against a real store

The demo catalog is synthetic, so nothing said whether it looks like a real
shelf, and it did not: its only yellow onion was American, so "nothing from
the United States" could not plan a bolognese, while a Real Canadian
Superstore sells yellow onions in 2, 3, 10 and 25 lb bags.
`python -m evals.reality_eval` (no LLM calls, no network) checks every
ingredient of the recipe library and writes `evals/reports/reality.md`:

- **exclusion**: whether a candidate is left once a common exclusion (United
  States, China) removes what is evidenced as coming from there;
- **choice**: the catalog's products for it against the reference store's;
- **price**: median unit price (per kg, L or each) within a factor of two of
  the reference store's;
- **origin**: every origin the reference store sells it from (read off a
  label) is one the catalog offers too.

The reference rows are in `evals/datasets/reference_basket.json`, captured by
hand: one product on one shelf on one day, with the link, and its origin only
when someone read the label (store sites rarely publish it for produce, and
refuse automated reads). The report lists the ingredients still to capture.
`--strict` exits 1 when a check fails.

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
