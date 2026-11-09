# Meal planning

The meal plan turns "3 Pepperoni Pizza + 2 Chicken Fried Rice + 3 chicken briyani + 7 mango
milkshakes in 2 weeks" into meals on a dated board, the shopping trips that keep each
perishable inside its cited storage time, and a shopping list per trip. The code is the
`pantry_planner/mealplan/` package. The plan itself lives in the browser (a `MealPlanDraft`,
localStorage key `pantry.mealplan.v1`); the server is stateless.

Every storage time comes from a cited row. A product with no cited row is never given a
number: its time is shown as unknown and planned with the shopper's own setting, labelled as
theirs. Store prices and stock, the demo starter recipes, the library's house amounts and
products 166-169 are synthetic demo data, and say so.

## Endpoints

| Endpoint | What it does | LLM | Limit per client |
| --- | --- | --- | --- |
| `POST /mealplan/selection/parse` | Quick add: counted dishes and the period, matched to recipes | no | 60/min |
| `POST /mealplan/resolve` | Which products each recipe buys, once per distinct recipe | the selector, when live | 6/min, burst 3; 503 above the daily LLM ceiling |
| `POST /mealplan/schedule` | Meals, trips for both strategies, warnings, lists | no | 120/min |
| `POST /mealplan/plan` | MCP `plan_meals` over REST: counted dishes drafted into a plan, as ops for a console | the selector, when live (new recipes only) | 6/min, burst 3; 503 above the daily LLM ceiling |
| `POST /mealplan/suggest-cook-days` | A freshness-aware layout, as a proposal | no | 60/min |
| `POST /mealplan/alternatives` | Options for one trip line (or, with no trip date, for every line buying a product; or for a line that buys nothing): ranked products and what each does to the plan | no | 60/min |
| `POST /mealplan/catalog` | The wizard's recipes: slot, diets, contains, fit and a cost estimate per meal | no | 60/min |
| `POST /mealplan/basket` | The wizard's ingredients: each chosen recipe's products, stores, prices and origins | no | 120/min |
| `POST /mealplan/stores` | The stores to turn off and on: nearest first, where each is, in range, off, and how many of the basket's products each sells | no | 120/min |
| `GET /mealplan/starters` | The 13 demo starter recipes, labelled "demo recipe" | no | none |
| `GET /shelf-life?product_id=` | Cited storage and thaw rows per product | no | none |

`schedule` also carries nutrition: each day's `nutrition` (one person's day, one serving of
every meal), `period_nutrition`, per-recipe receipts in `recipe_nutrition`, and the draft may
send its own `nutrition_targets`. See `docs/nutrition.md`.

`schedule` and `suggest-cook-days` take the draft itself as the body (at most 256 KB; 413
above). Their 422s are `{"detail": {"error": code, "detail": text}}` with code
`slot_capacity`, `unknown_recipe_key`, `stale_product`, `invalid_dates`, `pin_invalid` or
`version`. `invalid_dates` covers a date outside the plan and a date approved twice (one
shop a day, under either strategy). A draft whose numbers are out of range is a plain validation 422 naming the field:
a doc line's quantity is a finite number from 0 to 1,000,000 (`MAX_LINE_QUANTITY`), a pack
count (`packs_override`, an approved snapshot) at most `MAX_PACKS` (10 billion, past
anything the engine computes), and a price at approval finite and not negative.

## What a count means

A count is **one meal occasion that serves the household** (`prefs.household_servings`,
default 2). "3 Pepperoni Pizza" is three dinners for two people, and Quick add says so:
"3 × Pepperoni Pizza = 3 dinners for 2 people". A meal can set its own `servings`. Each need
is the recipe line's amount times meal servings / recipe servings, kept as an exact fraction
(three meals of a recipe for 3 eaten by 2 add up to exactly two batches). "7 mango
milkshakes" is seven snack occasions at household servings.

## Quick add: matching names

1. The period is read and cut out: "in 2 weeks", "for a fortnight", "10 days", or a leading
   "10 days of" ("10 days of grilled cheese" is one grilled cheese over 10 days, not ten).
   The rest is split on `+`, commas, semicolons, new lines, and "and" when a count follows
   it ("peanut butter and jelly sandwich" stays whole).
2. Counts: digits, number words, `2x`, `2 x`, `2 ×`, `×3`, `x3`. No count is one, with
   `count_stated: false`. A slot word after for, as or at ("for breakfast", "as a snack")
   becomes `slot_hint`. A bare one may be part of a title ("Breakfast Burrito", "Dinner
   Rolls"), so the name is matched with it first and it becomes the hint only when that
   finds nothing ("mango milkshakes breakfast").
3. Names are normalised (NFKD, accents stripped, case-folded, punctuation and the stopwords
   of, the, a, an, and, with dropped) and each word singularised by rule (-ies → y, -oes → o,
   -es after s/x/z/ch/sh, a final -s on words over 3 letters, an exceptions list).
4. Matching runs level by level over the library, the demo starters and the shopper's own
   recipes, and the first level with a match wins:
   - **exact**: the same set of words;
   - **plural**: the same after singularising;
   - **alias**: a whole alternative name (`mealplan_starters.json` per starter,
     `seeds/recipe_aliases.json` per library recipe) or a title word's listed misspellings
     (biryani: briyani, biriyani, ...);
   - **fuzzy**: every title word matched by its own input word within Damerau-Levenshtein
     distance 2 for words of 6 or more letters, 1 for 4-5 letters, 0 for shorter ones, with
     no input word left over.
5. Two recipes at the winning level (for fuzzy, at the smallest distance) are `ambiguous`,
   with the candidates listed and nothing chosen. With no match, recipes whose title holds
   every input word are offered as candidates ("chicken curry" offers Simple Chicken Curry;
   it is not matched).

A match carries its recipe's slot: a starter's own, a library recipe's from the starters
file's `library_meta` (the grilled cheese and the PB&J are lunches), and the shopper's own
recipe's as the console sends it (dinner when it sends none).

Only exact and plural matches may be accepted without asking. Alias and fuzzy matches come
back with `needs_confirmation: true`, and the console shows "chicken briyani → Chicken
Biryani (demo starter)?" with Use or Not this.

The route is public and calls no LLM, so its work is bounded. At most 20 dishes are read
from one text (a plan holds 12 recipes), and a warning says where the unread ones begin. A
dish name of more than 12 words is not matched. Runs of spaces read as one. The fuzzy level
is a minimum-cost assignment over the word-distance matrix (at most 7 × 7), not a search of
every word order. Each word distance is computed once per request, and only up to the
word's allowance. One request spends at most 50,000 rows of that distance
(`MAX_FUZZY_ROWS`), about eight times what 20 typed dishes against 50 recipes of real words
use. Past that, the dish it ran out on and every later one get no fuzzy match (never one
picked from the recipes compared so far), and a warning says from which dish.

## Resolve: products, once per recipe

A recipe is resolved when it enters the tray, never on a drag. Library recipes go through
`flow.run(slug)` (the classic pipeline). Every RecipeDoc (a demo starter, a pasted or
imported recipe, a dish the assistant wrote) goes through `flow.run_spec`: the reviewed lines
are planned exactly as given, with no parse, and only the product choice is the selector's.
Results are cached in process (64 entries) by the recipe and every knob that changes them. A
recipe that fails keeps its own status (`needs_servings`, `unconfirmed_lines`, `not_found`,
`unparseable`, `aborted`, `llm_error`) and the others still resolve.

The schedule uses only the product id per line from this result (or the shopper's pin). The
amounts are read again from the recipe itself on every schedule call.

The draft's `settings` may carry the plan's origin rules, `exclude_origin` and `preference`
(at most 50 known countries each), which the console passes to resolve with the shopping
point.

## Matching by meaning: which products are an ingredient

Products are matched to an ingredient by what they are, the way grocery platforms label
query-product pairs (Amazon's ESCI labels: Exact, Substitute, Complement, Irrelevant), not by
the words the two names share: by words, Options for "Butter" offered Canadian Peanut Butter
Cream as the same ingredient. The code is `pantry_planner/matching/`; the plan's first pick
and Options both use it, so the plan and its alternatives agree.

**What it knows, made once, offline** (`scripts/matching_seeds.py`, outputs committed as
seeds, re-runnable: every Gemini reply is cached on disk by a hash of what it was asked, so a
rerun asks only for what changed):

- `seeds/product_profiles.json`: per product, from Gemini (a flash model, batched, with a JSON
  schema): `product_type` (what a recipe calls it: "butter", "peanut butter", "olive oil",
  "canned tomatoes"), `taxonomy` [department, aisle, product_type], `form`, `variant`
  ("salted", "extra virgin"), `culinary_role`, the diets it suits, and `aka` (the other
  names the catalog's description gives it, "scallions" on Green Onions). `profile_text` is
  one sentence templated from those fields, and is what gets embedded. A second, separate
  Gemini pass answered yes or no to each profile's type, form and diet and to the catalog's
  contains-tags; every "no" is a `flag` for review, never applied. Gemini never changes the
  contains-tags (`dietary_tags`). Every profile was then reviewed by hand; fixes are recorded
  on the profile (`fixes`: field, from, to, why) and re-applied on every rerun. The review
  page is `~/Downloads/pantry-design/product-profiles-review.html`.
- `seeds/product_embeddings.json`: EmbeddingGemma vectors of each `profile_text`, asked as its
  model card says for retrieval (a document, "title: <product_type> | text: ..."), the first
  256 of 768 dimensions re-normalized (Matryoshka), rounded to 5 decimals.
- `seeds/ingredient_types.json`: per distinct ingredient name of the library and the demo
  starters, from Gemini against the catalog's types: `exact_types` (types that ARE it),
  `substitute_types` (types that work instead, best first), `unrelated_types` (types that
  only look related: "peanut butter" for "butter"), the form and variant the name asks for,
  and the name's EmbeddingGemma vector (a query, "task: search result | query: ...").

Each profile and vector carries the hash of the product row it was made from (name,
description, category, subcategory, size, contains-tags). A product without a current
profile (one pantry-db added since) is matched by its words until the seeds are made again:
every word of the name in its own makes it the same item, ranked after the profiled ones.

**At request time** (deterministic, no LLM call; `matching/matcher.py`):

1. *What the name means.* A seed name (compared by its tokens, so "yellow onions" is "Yellow
   Onion", and with the parser's purchase form, so "tomatoes" bought canned is "Canned
   Tomatoes") takes its seed. Any other name (imported, pasted, written by the assistant) is
   embedded with EmbeddingGemma through the local Ollama (`OLLAMA_URL`, `EMBEDDING_MODEL`;
   cached on disk by text under `PANTRY_DATA_DIR`) and compared with every product type's
   centroid (the normalized mean of its products' vectors): the nearest type at or above 0.56
   is the same item, the other types at or above 0.50 work instead (three at most), and with
   nothing at or above 0.50 the line is not stocked. The floors are measured: a name and its
   own type score 0.53-0.71, an unstocked name's nearest type 0.37-0.59, so a weak nearest
   type is offered as working instead, never as the same item ("unsalted butter", 0.53, is
   bought as salted butter, labelled a substitution; "cream cheese" is never cheddar). Such a
   name's form and variant are read from the catalog's own: a form among its words prefers a
   type within 0.05 of the nearest whose products have it ("cumin powder" is cumin, ground,
   though cumin seeds are a little nearer), and the variant is the longest of a candidate
   type's variants all of whose words it has ("extra-virgin olive oil" asks for extra virgin,
   "crushed tomatoes" for crushed, "boneless skinless chicken thighs" for boneless skinless).
   When a seed ingredient names its type as the same item, what works instead is that seed's
   list (Gemini's labels) rather than the type's embedding neighbours: pasta sauce does not
   work instead of "penne pasta".
2. *Candidates.* Every product whose type is the same item (**Same item**) or works instead
   (**Works instead**), and the 20 products nearest the name's vector at or above 0.47 whose
   type is neither unrelated nor already in a group (**Looser match**, Options only).
3. *Hard rules*, never choosable and reported with the reason (as `held_back` and
   `out_of_reach` already are): sold in range at a store that is on; passes the origin
   exclusion; adds no contains-tag the plan's own product lacks (for a line it bought nothing
   for: nothing the recipe's own diets leave out); contains nothing the plan's diet, avoid
   list or exclusions leave out. A dairy-free plan never sees a dairy product that works
   instead; butter's Options never offer peanut butter.
4. *Score within a group*, higher first, ties by catalog id: a weighted sum, each part 0..1:
   relevance (the cosine of the name's and the product's vectors; for a product that works
   instead, 10% less per place its type is behind the best substitute) 0.30, form and variant
   (the share of what the name asks for that the product has) 0.15, pack fit (covering the
   need exactly 1, falling to 0.6 at 6 times it; a catering pack beyond that 0.25 unless every
   pack is one; short 0.3; unknown 0.5) 0.20, value (the group's best unit price over its own)
   0.15, origin preference (only when the shopper gave one) 0.10, and the trip total after the
   swap against the group's cheapest 0.10 (falling to 0 at $10 more). The planner's first
   pick has no trip yet, so it reads what the product adds to the trip's spend (its price at
   its cheapest store in range), which is what the trip total moves by whenever the swap
   changes no stop.
5. *Every row keeps short reasons*: the group ("Same item: butter", "Works instead of
   butter: ghee; not the same item"), the pack, the origin, the trip, the rating, and
   `summary`, all on one line ("butter · 454g covers 80 g · no new allergens").

**The plan's first pick** uses the same candidates and order. A line is planned from its same
items; only when the catalog stocks none (White Chocolate) from what works instead, and the
pick then says "substitution". A line whose same item is stocked but not within the plan's
distance or price cap is out of range, as by words: what works instead is never silently
bought for it. Lines of one recipe that mean the same ingredient ("300 g yellow onion", "100 g
yellow onions") are ranked for their summed need, so they share one purchase. The demo
selector takes the first product of the order; the live selector chooses among the products
as before, which are now each line's candidates (a library recipe's selector used to see the
whole catalog). Every plan line says how it was matched (`matching`) and the group of its
product (`group`).

**When the embedder is down**, seeds still work (they are precomputed). A name no seed knows
is matched by today's words (the token levels, the size cap, the word ranking), and the
answer says so: the plan line's `matching: "words"`, and Options' `matching: "words"` with a
`matching_note` for the console ("Matched by words: ..."). Ollama is asked at most once a
minute after a failure. `/health`'s `matching` reports the seeds, the embedder and whether a
name no seed knows is matched by meaning or by words right now.

**Logging for learning to rank later.** Options served (`shown`: the ingredient, each row's
product id, rank, group and score) and the shopper's choices (`picked`: a cart swap, or a meal
plan pin the first time it is seen, with its rank and group and the products ranked above it,
`passed_over`) are appended to `<PANTRY_DATA_DIR>/matching/events.jsonl` (default
`~/.pantry-data`). No shopper, address, location or recipe is written, and times are to the
minute. `MATCHING_EVENTS=0` turns it off.

**Options' added fields** (nothing renamed): each row's `group` (`same_item`,
`works_instead`, `looser_match`, or `outside` for a cart's pick that is no candidate),
`group_label` ("Same item", "Works instead", "Looser match", "Not a candidate"), `score` (the
weighted sum; null when matched by words) and `summary`; the ranking's `matching`
(`meaning` | `words`), `matching_note`, `filtered` (`{product_id, product, group, why:
adds_tags | diet | constraints, tags[], text}`), and `counts.same_item`, `works_instead`,
`looser_match`. By meaning, `order` is `["group", "score", "id"]` and `ranking_text` says the
weights. The words-era fields stay and agree: a Same item is `tier: "same"`, `match:
"exact"`; Works instead `other`/`substitute`; a Looser match `other`/`related`. A plan line
(`PlanLineItem`) adds `matching` and `group`.

**Evaluation** (`python -m evals.substitutes_eval`, report `evals/reports/substitutes.md`).
For every distinct ingredient name of the library and the starters (47), a Gemini judge
(gemini-3-flash-preview) labelled the union of both methods' top 20 candidates (321 pairs)
Exact / Substitute / Complement / Irrelevant (`evals/substitutes_golden.json`). With ESCI's
gains (1 / 0.1 / 0.01 / 0), on Options' order with no amount, location or preference:

| Method | nDCG@3 | Hit@1 |
| --- | --- | --- |
| Words (today's fallback) | 0.924 | 96% |
| Meaning | 0.966 | 98% |

Words already put the item itself first for most of these names; meaning gains in what comes
after it ("Butter": ghee and oils instead of Canadian Peanut Butter Cream and Nestles PBJ;
"Garlic": shallots instead of an onion and ginger) and in the names words could not place
("Peanut Butter and Jelly Jam"). Both score 0 on "White Chocolate", which the catalog does not
sell: the judge calls neither chocolate a substitute. `tests/test_substitutes_eval.py` holds
matching by meaning to at least the words' score and above 0.95 (and Hit@1 above 95%) on the
committed labels.

**Remaking the seeds** (only when the catalog changes): `GEMINI_API_KEY=... python
scripts/matching_seeds.py profiles`, then `check`, review the flags and record fixes, then
`embed`, `types` and `review`; `golden` needs a seeded database (`DB_URL`). Each step prints
the Gemini calls spent; `--budget` stops it at a ceiling (40 for this catalog's first run, of
which 33 were used).

**Not matched by meaning yet**: the weekly menu optimizer (`POST /plan/week`) and the MCP
`lookup_products` tool still match by words.

### Describing names no seed knows

A bare name is a poor query: "cooking onion", "caster sugar", "jeera" or "nam pla" give an
embedding model a few words to guess from, and a bare name embedded against profile text that
says what a product is *for* ("aromatic base for savoury dishes") pulls onion, garlic and
shallots together. With `DESCRIBER_MODEL` set, a name no seed knows is first **identified** by
a low-cost model (`matching/describe.py`), in fields that say only what identifies an item, as
the catalog files its products:

| field | "cooking onion" | "jeera" | "lamb shoulder" |
|---|---|---|---|
| `product_type` (from the catalog's own type list) | yellow onions | cumin seeds | lamb shoulder *(no such type: unstocked)* |
| `form` (one of the catalog's forms) | whole | whole | raw |
| `variant` | | | |
| `look_alikes` (kept for review only) | red onion, shallots | black mustard seeds | pork shoulder roast |

The matcher takes `product_type` as the **Same item** when the catalog has that type, so the
decision is the catalog's type list, never a hand-kept rule; what works instead is the seed's
list when a seed names the same type, else the nearest other types by embedding; and
`form` and `variant` are compared as fields (`form_variant`). The look-alikes are stored so a
reviewer can see them; the matcher does not read them. No prose is asked for: flavour and use
are what blur similar items.

Only the **identity string** is embedded (`seeds.identity_text`: "butter | block | salted"),
with `EMBEDDING_MODEL=gemini:gemini-embedding-001` against `seeds/gemini_vectors.json`, whose
product vectors are identity strings too (`scripts/matching_seeds.py gemini-vectors`). The
embedding is the check and the tie-break: a described type whose identity string is far from
that type's centroid (below `GEMINI_FLOORS["check"]`) is offered as working instead, not as the
same item. Gemini's cosines sit higher and closer together than EmbeddingGemma's, so the
Gemini floors are their own (`matcher.GEMINI_FLOORS`), calibrated on the 47 seed names only:
FLOORS_TBD

Measured on names labelled before any model ran (none a seed name), Same item right / wrong
/ unstocked wrongly given one:

MEASURED_TBD

Every answer is cached in the database (`matching/cache.py`; pantry-db 0010): a name is
described once, keyed by the prompt version, the model, the catalog's type list and the
normalized name, and an identity string embedded once, keyed by the model, its dimensions and
the text. A reviewed or corrected row (`status`) is never replaced. Requests are counted per
UTC day across instances in `matching_usage` against `MATCHING_DAILY_EMBED_TEXTS` and
`MATCHING_DAILY_DESCRIBE_CALLS`; past an allowance the describer or embedder reports itself
unavailable and the name falls back (embedded as written, then words), and `/health` says
so under `matching.cache.budget`. `scripts/prefill_matching.py` describes and embeds
`seeds/common_ingredients.json` (511 names) ahead of any import, within the allowance, and a
rerun costs nothing for what is cached.

## Options for a trip line, and pins

Every trip line has the chat cart's Options dialog. `POST /mealplan/alternatives` takes
`{draft, trip_date?, product_id | line, strategy?, limit? 1..25 = 12}` (exactly one of
`product_id` and `line: {recipe_key, line_no}`, else 422) and answers
`{rev, strategy, trip_date, product_id, product, stocked, pinned, lines[], plan_total,
ranking}` (`mealplan/options.py`, no LLM):

- `lines` are the recipe lines the purchase covers, which can span recipes (salt for the
  pizza and the biryani), each `{recipe_key, title, line_no, ingredient, planner_product_id,
  pinned_product_id}`. Choosing a product pins it on every one of them
  (`pins[recipe_key][line_no]`); a pin equal to `planner_product_id` is no pin, which is how
  "Back to the planner's pick" works.
- `ranking` is an `AlternativeRanking` from `alternatives.rank_alternatives` over those
  lines, each rebuilt as resolve planned it (`pins.meal_basis`): the candidates (by meaning,
  in their groups, above), the held-back and `filtered` products, `unavailable`, the facts,
  reasons and order are the chat cart's, and no row contains what the plan's diet or avoid
  list leaves out, or what the covered recipes' own diets do (`pins.forbidden`). Its
  need is what these meals need on this trip. Each row's `trip` is the plan re-scheduled with
  that product pinned, under the same strategy, by the schedule's own code: `trip.total` and
  `trip.delta` are the strategy's total and its change ("$0.40 less on your plan's trips"),
  `packs` and `cost_for_need` what the trip line then buys and charges, `trip.buys_at` the
  store and pack price there. A row whose price the plan cannot work out (its pack is in
  other units than the need) has no total and says so; it is never shown as cheaper.
- `stocked: false` when no store in range sells the line's product any more: its options are
  listed and it is not among the rows. The `no_longer_stocked` and `not_stocked` warnings'
  `open_options` remedy opens exactly this.
- 422 with the schedule's codes for a draft it refuses; 409 `no_trip_line` when the trip buys
  no such product (the plan changed).
- With no `trip_date` (the wizard's Ingredients stage, before any trip exists) the purchase is
  every line of the plan that buys the product, for all its recipe's meals, as
  `/mealplan/basket` lists them, and choosing still pins them all. No trip is worked out:
  every row's `trip` is null with a reason saying so, its `packs` are what those meals need
  and its `cost_for_need` those packs at its cheapest offer in range; `trip_date` and
  `plan_total` are null.
- **A line that buys nothing** (the basket's rows with no product: the resolve left it out
  for the origin rule, out of range or not stocked, and no pin fills it) is asked for by
  `line: {recipe_key, line_no}` instead of a product. The answer has `product_id: null`,
  `product: ""`, `stocked: false`, and `lines` that line and every other line of the plan with
  the same ingredient (by name) that buys nothing either, each with `planner_product_id`
  null. `ranking` is the same ranking with no current pick (`rank_alternatives(...,
  open_lines=True)`): `items` are the choosable rows (sold in range at a store
  that is on, not held back), none `current`; `held_back` the products the origin exclusion
  holds back, with their evidence, never choosable; `filtered` what a hard rule keeps out;
  and `ranking.out_of_reach` the products
  that would fill it but are sold only at stores turned off or beyond `max_km`, each
  `{product_id, product, why: "stores_off" | "out_of_range" | "not_stocked", stores[],
  nearest: {store, price, distance_km} | null}` (`stores` the stores turned off that sell it
  within the distance, nearest first; `nearest` the nearest offer beyond it, at a store that
  is on when one sells it), at most `limit` of them, best match first; `unavailable` counts
  them all. `out_of_reach` is filled for the meal plan's Options only (a chat cart's is
  empty). Choosing a row pins every covered line, and the line then buys it everywhere
  (basket, schedule, catalog). With a `trip_date` each row's `trip` is the plan re-scheduled
  with that product bought for those lines' placed meals, so `trip.delta` is what adding it
  costs. 409 `no_trip_line` when the line buys a product (ask by `product_id`), is not a
  line of the recipe, or its recipe has no meal or is not resolved; 422 `unknown_recipe_key`
  for a recipe the plan does not have. With every yellow onion evidenced as grown in the
  United States and that country excluded, the pilaf's and the biryani's Yellow Onion lines
  get Red Onion, Shallots and Green Onions as rows that work instead (the seed's substitute
  types, in its order); with every one of them held back, `items` is empty and `held_back`
  lists them (tests/test_mealplan_options.py).

Every schedule call checks every pin of a resolved recipe as a cart swap is checked
(`alternatives.validate_pins`): the line is one of the recipe's (one the resolve bought
nothing for included: the pin fills it, and the basket, the schedule and the catalog then buy
it there), the product is one of its candidates, no hard rule keeps it out (it adds no
contains-tag the resolve's product lacks, and contains nothing the plan's diet, avoid list or
the recipe's diets leave out), the plan's origin exclusion does not hold it back, and a store
in range sells it. Otherwise it is 422 `pin_invalid` with `recipe_key`, `line_no` and the reason
("... is evidenced as United States (made or packed there), which this plan excludes"). A pin
applies to every meal of its recipe, wherever the meals move. Pins are bounded: at most 40
lines per recipe, line numbers of at most 4 digits, product ids from 1 to 2^31 - 1. An
approved trip whose products change through a pin becomes `needs_review` by the fingerprint
rule below.

### Unknown servings

A recipe that does not say how many it serves resolves with status `needs_servings`. Until
the shopper answers ("How many does this recipe serve?"), its lines have no packs, no need,
no leftover and no price, and the trip total is marked as a floor ("Total at least"). When
no line on a trip has a price, its total is unknown (`total_cost: null`, "Total unknown (no
line has a price yet)"), never $0.00, and so is the strategy's when no trip has one. A
`must_fix` warning offers `set_servings`. The answer is stored as `recipes[key].servings`
(or `ref.servings` when resolving) and labelled `servings_basis: "your_setting"`.

## The wizard: budget, diets, catalog and basket

The Meal plan tab's six stages (Basics, Meals, Ingredients, Schedule, Shopping plan, Calendar)
read the endpoints above and two more, both pure code with no model call in demo mode or live.

**Prefs.** `prefs` gains `budget` (groceries for the whole plan, 0 < budget ≤ 10,000; null
for none), `diet` (`any`, `pescatarian`, `vegetarian` or `vegan`) and `avoid` (any of
`dairy`, `gluten`, `nuts`, `soy`, `egg`: the products' own tags). All three have defaults, so
a draft saved before them loads. The schedule warns `over_budget` (should fix) for each
strategy whose known total is more than the budget, naming both ("at least" when a line has
no price), with the other strategy as a remedy when it is within the budget and a budget
that covers the total; there is no warning without a budget.

**Origin rules.** The wizard's defaults are to exclude the United States and to prefer
Canada, then `DEFAULT_ALLIES` (mealplan/models.py: United Kingdom, France, Germany, Italy,
Spain, Netherlands, Ireland, Denmark, Norway, Sweden, Australia, New Zealand, Japan, South
Korea, Mexico). A preference only orders products; only the exclusion leaves one out.

**Stores turned off.** `settings.stores_off` (store ids, at most 50) are stores the shopper
turned off in the Ingredients stage. No offer of theirs is priced or put on a trip: the
catalog's `pack_price`, the basket, the schedule's trips (and so the calendar's store names)
and both kinds of Options list read the price matrix without them
(`build_price_matrix_sql(..., exclude_stores=)`). Resolve ignores them, and so do the
catalog's cached picks, so turning a store off or on never resolves a recipe again.

**`POST /mealplan/stores`** takes `{draft}` and answers `{v, stores[], products_total,
synthetic, origin, max_km}` (`mealplan/stores.py`), each store `{id, name, address, lat, lon,
distance_km, in_range, off, sells}`: every store, nearest first from the plan's shopping point
(measured as the price matrix measures it), `in_range` within `max_km`, `off` from
`stores_off`, and `sells` how many of the basket's products it stocks, whatever its distance
(null while the plan buys nothing). `products_total` counts those products: the draft's chosen
ones, pins included, less any the origin exclusion holds back (`basket.purchases`). `address`,
`lat` and `lon` are the stores table's, always, for the console's list and map; `synthetic` is
`STORES_SYNTHETIC` (on by default: the stores and their addresses are made up), so the console
labels them demo stores. The calendar export's policy is unchanged: it never writes a
synthetic store's address. `origin` is the plan's shopping point `{lat, lon}` (its own, or the
server's) and `max_km` its distance limit (null: none), the map's centre and range.

**Diets.** Every demo starter says its `slot`, its `diets` (which of vegan, vegetarian and
pescatarian it satisfies) and `contains`; the library's are the starters file's
`library_meta`, since `recipes.json` is pantry-db's byte for byte. `mealplan/diets.py` is the
rule: a recipe contains its products' tags, a seafood product's `meat` read as `fish` (its
aisle, or a fish word in its name or description, as for canned tuna and fish sauce), and
what its lines' names say whatever the product's tags (milk, cheese, butter, cream, yogurt,
ghee, paneer, white or milk chocolate are dairy; eggs egg; fish, salmon, tuna, shrimp, prawn,
anchovy fish; chicken, beef, pork, bacon, ham, pepperoni, sausage, lamb, turkey meat; honey
`honey`), unless a plant word comes right before ("peanut butter", "coconut milk"). Vegan
leaves out meat, fish, dairy, egg and honey, vegetarian meat and fish, pescatarian meat. A
test resolves every recipe in demo mode and checks the file against its products and names. Salmon and tuna
have no row in `nutrients.json` yet, so the three fish starters' nutrition counts those lines
as `no_reference`.

| Slot | Recipes (diets: vg vegan, v vegetarian, p pescatarian) |
| --- | --- |
| breakfast | Ginger Congee (vg v p), Tomato Garlic Toast (vg v p), Scrambled Eggs on Toast (v p), Yogurt Fruit Bowl (v p), Salmon Scrambled Eggs (p) |
| lunch | Garlic Broccoli Penne (vg v p), Tomato Pilaf (vg v p), Peanut Butter & Jelly Sandwich (v p: its White Chocolate is dairy), Grilled Cheese Sandwich (v p), Tuna Melt (p) |
| dinner | Vegetable Stir Fry (vg v p), Tomato Penne (v p), Ginger Soy Salmon (p), Pepperoni Pizza, Chicken Fried Rice, Chicken Biryani, Simple Chicken Curry, Spaghetti Bolognese, Beef & Broccoli Rice Bowl |
| snack | Mango Milkshake (v p) |

**`POST /mealplan/catalog`** takes `{household_servings, diet, avoid, settings, slots?,
recipes?, pins?}` (`recipes`: the shopper's own refs, at most 24; `pins`: the plan's, as the
draft holds them, so a pinned line is costed at the shopper's product) and answers `{v, recipes[],
est_basis, latency_ms}`, each recipe `{key, ref, title, image, slot, diets, contains,
servings, fits, why_not, est_cost_per_meal, lines_total, lines_priced, unavailable, held_back,
lines[]}`, ordered by slot then title (`mealplan/catalog.py`). Each line's product is the
one resolve picks in demo mode: the resolve's own retrieval and origin filter, then the demo
selector; live, the resolve's model may pick another product later.

- `est_cost_per_meal` prices each line at its product's cheapest offer in range as its share
  of the pack (what the recipe uses for the household over the pack size); `lines_priced` of
  `lines_total` (water and ice are not bought) are. It is a per-meal figure, well under what
  whole packs cost.
- `lines` are `{ingredient, product_id, pack_price, pack_qty, pack_uom, need_per_meal,
  held_back}`, `need_per_meal` in the pack's unit for one meal for the household. The console
  costs any selection from them by **whole packs**: per product, the total need (meals ×
  `need_per_meal`) over `pack_qty` rounded up, and at least one pack (one in all, however many
  lines share it) when a line's need or the pack size is unknown, at `pack_price`; a line with no `pack_price` is not bought.
  `/mealplan/basket`'s `est_total` counts the same way (`catalog.whole_packs`), so stage 2's
  budget bar and stage 3's total agree.
- An ingredient no store in range sells, or that the origin exclusion empties, is in
  `unavailable` and never priced; `held_back` names those the origin rule left out (every
  candidate in range evidenced as from an excluded country), so the console never calls them
  "not sold nearby".
- `why_not` names the diet's tags first, then the avoided ones ("contains fish and gluten").
  A shopper's own recipe is classified from its picks and line names when every line has a
  product; otherwise `diets` and `contains` are empty and it fits only with no diet and
  nothing avoided.

The picks are cached in process per recipe and shopping point, so a diet or household change
answers from the cache.

**Pictures.** A catalog or basket recipe names its picture as `image`, and never holds one:

- `{"kind": "wiki", "title"}` for a demo starter or library recipe: its `image_wiki` in the
  starters file (`library_meta` for the library), the English Wikipedia article whose lead
  image shows the dish or the closest one ("Congee", "Biryani", "Melt sandwich" for the Tuna
  Melt, "Pa amb tomàquet" for the Tomato Garlic Toast). The console shows that image through
  the demo hub (`GET /hub/images/wiki?title=`), credited to Wikipedia and linked back. Each
  title was checked by hand to be an article with a lead image; the tests never call Wikipedia.
- `{"kind": "url", "url"}` for a shopper's own recipe whose doc has `source.image_url`: the
  link to the preview image of the page it was imported from (its schema.org Recipe image or
  `og:image`, a video's thumbnail; an absolute http(s) URL of at most 2,000 characters), shown
  through the hub's guarded `GET /hub/images/remote?url=`, credited to the site.
- `null` otherwise (a pasted recipe, a page with no preview image).

**`POST /mealplan/basket`** takes `{draft}` and answers `{v, rev, recipes[],
unavailable_count, est_total, budget, latency_ms}` for every recipe with a meal
(`mealplan/basket.py`): `{key, title, image, status, meals, lines[]}` (`image` as the
catalog's), each line `{line_no, ingredient, need, product, store, price, cost_for_need,
origin, pinned, shared_with, available, unavailable_kind, why_unavailable}`. Products are the
draft's (resolved, then pinned, checked as the schedule checks them); a recipe not resolved yet
lists no lines. Each product is bought once for every meal at its cheapest offer in range
(`store`, and `price` one pack there: the catalog's `pack_price`), by the whole-pack rule
above; each line's `cost_for_need` is its share of the packs by need (one pack for a line of
unknown need), so the lines add up to `est_total`. Trips that buy on several dates can buy
more; stage 5 shows theirs. `origin` is `{country, claim, status: evidenced | conflicting | unknown,
preferred_rank}`: `claim` in plain words ("From Thailand", "Made in Canada, ingredients may
be imported", "Origin unclear: sources disagree", "Origin unknown"), and `preferred_rank` the
place (1 first) of the preference entry it matches. An excluded product is not available and never
priced; a product no store in range sells names the nearest offer. `unavailable_kind` is
`origin_excluded`, `stores_off` ("Sold only at stores you turned off: ..."), `out_of_range` or
`not_stocked` (null when available); for a line the
resolve bought nothing for, the kind and the reason are the planner's ("its only candidate
is evidenced as coming from United States, which this plan excludes").

## Storage times: sources, day semantics, safety and quality

`seeds/shelf_life.json` holds rows from two public pages, quoted verbatim, and a map from
catalog product to the rows that apply:

- **FoodSafety.gov (U.S. Department of Health and Human Services), Cold Food Storage Chart**,
  reviewed 2023-09-19, retrieved 2026-10-08: fridge and freezer times for ground meat, steaks
  and roasts, whole poultry and pieces, fatty and lean fish, shrimp, eggs in the shell,
  leftovers, pizza, soups and stews.
- **U.S. Department of Agriculture, Food Safety and Inspection Service, "The Big Thaw — Safe
  Defrosting Methods"**, updated 2013-06-15, retrieved 2026-10-08: thaw times and the
  after-thaw fridge times.

Rules of use:

- **Day semantics.** The purchase day is day 0. A cited fridge time of N days allows eating
  up to day N, so a meal on day d may be bought on days d − N to d. A breakfast is bought by
  the day before. Plans use the **lower bound** of a range ("1 to 2 days" is planned as 1);
  the console always shows the verbatim text.
- A time in months or years has no day count. It only ever counts as longer than the plan.
- Where a product cites more than one row for one kind of storage, the shorter time is
  planned and every row is cited (the white fish bag cites both fish freezer rows).
- **Safety versus quality.** The chart says its short fridge limits "will help keep them from
  spoiling or becoming dangerous to eat", and that its freezer times "are for quality only"
  ("frozen foods stored continuously at 0°F (-18°C) or below can be kept indefinitely").
  `basis` (safety for fridge rows, quality for freezer rows) is the file's own label for
  that; the console shows the verbatim text, not the label. The plan freezes a product on arrival only when its cited freezer time is
  longer than the plan and a cited thaw time exists; eggs ("Do not freeze in shell") are
  never frozen.
- **Thawing.** In the fridge: a full day for a small amount; at least 24 hours per 5 lb for a
  large item (2.268 kg, the file's conversion; the page states pounds only), rounded up. The
  thaw reminder is that many days before the meal, and it quotes the row. **The rest of a
  thawed pack is never planned for a later meal**: each thawed meal has a purchase of its
  own. This is the planner's choice, not a source statement (FSIS allows a day or two more,
  and refreezing).
- **Unknown storage times.** Most products have no cited row (dairy, produce, bread, cured
  sausage, every pantry item). They are labelled unknown and never given a number:
  - chilled or fresh: planned with the shopper's **buy-ahead setting** (`buy_ahead_days`,
    default 7), shown as "your setting";
  - shelf-stable or bought frozen (the file's own storage class, not a cited time): not
    limited; one purchase can serve the whole period, and the line says no time is claimed.
- The coverage block counts products with cited times, with your setting, and with neither.

`thaw_reminder` (evening before or morning of) is the setting for a product frozen with no
cited thaw time; every product the current data lets the plan freeze has one, so it is not
used yet.

### Adding a product mapping

Add the product to `products` in `shelf_life.json` (and remove it from `unmapped`) with its
`storage_class`, `bought_state` and the ids of the rows that apply under `fridge`, `freezer`,
`after_thaw_fridge` and `thaw`, plus a `note` when the choice needs explaining. Use a row only
when the chart names that food; when in doubt, leave the product unmapped.
`tests/test_mealplan_seeds.py` checks that every catalog product is listed once and every
rule id exists.

## Trips: the stabbing search

Both strategies are computed on every call:

- **fresh**: nothing is frozen unless the shopper set a product to the freezer
  (`storage_overrides`);
- **fewest_trips**: a product with a cited freezer time and thaw time may be frozen on
  arrival, so its window opens at the start of the plan. It is bought fresh when a trip falls
  inside its fridge window and frozen otherwise, with a freeze action on the trip date and a
  thaw action citing the thaw row.

Each need has a window of days it may be bought on (above). Windows are clipped to the days
the shopper shops (`shop_weekdays`, less `dismissed_dates`, plus `fixed_dates` and approved
trip dates). Choosing dates is **minimum interval stabbing**: windows sorted by their last
day; a window no chosen date falls in gets a trip on its last allowed day. That is the fewest
trips (the tests check it against brute force on 200 seeded instances), and it buys as late
as possible. A need whose window holds no shopping day is bought on the last trip before it
and raises a warning instead of being silently stretched.

Purchases: per product in date order, a need joins the current purchase when that trip lies
inside its window, and the packs are recomputed for the summed need with the shared
`packs.pack_count` (one line is enough: 1200 g against 450 g packs is 3). A need in no
measurable unit has unknown packs and price (`packs_basis: amount_unknown`), and the shopper
can set the packs (`packs_override["<date>:<product_id>"]`, 0 to drop the line).

Each trip is priced with the trip optimizer over its stores in range (`tripopt`), and lists
its `frontier` and `recommended` store split. `fresh` is recommended unless it has must-fix
warnings that `fewest_trips` does not.

### Warnings

| Level | Codes |
| --- | --- |
| must_fix | `unplaced`, `fridge_window_exceeded`, `meal_before_trip`, `needs_servings`, `no_longer_stocked` |
| should_fix | `over_budget` |
| decide | `needs_review`, `not_stocked`, `buy_ahead_exceeded`, `trip_cap_exceeded`, `unresolved_recipe` |
| note | `amount_unknown`, `shelf_life_unknown`, `price_changed` |

Each carries at most three remedies: edits the console can apply (`move_meal`,
`set_storage`, `add_trip`, `set_servings`, `set_packs`, `open_options`, `resolve`,
`set_strategy`, `set_pref`, `approve_trip`, `remove_meal`). The engine never applies them.

## Placing meals and "Suggest cook days"

The board has one breakfast, lunch and dinner a day and two snacks. A meal with a date, or
pinned, never moves. Undated meals are spread evenly: the i-th of a recipe's n meals aims at
day floor((i + 0.5) × days / n), recipes with more meals first, collisions probed d+1, d−1,
d+2, ... Overflow is unplaced with a must-fix warning.

"Suggest cook days" (`place.freshness_layout`) proposes a layout and applies nothing:

- candidate trip days are the approved and fixed dates, then every day the shopper shops;
- each unpinned meal gets a horizon: the shortest cited fridge time among its products, else
  the buy-ahead setting for a product with no cited time, else no limit;
- meals with a horizon go first, shortest first (earliest deadline first), each to the free
  cell of its slot nearest its evenly spread day among the days [trip, trip + horizon] after
  some candidate trip; the rest are spread from the first candidate trip on;
- the response lists `move_meal` ops, each with a reason built by code, for example
  "Chicken Biryani moved to Sun 18 Oct: Chicken Thighs Bone-In keeps 1 to 2 days in the
  fridge (FoodSafety.gov, Cold Food Storage Chart, fs-poultry-pieces-fridge); bought on the
  Sat 17 Oct shop.", and the warning counts before and after.

With Saturday-only shopping the chicken meals land on Saturday and Sunday, because the chart
gives poultry pieces 1 to 2 days.

## Approval, fingerprints, prices and stock

The shopper approves a trip; the console keeps `{date, fingerprint, strategy, snapshot[
{product_id, packs, storage, price_at_approval}]}` in `draft.trips`. The engine never
rewrites it:

- the **fingerprint** is sha256 of `"<date>|<pid>:<packs>:<storage>;..."` over the sorted
  lines (packs `?` when unknown). **Price is not in it**;
- a recomputed trip with another fingerprint is `needs_review`, with the diff
  (`+1 Whole Milk 1L (fridge)`, `-1 Chicken Thighs Bone-In (fridge)`) and a decide warning
  offering to re-approve;
- a **price change** never changes the status: each line shows `price_at_approval` and
  `price_delta`, the trip its summed delta, and a note says "Price changed since you approved
  ...: +$1.20 (demo prices)". `price_at_approval` is the line's price for all its packs, so
  the delta compares unit prices: (unit price now − `price_at_approval` / packs at approval)
  × packs now. A changed pack count is a diff, not a price change, and a line whose packs are
  unknown or 0 on either side has no delta;
- a product with **no offer in range any more** is `stocked: false`, the trip becomes
  `needs_review`, and a must-fix `no_longer_stocked` warning offers Options for that line or
  dropping it.

`approved_schedule` collects the approved trips with their lists and their shop, freeze, thaw
and cook actions for the calendar export, with a cook event for every placed meal;
`exportable` is false while any of them needs review. Its shape and the export itself: the
README's "Calendar export" section.

## Trip lists, and what "ordered" means

`list_text` is built by code: grouped by store in the order of the recommended stops, then by
aisle (the product's catalog category), each line with packs, size, need and price, lines
not stocked in range last, then the total ("Total at least" when a price is unknown, "Total
unknown (no line has a price yet)" when none is known) and the footer "Prices and stock are demo data." The console's Copy list and Print list use it, and
the calendar export puts it in the trip's description.

"Ordered together" means exactly this: one list per store per trip. No store ordering,
pickup or delivery API exists; a retailer integration would need the user's permission and
an account.

## Sources and credit

- Source: FoodSafety.gov (U.S. Department of Health and Human Services), Cold Food Storage
  Chart, reviewed 2023-09-19. A work of the U.S. federal government (17 U.S.C. 105); the page
  states no reuse terms of its own.
- Source: U.S. Department of Agriculture, Food Safety and Inspection Service, The Big Thaw —
  Safe Defrosting Methods, updated 2013-06-15. USDA asks for a credit line for its public
  domain information (Policies and Links page).
- Nutrition: Health Canada, Canadian Nutrient File (API, edition not stated; most likely the
  2015 CNF). Contains information licensed under the Open Government Licence – Canada. See
  `docs/nutrition.md`.
- Demo starter recipes, the library's house amounts, products 166-169 and every store price
  and stock level: written for this demo, synthetic, and labelled wherever they are shown.

Storage times are general food-safety guidance, not advice for a particular food or
household.

## The assistant's draft (`plan_meals`, `POST /mealplan/plan`)

An agent drafts a plan with the MCP tool `plan_meals` (`pantry_planner/mealplan/assistant.py`):
counted `dishes` (`{recipe, count, slot?}`, the recipe a key, a library slug, a starter key or
a title; or a dish the assistant writes, with `lines` and `servings`, its amounts labelled
`written_by_assistant`), the shopper's `current` plan in brief, and `proposed`, the dishes an
app read from the shopper's words that need their yes. The same rule as Quick add holds: only
an exact or plural title match is placed. An alias or fuzzy match ("chicken briyani") comes
back in `summary.proposals` with its question and the op a console applies on "Use", never
placed; a name that fits several recipes, or none, is in `summary.unmatched` with what it could
be.

New meals are spread over free slots by the schedule's own placement; the current plan's meals
never move, its window is lengthened but never shortened, and its approved trips are never
touched (the summary warns that new meals will put them up for review). The summary's trips,
totals, warnings and `nutrition` line come from `schedule.compute` on the draft, so they are
what the Meal plan shows once the shopper applies `summary.ops` (`set_window`, `add_recipe`,
`add_meals`, `place`) as one undo step. The tool saves nothing and has no way to approve a
trip. Tests: `tests/test_mcp_meals.py`.
