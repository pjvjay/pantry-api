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
| `POST /mealplan/suggest-cook-days` | A freshness-aware layout, as a proposal | no | 60/min |
| `GET /mealplan/starters` | The 4 demo starter recipes, labelled "demo recipe" | no | none |
| `GET /shelf-life?product_id=` | Cited storage and thaw rows per product | no | none |

`schedule` and `suggest-cook-days` take the draft itself as the body (at most 256 KB; 413
above). Their 422s are `{"detail": {"error": code, "detail": text}}` with code
`slot_capacity`, `unknown_recipe_key`, `stale_product`, `invalid_dates`, `pin_invalid` or
`version`. A draft whose numbers are out of range is a plain validation 422 naming the field:
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

1. The period is read and cut out: "in 2 weeks", "for a fortnight", "10 days". The rest is
   split on `+`, commas, semicolons, new lines, and "and" when a count follows it ("peanut
   butter and jelly sandwich" stays whole).
2. Counts: digits, number words, `2x`, `2 x`, `2 ×`, `×3`, `x3`. No count is one, with
   `count_stated: false`. "for breakfast", "snacks" and the like become `slot_hint`.
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

Only exact and plural matches may be accepted without asking. Alias and fuzzy matches come
back with `needs_confirmation: true`, and the console shows "chicken briyani → Chicken
Biryani (demo starter)?" with Use or Not this.

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

### Unknown servings

A recipe that does not say how many it serves resolves with status `needs_servings`. Until
the shopper answers ("How many does this recipe serve?"), its lines have no packs, no need,
no leftover and no price, the trip total is marked as a floor ("Total at least"), and a
`must_fix` warning offers `set_servings`. The answer is stored as `recipes[key].servings`
(or `ref.servings` when resolving) and labelled `servings_basis: "your_setting"`.

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
  ...: +$1.20 (demo prices)";
- a product with **no offer in range any more** is `stocked: false`, the trip becomes
  `needs_review`, and a must-fix `no_longer_stocked` warning offers Options for that line or
  dropping it.

`approved_schedule` collects the approved trips with their lists and their shop, freeze, thaw
and cook actions for the calendar export; `exportable` is false while any of them needs
review.

## Trip lists, and what "ordered" means

`list_text` is built by code: grouped by store in the order of the recommended stops, then by
aisle (the product's catalog category), each line with packs, size, need and price, lines
not stocked in range last, then the total ("Total at least" when a price is unknown) and the
footer "Prices and stock are demo data." The console's Copy list and Print list use it, and
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
- Demo starter recipes, the library's house amounts, products 166-169 and every store price
  and stock level: written for this demo, synthetic, and labelled wherever they are shown.

Storage times are general food-safety guidance, not advice for a particular food or
household.
