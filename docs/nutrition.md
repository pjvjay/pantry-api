# Nutrition

Energy, protein, fat, saturated fat, carbohydrate, fibre, sugars and sodium per portion
eaten, computed by code (`pantry_planner/nutrition.py`, no LLM) from a recipe's amounts and
one reference food per ingredient. Every value is a published reference value for a generic
food, never a product's label: the catalog is synthetic and has no nutrition facts.

Not medical or dietary advice. Totals add up raw ingredients as written and do not count
cooking losses.

## Where the numbers come from

- **Reference values:** `seeds/nutrients.json` (byte-identical to pantry-db's, loaded into
  the tables of pantry-db migrations 0008 and 0009). It holds 41 reference foods, up to 8
  nutrients per 100 g each, 40 measures, and a 46-key ingredient map. Every value was read
  from Health Canada's Canadian Nutrient File (CNF) API, one request per food, on 2026-10-08.
- **Edition:** the API does not say which edition it serves. Health Canada's CNF online
  search serves the 2015 CNF, so these are most likely 2015 values, **not** the CNF 2026
  files. The source's `edition` field says this and travels with every number.
- **Licence:** the CNF 2026 record on open.canada.ca states the Open Government Licence –
  Canada. The API documentation states no licence, so the file says that the licence's
  coverage of the API data is not verified. Credit line, shown wherever a number is: "Contains
  information licensed under the Open Government Licence – Canada."
- **Amounts:** the recipe's own lines. Library recipes and the demo starters use demo house
  amounts (synthetic, labelled); pasted and imported recipes use the amounts the shopper
  reviewed. Packs bought never feed nutrition.

`pantry-db/scripts/cnf-subset.py` can rebuild the reference values from a CNF download if
one is ever approved. It works offline and fetches nothing.

## The rules

**Unknown is never zero.** A nutrient the source did not publish for a food has no row, and
its total for any meal using that food is a lower bound. Three foods publish no total sugars:
cardamom, all-purpose flour and spearmint. A line is not counted when it has no amount, no
weight, or no reference food, and it is named. Each nutrient's total carries one of three
statuses:

| status | amount | shown as |
| --- | --- | --- |
| `complete` | a number | `642 kcal` |
| `at_least` | a lower bound | `≥ 610 kcal`, with `gaps` naming the lines it misses |
| `unknown` | `null` | `unknown`, never 0 |

**Grams, never guessed.**
- A weight converts directly (`g`, `kg`, `lb` 454 g, `oz` 28 g).
- A millilitre or count line converts only through a measure the source publishes for that
  food, quoted in the receipt. For example, 30 ml olive oil uses the source's measure
  "15ml" = 13.682 g. No density is ever assumed, not even 1 g/ml for water or milk.
- A volume uses the published measure nearest in volume, and that row's own grams per
  millilitre.
- A count uses the size the recipe names ("large eggs" uses "1 large egg"). When the recipe
  names no size and the source lists several, the line stays unconverted (the fried-rice
  starter's "3 each Eggs").
- A `can` has no stated size, so it is not converted. `pinch`, `dash` and `to taste` are not
  amounts.

**One map, keyed by ingredient.** The key is `units.tokens()` of the line's name joined by
spaces (`"Chicken Thighs"` gives `chicken thigh`). Lookup tries the full key, then the
generic key (descriptor words dropped, only when one was), then the head noun, and stops at
the first key with a row. `match_kind` is one of:
- `generic`: the same food;
- `close`: a near food, with a note saying how it differs;
- `none`: reviewed, and nothing fits ("garam masala", "peanut butter and jelly jam").

A key with no row at all has not been reviewed, and is reported differently. Water has a
reviewed row (CNF municipal water), so a water line is counted. A non-purchase line with no
row ("ice cubes") is excluded and does not count against coverage.

**Coverage, with a floor.** Each meal reports coverage by line count and by mass, the mass
taken over the lines whose weight is known. Below `NUTRITION_MIN_COVERAGE` (default 0.8) on
either measure, the meal is `below_floor`, and its note reads "Nutrition known for only K of
N ingredients - totals are minimums".

**Demo amounts are labelled.**
- Each line copies its `amount_basis` from the recipe line.
- A meal's `amounts_basis` is the set of values over its counted lines.
- `demo_amounts` is true when any of those is `demo_house_amounts`. It is true for every
  library recipe and every demo starter, and false for a pasted or imported recipe.
- Days and the period combine their meals' values.

Every surface that shows a number shows a "demo amounts" badge whenever `demo_amounts` is
true (title: "Recipe amounts are demo house amounts, not from a published recipe").

## Endpoints

| Endpoint | What it returns |
| --- | --- |
| `GET /nutrition/recipes` | `{sources, recipes: RecipeNutrition[], note}`: each library recipe per serving |
| `GET /recipes/{slug}/nutrition?portions=1` | `RecipeNutrition` with every line's receipt and its `sources`; the same 404 as `GET /recipes/{slug}` |
| `POST /plan/week` (additive) | `days[].nutrition` (MealNutrition per serving), `days[].day_totals` (dinner only), `nutrition_sources`; optional body `targets` |
| `POST /mealplan/schedule` (additive) | `days[].nutrition` (DayNutrition), `period_nutrition`, `recipe_nutrition`, `coverage.nutrition`; optional draft `nutrition_targets` |

None of them calls an LLM. Before pantry-db 0008 is deployed, plans and schedules still work:
- every nutrition field is `null`;
- `/plan/week` adds the note "nutrition not shown: nutrition tables not deployed (pantry-db
  0008)";
- the recipe endpoints say the same in `note`;
- the schedule's `coverage.nutrition` is `not_deployed`.

With 0008 deployed but not 0009, millilitre and count lines say that no source measures are
deployed.

## Shapes

```text
NutrientTotal  {amount: float|null, unit: kcal|g|mg, status: complete|at_least|unknown,
                complete: bool, lines_counted, lines_total, gaps: [str]}
NutritionLine  {line_no, ingredient, quantity, unit, grams|null (in the whole recipe),
                status: counted|no_quantity|no_conversion|no_reference|excluded, reason,
                key, match_kind|null, match_note, ref_id|null, ref_description (verbatim),
                state_note, source, conversion (quoted), values {nutrient: amount}
                (for the meal's basis), absent [nutrient], amount_basis}
NutritionCoverage {lines_total, lines_counted, count_fraction|null, grams_weighed,
                grams_known, mass_fraction|null, lines_mass_unknown, floor, meets_floor, note}
MealNutrition  {basis: per_serving|per_recipe, servings|null, portions,
                totals {nutrient: NutrientTotal}, status: complete|incomplete|below_floor,
                coverage, missing [{line_no, ingredient, reason, detail}], lines,
                source_ids, amounts_basis [str], demo_amounts: bool, note}
RecipeNutrition {slug, name, servings, nutrition: MealNutrition|null, note, sources}
NutritionSource {source, name, publisher, edition, licence, licence_url, attribution, url,
                retrieved_at}
DayMeal        {meal_id, recipe_key, title, slot, basis, status, totals, amounts_basis,
                demo_amounts}
DayNutrition   {date|null, meals [DayMeal], meals_counted [slot], all_meals_planned,
                totals, complete, amounts_basis, demo_amounts, note,
                targets {nutrient: TargetCheck}|null}
PeriodNutrition {days_total, days_complete, incomplete_days [date],
                per_day_average_over_complete_days {nutrient: amount}|null,
                lower_bound_total {nutrient: {amount|null, complete}}, slots_counted,
                amounts_basis, demo_amounts, note}
NutritionTarget {min?: float, max?: float, source: you|health_canada_dv}
TargetCheck    {amount|null, complete, min: Verdict|null, max: Verdict|null}
```

The meanings that are easy to get wrong:

- **basis:** `per_serving` divides by the recipe's servings and multiplies by `portions`.
  `per_recipe` is the whole recipe, used when the recipe does not say how many it serves (a
  meal-plan recipe in `needs_servings`). Such a meal adds nothing to a day until the shopper
  answers.
- **A day** is what one person eats: one serving of every meal on that date. It is complete
  only when every slot in `prefs.slots_on` has a meal and every meal's totals are complete.
  On `/plan/week` the day counts dinner only, so it is never complete.
- **The period** counts a day as complete only when all eight of its totals are. The
  per-day average is over complete days only, and is `null` when none is. The band reads,
  for example, "9 of 14 days complete · average 1,840 kcal per day over complete days (demo
  amounts)".

## Targets

Daily targets are the shopper's own, kept in the browser and sent with each request
(`/plan/week` body `targets`, or the draft's `nutrition_targets`). The keys are the eight
nutrient names; any other key is a 422. A verdict is given only where the data proves it:

| bound | verdict | when |
| --- | --- | --- |
| max | `over` | the amount (even a lower bound) is above it |
| max | `within` | the total is complete and at or below it |
| min | `met` | the amount (even a lower bound) reaches it |
| min | `short` | the total is complete and below it |
| either | `unknown` | anything else |

A dinner-only day therefore never says `within` or `short`. Health Canada publishes no Daily
Value for energy or protein, so a target labelled `health_canada_dv` for either is refused.
No Daily Value presets are shipped: a preset would need its own cited source.
