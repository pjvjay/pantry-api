# Reality check: the catalog against a real store

Generated 2026-10-07 by `python -m evals.reality_eval` from the database it ran against and `evals/datasets/reference_basket.json`. No LLM calls, no network.

- **Reference:** 4 observation(s) of 1 of the library's 22 ingredients, at Real Canadian Superstore, Eaton Center and Kingsway (store 1518, the website's default for a Vancouver visitor), 2026-10-07
- **Failing:** 2 of 22 ingredients
- **Without United States:** every ingredient keeps a candidate
- **Without China:** 2 ingredient(s) have no candidate left: Garlic, Ginger
- **One product only:** 8 ingredient(s): Basmati Rice, Garlic, Canola Oil, Ginger, Garam Masala, Turmeric, Cheddar Cheese, Spaghetti

## Per ingredient

| Ingredient | Catalog options | Reference options | Catalog unit price | Reference unit price | Without United States | Without China | Result |
|---|---|---|---|---|---|---|---|
| Ground Beef | 3 | – | $11.73–20.90/kg | – | 2 left | 3 left | ok |
| Basmati Rice | 1 | – | $5.82/kg | – | 1 left | 1 left | ok |
| Broccoli | 2 | – | $5.68–6.00/kg | – | 1 left | 2 left | ok |
| Garlic | 1 | – | $13.60/kg | – | 1 left | 0 left | no candidate without China |
| Canola Oil | 1 | – | $3.90/L | – | 1 left | 1 left | ok |
| Chicken Thighs | 2 | – | $8.23–14.64/kg | – | 2 left | 2 left | ok |
| Yellow Onion | 4 | 4 | $1.31–4.35/kg | $0.93–3.31/kg | 2 left | 4 left | ok |
| Ginger | 1 | – | $11.40/kg | – | 1 left | 0 left | no candidate without China |
| Garam Masala | 1 | – | $44.20/kg | – | 1 left | 1 left | ok |
| Turmeric | 1 | – | $35.40/kg | – | 1 left | 1 left | ok |
| Diced Tomatoes | 2 | – | $2.20–2.74/L | – | 1 left | 2 left | ok |
| Sliced Bread | 3 | – | $5.93–16.75/kg | – | 3 left | 3 left | ok |
| Cheddar Cheese | 1 | – | $17.03/kg | – | 1 left | 1 left | ok |
| Butter | 3 | – | $5.18–20.12/kg | – | 3 left | 3 left | ok |
| Peanut Butter and Jelly Jam | 0 | – | – | – | – | – | no product matches this name |
| Wheat Bread | 2 | – | $5.93–7.02/kg | – | 2 left | 2 left | ok |
| White Chocolate | 0 | – | – | – | – | – | no product matches this name |
| Spaghetti | 1 | – | $4.42/kg | – | 1 left | 1 left | ok |
| Canned Tomatoes | 2 | – | $2.20–2.89/L | – | 1 left | 2 left | ok |
| Olive Oil | 2 | – | $11.91–18.66/L | – | 2 left | 2 left | ok |
| Mozzarella | 2 | – | $22.45–26.45/kg | – | 2 left | 2 left | ok |
| Penne | 2 | – | $3.94–11.88/kg | – | 1 left | 2 left | ok |

## To capture

Ingredients with no reference row yet: Ground Beef, Basmati Rice, Broccoli, Garlic, Canola Oil, Chicken Thighs, Ginger, Garam Masala, Turmeric, Diced Tomatoes, Sliced Bread, Cheddar Cheese, Butter, Peanut Butter and Jelly Jam, Wheat Bread, White Chocolate, Spaghetti, Canned Tomatoes, Olive Oil, Mozzarella, Penne.

Reference rows without an origin (read it off the label or shelf sign): Yellow Onion.

A row is one product on one shelf on one day: `ingredient` (the library's name), `store`, `branch`, `product`, `size`, `qty` and `uom` (g, ml or each), `price` (the shelf price that day), `regular_price` when on sale, `origin` and `origin_source` ("label" or "shelf sign") or null, `observed_at`, `link`.
