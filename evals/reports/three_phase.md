# Eval Report — routing strategy: `three_phase`

- **Recipes evaluated:** 5
- **Overall accuracy:** 100.00%
- **Total LLM cost:** $0.0442
- **Total wall-clock latency:** 33358 ms
- **Escalated (cascade only):** 0/5

## Per-recipe results

### Peanut Butter & Jelly Sandwich (`pbj_sandwich`)

- Accuracy: **100.00%**
- Grocery total: $21.00
- LLM cost: $0.0086
- Latency: 7699 ms
- Preselected: `claude-haiku-4-5-20251001`  |  Escalated: False

| # | Ingredient | Chosen | $ | Conf | Score | Verdict |
|---|---|---|---|---|---|---|
| 1 | Peanut Butter and Jelly Jam | Nestles PBJ `#5` | $8.00 | 1.00 | 1.00 | ☑️ correct |
| 2 | Wheat Bread | Wheat Bread 2 `#2` | $3.00 | 1.00 | 1.00 | ✅ perfect |
| 3 | White Chocolate | Milk Chocolate Cadbury `#4` | $10.00 | 0.60 | 1.00 | ☑️ correct |

### Grilled Cheese Sandwich (`grilled_cheese`)

- Accuracy: **100.00%**
- Grocery total: $15.98
- LLM cost: $0.0074
- Latency: 4659 ms
- Preselected: `claude-haiku-4-5-20251001`  |  Escalated: False

| # | Ingredient | Chosen | $ | Conf | Score | Verdict |
|---|---|---|---|---|---|---|
| 1 | Sliced Bread | Wheat Bread 2 `#2` | $3.00 | 1.00 | 1.00 | ✅ perfect |
| 2 | Cheddar Cheese | Cheddar Cheese Block 300g `#23` | $5.99 | 1.00 | 1.00 | ☑️ correct |
| 3 | Butter | Butter Salted 454g `#18` | $6.99 | 1.00 | 1.00 | ☑️ correct |

### Spaghetti Bolognese (`spaghetti_bolognese`)

- Accuracy: **100.00%**
- Grocery total: $29.23
- LLM cost: $0.0096
- Latency: 7093 ms
- Preselected: `claude-haiku-4-5-20251001`  |  Escalated: False

| # | Ingredient | Chosen | $ | Conf | Score | Verdict |
|---|---|---|---|---|---|---|
| 1 | Spaghetti | Spaghetti Pasta 500g `#21` | $2.49 | 1.00 | 1.00 | ☑️ correct |
| 2 | Ground Beef | Ground Beef Lean `#22` | $7.99 | 1.00 | 1.00 | ☑️ correct |
| 3 | Yellow Onion | Yellow Onion `#12` | $0.99 | 1.00 | 1.00 | ☑️ correct |
| 4 | Garlic | Fresh Garlic `#13` | $0.79 | 1.00 | 1.00 | ☑️ correct |
| 5 | Canned Tomatoes | Canned Diced Tomatoes `#15` | $1.99 | 1.00 | 1.00 | ☑️ correct |
| 6 | Olive Oil | Extra Virgin Olive Oil 500ml `#16` | $9.99 | 1.00 | 1.00 | ☑️ correct |
| 7 | Mozzarella | Mozzarella Shredded 200g `#24` | $4.99 | 1.00 | 1.00 | ☑️ correct |

### Simple Chicken Curry (`chicken_curry`)

- Accuracy: **100.00%**
- Grocery total: $37.02
- LLM cost: $0.0106
- Latency: 8070 ms
- Preselected: `claude-haiku-4-5-20251001`  |  Escalated: False

| # | Ingredient | Chosen | $ | Conf | Score | Verdict |
|---|---|---|---|---|---|---|
| 1 | Chicken Thighs | Chicken Thighs Bone-In `#11` | $6.49 | 1.00 | 1.00 | ☑️ correct |
| 2 | Basmati Rice | Basmati Rice 2kg `#8` | $12.00 | 1.00 | 1.00 | ☑️ correct |
| 3 | Yellow Onion | Yellow Onion `#12` | $0.99 | 1.00 | 1.00 | ☑️ correct |
| 4 | Garlic | Fresh Garlic `#13` | $0.79 | 1.00 | 1.00 | ☑️ correct |
| 5 | Ginger | Fresh Ginger `#25` | $1.29 | 1.00 | 1.00 | ☑️ correct |
| 6 | Garam Masala | Garam Masala 100g `#29` | $4.99 | 1.00 | 1.00 | ☑️ correct |
| 7 | Turmeric | Turmeric Ground 100g `#27` | $3.99 | 1.00 | 1.00 | ☑️ correct |
| 8 | Diced Tomatoes | Canned Diced Tomatoes `#15` | $1.99 | 1.00 | 1.00 | ☑️ correct |
| 9 | Canola Oil | Canola Oil 1L `#17` | $4.49 | 1.00 | 1.00 | ☑️ correct |

### Vegetable Stir Fry (`veggie_stirfry`)

- Accuracy: **100.00%**
- Grocery total: $22.06
- LLM cost: $0.0081
- Latency: 5837 ms
- Preselected: `claude-haiku-4-5-20251001`  |  Escalated: False

| # | Ingredient | Chosen | $ | Conf | Score | Verdict |
|---|---|---|---|---|---|---|
| 1 | Broccoli | Frozen Broccoli 500g `#30` | $3.49 | 0.90 | 1.00 | ☑️ correct |
| 2 | Basmati Rice | Basmati Rice 2kg `#8` | $12.00 | 1.00 | 1.00 | ☑️ correct |
| 3 | Garlic | Fresh Garlic `#13` | $0.79 | 1.00 | 1.00 | ☑️ correct |
| 4 | Ginger | Fresh Ginger `#25` | $1.29 | 1.00 | 1.00 | ☑️ correct |
| 5 | Canola Oil | Canola Oil 1L `#17` | $4.49 | 1.00 | 1.00 | ☑️ correct |
