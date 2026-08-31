# Origin resolution eval

Measured, not asserted. No LLM calls, no network — this scores the
evidence currently in the database, so a low coverage number is a
statement about the corpus, not about the resolver.

- **Coverage**: 62% (5 of 8 products have usable evidence)
- **Accuracy**: 100% of 4 resolved products name the right country
- **Holdout**: 75% of 4 products that SHOULD be held out (unknown/conflicting) correctly were

## Exclusion filter — United States

Recall and precision are reported separately and never averaged: a
false pass (an American product that ships) and a false exclude (a
domestic product wrongly removed) cost the user different things.

- Recall: 100% — of 1 genuinely US-linked products, 1 were caught
- Precision: 100% — of 1 flagged, 1 genuinely are

## Per-case

| Product | Expected | Got | Country | OK |
|---|---|---|---|---|
| Basmati Rice 2kg | resolved | resolved | India | yes |
| Whole Milk 1L | resolved | resolved | Canada | yes |
| Nestles PBJ | resolved | resolved | Canada | yes |
| Parmesan Wedge 200g | resolved | resolved | Italy | yes |
| Extra Virgin Olive Oil 500ml | conflicting | conflicting | Greece, Italy | yes |
| Atlantic Salmon Fillet 400g | unknown | unknown | — | yes |
| Roma Tomato | unknown | unknown | — | yes |
| Dark Chocolate Cadbury | unknown | resolved | United Kingdom | NO |
