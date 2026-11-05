# Substitutes eval: word matching against matching by meaning

47 ingredient names (every distinct one of the library and the demo starters),
each method's first three products scored on labels a Gemini judge gave the union
of both methods' top 20 (evals/substitutes_golden.json; ESCI gains E 1, S 0.1,
C 0.01, I 0). No amount, location or origin preference: only matching and price.

| Method | nDCG@3 | Hit@1 |
|---|---|---|
| Words (today's fallback) | 0.924 | 96% |
| Meaning (EmbeddingGemma + Gemini seeds) | 0.966 | 98% |

## How sure these numbers are

95% intervals from a paired bootstrap over the 47 names (10,000 draws,
seed fixed). The difference is meaning minus words on the same names.

| Metric | Words | Meaning | Difference | Draws where meaning is not better |
|---|---|---|---|---|
| nDCG@3 | 0.858–0.975 | 0.915–0.996 | +0.041 (0.011–0.091) | 0.0% |
| Hit@1 | 89%–100% | 94%–100% | +2.1% (0%–6%) | 36.3% |

Caveats: the same Gemini family wrote the product profiles and judged the labels, and
the matcher's weights were tuned on this set, so these are optimistic for new names.

## When the exact item is missing

46 names with an exact product: every product labelled E removed, Options' hard
rule applied (nothing that adds an allergen or diet tag the exact item lacks), and the
first product left scored. Unlabelled products count as not a substitute.

| Method | Sub@1 (first offer works instead) | Something offered | First offer would add an allergen without the rule |
|---|---|---|---|
| Words | 52% | 87% | 13% |
| Meaning | 74% | 85% | 13% |

| Ingredient | Words: first offer | Meaning: first offer |
|---|---|---|
| All-Purpose Flour | Flour Tortillas 10-Pack (I) | nothing |
| Basmati Rice | Long Grain White Rice 1kg (S) | Long Grain White Rice 1kg (S) |
| Broccoli | Fresh Garlic (I) | Bok Choy (S) |
| Butter | nothing | Ghee 400g (S) |
| Canned Tomatoes | Roma Tomato (S) | Roma Tomato (S) |
| Canned Tuna | Canned Corn Kernels 341ml (I) | Frozen White Fish 400g (S) |
| Canola Oil | Toasted Sesame Oil 250ml (I) | Extra Virgin Olive Oil 500ml (S) |
| Carrots | Fresh Garlic (I) | nothing |
| Cheddar Cheese | Paneer 350g (S) | Mozzarella Shredded 200g (S) |
| Chicken Breast | Chicken Thighs Bone-In (S) | Chicken Thighs Bone-In (S) |
| Chicken Thighs | Chicken Drumsticks 1kg (S) | Chicken Drumsticks 1kg (S) |
| Chili Powder | Sichuan Chili Flakes 100g (S) | Paprika Ground 100g (S) |
| Cilantro | Flat-Leaf Parsley (S) | Flat-Leaf Parsley (S) |
| Diced Tomatoes | Crushed Tomatoes Canned 796ml (S) | Crushed Tomatoes Canned 796ml (S) |
| Dried Oregano | Bay Leaves 10g (I) | Dried Thyme 10g (S) |
| Eggs | nothing | nothing |
| Fresh Mint | Cilantro (I) | Cilantro (I) |
| Frozen Mango | Limes 4-Pack (I) | nothing |
| Garam Masala | Cumin Ground 100g (I) | Curry Powder 100g (S) |
| Garlic | Yellow Onion (I) | Shallots 227g (S) |
| Ghee | Butter Salted 454g (S) | Butter Salted 454g (S) |
| Ginger | Fresh Garlic (I) | nothing |
| Granulated Sugar | Icing Sugar 1kg (S) | Brown Sugar 1kg (S) |
| Green Cardamom Pods | Star Anise Whole 30g (I) | Green Chilies 100g (I) |
| Green Chilies | Dried Red Chilies 100g (S) | Green Bell Pepper (S) |
| Green Onions | Yellow Onion (S) | Yellow Onions 10lb Bag (S) |
| Ground Beef | Beef Sirloin Steak 450g (S) | Ground Pork 450g (S) |
| Instant Yeast | Baking Soda 500g (I) | nothing |
| Jasmine Rice | Long Grain White Rice 1kg (S) | Long Grain White Rice 1kg (S) |
| Mozzarella | Cheddar Cheese Block 300g (S) | Cheddar Cheese Block 300g (S) |
| Olive Oil | Canola Oil 1L (S) | Canola Oil 1L (S) |
| Passata | Tomato Sauce Jar 650ml (S) | Canned Diced Tomatoes (S) |
| Peanut Butter and Jelly Jam | nothing | Canadian Peanut Butter Cream (S) |
| Penne | Spaghetti Pasta 500g (S) | Fusilli Family Pack 900g (S) |
| Pepperoni | nothing | Crushed Red Pepper Flakes 60g (I) |
| Plain Yogurt | Coconut Yogurt 500g (S) | Coconut Yogurt 500g (S) |
| Salmon Fillet | Frozen White Fish 400g (S) | Frozen White Fish 400g (S) |
| Salt | nothing | nothing |
| Sesame Oil | Canola Oil 1L (I) | Canola Oil 1L (I) |
| Sliced Bread | nothing | Sourdough Loaf (S) |
| Soy Sauce | Tomato Sauce Jar 650ml (I) | Vegetable Oil 1.42L (I) |
| Spaghetti | Penne Rigate 500g (S) | Fusilli Family Pack 900g (S) |
| Turmeric | Cumin Ground 100g (I) | Curry Powder 100g (S) |
| Wheat Bread | Sourdough Loaf (S) | Sourdough Loaf (S) |
| Whole Milk | Coconut Milk Canned 400ml (S) | Oat Milk 1L (S) |
| Yellow Onion | Green Onions (S) | Red Onion (S) |

## Per ingredient

| Ingredient | Words nDCG@3 | Words top 3 | Meaning nDCG@3 | Meaning top 3 |
|---|---|---|---|---|
| Salmon Fillet | 1.00 | Atlantic Salmon Fillet 400g (E); Frozen White Fish 400g (S); Raw Shrimp 340g (S) | 0.96 | Atlantic Salmon Fillet 400g (E); Frozen White Fish 400g (S) |
| All-Purpose Flour | 0.94 | All-Purpose Flour 2.5kg (E); Flour Tortillas 10-Pack (I); Baking Soda 500g (I) | 0.94 | All-Purpose Flour 2.5kg (E) |
| Basmati Rice | 1.00 | Basmati Rice 2kg (E); Long Grain White Rice 1kg (S); Jasmine Rice 2kg (S) | 1.00 | Basmati Rice 2kg (E); Long Grain White Rice 1kg (S); Jasmine Rice 2kg (S) |
| Canned Tomatoes | 1.00 | Canned Diced Tomatoes (E); Crushed Tomatoes Canned 796ml (E); Petite Diced Tomatoes 796ml (E) | 1.00 | Canned Diced Tomatoes (E); Crushed Tomatoes Canned 796ml (E); Petite Diced Tomatoes 796ml (E) |
| Carrots | 1.00 | Carrots 1kg (E); Fresh Garlic (I); Yellow Onion (I) | 1.00 | Carrots 1kg (E) |
| Chicken Breast | 1.00 | Boneless Skinless Chicken Breast (E); Chicken Thighs Bone-In (S); Chicken Thighs Boneless 450g (S) | 1.00 | Boneless Skinless Chicken Breast (E); Chicken Thighs Bone-In (S); Chicken Drumsticks 1kg (S) |
| Chicken Thighs | 1.00 | Chicken Thighs Bone-In (E); Chicken Thighs Boneless 450g (E); Chicken Drumsticks 1kg (S) | 1.00 | Chicken Thighs Bone-In (E); Chicken Thighs Boneless 450g (E); Chicken Drumsticks 1kg (S) |
| Cilantro | 0.96 | Cilantro (E); Flat-Leaf Parsley (S); Fresh Basil 30g (I) | 0.96 | Cilantro (E); Flat-Leaf Parsley (S); Fresh Mint 30g (I) |
| Diced Tomatoes | 1.00 | Canned Diced Tomatoes (E); Petite Diced Tomatoes 796ml (E); Crushed Tomatoes Canned 796ml (S) | 1.00 | Canned Diced Tomatoes (E); Petite Diced Tomatoes 796ml (E); Crushed Tomatoes Canned 796ml (S) |
| Eggs | 1.00 | Eggs Large Dozen (E) | 1.00 | Eggs Large Dozen (E) |
| Fresh Mint | 1.00 | Fresh Mint 30g (E); Cilantro (I); Flat-Leaf Parsley (I) | 1.00 | Fresh Mint 30g (E); Cilantro (I); Fresh Basil 30g (I) |
| Frozen Mango | 1.00 | Frozen Mango Chunks 600g (E); Limes 4-Pack (I); Lemons 3-Pack (I) | 1.00 | Frozen Mango Chunks 600g (E) |
| Ghee | 1.00 | Ghee 400g (E); Butter Salted 454g (S) | 1.00 | Ghee 400g (E); Butter Salted 454g (S) |
| Ginger | 1.00 | Fresh Ginger (E); Fresh Garlic (I); Yellow Onion (I) | 1.00 | Fresh Ginger (E) |
| Granulated Sugar | 1.00 | Granulated Sugar 1kg (E); Icing Sugar 1kg (S); Brown Sugar 1kg (S) | 1.00 | Granulated Sugar 1kg (E); Brown Sugar 1kg (S); Pure Maple Syrup 540ml (S) |
| Green Cardamom Pods | 1.00 | Green Cardamom Pods 50g (E); Star Anise Whole 30g (I); Black Mustard Seeds 100g (I) | 1.00 | Green Cardamom Pods 50g (E); Green Chilies 100g (I) |
| Green Onions | 1.00 | Green Onions (E); Yellow Onion (S); Red Onion (S) | 1.00 | Green Onions (E); Yellow Onions 10lb Bag (S); Yellow Onions 3lb Bag (S) |
| Ground Beef | 1.00 | Ground Beef Extra Lean 300g (E); Ground Beef Lean (E); Ground Beef Medium 900g (E) | 1.00 | Ground Beef Lean (E); Ground Beef Medium 900g (E); Ground Beef Extra Lean 300g (E) |
| Instant Yeast | 0.94 | Instant Yeast 3-Pack (E); Baking Soda 500g (I); Cornstarch 450g (I) | 0.94 | Instant Yeast 3-Pack (E) |
| Jasmine Rice | 1.00 | Jasmine Rice 2kg (E); Long Grain White Rice 1kg (S); Basmati Rice 2kg (S) | 1.00 | Jasmine Rice 2kg (E); Long Grain White Rice 1kg (S); Basmati Rice 2kg (S) |
| Mozzarella | 1.00 | Mozzarella Shredded 200g (E); Vegan Cheese Shreds 200g (S); Cheddar Cheese Block 300g (S) | 1.00 | Mozzarella Shredded 200g (E); Vegan Cheese Shreds 200g (S); Cheddar Cheese Block 300g (S) |
| Olive Oil | 1.00 | Extra Virgin Olive Oil 500ml (E); Olive Oil 1L (E); Canola Oil 1L (S) | 1.00 | Olive Oil 1L (E); Extra Virgin Olive Oil 500ml (E); Vegetable Oil 1.42L (S) |
| Penne | 1.00 | Penne Rigate 500g (E); Gluten-Free Penne 340g (E); Spaghetti Pasta 500g (S) | 1.00 | Penne Rigate 500g (E); Gluten-Free Penne 340g (E); Fusilli Family Pack 900g (S) |
| Pepperoni | 1.00 | Sliced Pepperoni 175g (E) | 1.00 | Sliced Pepperoni 175g (E); Crushed Red Pepper Flakes 60g (I) |
| Salt | 1.00 | Table Salt 1kg (E); Fine Sea Salt 750g (E); Kosher Salt 1.36kg (E) | 1.00 | Table Salt 1kg (E); Fine Sea Salt 750g (E); Kosher Salt 1.36kg (E) |
| Sesame Oil | 1.00 | Toasted Sesame Oil 250ml (E); Canola Oil 1L (I); Vegetable Oil 1.42L (I) | 1.00 | Toasted Sesame Oil 250ml (E); Vegetable Oil 1.42L (I); Canola Oil 1L (I) |
| Sliced Bread | 1.00 | Wheat Bread 2 (E); Wheat Bread 1 (E); Gluten-Free Bread 400g (E) | 1.00 | Wheat Bread 2 (E); Wheat Bread 1 (E); Gluten-Free Bread 400g (E) |
| Soy Sauce | 1.00 | Soy Sauce 500ml (E); Light Soy Sauce 500ml (E); Dark Soy Sauce 500ml (E) | 1.00 | Soy Sauce 500ml (E); Light Soy Sauce 500ml (E); Dark Soy Sauce 500ml (E) |
| Spaghetti | 1.00 | Spaghetti Pasta 500g (E); Penne Rigate 500g (S); Fusilli Family Pack 900g (S) | 1.00 | Spaghetti Pasta 500g (E); Fusilli Family Pack 900g (S); Penne Rigate 500g (S) |
| Wheat Bread | 1.00 | Wheat Bread 2 (E); Wheat Bread 1 (E); Gluten-Free Bread 400g (S) | 1.00 | Wheat Bread 2 (E); Wheat Bread 1 (E); Sourdough Loaf (S) |
| White Chocolate | 0.00 | Dark Chocolate Cadbury (I); Milk Chocolate Cadbury (I) | 0.00 | Dark Chocolate Cadbury (I); Milk Chocolate Cadbury (I) |
| Whole Milk | 1.00 | Whole Milk 1L (E); Coconut Milk Canned 400ml (S); Almond Milk 946ml (S) | 1.00 | Whole Milk 1L (E); Oat Milk 1L (S); Almond Milk 946ml (S) |
| Yellow Onion | 1.00 | Yellow Onion (E); Yellow Onions 3lb Bag (E); Organic Yellow Onions 2lb Bag (E) | 1.00 | Yellow Onions 10lb Bag (E); Yellow Onions 3lb Bag (E); Yellow Onion (E) |
| Dried Oregano | 0.99 | Dried Oregano 10g (E); Bay Leaves 10g (I); Dried Thyme 10g (S) | 1.00 | Dried Oregano 10g (E); Dried Thyme 10g (S); Flat-Leaf Parsley (I) |
| Chili Powder | 0.63 | Chili Powder 100g (E); Sichuan Chili Flakes 100g (S); Doubanjiang Chili Bean Paste 500g (I) | 0.66 | Chili Powder 100g (E); Paprika Ground 100g (S); Sichuan Chili Flakes 100g (S) |
| Broccoli | 0.97 | Broccoli Crown (E); Frozen Broccoli 500g (E); Fresh Garlic (I) | 1.00 | Broccoli Crown (E); Frozen Broccoli 500g (E); Bok Choy (S) |
| Canola Oil | 0.96 | Canola Oil 1L (E); Vegetable Oil 1.42L (S); Toasted Sesame Oil 250ml (I) | 1.00 | Canola Oil 1L (E); Vegetable Oil 1.42L (S); Peanut Oil 1L (S) |
| Passata | 0.96 | Passata Strained Tomatoes 680ml (E); Tomato Sauce Jar 650ml (S); Soy Sauce 500ml (I) | 1.00 | Passata Strained Tomatoes 680ml (E); Canned Diced Tomatoes (S); Crushed Tomatoes Canned 796ml (S) |
| Plain Yogurt | 0.96 | Plain Greek Yogurt 750g (E); Coconut Yogurt 500g (S) | 1.00 | Plain Greek Yogurt 750g (E); Coconut Yogurt 500g (S); Sour Cream 500ml (S) |
| Garam Masala | 0.94 | Garam Masala 100g (E); Cumin Ground 100g (I); Chinese Five-Spice Powder 50g (I) | 1.00 | Garam Masala 100g (E); Curry Powder 100g (S); Chili Powder 100g (I) |
| Garlic | 0.94 | Fresh Garlic (E); Yellow Onion (I); Fresh Ginger (I) | 1.00 | Fresh Garlic (E); Shallots 227g (S) |
| Turmeric | 0.94 | Turmeric Ground 100g (E); Cumin Ground 100g (I); Chinese Five-Spice Powder 50g (I) | 1.00 | Turmeric Ground 100g (E); Curry Powder 100g (S); Cumin Seeds 100g (I) |
| Butter | 0.90 | Butter Salted 454g (E); Canadian Peanut Butter Cream (I); Nestles PBJ (I) | 1.00 | Butter Salted 454g (E); Ghee 400g (S); Vegetable Oil 1.42L (S) |
| Canned Tuna | 0.90 | Canned Tuna Flaked 170g (E); Canned Corn Kernels 341ml (I); Black Beans Canned 540ml (I) | 1.00 | Canned Tuna Flaked 170g (E); Frozen White Fish 400g (S); Atlantic Salmon Fillet 400g (S) |
| Cheddar Cheese | 0.89 | Cheddar Cheese Block 300g (E); Vegan Cheese Shreds 200g (I); Cheddar Shredded 320g (E) | 1.00 | Cheddar Cheese Block 300g (E); Cheddar Shredded 320g (E); Mozzarella Shredded 200g (S) |
| Green Chilies | 0.63 | Green Chilies 100g (E); Dried Red Chilies 100g (S); Fresh Garlic (I) | 0.93 | Green Chilies 100g (E); Green Bell Pepper (S); Jalapeno Peppers (E) |
| Peanut Butter and Jelly Jam | 0.00 | none | 1.00 | Nestles PBJ (E); Canadian Peanut Butter Cream (S) |
