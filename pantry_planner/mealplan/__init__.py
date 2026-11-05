"""Meal plans: counted recipes on a dated board, cited storage times and trips the shopper
approves.

The shopper picks recipes with counts ("3 Pepperoni Pizza + 2 Chicken Fried Rice ... in 2
weeks"), the console holds the plan (a MealPlanDraft in the browser), and the server answers
two kinds of question:

- slow, once per distinct recipe (resolve.py): which products does this recipe buy? The
  pipeline's selector picks them, as for a chat plan;
- fast, after every edit (schedule.py): given where the meals sit, what does each day need,
  on which dates should the shopper shop, and what does each trip buy? Pure code: no LLM,
  no writes, facts re-read from the database by id, the same draft giving the same bytes.

Every storage time comes from a cited row of seeds/shelf_life.json (shelf.py). A product with
no cited row is never given a number: its time is unknown, and the plan uses the shopper's
own "buy at most N days ahead" setting, labelled as theirs.
"""
