---
name: recipe-shopper
description: Prices a recipe's ingredients at nearby grocery stores with the pantry MCP server and finds the cheapest basket. Use when the user shares a recipe link (URL) or pastes a recipe and asks what the ingredients cost, where to buy them, or for the cheapest ingredients nearby. Needs the pantry server's plan_from_text tool.
---

# Recipe shopper

Turn a recipe link into a priced shopping list: the cheapest stocked product
for each ingredient, the stores to visit, and an honest list of what the
catalog does not carry. You fetch the page; the pantry server only ever
receives ingredient text. It never fetches a URL.

## Prerequisites

- The pantry MCP server is connected, for example:

  ```bash
  claude mcp add --transport http pantry https://<host>/mcp \
    --header "Authorization: Bearer <token>"
  ```

  For a local `uvicorn pantry_planner.api:app` the URL is
  `http://localhost:8000/mcp`.
- Tool names depend on the client and any gateway in between:
  `plan_from_text` may be listed as `mcp__pantry__plan_from_text`, or with a
  gateway prefix such as `pantry-plan-from-text` (ContextForge). Use whichever
  name this session lists. If none is listed, stop and say the pantry server
  is not connected.
- `python3` for the extractor (standard library only).

## Steps

### 1. Get the ingredient lines

For a link, run the extractor from this skill's directory:

```bash
python3 scripts/extract_recipe.py "<url>"
```

It fetches the page on this machine and reads the schema.org Recipe the page
publishes (JSON-LD, else microdata). It prints JSON with `name`, `yield`,
`servings`, `ingredients` (the lines verbatim), `recipe_text` (built as in
step 2) and `omitted`. It never runs the page's scripts.

- **Exit 2** (no structured recipe on the page): read the page with WebFetch
  and copy the ingredient lines verbatim: every line, quantities included,
  nothing added, merged or reworded. Then build `recipe_text` yourself.
- **Exit 1** (the fetch failed: HTTP error, timeout, blocked): tell the user,
  try WebFetch, and if that fails too ask them to paste the ingredient list.
- **A pasted recipe**: skip the script and use the user's lines verbatim.

Before planning, tell the user the recipe name and how many ingredient lines
you found, so a wrong page is caught before it costs anything.

### 2. Build recipe_text

```text
<name> (serves N)
- <ingredient line 1>
- <ingredient line 2>
```

One `- ` line per ingredient, as written. Use "(makes <yield>)" when the
yield is not a serving count, and only the name when there is no yield. At
most 8000 characters: the tool rejects more. The extractor's `omitted` list
names any line that did not fit; tell the user those were not planned. If
the user gave shopping notes ("under $40", "no dairy"), add a last line
`Notes: <their words>`. Never put the URL in `recipe_text`.

### 3. Plan it

Tell the user this runs Claude on the pantry server (it costs API credits)
and takes 10-60 s. Then call `plan_from_text` once:

```json
{"recipe_text": "<from step 2>", "allow_partial": true, "lat": 49.2827, "lon": -123.1207, "max_km": 5}
```

- `allow_partial`: always true for a real recipe. Without it, the first
  ingredient the catalog lacks fails the whole call.
- `lat` / `lon`: only when the user gave a location (coordinates, or a place
  you can locate confidently). Otherwise leave both out and say the server's
  default location was used.
- `max_km` (0.5-100): only when the user gave a distance. If they said
  something vague ("walking distance"), pick a number and say which. Without
  `max_km` every store counts, however far.
- `exclude_origin` / `preference`: only for a buy-local or boycott request;
  check the spellings against the `pantry://countries` resource first.

Call it once. Each call costs credits, so do not re-run it to "improve" the
result. If it returns an error even with `allow_partial`, nothing could be
planned: report the error text and the ingredients it names. On a validation
error (text too long, `max_km` out of range), fix that argument and retry
once.

### 4. Out of range

If `summary.out_of_range` is not empty, those ingredients are stocked but
not within the limits, and each `reason` names the nearest offer. Offer to
plan again with a larger `max_km` that reaches it. Re-plan only if the user
says yes.

### 5. Report

Use only what the tool returned. Never invent or estimate a product, price,
store or distance, and never price a not-stocked ingredient from general
knowledge.

1. **A table**, one row per `summary.lines` entry: ingredient, `product`
   (`size`), `store`, `price`. When a line's `match` is "generic", flag it as
   a substitution: descriptor words such as light/dark/toasted were dropped
   to find it (e.g. "light brown sugar" bought as Brown Sugar 1kg), so the
   user should check it suits the recipe.
2. **`summary.total_cost` exactly as returned.** If anything was dropped,
   say it covers the planned lines only, and give the "planned K of N
   ingredients" count from the notes.
3. **The recommended trip** from `summary.trip`: `stores`, `basket_cost`,
   `travel_cost` and its own `total_cost`. The trip prices one realistic
   route at the stores it visits, while `summary.total_cost` buys every
   line at its cheapest store, so the two can differ; say which is which.
   If `trip` is null, say no trip was computed.
4. **Not stocked**: every `summary.not_stocked` entry, with its
   `suggestions` (products the catalog does carry; offer them, never assume
   them).
5. **Out of range**: every `summary.out_of_range` entry with its `reason`.
6. **Provenance**: `summary.origin_status` and, when `summary.coverage` is
   present, `coverage.spend_fraction`, both verbatim as returned. Follow the
   server's `plan_dinner` prompt: never call the basket clean, verified or
   free of a country unless `origin_status` is "verified" and
   `coverage.meets_floor` is true. A line with no evidence is unknown, not
   foreign and not domestic.
7. **Every entry in `summary.notes`** (interpretation, generic matches,
   substitutions, warnings), named rather than summarised away.

## Example

> **User:** What would https://example.com/mapo-tofu cost me within 3 km of
> Granville and Georgia in Vancouver?

1. `python3 scripts/extract_recipe.py "https://example.com/mapo-tofu"`:
   "Mapo Tofu", serves 4, 14 ingredient lines.
2. "Planning 14 ingredients within 3 km; this calls Claude on the pantry
   server and takes up to a minute."
3. `plan_from_text` with that `recipe_text`, `allow_partial` true, the
   intersection's `lat` / `lon`, and `max_km` 3.
4. The reply: the line table (generic matches flagged), `total_cost` for
   the planned lines, the recommended trip, then what was not stocked or
   out of range and why, then `origin_status` and the notes. If something
   was out of range, ask whether to widen `max_km`.
