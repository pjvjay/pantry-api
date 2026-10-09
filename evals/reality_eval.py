"""
Check the catalog against what a real store sells. Writes evals/reports/reality.md.

The demo catalog is synthetic (four made-up Vancouver stores) and nothing measured whether it
looks like a real shelf. It did not: its only yellow onion was American, so "nothing from the
United States" could not plan a bolognese, while a Real Canadian Superstore sells yellow onions
loose and in 2, 3, 10 and 25 lb bags. Per ingredient of the recipe library:

  exclusion   catalog only, always runs: is a candidate left once a common exclusion removes
              what is evidenced as coming from there? None left means no plan with that
              ingredient can honour the exclusion.
  choice      how many products the catalog offers for it, against the reference store.
  price       the median unit price (per kg, L or each) of the catalog's candidates against
              the reference store's: flagged outside a factor of two.
  origin      for each origin a reference row read off a label, whether the catalog offers
              that origin too (a store selling Canadian onions against a catalog of American
              ones fails).

The reference basket (evals/datasets/reference_basket.json) is captured by hand: a store's site
shows packs and prices but rarely origin, and refuses automated reads. The report's "to
capture" section lists the library's ingredients without a reference row yet.

Run:  python -m evals.reality_eval            (no LLM calls, no network)
      python -m evals.reality_eval --strict   (exit 1 when any check fails)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from statistics import median
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
BASKET = HERE / "datasets" / "reference_basket.json"
REPORTS = HERE / "reports"
EXCLUSIONS = ["United States", "China"]
PRICE_FACTOR = 2.0
REQUIRED = ("ingredient", "store", "product", "qty", "uom", "price", "observed_at")


def load_basket(path: Path = BASKET) -> list[dict]:
    """The reference observations, each checked for the fields the checks read."""
    data = json.loads(path.read_text())
    rows = data["observations"]
    for i, row in enumerate(rows):
        missing = [k for k in REQUIRED if row.get(k) in (None, "")]
        if missing:
            raise ValueError(f"{path.name} observation {i} ({row.get('product')!r}) "
                             f"lacks {', '.join(missing)}")
    return rows


def unit_price(price: float | None, qty: float | None, uom: str) -> tuple[float, str] | None:
    """Price per kg, per L or per item; None when the pack size is unknown."""
    if price is None or not qty:
        return None
    if uom == "g":
        return price / qty * 1000, "kg"
    if uom == "ml":
        return price / qty * 1000, "L"
    if uom == "each":
        return price / qty, "each"
    return None


def library_ingredients() -> list[str]:
    """Every ingredient name the recipe library uses, once, in recipe order."""
    from pantry_planner import db

    names: list[str] = []
    for recipe in db.load_all_recipes():
        for ing in recipe.ingredients:
            if ing.name not in names:
                names.append(ing.name)
    return names


@dataclass
class Row:
    ingredient: str
    candidates: list = field(default_factory=list)       # catalog products, store-priced
    without: dict[str, list] = field(default_factory=dict)   # exclusion -> candidates left
    origins: dict = field(default_factory=dict)              # product id -> ProductOrigin
    reference: list[dict] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)

    def unit_prices(self) -> tuple[list[float], str]:
        units: dict[str, list[float]] = {}
        for p in self.candidates:
            up = unit_price(p.store_price if p.store_price is not None else p.price,
                            p.unit_qty, p.unit_uom)
            if up:
                units.setdefault(up[1], []).append(up[0])
        unit = max(units, key=lambda u: len(units[u])) if units else ""
        return units.get(unit, []), unit

    def reference_prices(self, unit: str) -> list[float]:
        out = []
        for r in self.reference:
            up = unit_price(r["price"], r["qty"], r["uom"])
            if up and up[1] == unit:
                out.append(up[0])
        return out


def catalog_rows(ingredients: list[str], exclusions: list[str] = EXCLUSIONS) -> list[Row]:
    """The catalog's candidates per ingredient, and what each exclusion leaves of them."""
    from pantry_planner import flow
    from pantry_planner import origins as origins_mod
    from pantry_planner.config import settings

    cfg = settings()
    recipe = SimpleNamespace(ingredients=[SimpleNamespace(name=n) for n in ingredients])
    pools = flow._ingredient_pools(recipe, cfg.default_lat, cfg.default_lon)
    rows = [Row(ingredient=n, candidates=list(pools.get(i, {}).get("direct", [])))
            for i, n in enumerate(ingredients)]
    origins = origins_mod.resolve_all(sorted({p.id for r in rows for p in r.candidates}))
    for row in rows:
        row.origins = {p.id: origins.get(p.id) for p in row.candidates}
        for country in exclusions:
            kept, _ = origins_mod.filter_pool(row.candidates, exclude=[country], origins=origins)
            row.without[country] = kept
            if row.candidates and not kept:
                row.failures.append(f"no candidate without {country}")
    return rows


def compare(rows: list[Row], observations: list[dict]) -> None:
    """Add the reference store's rows and the checks that need them."""
    by_name = {r.ingredient.lower(): r for r in rows}
    for obs in observations:
        row = by_name.get(obs["ingredient"].lower())
        if row is not None:
            row.reference.append(obs)
    for row in rows:
        if not row.reference:
            continue
        prices, unit = row.unit_prices()
        ref = row.reference_prices(unit)
        if prices and ref:
            ratio = median(prices) / median(ref)
            if not 1 / PRICE_FACTOR <= ratio <= PRICE_FACTOR:
                row.failures.append(f"median price {ratio:.1f}x the reference")
        offered = {o.country for o in row.origins.values()
                   if o is not None and o.status == "resolved" and o.country}
        for origin in sorted({o["origin"] for o in row.reference if o.get("origin")}):
            if origin not in offered:
                row.failures.append(f"the reference store sells it from {origin}; "
                                    "the catalog does not")


def _money(v: float | None, unit: str) -> str:
    return "–" if v is None else f"${v:.2f}/{unit}"


def _range(values: list[float], unit: str) -> str:
    if not values:
        return "–"
    lo, hi = min(values), max(values)
    return _money(lo, unit) if lo == hi else f"${lo:.2f}–{hi:.2f}/{unit}"


def report(rows: list[Row], observations: list[dict], exclusions: list[str] = EXCLUSIONS) -> str:
    stores = sorted({f"{o['store']}, {o.get('branch') or 'branch not recorded'}"
                     for o in observations})
    days = sorted({o["observed_at"] for o in observations})
    referenced = [r for r in rows if r.reference]
    failing = [r for r in rows if r.failures]
    out = [
        "# Reality check: the catalog against a real store",
        "",
        f"Generated {date.today().isoformat()} by `python -m evals.reality_eval` from the "
        "database it ran against and `evals/datasets/reference_basket.json`. No LLM calls, "
        "no network.",
        "",
        f"- **Reference:** {len(observations)} observation(s) of {len(referenced)} of the "
        f"library's {len(rows)} ingredients"
        + (f", at {'; '.join(stores)}, {', '.join(days)}" if observations else ""),
        f"- **Failing:** {len(failing)} of {len(rows)} ingredients",
    ]
    for country in exclusions:
        names = [r.ingredient for r in rows if r.candidates and not r.without[country]]
        out.append(f"- **Without {country}:** "
                   + (f"{len(names)} ingredient(s) have no candidate left: {', '.join(names)}"
                      if names else "every ingredient keeps a candidate"))
    single = [r.ingredient for r in rows if len(r.candidates) == 1]
    out.append(f"- **One product only:** {len(single)} ingredient(s)"
               + (f": {', '.join(single)}" if single else ""))
    out += ["", "## Per ingredient", "",
            "| Ingredient | Catalog options | Reference options | Catalog unit price | "
            "Reference unit price | " + " | ".join(f"Without {c}" for c in exclusions)
            + " | Result |",
            "|---" * (6 + len(exclusions)) + "|"]
    for r in rows:
        prices, unit = r.unit_prices()
        ref = r.reference_prices(unit)
        left = [f"{len(r.without[c])} left" if r.candidates else "–" for c in exclusions]
        # A name no product carries every word of ("Peanut Butter and Jelly Jam") is a recipe
        # wording to fix, not a shelf gap: the classic selector picks from the whole catalog.
        result = "; ".join(r.failures) or ("ok" if r.candidates else "no product matches this name")
        out.append(f"| {r.ingredient} | {len(r.candidates)} | "
                   f"{len(r.reference) if r.reference else '–'} | {_range(prices, unit)} | "
                   f"{_range(ref, unit)} | " + " | ".join(left) + f" | {result} |")
    todo = [r.ingredient for r in rows if not r.reference]
    origin_todo = sorted({o["ingredient"] for o in observations if not o.get("origin")})
    out += ["", "## To capture", "",
            "Ingredients with no reference row yet: " + (", ".join(todo) or "none") + ".",
            "",
            "Reference rows without an origin (read it off the label or shelf sign): "
            + (", ".join(origin_todo) or "none") + ".",
            "",
            "A row is one product on one shelf on one day: `ingredient` (the library's name), "
            "`store`, `branch`, `product`, `size`, `qty` and `uom` (g, ml or each), `price` "
            "(the shelf price that day), `regular_price` when on sale, `origin` and "
            "`origin_source` (\"label\" or \"shelf sign\") or null, `observed_at`, `link`."]
    return "\n".join(out) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--basket", type=Path, default=BASKET)
    parser.add_argument("--out", type=Path, default=REPORTS / "reality.md")
    parser.add_argument("--strict", action="store_true", help="exit 1 when any check fails")
    args = parser.parse_args(argv)
    observations = load_basket(args.basket)
    rows = catalog_rows(library_ingredients())
    compare(rows, observations)
    text = report(rows, observations)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text)
    print(text)
    return 1 if args.strict and any(r.failures for r in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
