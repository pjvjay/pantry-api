"""
Measure origin resolution quality. Writes evals/reports/origins.md.

Until this existed there was no number at all for the feature the whole
provenance stack rests on. origins.py's own docstring asserts as fact that
"European imports resolve, Canadian goods mostly do not" — that was a
comment, not a measurement.

Four axes, because they fail differently and a single accuracy figure
would hide the one that matters:

  coverage        of the catalog, how much has usable evidence at all
  accuracy        of what resolved, how much names the right country
  holdout         of what SHOULD be held out (unknown/conflicting), how
                  much correctly was — guessing scores zero here
  exclusion       recall and precision of the filter itself:
                    recall    — of products that really are from X, how
                                many the filter catches
                    precision — of products the filter removes, how many
                                really are from X

The asymmetry is the point. For a boycott filter a false PASS (an
American product that ships) and a false EXCLUDE (a Canadian product
wrongly removed) have completely different costs to the user, so they are
never averaged into one number.

Run:  python -m evals.origin_eval        (no LLM calls, no network)
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATASET = HERE / "datasets" / "origins.jsonl"
REPORTS = HERE / "reports"


def load_cases() -> list[dict]:
    with DATASET.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def score(cases: list[dict], resolved: dict) -> dict:
    """Score resolved origins against ground truth."""
    from pantry_planner.origins import country_matches

    total = len(cases)
    should_resolve = [c for c in cases if c["expected_status"] == "resolved"]
    should_hold = [c for c in cases if c["expected_status"] != "resolved"]

    got_resolved = [c for c in cases
                    if (o := resolved.get(c["product_id"])) and o.status == "resolved"]
    coverage = len(got_resolved) / total if total else 0.0

    # Accuracy: of the ones we resolved AND that should resolve, is the
    # country right? Both fields are checked — a product read as simply
    # "Canada" when its ingredients are American is wrong, not partial.
    correct = 0
    for c in should_resolve:
        o = resolved.get(c["product_id"])
        if not o or o.status != "resolved":
            continue
        mfg_ok = (not c["truth_manufactured_in"]
                  or country_matches(o.manufactured_in, c["truth_manufactured_in"]))
        ing_truth = c["truth_ingredient_origin"]
        ing_ok = (not ing_truth
                  or country_matches(o.ingredient_origin, ing_truth)
                  or country_matches(o.manufactured_in, ing_truth))
        if mfg_ok and ing_ok:
            correct += 1
    resolvable_hits = [c for c in should_resolve
                       if (o := resolved.get(c["product_id"])) and o.status == "resolved"]
    accuracy = correct / len(resolvable_hits) if resolvable_hits else 0.0

    # Holdout: cases that must NOT resolve. Inventing an answer here is the
    # failure the module exists to prevent, so it is scored separately.
    held = sum(1 for c in should_hold
               if not (o := resolved.get(c["product_id"])) or o.status != "resolved")
    holdout = held / len(should_hold) if should_hold else 1.0

    return {
        "total": total,
        "coverage": coverage,
        "resolved_count": len(got_resolved),
        "accuracy": accuracy,
        "accuracy_denominator": len(resolvable_hits),
        "holdout": holdout,
        "holdout_denominator": len(should_hold),
    }


def score_exclusion(cases: list[dict], resolved: dict, country: str) -> dict:
    """Recall/precision of the exclusion filter for one country."""
    from pantry_planner.origins import _match_exclusion

    truly = {c["product_id"] for c in cases
             if country.lower() in (c["truth_manufactured_in"].lower()
                                    + " " + c["truth_ingredient_origin"].lower())}
    flagged = {pid for pid, o in resolved.items()
               if o.status == "resolved" and _match_exclusion(o, [country])}
    tp = len(truly & flagged)
    recall = tp / len(truly) if truly else 1.0
    precision = tp / len(flagged) if flagged else 1.0
    return {"country": country, "truly": len(truly), "flagged": len(flagged),
            "recall": recall, "precision": precision,
            "missed": sorted(truly - flagged), "false": sorted(flagged - truly)}


def main() -> int:
    from pantry_planner.origins import resolve_all

    cases = load_cases()
    resolved = resolve_all([c["product_id"] for c in cases])
    s = score(cases, resolved)
    us = score_exclusion(cases, resolved, "United States")

    REPORTS.mkdir(exist_ok=True)
    lines = [
        "# Origin resolution eval",
        "",
        "Measured, not asserted. No LLM calls, no network — this scores the",
        "evidence currently in the database, so a low coverage number is a",
        "statement about the corpus, not about the resolver.",
        "",
        f"- **Coverage**: {s['coverage']:.0%} "
        f"({s['resolved_count']} of {s['total']} products have usable evidence)",
        f"- **Accuracy**: {s['accuracy']:.0%} of {s['accuracy_denominator']} resolved "
        f"products name the right country",
        f"- **Holdout**: {s['holdout']:.0%} of {s['holdout_denominator']} products that "
        f"SHOULD be held out (unknown/conflicting) correctly were",
        "",
        "## Exclusion filter — United States",
        "",
        "Recall and precision are reported separately and never averaged: a",
        "false pass (an American product that ships) and a false exclude (a",
        "domestic product wrongly removed) cost the user different things.",
        "",
        f"- Recall: {us['recall']:.0%} — of {us['truly']} genuinely US-linked "
        f"products, {us['truly'] - len(us['missed'])} were caught",
        f"- Precision: {us['precision']:.0%} — of {us['flagged']} flagged, "
        f"{us['flagged'] - len(us['false'])} genuinely are",
    ]
    if us["missed"]:
        lines.append(f"- **Missed (false pass)**: product ids {us['missed']}")
    if us["false"]:
        lines.append(f"- **Wrongly excluded**: product ids {us['false']}")
    lines += ["", "## Per-case", "",
              "| Product | Expected | Got | Country | OK |", "|---|---|---|---|---|"]
    for c in cases:
        o = resolved.get(c["product_id"])
        got = o.status if o else "missing"
        ok = "yes" if (got == c["expected_status"]
                       or (c["expected_status"] != "resolved" and got != "resolved")) else "NO"
        country = (o.manufactured_in or o.ingredient_origin or "—") if o else "—"
        lines.append(f"| {c['product_name']} | {c['expected_status']} | {got} | "
                     f"{country} | {ok} |")

    report = "\n".join(lines) + "\n"
    (REPORTS / "origins.md").write_text(report)
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
