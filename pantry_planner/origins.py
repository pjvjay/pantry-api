"""
Provenance: evidence in, ranking out.

Origin is never inferred here. Evidence arrives from the companion
claude-chrome-container tooling (Open Food Facts lookups, package-label
vision reads, retailer pages) via ingest.py, and this module does three
things with it:

  resolve_origin()  collapse a product's evidence rows into one summary,
                    keeping `conflicting` as a status rather than picking
                    a winner between sources that disagree
  rank_products()   place products against a CALLER-SUPPLIED country
                    preference order — no policy list lives in this repo
  triage_*          use the old name/brand heuristics for the only thing
                    they are fit for: choosing what to photograph next

Two rules the shape of this module exists to enforce:

1. A product with no evidence is never ranked. Coverage is thin and biased
   (measured: European imports resolve, Canadian goods mostly do not), so
   sorting "nobody published this" alongside real findings would silently
   turn absence into a verdict. Those products go to `unranked` with a
   reason, and the count is always reported.

2. Exclusion is checked against ingredient origin AND manufacturing origin.
   "Made in Canada" legally permits imported ingredients — Kraft Smooth
   Peanut Butter is American peanuts processed in Canada — so a filter that
   reads only the manufacturing country passes exactly the products a
   provenance-conscious shopper is trying to avoid.
"""
from __future__ import annotations

import re
import unicodedata

from .models import (
    ExcludedProduct,
    OriginRanking,
    Product,
    ProductOrigin,
    RankedProduct,
    UnrankedProduct,
)

# ─── Claim strength ──────────────────────────────────────────
# Under CFIA rules these are different facts, and the difference is the
# whole point of ranking by provenance rather than by country:
#   "Product of Canada"  >=98% Canadian content, ingredients included
#   "Made in Canada"     last substantial transformation here; ingredients
#                        may be imported and must be qualified as such
FULL_CLAIMS = {"product-of", "grown-in", "farmed-in", "harvested-in", "caught-in"}
PROCESSING_CLAIMS = {"made-in", "prepared-in", "packaged-in"}
NON_CLAIMS = {"unknown", "conflicting", "none", "imported"}

CONF_ORDER = {"high": 3, "medium": 2, "low": 1}


# ─── Country matching ────────────────────────────────────────
# Evidence is free text written by contributors and label transcribers:
# "Quebec, Canada", "New Hampshire, Stratham, USA", "Made in Italy". So
# matching is by surface form, but with word boundaries — a naive substring
# test makes "us" match "Australia" and "Aus" match nothing useful.

_ALIASES: dict[str, set[str]] = {
    "united states": {
        "united states", "united states of america", "usa", "u.s.a", "u.s",
        "us",
        # "america"/"american" are deliberately absent: they match
        # "South America" and "Central America", which are not the US.
        # Sub-national names appear in label transcriptions ("New Hampshire,
        # Stratham"). Georgia is deliberately absent — it is also a country.
        "alabama", "alaska", "arizona", "arkansas", "california", "colorado",
        "connecticut", "delaware", "florida", "hawaii", "idaho", "illinois",
        "indiana", "iowa", "kansas", "kentucky", "louisiana", "maine",
        "maryland", "massachusetts", "michigan", "minnesota", "mississippi",
        "missouri", "montana", "nebraska", "nevada", "new hampshire",
        "new jersey", "new mexico", "new york", "north carolina",
        "north dakota", "ohio", "oklahoma", "oregon", "pennsylvania",
        "rhode island", "south carolina", "south dakota", "tennessee",
        "texas", "utah", "vermont", "virginia", "washington", "west virginia",
        "wisconsin", "wyoming",
    },
    "canada": {
        "canada", "canadian",
        "alberta", "british columbia", "manitoba", "new brunswick",
        "newfoundland", "nova scotia", "prince edward island", "quebec",
        "saskatchewan",
        # "Ontario" is omitted: it is also a city in California, and an
        # origin string naming it alone is not worth a false Canadian match.
    },
    "united kingdom": {
        "united kingdom", "uk", "u.k", "great britain", "britain",
        "england", "scotland", "wales", "northern ireland",
        # "british" is deliberately absent: it matches "British Columbia".
    },
    "netherlands": {"netherlands", "holland", "dutch"},
    "south korea": {"south korea", "korea, south", "republic of korea"},
}

# Multi-word country names whose components are themselves countries or
# regions. A shorter name matching INSIDE one of these is a false positive:
# "Guinea" inside "Papua New Guinea", "Ireland" inside "Northern Ireland".
_SUPERSETS = (
    "papua new guinea", "equatorial guinea", "guinea-bissau",
    "northern ireland", "south africa", "south korea", "north korea",
    "south sudan", "dominican republic", "trinidad and tobago",
    "central african republic", "united arab emirates", "new zealand",
    "united states", "united kingdom", "south america", "central america",
    "british columbia", "west virginia", "new hampshire", "new jersey",
    "new mexico", "new york", "north carolina", "south carolina",
    "north dakota", "south dakota", "rhode island",
)

# surface form -> canonical country
_SURFACE_TO_CANON: dict[str, str] = {}
for _canon, _forms in _ALIASES.items():
    for _f in _forms:
        _SURFACE_TO_CANON[_f] = _canon
    _SURFACE_TO_CANON[_canon] = _canon


def _normalize(text: str) -> str:
    """Lowercase, strip accents, drop periods, collapse whitespace.

    Periods are removed rather than replaced with a space so that "U.S.A."
    folds to "usa"; replacing them produced "u s a", which matched nothing.
    Accent folding lets "M\u00e9xico" and "Per\u00fa" match.
    """
    t = unicodedata.normalize("NFKD", (text or "").lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", t.replace(".", "")).strip()


def canonical_country(name: str) -> str:
    """Fold a surface form to a canonical country name, lowercased."""
    n = _normalize(name)
    return _SURFACE_TO_CANON.get(n, n)


def _forms_for(country: str) -> set[str]:
    canon = canonical_country(country)
    return _ALIASES.get(canon, {canon})


def country_matches(evidence_text: str, country: str) -> bool:
    """Does free-text origin evidence name this country?

    Word-boundary matched, so "us" does not fire inside "Australia" and
    "india" does not fire inside "indiana".
    """
    if not evidence_text or not country:
        return False
    hay = _normalize(evidence_text)
    forms = {_normalize(f) for f in _forms_for(country)}
    forms.add(canonical_country(country))
    # Spans covered by a longer name that is NOT one of this country's own
    # forms. "Guinea" must not fire inside "Papua New Guinea", but
    # "New Hampshire" is itself a US form and must still count as the US.
    blocked = [
        (m.start(), m.end())
        for s in _SUPERSETS if s not in forms
        for m in re.finditer(rf"(?<![a-z]){re.escape(s)}(?![a-z])", hay)
    ]
    for form in forms:
        f = _normalize(form)
        for m in re.finditer(rf"(?<![a-z]){re.escape(f)}(?![a-z])", hay):
            if not any(b0 <= m.start() and m.end() <= b1 for b0, b1 in blocked):
                return True
    return False


# ─── Evidence -> resolved summary ────────────────────────────

def _conf_label(value: float) -> str:
    return "high" if value >= 0.8 else "medium" if value >= 0.5 else "low"


def _conf_float(label: str) -> float:
    return {"high": 0.9, "medium": 0.6, "low": 0.3}.get(label, 0.3)


def _row_countries(row, field: str) -> frozenset[str]:
    """Countries named by ONE evidence row's field.

    Split on separators only. " and " is deliberately not a separator:
    it would break "Trinidad and Tobago".
    """
    val = (getattr(row, field, "") or "").strip()
    if not val:
        return frozenset()
    return frozenset(
        canonical_country(part) for part in re.split(r"[,;/]", val) if part.strip())


def _countries_in(rows, field: str) -> set[str]:
    out: set[str] = set()
    for r in rows:
        out |= _row_countries(r, field)
    return out


def _rows_disagree(rows, field: str) -> bool:
    """True when two rows name different countries for the same field.

    A single row listing four countries is a blended product (Bertolli
    olive oil genuinely lists Italy, Spain, Argentina and Peru) — that is
    one source being precise, not two sources conflicting.
    """
    seen = [c for c in (_row_countries(r, field) for r in rows) if c]
    return any(a != b for a in seen for b in seen)


def resolve_origin(product_id: int, product_name: str, evidence: list
                   ) -> ProductOrigin:
    """Collapse one product's evidence into a summary.

    Disagreement is reported, not resolved: if two sources name different
    manufacturing countries the status is `conflicting` and nothing is
    ranked. Picking a winner here would be inventing a fact.
    """
    usable = [e for e in evidence
              if e.source != "guess"
              and (e.claim_type or "unknown") not in NON_CLAIMS
              and not e.importer_only
              and ((e.manufactured_in or "").strip()
                   or (e.ingredient_origin or "").strip())]

    if not usable:
        has_guess = any(e.source == "guess" for e in evidence)
        failed = any((e.claim_type or "") == "lookup_failed" for e in evidence)
        status = ("lookup_failed" if failed
                  else "guess" if has_guess
                  else "unknown")
        note = ""
        if has_guess:
            g = next(e for e in evidence if e.source == "guess")
            note = f"heuristic guess only: {g.note or g.manufactured_in}"
        return ProductOrigin(
            product_id=product_id, product_name=product_name, status=status,
            evidence_count=len(evidence), note=note)

    mfg = _countries_in(usable, "manufactured_in")
    ing = _countries_in(usable, "ingredient_origin")
    if _rows_disagree(usable, "manufactured_in") or _rows_disagree(
            usable, "ingredient_origin"):
        return ProductOrigin(
            product_id=product_id, product_name=product_name,
            status="conflicting", evidence_count=len(evidence),
            ingredient_origin=", ".join(c.title() for c in sorted(ing)),
            manufactured_in=", ".join(c.title() for c in sorted(mfg)),
            note="sources disagree; not ranked")

    # Strongest claim wins as the representative row: a transcribed label
    # beats a crowd-sourced database record at equal confidence.
    def strength(e):
        return (
            1 if (e.claim_type or "") in FULL_CLAIMS else 0,
            CONF_ORDER.get(e.confidence or "low", 1),
            1 if e.source == "label-photo" else 0,
        )

    best = max(usable, key=strength)
    # Fields are merged across rows, not read off the strongest one. One
    # source often knows only where the ingredients came from and another
    # only where it was processed; keeping just the winner's fields drops
    # half the provenance and lets an excluded ingredient origin through.
    ranked_rows = sorted(usable, key=strength, reverse=True)
    merged_ing = next((r.ingredient_origin.strip() for r in ranked_rows
                       if (r.ingredient_origin or "").strip()), "")
    merged_mfg = next((r.manufactured_in.strip() for r in ranked_rows
                       if (r.manufactured_in or "").strip()), "")
    return ProductOrigin(
        product_id=product_id, product_name=product_name, status="resolved",
        claim_type=best.claim_type or "unknown",
        country=(merged_mfg or merged_ing),
        ingredient_origin=merged_ing,
        manufactured_in=merged_mfg,
        verbatim=best.verbatim or "", confidence=best.confidence or "low",
        source=best.source, note=best.note or "", evidence_count=len(evidence))


def resolve_all(product_ids: list[int] | None = None) -> dict[int, ProductOrigin]:
    """Resolved summaries for the given products (or the whole catalog)."""
    from . import db

    products = {p.id: p for p in db.load_all_products()}
    if product_ids is not None:
        products = {k: v for k, v in products.items() if k in product_ids}
    evidence = db.load_origin_evidence(list(products))
    return {
        pid: resolve_origin(pid, p.name, evidence.get(pid, []))
        for pid, p in products.items()
    }


# ─── Ranking ─────────────────────────────────────────────────

def _match_preference(origin: ProductOrigin, preference: list[str]
                      ) -> tuple[int, str, str] | None:
    """First preference entry this product's origin satisfies."""
    for i, country in enumerate(preference):
        if country_matches(origin.manufactured_in, country):
            return i, country, "manufactured_in"
        if country_matches(origin.ingredient_origin, country):
            return i, country, "ingredient_origin"
    return None


def _match_exclusion(origin: ProductOrigin, exclude: list[str]
                     ) -> tuple[str, str] | None:
    """Excluded country evidenced anywhere in this product's provenance.

    Both fields are checked. A product manufactured in Canada from American
    ingredients matches an exclusion of the United States — that is the
    intended behaviour, not an over-reach.
    """
    for country in exclude:
        if country_matches(origin.ingredient_origin, country):
            return country, "ingredient_origin"
        if country_matches(origin.manufactured_in, country):
            return country, "manufactured_in"
    return None


def rank_products(products: list[Product], *, preference: list[str] | None = None,
                  exclude: list[str] | None = None,
                  origins: dict[int, ProductOrigin] | None = None
                  ) -> OriginRanking:
    """Rank products against a caller-supplied country preference order.

    `preference` is ordered, most-preferred first. `exclude` removes
    products with positive evidence of an excluded origin — it never
    removes a product for lacking evidence.
    """
    preference = [p for p in (preference or []) if p.strip()]
    exclude = [e for e in (exclude or []) if e.strip()]
    if origins is None:
        origins = resolve_all([p.id for p in products])

    ranked: list[RankedProduct] = []
    excluded: list[ExcludedProduct] = []
    unranked: list[UnrankedProduct] = []

    for p in products:
        origin = origins.get(p.id) or ProductOrigin(
            product_id=p.id, product_name=p.name, status="unknown")

        if origin.status != "resolved":
            reason = {"unknown": "no_evidence", "conflicting": "conflicting",
                      "lookup_failed": "lookup_failed", "guess": "guess_only",
                      }.get(origin.status, "no_evidence")
            detail = origin.note
            # Unresolved is not the same as clean. If any evidence names an
            # excluded country — a conflicting record, say — surface it here
            # rather than letting the product pass unremarked.
            flagged = _match_exclusion(origin, exclude)
            if flagged:
                detail = (f"WARNING: some evidence names {flagged[0]} "
                          f"({flagged[1]}), but sources are unresolved. "
                          f"{detail}").strip()
            unranked.append(UnrankedProduct(
                product_id=p.id, product_name=p.name, price=p.price,
                reason=reason, detail=detail))
            continue

        hit = _match_exclusion(origin, exclude)
        if hit:
            country, field = hit
            excluded.append(ExcludedProduct(
                product_id=p.id, product_name=p.name, price=p.price,
                excluded_country=country, matched_field=field,
                claim_type=origin.claim_type, verbatim=origin.verbatim,
                confidence=origin.confidence))
            continue

        pref = _match_preference(origin, preference)
        full = origin.claim_type in FULL_CLAIMS
        if pref is not None:
            idx, country, field = pref
            rank = idx * 2 + (0 if full else 1)
            qualifier = ("origin" if full
                         else "processed here, ingredients may be imported")
            label = f"{country} — {qualifier}"
            matched_country, matched_field = country, field
        else:
            rank = len(preference) * 2
            label = "Other country"
            matched_country, matched_field = "", ""

        ranked.append(RankedProduct(
            product_id=p.id, product_name=p.name, price=p.price, rank=rank,
            tier_label=label, origin=origin,
            matched_country=matched_country, matched_field=matched_field))

    ranked.sort(key=lambda r: (r.rank, r.price, r.product_name))
    excluded.sort(key=lambda e: e.product_name)
    unranked.sort(key=lambda u: (u.reason, u.product_name))

    total = len(products)
    note = (f"{len(ranked)} of {total} products carry origin evidence. "
            f"{len(unranked)} are unverified and are NOT ranked — no source "
            f"published an origin for them, which is not evidence that they "
            f"are foreign or domestic.")
    return OriginRanking(
        preference=preference, exclude=exclude, ranked=ranked,
        excluded=excluded, unranked=unranked,
        counts={"total": total, "ranked": len(ranked),
                "excluded": len(excluded), "unranked": len(unranked)},
        coverage_note=note)


# ─── Triage: what is worth photographing next ────────────────
# The old keyword rules live on here and nowhere else. As origin they are
# brand-nationality and category inference — the reconciliation pass in the
# companion tooling found Lindt Excellence 70% is made in New Hampshire, not
# the France or Switzerland such a rule would confidently produce. As a
# *work queue* they are fine: a guess that something is imported is a good
# reason to go read its label.

_TRIAGE_HINTS: list[tuple[tuple[str, ...], str]] = [
    (("cadbury",), "British brand — verify, brands manufacture across regions"),
    (("basmati",), "Basmati is grown in India/Pakistan — verify which"),
    (("garam masala", "turmeric", "cumin", "coriander"), "Likely imported spice"),
    (("parmesan", "parmigiano", "passata"), "Italian designation — verify"),
    (("olive oil",), "Blended oils are often multi-country — verify"),
    (("soy sauce", "coconut milk", "coconut yogurt"), "Likely imported"),
    (("shrimp", "prawn", "tilapia", "salmon", "haddock", "tuna"),
     "Seafood origin is not published online — needs a label photo"),
    (("rice", "pasta", "chocolate"), "Commonly imported category"),
]


def triage_candidates(products: list[Product],
                      origins: dict[int, ProductOrigin] | None = None
                      ) -> list[dict]:
    """Products worth spending a label photograph on, most useful first.

    Returns hints, never origins. Anything already resolved is skipped.
    """
    if origins is None:
        origins = resolve_all([p.id for p in products])
    out = []
    for p in products:
        o = origins.get(p.id)
        if o and o.status in {"resolved", "conflicting"}:
            continue
        hay = f"{p.name} {p.brand} {p.subcategory or ''}".lower()
        for keywords, why in _TRIAGE_HINTS:
            if any(k in hay for k in keywords):
                out.append({"product_id": p.id, "product_name": p.name,
                            "reason": why,
                            "status": o.status if o else "unknown"})
                break
    return out
