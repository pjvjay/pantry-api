"""Matching by meaning: which products ARE an ingredient, which work instead, and how close each
one is, as grocery platforms label query-product pairs (Amazon's ESCI: Exact, Substitute,
Complement, Irrelevant) rather than by the words two names share.

An ingredient name is first understood (meaning_of / meanings): which product types are the
same item (exact), which work instead in cooking (substitutes, best first) and which only look
related (unrelated: "peanut butter" for "butter"), and the form and variant the name asks for.

  seed        a name of the library or the demo starters: ingredient_types.json, labelled by
              Gemini once, offline
  embedding   any other name (imported, pasted, written by the assistant): embedded with
              EmbeddingGemma and compared with every product type's centroid. The nearest type
              at or above EXACT_FLOOR is the same item (with any type within TIE of it); a
              form the name asks for (asked_form: "cumin powder") prefers a type within
              FORM_TIE of the nearest whose products have it. What works instead is a seed's
              substitutes when a seed ingredient names the same type as the same item, else
              the other types at or above TYPE_FLOOR (at most MAX_SUBSTITUTES). Nothing at or
              above TYPE_FLOOR: the name is not stocked. The variant it asks for is the
              longest variant of a candidate type's products all of whose words it has
              (asked_variant: "extra-virgin olive oil" asks for extra virgin).
  words       no seed knows the name and the embedder cannot be reached: the caller matches it
              by today's words (nlsearch's token levels), and says so (matching "words")

The floors were set from measured cosines (docs/meal-planning.md, "Matching by meaning"): a
name and its own product type score 0.53-0.71, and an unstocked name's nearest type 0.37-0.59,
so a weak nearest type is offered as working instead, never as the same item.

Candidates (candidates): every product whose type is exact (Same item) or a substitute (Works
instead), and the LOOSER_K products nearest the name's vector at or above LOOSER_FLOOR whose
type is neither unrelated nor already in a group (Looser match). Only products with a profile
made from their current catalog row are matched by meaning (seeds.py); one the catalog gained
since the seeds were made is matched by its words until it has one: when every word of the
name is among its own (its name and description, as the word index reads them) it is the same
item, ranked after the profiled ones (relevance 0).

Within a group, products are ordered by a documented weighted sum (score, WEIGHTS), higher
first, ties by catalog id:

  relevance     the cosine of the name's vector and the product's profile vector; for a
                product that works instead, scaled down 10% per place its type is behind the
                best substitute (SUBSTITUTE_STEP)
  form_variant  the share of the form and variant the name asks for that the product has
                (1 when it asks for none)
  pack_fit      the packs against the recipe's need: covering it exactly 1, falling to 0.6
                at CATERING x the need; a catering pack (more than that) 0.25, short 0.3,
                unknown (no need, or another unit) 0.5. When even the group's smallest pack
                is over CATERING x the need (a tablespoon of oil, 5 g of salt), the cook buys
                a normal pack: those up to CATERING x the smallest are 0.6, as the word
                planner's size cap scales to the smallest pack
  value         the lowest unit price among the group's products over this one's (1 for the
                best value, 0.5 when it has no unit price)
  origin        the shopper's origin preference: 1 for the first preferred country, falling
                to 0 for none; 0 for every product when there is no preference
  trip          what choosing it does to the trip total: 1 for the group's cheapest trip,
                falling by TRIP_SPAN dollars to 0; 0 when the trip is not worked out. The
                planner's first pick is made before any trip, so it reads what the product
                adds to the trip's spend instead (its price at its cheapest store in range):
                the trip total moves by exactly that whenever the swap changes no stop

The planner's first pick takes the first product of this order that may be bought (rank_pool;
demo mode's selector and the meal plan's catalog and basket picks), and Options shows the same
order with the trip filled in, so the plan and its alternatives agree.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Literal

from ..models import Product
from ..nlsearch.units import index_text, tokens
from . import cache, describe, embed, seeds

Group = Literal["same_item", "works_instead", "looser_match"]
GROUPS: tuple[Group, ...] = ("same_item", "works_instead", "looser_match")
GROUP_LABELS = {"same_item": "Same item", "works_instead": "Works instead",
                "looser_match": "Looser match"}
Source = Literal["seed", "described", "embedding", "words"]

EXACT_FLOOR = 0.56
TYPE_FLOOR = 0.50
TIE = 0.01
MAX_SUBSTITUTES = 3
LOOSER_K = 20
LOOSER_FLOOR = 0.47
SUBSTITUTE_STEP = 0.10
# The floors above are EmbeddingGemma's (seeds' "profile" vectors). Gemini's identity-string
# vectors sit higher and closer together, so they have their own, calibrated on the 47 seed
# names only (docs/meal-planning.md, "Describing names no seed knows"); CHECK is the least
# cosine a described name's identity string must have with its Same item type's centroid, or the
# type is offered as working instead rather than as the same item.
GEMINI_FLOORS = {"exact": 0.0, "type": 0.0, "looser": 0.0, "check": 0.0}


def floors() -> dict[str, float]:
    """The floors for the vectors the seeds hold (seeds.load().model)."""
    if seeds.load().model.startswith("gemini"):
        return dict(GEMINI_FLOORS)
    return {"exact": EXACT_FLOOR, "type": TYPE_FLOOR, "looser": LOOSER_FLOOR, "check": 0.0}
CATERING = 6            # as the word planner's SIZE_CAP_FACTOR
TRIP_SPAN = 10.0        # dollars of trip total over which `trip` falls from 1 to 0

WEIGHTS = {"relevance": 0.30, "form_variant": 0.15, "pack_fit": 0.20, "value": 0.15,
           "origin": 0.10, "trip": 0.10}
# Forms one name and another product call the same thing.
SAME_FORM = {"powder": "ground", "powdered": "ground", "ground": "ground", "dry": "dried"}
# The forms an embedded name may ask for in its words (the profiles' forms a recipe names;
# not "whole", which is whole milk and whole wheat as often as a whole spice).
ASKED_FORMS = frozenset({"fresh", "frozen", "canned", "dried", "dry", "ground", "powder",
                         "powdered", "sliced", "shredded", "flakes", "paste"})
FORM_TIE = 0.05


# ─── What a name means ───────────────────────────────────────

@dataclass(frozen=True)
class Meaning:
    """What an ingredient name is, for matching: see the module docstring. `nearest` is, for
    an embedded name, its nearest product type and cosine; `why_words` says why a name fell
    back to words."""
    name: str
    source: Source
    exact: tuple[str, ...] = ()
    substitutes: tuple[str, ...] = ()
    unrelated: tuple[str, ...] = ()
    form: str = ""
    variant: str = ""
    vector: tuple[float, ...] | None = None
    nearest: tuple[str, float] | None = None
    why_words: str = ""

    @property
    def matching(self) -> Literal["meaning", "words"]:
        return "words" if self.source == "words" else "meaning"

    @property
    def stocked(self) -> bool:
        """Some product type in the catalog is the same item or works instead."""
        have = seeds.load().types()
        return any(t in have for t in (*self.exact, *self.substitutes))


def _label(name: str, form: str | None) -> str:
    return f"{form} {name}" if form and form.lower() not in name.lower() else name


def _seed(name: str, form: str | None) -> seeds.SeedIngredient | None:
    known = seeds.load().ingredients
    for text in dict.fromkeys([_label(name, form), name]):
        hit = known.get(seeds.ingredient_key(text))
        if hit is not None:
            return hit
    return None


def meanings(items: list[tuple[str, str | None]]) -> list[Meaning]:
    """(name, form) -> Meaning, for every item: seeds first, then every other name embedded
    in one request. When the embedder cannot be reached (or there is none), those names are
    Meaning(source="words") with the reason."""
    s = seeds.load()
    out: list[Meaning | None] = []
    todo: dict[str, list[int]] = {}
    for i, (name, form) in enumerate(items):
        hit = _seed(name, form)
        if hit is not None:
            out.append(Meaning(name=name, source="seed", exact=hit.exact_types,
                               substitutes=hit.substitute_types, unrelated=hit.unrelated_types,
                               form=hit.form or (form or "").lower(), variant=hit.variant,
                               vector=hit.vector))
        else:
            out.append(None)
            todo.setdefault(_label(name, form), []).append(i)
    if todo:
        why = ""
        identities: dict[str, describe.Identity] = {}
        describer = describe.get_describer()
        if describer is not None and s.profiles:
            try:
                identities = describer.describe(list(todo))
            except describe.DescriberUnavailableError:
                identities = {}         # embedded as they are, below
        vectors: dict[str, list[float]] = {}
        embedder = embed.get_embedder()
        if not s.centroids:
            why = "the matching seeds are missing"
        elif embedder is None:
            why = "no embedding model is configured (OLLAMA_URL is empty)"
        else:
            # what is embedded: a described name's identity string when the seeds hold identity
            # vectors ("butter | block | salted"), else the name as written
            query = {text: (identities[text].text if text in identities and s.text == "identity"
                            else text) for text in todo}
            try:
                agrees, why = _agrees(embedder)
                if agrees:
                    asked = list(dict.fromkeys(query.values()))
                    got = dict(zip(asked, embedder.embed(asked, "query"), strict=True))
                    vectors = {text: got[q] for text, q in query.items()}
            except embed.EmbedderUnavailableError as e:
                why = str(e)
        for text, idxs in todo.items():
            for i in idxs:
                name, form = items[i]
                if text in identities:
                    out[i] = _described(name, form, identities[text], vectors.get(text))
                elif text in vectors:
                    out[i] = _embedded(name, form, vectors[text])
                else:
                    out[i] = Meaning(name=name, source="words", form=(form or "").lower(),
                                     why_words=why)
    return [m for m in out if m is not None]


def _agrees(embedder) -> tuple[bool, str]:
    """Whether `embedder` reproduces the seeds' vectors (OllamaEmbedder.agrees_with_seeds); an
    embedder with no such check (a test's) is taken at its word."""
    check = getattr(embedder, "agrees_with_seeds", None)
    return check() if check is not None else (True, "")


def meaning_of(name: str, form: str | None = None) -> Meaning:
    return meanings([(name, form)])[0]


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def asked_form(name: str, form: str | None) -> str:
    """The purchase form a name asks for: the parser's, else one of ASKED_FORMS among its
    words ("frozen broccoli", "cumin powder" -> ground); "" when none."""
    if form:
        return SAME_FORM.get(form.lower(), form.lower())
    for w in _words(name):
        if w in ASKED_FORMS:
            return SAME_FORM.get(w, w)
    return ""


def asked_variant(name: str, types: set[str]) -> str:
    """The variant a name asks for: the longest variant of a product of `types` whose words
    are all in the name ("extra virgin" for "extra-virgin olive oil", "crushed" for "crushed
    tomatoes"); "" when none is."""
    words = set(tokens(" ".join(_words(name))))
    found = {p.variant for p in seeds.load().profiles.values()
             if p.product_type in types and p.variant
             and set(tokens(p.variant)) <= words and tokens(p.variant)}
    return max(found, key=lambda v: (len(tokens(v)), v)) if found else ""


def _embedded(name: str, form: str | None, vector: list[float]) -> Meaning:
    """An unknown name mapped to the nearest product types by cosine (module docstring)."""
    s = seeds.load()
    sims = sorted(((embed.cosine(vector, c), t) for t, c in s.centroids.items()),
                  key=lambda st: (-st[0], st[1]))
    nearest = (sims[0][1], round(sims[0][0], 4)) if sims else None
    want = asked_form(name, form)
    if want and sims:
        # A form the name asks for chooses between near types: "cumin powder" is cumin
        # (ground), not cumin seeds, a little nearer.
        forms: dict[str, set[str]] = {}
        for p in s.profiles.values():
            forms.setdefault(p.product_type, set()).add(SAME_FORM.get(p.form, p.form))
        near = [st for st in sims if st[0] >= sims[0][0] - FORM_TIE]
        if want not in forms.get(sims[0][1], set()):
            hit = next((st for st in near if want in forms.get(st[1], set())), None)
            if hit is not None:
                sims = [(sims[0][0], hit[1])] + [st for st in sims if st[1] != hit[1]]
    best = sims[0][0] if sims else 0.0
    f = floors()
    exact = tuple(t for sim, t in sims if sim >= f["exact"] and sim >= best - TIE)
    # What works instead of a type a seed ingredient names as the same item is that seed's
    # (Gemini's labels, offline) rather than the type's embedding neighbours.
    seeded = [i for i in s.ingredients.values() if set(i.exact_types) & set(exact)]
    if seeded:
        subs = tuple(dict.fromkeys(t for i in seeded for t in i.substitute_types
                                   if t not in exact))
        unrelated = tuple(dict.fromkeys(t for i in seeded for t in i.unrelated_types))
    else:
        subs = tuple(t for sim, t in sims if sim >= f["type"]
                     and t not in exact)[:MAX_SUBSTITUTES]
        unrelated = ()
    return Meaning(name=name, source="embedding", exact=exact, substitutes=subs,
                   unrelated=unrelated, form=want,
                   variant=asked_variant(name, set(exact) | set(subs)),
                   vector=tuple(vector), nearest=nearest)


def _described(name: str, form: str | None, ident: describe.Identity,
               vector: list[float] | None) -> Meaning:
    """A name the describer identified (describe.py). Its product_type is the Same item when the
    catalog has that type, unless its identity string is far from that type's centroid (below
    floors()["check"]: the model may have filed it wrongly), when the type is offered as working
    instead. What works instead is a seed's substitutes when a seed names the same type as the
    same item, else the nearest other types at or above floors()["type"]. Form and variant are
    the describer's, read as asked_form and asked_variant read a name's own words."""
    s = seeds.load()
    f = floors()
    have = s.types()
    exact: tuple[str, ...] = (ident.product_type,) if ident.product_type in have else ()
    sims = (sorted(((embed.cosine(vector, c), t) for t, c in s.centroids.items()),
                   key=lambda st: (-st[0], st[1])) if vector is not None else [])
    demoted: tuple[str, ...] = ()
    if exact and vector is not None and f["check"]:
        own = embed.cosine(vector, s.centroids[exact[0]]) if exact[0] in s.centroids else 0.0
        if own < f["check"]:
            demoted, exact = exact, ()
    seeded = [i for i in s.ingredients.values() if set(i.exact_types) & set(exact)]
    if seeded:
        subs = tuple(dict.fromkeys(t for i in seeded for t in i.substitute_types
                                   if t not in exact))
        unrelated = tuple(dict.fromkeys(t for i in seeded for t in i.unrelated_types))
    else:
        near = tuple(t for sim, t in sims if sim >= f["type"] and t not in exact
                     and t not in demoted)
        subs = (demoted + near)[:MAX_SUBSTITUTES]
        unrelated = ()
    want = SAME_FORM.get(ident.form, ident.form) if ident.form else asked_form(name, form)
    return Meaning(name=name, source="described", exact=exact, substitutes=subs,
                   unrelated=unrelated, form=want,
                   variant=ident.variant or asked_variant(name, set(exact) | set(subs)),
                   vector=tuple(vector) if vector is not None else None,
                   nearest=(sims[0][1], round(sims[0][0], 4)) if sims else None)


def describe_state() -> dict:
    """The describer for /health: its model, and how many names it has described (cached)."""
    d = describe.get_describer()
    if d is None:
        return {"model": None, "detail": "DESCRIBER_MODEL is empty: names no seed knows are "
                                          "embedded as they are"}
    return {"model": d.model, "calls": d.calls, "cached": d.cached}


# ─── Candidates ──────────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    """A product for a name: its group, its type, and the facts that do not depend on a price:
    relevance (the cosine, scaled for a substitute's place) and form_variant."""
    product_id: int
    group: Group
    product_type: str
    cosine: float
    relevance: float
    form_variant: float


def matched(catalog: dict[int, Product]) -> dict[int, seeds.Profile]:
    """The catalog's products that have a profile made from their current row."""
    s = seeds.load()
    return {pid: s.profiles[pid] for pid, p in catalog.items()
            if pid in s.profiles and pid in s.vectors
            and s.profiles[pid].hash == seeds.product_hash(p)}


def form_variant(m: Meaning, prof: seeds.Profile) -> float:
    """The share of the form and variant `m` asks for that the product has; 1 when it asks for
    neither. A variant is had when all its words are in the product's variant or type."""
    asked: list[bool] = []
    if m.form:
        asked.append(SAME_FORM.get(m.form, m.form) == SAME_FORM.get(prof.form, prof.form))
    if m.variant:
        have = set(tokens(f"{prof.variant} {prof.product_type}"))
        asked.append(set(tokens(m.variant)) <= have)
    return sum(asked) / len(asked) if asked else 1.0


def candidates(m: Meaning, catalog: dict[int, Product], *,
               looser: bool = True) -> list[Candidate]:
    """Every product of `catalog` that is the same item as `m` or works instead, and (with
    `looser`) its nearest other products: Looser match. Best group first, then relevance."""
    if m.source == "words":
        return []
    s = seeds.load()
    loose = floors()["looser"]
    profiles = matched(catalog)
    words = set(tokens(m.name if not m.form or m.form in m.name.lower()
                       else f"{m.form} {m.name}"))
    exact, subs = set(m.exact), list(dict.fromkeys(t for t in m.substitutes
                                                   if t not in m.exact))
    out: list[Candidate] = []
    near: list[Candidate] = []
    for pid, prof in profiles.items():
        cos = embed.cosine(m.vector, s.vectors[pid]) if m.vector is not None else 0.0
        fv = form_variant(m, prof)
        if prof.product_type in exact:
            out.append(Candidate(pid, "same_item", prof.product_type, cos, cos, fv))
        elif prof.product_type in subs:
            step = 1.0 - SUBSTITUTE_STEP * subs.index(prof.product_type)
            out.append(Candidate(pid, "works_instead", prof.product_type, cos,
                                 cos * max(step, 0.0), fv))
        elif (looser and cos >= loose and prof.product_type not in m.unrelated):
            near.append(Candidate(pid, "looser_match", prof.product_type, cos, cos, fv))
    for pid, p in catalog.items():
        if pid not in profiles and words and words <= set(tokens(index_text(p.name,
                                                                            p.description))):
            out.append(Candidate(pid, "same_item", "", 0.0, 0.0, 1.0))
    near.sort(key=lambda c: (-c.cosine, c.product_id))
    out += near[:LOOSER_K]
    return sorted(out, key=lambda c: (GROUPS.index(c.group), -c.relevance, c.product_id))


# ─── Scores ──────────────────────────────────────────────────

@dataclass
class Facts:
    """One candidate's score components, each 0..1, higher better (module docstring)."""
    relevance: float
    form_variant: float
    pack_fit: float = 0.5
    value: float = 0.5
    origin: float = 0.0
    trip: float = 0.0

    def score(self) -> float:
        return round(sum(w * getattr(self, k) for k, w in WEIGHTS.items()), 6)


def pack_fit(product: Product, need: tuple[float, str] | None, packs: int = 1,
             smallest: float | None = None) -> float:
    """The pack_fit component: `packs` of the product against the need, `smallest` being the
    smallest pack in the need's unit among the products it is compared with (module
    docstring)."""
    if need is None or need[0] <= 0 or not product.unit_qty or product.unit_uom != need[1]:
        return 0.5
    have = product.unit_qty * packs
    if have + 1e-9 < need[0]:
        return 0.3
    if product.unit_qty > need[0] * CATERING:
        if smallest is not None and smallest > need[0] * CATERING \
                and product.unit_qty <= smallest * CATERING:
            return 0.6
        return 0.25
    return round(1.0 - 0.4 * min(1.0, (have - need[0]) / (need[0] * (CATERING - 1))), 6)


def smallest_pack(products: list[Product], need: tuple[float, str] | None) -> float | None:
    """The smallest pack size among `products` in the need's unit (None when none has one)."""
    if need is None:
        return None
    sizes = [p.unit_qty for p in products if p.unit_qty and p.unit_uom == need[1]]
    return min(sizes) if sizes else None


def unit_price(product: Product, price: float) -> tuple[float, str] | None:
    """(price per 100 g / 100 ml / each, its basis) or None when the pack size is unknown."""
    if not product.unit_qty or product.unit_uom not in {"g", "ml", "each"}:
        return None
    per = 1 if product.unit_uom == "each" else 100
    return price / product.unit_qty * per, product.unit_uom


def values(prices: dict[int, tuple[float, str] | None]) -> dict[int, float]:
    """The value component for each product: the lowest unit price on its basis over its own
    (1 for the best value), 0.5 when it has none."""
    best: dict[str, float] = {}
    for up in prices.values():
        if up is not None and up[0] > 0:
            best[up[1]] = min(best.get(up[1], math.inf), up[0])
    return {pid: (best[up[1]] / up[0] if up is not None and up[0] > 0 else 0.5)
            for pid, up in prices.items()}


def origin_component(rank: int, preference: list[str]) -> float:
    """origins.preference_rank's rank (0 best, len*2 for none) as 1..0; 0 with no preference."""
    worst = len(preference) * 2
    return 0.0 if not preference else max(0.0, 1.0 - rank / worst)


def trip_component(total: float | None, best: float | None) -> float:
    if total is None or best is None:
        return 0.0
    return max(0.0, 1.0 - (total - best) / TRIP_SPAN)


@dataclass
class Ranked:
    """A candidate with its score, in rank_pool's order."""
    product_id: int
    group: Group
    score: float
    facts: Facts
    candidate: Candidate


def rank_pool(m: Meaning, pool: list[Product], *, need: tuple[float, str] | None = None,
              preference: list[str] | None = None, origins: dict | None = None,
              catalog: dict[int, Product] | None = None,
              prices: dict[int, float] | None = None, looser: bool = False) -> list[Ranked]:
    """The products of `pool` that are candidates for `m` (with `looser`, its looser matches
    too), best first: group, then score, then catalog id. A product's price is `prices`' when
    given, else its store offer when it has one, else its catalog price; one pack is bought,
    as a cart buys for one line. The planner's first pick is the first of these it may
    buy."""
    from ..origins import preference_rank

    preference = [c for c in (preference or []) if c.strip()]
    by_id = {p.id: p for p in pool}
    cands = {c.product_id: c for c in candidates(m, catalog or by_id, looser=looser)
             if c.product_id in by_id}
    out: list[Ranked] = []
    for group in GROUPS:
        members = [c for c in cands.values() if c.group == group]
        price = {c.product_id: (prices or {}).get(c.product_id, _price(by_id[c.product_id]))
                 for c in members}
        value = values({pid: unit_price(by_id[pid], pr) for pid, pr in price.items()})
        cheapest = min(price.values(), default=None)
        smallest = smallest_pack([by_id[c.product_id] for c in members], need)
        for c in members:
            p = by_id[c.product_id]
            f = Facts(relevance=c.relevance, form_variant=c.form_variant,
                      pack_fit=pack_fit(p, need, smallest=smallest), value=value[c.product_id],
                      origin=origin_component(
                          preference_rank((origins or {}).get(p.id), preference), preference),
                      trip=trip_component(price[c.product_id], cheapest))
            out.append(Ranked(c.product_id, group, f.score(), f, c))
    out.sort(key=lambda r: (GROUPS.index(r.group), -r.score, r.product_id))
    return out


def _price(p: Product) -> float:
    return p.store_price if p.store_price is not None else p.price


# ─── State, for /health ──────────────────────────────────────

def state() -> dict:
    """What the matcher knows and whether it can embed a name no seed knows: `mode` is
    "meaning" with seeds, `unknown_names` how a name no seed knows is matched right now
    ("meaning" when the embedder answers, else "words")."""
    s = seeds.load()
    e = embed.get_embedder()
    if e is None:
        up, detail = False, "no embedding model is configured (OLLAMA_URL is empty)"
    elif hasattr(e, "ping"):
        up, detail = e.ping()
    else:
        up, detail = True, ""
    agrees = None
    if up and e is not None:
        try:
            agrees, why = _agrees(e)
        except embed.EmbedderUnavailableError as err:
            up, why = False, str(err)
        if not up or not agrees:
            up, detail = False, why
    return {"mode": "meaning" if s.profiles else "words", "profiles": len(s.profiles),
            "product_types": len(s.centroids), "seed_ingredients": len(s.ingredients),
            "embedding_model": s.model, "dims": s.dims,
            "embedder": getattr(e, "name", None), "embedder_up": up,
            "embedder_matches_seeds": agrees,
            "unknown_names": "meaning" if up and s.centroids else "words",
            "detail": detail,
            "describer": describe_state(),
            "cache": cache.state(),
            "floors": floors()}
