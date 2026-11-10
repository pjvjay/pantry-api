"""The matching seeds, read once per process: what each product is, its vector, and what each
known ingredient name maps to.

  product_profiles.json    per product: product_type, taxonomy, form, variant, culinary_role,
                           diet and profile_text, from Gemini once (scripts/matching_seeds.py),
                           checked by a second pass and reviewed by hand
  product_embeddings.json  per product: the EmbeddingGemma document vector of its profile_text
  ingredient_types.json    per ingredient name of the library and the demo starters: the
                           product types that ARE it (exact), that work instead (substitute,
                           best first) and that only look related (unrelated), the form and
                           variant its name asks for, and its EmbeddingGemma query vector

Every profile and vector carries the hash of the product row it was made from (name,
description, category, subcategory, size, contains-tags). A product whose row has changed
since, or that has no profile, is not matched by meaning (`stale` says which); the catalog the
database serves is pantry-db's copy of seeds/products.json, so in practice none is.

A type's centroid is the normalized mean of its products' vectors: an ingredient name no seed
knows is mapped to the types whose centroids are nearest (matcher.py).

Two sets of vectors, chosen by EMBEDDING_MODEL:

  embeddinggemma (default)   product_embeddings.json and ingredient_types.json's vectors: each
                             product's profile_text and each seed name, as EmbeddingGemma embeds
                             them through Ollama
  gemini:<model>             gemini_vectors.json: each product's and each seed name's identity
                             string (identity_text: "butter | block | salted"), as Gemini's
                             embedding API embeds it. Only what identifies an item is embedded,
                             never its culinary role or aisle, so look-alikes stay apart.
"""
from __future__ import annotations

import functools
import hashlib
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..nlsearch.units import tokens

SEEDS_DIR = Path(__file__).resolve().parents[2] / "seeds"

PROFILES_ABOUT = (
    "What each product is, for matching by meaning: product_type (what a recipe calls it), "
    "taxonomy [department, aisle, product_type], form, variant, culinary_role and the diets it "
    "suits, from Gemini in batches (scripts/matching_seeds.py profiles); aka, the other names "
    "the catalog description gives it in parentheses. profile_text is templated from those "
    "fields and is what gets embedded, as a document titled with its product_type. flags "
    "are a second, separate Gemini pass's disagreements, for review; fixes are corrections "
    "made by hand after review "
    "(field, from, to, why). The catalog's dietary_tags (contains-tags) are never changed "
    "here. hash: the product row the profile was made from.")
EMBEDDINGS_ABOUT = (
    "EmbeddingGemma vectors of each product's profile_text as a retrieval document titled with "
    "its product_type (\"title: <product_type> | text: <profile_text>\"), the first "
    "256 of 768 dimensions re-normalized to unit length (Matryoshka), rounded to 5 decimals.")
TYPES_ABOUT = (
    "Every distinct ingredient name of the library and the demo starters, keyed by its "
    "tokens: the product types that ARE it (exact_types), that work instead in cooking "
    "(substitute_types, best first) and that only look related and must never be offered "
    "(unrelated_types), as Gemini labelled them against the catalog's types (ESCI-style), "
    "with the form and variant the name asks for and the name's EmbeddingGemma vector as a "
    "retrieval query.")


def product_hash(p) -> str:
    """The hash of what a profile is made from: a products.json row or a Product."""
    get = (p.get if isinstance(p, dict) else lambda k, d="": getattr(p, k, d))
    body = [get("name", ""), get("description", "") or "", get("category", "") or "",
            get("subcategory", "") or "", get("unit_size", "") or "",
            get("dietary_tags", "") or ""]
    return hashlib.sha256(json.dumps(body).encode()).hexdigest()[:16]


def profile_text(prof: dict) -> str:
    """The sentence a product is embedded as, templated from its profile: "green onions
    (fresh): mild onion garnish or aromatic base. Aisle: produce, fresh herbs. Also called
    scallions, spring onions.\" (the last part only when the catalog names it otherwise)."""
    variant = f", {prof['variant']}" if prof.get("variant") else ""
    dept, aisle = (list(prof["taxonomy"]) + ["", ""])[:2]
    aka = f" Also called {', '.join(prof['aka'])}." if prof.get("aka") else ""
    return (f"{prof['product_type']}{variant} ({prof['form']}): "
            f"{prof['culinary_role'].rstrip('.')}. Aisle: {dept}, {aisle}.{aka}")


def aka(description: str) -> list[str]:
    """The other names a catalog description gives a product, in its parentheses: "Fresh green
    onions (scallions, spring onions), 1 bunch (~100g)" -> [scallions, spring onions]. A
    parenthesis with a number in it is a size, not a name."""
    out: list[str] = []
    for inner in re.findall(r"\(([^()]*)\)", description or ""):
        if not re.search(r"\d", inner):
            out += [w.strip() for w in re.split(r"[,;]", inner) if w.strip()]
    return list(dict.fromkeys(out))


def ingredient_key(name: str) -> str:
    """How an ingredient name is looked up: its match tokens (lower case, stemmed, stopwords
    out), so "Canned Tomatoes", "canned tomato" and "canned  tomatoes" are one name."""
    return " ".join(tokens(name))


@dataclass(frozen=True)
class Profile:
    id: int
    hash: str
    product_type: str
    taxonomy: tuple[str, ...]
    form: str
    variant: str
    culinary_role: str
    diet: tuple[str, ...]
    aka: tuple[str, ...]
    profile_text: str


@dataclass(frozen=True)
class SeedIngredient:
    name: str
    exact_types: tuple[str, ...]
    substitute_types: tuple[str, ...]
    unrelated_types: tuple[str, ...]
    form: str
    variant: str
    vector: tuple[float, ...]
    identity: str = ""          # the identity string its Gemini vector embeds ("" otherwise)


@dataclass
class Seeds:
    profiles: dict[int, Profile] = field(default_factory=dict)
    vectors: dict[int, tuple[float, ...]] = field(default_factory=dict)
    ingredients: dict[str, SeedIngredient] = field(default_factory=dict)
    centroids: dict[str, tuple[float, ...]] = field(default_factory=dict)
    model: str = ""
    dims: int = 0
    text: str = "profile"       # what the vectors embed: "profile" text or "identity" strings

    def types(self) -> set[str]:
        return {p.product_type for p in self.profiles.values()}


GEMINI_VECTORS = "gemini_vectors.json"
GEMINI_ABOUT = (
    "Gemini embedding vectors (OpenAI-compatible embeddings endpoint, unit length) of each "
    "product's identity string, product_type | form | variant (identity_text), and of each seed "
    "ingredient name's identity string as the describer (matching/describe.py) identified it. "
    "Kept beside the EmbeddingGemma seeds; EMBEDDING_MODEL chooses which set is read.")


def identity_text(product_type: str, form: str, variant: str) -> str:
    """What identifies an item, and nothing else: "butter | block | salted"."""
    return " | ".join([product_type.strip().lower(), form.strip().lower(), variant.strip().lower()])


def _embedding_model() -> str:
    from ..config import settings

    return settings().embedding_model


def _read(name: str) -> dict:
    path = SEEDS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


@functools.lru_cache(maxsize=1)
def load() -> Seeds:
    """The three seed files, as committed. Missing files load as empty: nothing is then known
    by meaning and every name is matched by words."""
    prof = _read("product_profiles.json")
    types = _read("ingredient_types.json")
    gemini = _embedding_model().startswith("gemini:")
    vecs = _read(GEMINI_VECTORS) if gemini else _read("product_embeddings.json")
    out = Seeds(model=vecs.get("model", ""), dims=int(vecs.get("dims", 0) or 0),
                text="identity" if gemini else "profile")
    gem_ing = vecs.get("ingredients", {}) if gemini else {}
    for p in prof.get("products", []):
        out.profiles[p["id"]] = Profile(
            id=p["id"], hash=p["hash"], product_type=p["product_type"],
            taxonomy=tuple(p["taxonomy"]), form=p["form"], variant=p["variant"],
            culinary_role=p["culinary_role"], diet=tuple(p["diet"]),
            aka=tuple(p.get("aka", ())), profile_text=p["profile_text"])
    for v in vecs.get("products", []):
        p = out.profiles.get(v["id"])
        if p is not None and p.hash == v["hash"]:
            out.vectors[v["id"]] = tuple(v["vector"])
    for key, i in types.get("ingredients", {}).items():
        out.ingredients[key] = SeedIngredient(
            name=i["name"], exact_types=tuple(i["exact_types"]),
            substitute_types=tuple(i["substitute_types"]),
            unrelated_types=tuple(i["unrelated_types"]), form=i["form"], variant=i["variant"],
            vector=(tuple(gem_ing[key]["vector"]) if key in gem_ing else ()) if gemini
            else tuple(i["vector"]),
            identity=gem_ing.get(key, {}).get("identity", ""))
    by_type: dict[str, list[tuple[float, ...]]] = {}
    for pid, v in out.vectors.items():
        by_type.setdefault(out.profiles[pid].product_type, []).append(v)
    for t, vs in by_type.items():
        mean = [sum(col) / len(vs) for col in zip(*vs, strict=True)]
        norm = math.sqrt(sum(x * x for x in mean)) or 1.0
        out.centroids[t] = tuple(x / norm for x in mean)
    return out


def clear_cache() -> None:
    load.cache_clear()


def seed_ingredient_names() -> list[str]:
    """Every distinct ingredient name of the library and the demo starters that is bought (not
    water or ice), in first-seen order: the names ingredient_types.json maps, and the golden
    set's (evals/substitutes_eval.py)."""
    from ..nlsearch.units import is_non_purchase

    names = [i["name"] if isinstance(i, dict) else i
             for r in json.loads((SEEDS_DIR / "recipes.json").read_text(encoding="utf-8"))
             for i in r["ingredients"]]
    names += [ln["name"] for s in json.loads((SEEDS_DIR / "mealplan_starters.json")
                                             .read_text(encoding="utf-8"))["starters"]
              for ln in s["lines"]]
    seen: dict[str, str] = {}
    for n in names:
        if not is_non_purchase(n):
            seen.setdefault(ingredient_key(n), n)
    return list(seen.values())
