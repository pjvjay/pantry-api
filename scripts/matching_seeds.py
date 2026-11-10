"""The offline pipeline behind product matching by meaning. Run by hand, once; its outputs are
committed seeds that pantry-api reads at request time with no model call.

  profiles   seeds/product_profiles.json: what each product is (product_type, taxonomy, form,
             variant, culinary_role, diet), from Gemini with a JSON schema, in batches, and
             the templated profile_text that gets embedded
  check      a second, separate batched yes/no pass over each profile's type, form and diet
             (and the catalog's contains-tags); every "no" becomes a flag on the profile for
             review, never a silent change
  embed      seeds/product_embeddings.json: EmbeddingGemma vectors of each profile_text
  types      seeds/ingredient_types.json: for every distinct ingredient name of the library
             and the demo starters, its exact, substitute and unrelated product types
             (Gemini), and its EmbeddingGemma query vector
  review     the review page (one row per product, flagged rows first)
  gemini-vectors  seeds/gemini_vectors.json: Gemini embedding vectors of each product's and
             each seed name's identity string, for EMBEDDING_MODEL=gemini:<model>
  golden     evals/substitutes_golden.json: a Gemini judge labels each seed ingredient's top
             candidates, old word matching and new matching together, Exact / Substitute /
             Complement / Irrelevant (evals/substitutes_eval.py scores them)

Every Gemini reply is cached on disk under --cache-dir by a hash of what it was asked: a
profile by its product's name, description, category, subcategory, size and contains-tags, a
check by that and the profile, a mapping by the ingredient name and the catalog's types, a
judgement by the ingredient and its candidates. A rerun asks only for what changed, and costs
no call when nothing did. Every HTTP attempt is counted in <cache-dir>/calls.jsonl, and no call
is made once --budget attempts have been spent.

The profiles are reviewed by hand after `check`; a fix is recorded on the profile ("fixes":
field, from, to, why) and re-applied on every rerun, so a fix is never lost to the cache.

The Gemini key is read from GEMINI_API_KEY and never printed or written.

  GEMINI_API_KEY=... python scripts/matching_seeds.py profiles --cache-dir DIR
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pantry_planner.matching import embed as emb  # noqa: E402
from pantry_planner.matching import seeds as mseeds  # noqa: E402

SEEDS = ROOT / "seeds"
GOLDEN = ROOT / "evals" / "substitutes_golden.json"
MODEL = "gemini:gemini-flash-latest"
PROMPT_VERSION = "1"
PROFILE_BATCH = 43
CHECK_BATCH = 85
TYPES_BATCH = 48
GOLDEN_BATCH = 24
FORMS = ["fresh", "frozen", "canned", "dried", "ground", "whole", "block", "shredded",
         "sliced", "liquid", "paste", "powder", "flakes", "jarred", "bottled", "baked",
         "cured", "raw"]
DIETS = ["vegan", "vegetarian", "pescatarian"]


# ─── Gemini, cached and counted ──────────────────────────────

class BudgetSpentError(RuntimeError):
    pass


class Gemini:
    """forced_tool_call (pantry_planner/llm.py) with a disk cache and a call budget."""

    def __init__(self, cache_dir: Path, budget: int, retries: int = 2):
        self.retries = retries
        self.dir = cache_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.budget = budget
        from pantry_planner import llm

        # One attempt per call: a free-tier key's 429 (or a busy model's 503) is waited out
        # here, a minute and a half at a time, rather than retried at once.
        llm.GEMINI_MAX_ATTEMPTS = 1

    def spent(self) -> int:
        path = self.dir / "calls.jsonl"
        if not path.exists():
            return 0
        return sum(json.loads(ln)["attempts"] for ln in path.read_text().splitlines() if ln)

    def get(self, kind: str, key: str) -> dict | None:
        path = self.dir / kind / f"{key}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def put(self, kind: str, key: str, value: dict) -> None:
        path = self.dir / kind / f"{key}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=1, sort_keys=True))

    def _count(self, step: str, attempts: int, error: str | None) -> None:
        with (self.dir / "calls.jsonl").open("a") as f:
            f.write(json.dumps({"at": datetime.now(UTC).isoformat(timespec="seconds"),
                                "step": step, "attempts": attempts, "error": error}) + "\n")

    def call(self, step: str, system: str, user: str, tool: dict, max_tokens: int) -> dict:
        from pantry_planner import llm

        request = json.dumps([MODEL, system, user, tool], sort_keys=True)
        key = hashlib.sha256(request.encode()).hexdigest()
        hit = self.get("replies", key)
        if hit is not None:
            return hit
        for wait in (0, *[90] * self.retries):
            if self.spent() >= self.budget:
                raise BudgetSpentError(f"the budget of {self.budget} Gemini calls is spent")
            if wait:
                print(f"  {step}: waiting {wait}s before trying again", file=sys.stderr)
                time.sleep(wait)
            try:
                reply = llm.forced_tool_call(model=MODEL, system=system,
                                             messages=[{"role": "user", "content": user}],
                                             tool=tool, max_tokens=max_tokens)
            except llm.LLMError as e:
                # A 429 or 5xx was retried inside llm.py: count every attempt it may have made.
                retried = isinstance(e, llm.LLMRateLimitError) or (e.status or 0) >= 500
                self._count(step, llm.GEMINI_MAX_ATTEMPTS if retried else 1, type(e).__name__)
                if not retried:
                    raise
                print(f"  {step}: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
                continue
            self._count(step, len((reply.http or {}).get("attempts") or [1]), None)
            out = {"input": reply.input, "input_tokens": reply.input_tokens,
                   "output_tokens": reply.output_tokens, "model": reply.model}
            self.put("replies", key, out)
            return out
        raise RuntimeError(f"{step}: Gemini kept failing; rerun later (replies so far are "
                           "cached)")


def _batches(items: list, size: int) -> list[list]:
    """`items` in even batches of at most `size`."""
    if not items:
        return []
    n = -(-len(items) // size)
    per = -(-len(items) // n)
    return [items[i:i + per] for i in range(0, len(items), per)]


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _products() -> list[dict]:
    return json.loads((SEEDS / "products.json").read_text(encoding="utf-8"))


def _row(p: dict) -> dict:
    return {"id": p["id"], "name": p["name"], "description": p.get("description", ""),
            "category": p.get("category"), "subcategory": p.get("subcategory", ""),
            "size": p.get("unit_size", ""), "contains": p.get("dietary_tags", "")}


# ─── profiles ────────────────────────────────────────────────

PROFILE_SYSTEM = f"""You write product profiles for a grocery catalog's matching engine, which \
decides which products ARE the ingredient a recipe line names, which work instead, and which \
are a different food. For each product you get its id, name, description, category, \
subcategory, size and contains-tags. Return, for every product:

- product_type: the canonical, lower-case name of what the product is, as a recipe names the \
ingredient: no brand, size, pack or count words. Use the most specific name a recipe would use, \
but leave out what recipes treat as interchangeable, which goes in variant: salted is a variant \
of "butter", extra virgin and pure are variants of "olive oil", lean and medium are variants of \
"ground beef", diced, crushed and petite diced are variants of "canned tomatoes", boneless \
skinless and bone-in are variants of "chicken thighs". A different food gets a different type \
even when it shares a word: peanut butter is not butter, tomato paste is not canned tomatoes, \
coconut milk is not milk, a jarred pasta sauce is not passata. Use the plural when recipes buy \
it so ("chicken thighs", "eggs", "carrots"), the singular otherwise ("butter", "garlic").
- department and aisle: where a North American supermarket shelves it, lower case.
- form: one word for its physical form, from: {", ".join(FORMS)}.
- variant: what sets it apart within its type, lower case ("salted", "extra virgin", \
"low sodium", "boneless skinless", "lean"), or "" for the plain one.
- culinary_role: one short phrase (at most 8 words): how a cook uses it.
- diet: which of vegan, vegetarian, pescatarian it suits, from what it is: anything with meat \
suits none; fish or shellfish suits pescatarian; dairy, egg or honey suits vegetarian and \
pescatarian; a plant food suits all three.

Base every answer on the product as described; never guess a brand's recipe."""

PROFILE_TOOL = {
    "name": "submit_profiles",
    "description": "One profile per product, every product given, in any order.",
    "input_schema": {
        "type": "object", "required": ["profiles"],
        "properties": {"profiles": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "product_type", "department", "aisle", "form", "variant",
                         "culinary_role", "diet"],
            "properties": {
                "id": {"type": "integer"},
                "product_type": {"type": "string"},
                "department": {"type": "string"},
                "aisle": {"type": "string"},
                "form": {"type": "string", "enum": FORMS},
                "variant": {"type": "string"},
                "culinary_role": {"type": "string"},
                "diet": {"type": "array", "items": {"type": "string", "enum": DIETS}},
            }}}},
    },
}

PROFILE_FIELDS = ("product_type", "taxonomy", "form", "variant", "culinary_role", "diet", "aka")


def _profile_of(raw: dict) -> dict:
    """A Gemini profile as the seed stores it: lower case, trimmed, taxonomy as a path."""
    t = " ".join(str(raw["product_type"]).lower().split())
    return {"product_type": t,
            "taxonomy": [" ".join(str(raw["department"]).lower().split()),
                         " ".join(str(raw["aisle"]).lower().split()), t],
            "form": str(raw["form"]).lower().strip(),
            "variant": " ".join(str(raw["variant"]).lower().split()),
            "culinary_role": " ".join(str(raw["culinary_role"]).split()),
            "diet": [d for d in DIETS if d in set(raw["diet"])]}


def _apply_fixes(profile: dict, fixes: list[dict]) -> dict:
    out = dict(profile)
    for f in fixes:
        out[f["field"]] = f["to"]
    if any(f["field"] == "product_type" for f in fixes):
        out["taxonomy"] = [*out["taxonomy"][:2], out["product_type"]]
    return out


def cmd_profiles(g: Gemini, args) -> None:
    products = _products()
    path = SEEDS / "product_profiles.json"
    old = {p["id"]: p for p in json.loads(path.read_text())["products"]} \
        if path.exists() else {}
    keyof = {p["id"]: f"{PROMPT_VERSION}-{mseeds.product_hash(p)}" for p in products}
    todo = [p for p in products if g.get("profile", keyof[p["id"]]) is None]
    print(f"profiles: {len(products) - len(todo)} cached, {len(todo)} to ask")
    for i, batch in enumerate(_batches(todo, PROFILE_BATCH)):
        if not batch:
            continue
        user = json.dumps({"products": [_row(p) for p in batch]}, ensure_ascii=False)
        reply = g.call(f"profiles {i + 1}", PROFILE_SYSTEM, user, PROFILE_TOOL,
                       max_tokens=200 * len(batch) + 1000)
        got = {int(r["id"]): r for r in reply["input"].get("profiles", [])}
        for p in batch:
            if p["id"] in got:
                g.put("profile", keyof[p["id"]], _profile_of(got[p["id"]]))
        missing = [p["id"] for p in batch if p["id"] not in got]
        if missing:
            print(f"  batch {i + 1}: no profile for {missing}; rerun to ask again")
    out = []
    for p in products:
        prof = g.get("profile", keyof[p["id"]])
        if prof is None:
            continue
        prev = old.get(p["id"], {})
        fixes = prev.get("fixes", []) if prev.get("hash") == mseeds.product_hash(p) else []
        prof = _apply_fixes({**prof, "aka": mseeds.aka(p.get("description", ""))}, fixes)
        same = prev.get("hash") == mseeds.product_hash(p)
        out.append({"id": p["id"], "name": p["name"], "hash": mseeds.product_hash(p),
                    **{k: prof[k] for k in PROFILE_FIELDS},
                    "profile_text": mseeds.profile_text(prof),
                    "flags": prev.get("flags", []) if same else [], "fixes": fixes,
                    **({"review": prev["review"]} if same and prev.get("review") else {})})
    _write(path, {"v": 1, "model": MODEL, "prompt_version": PROMPT_VERSION,
                  "about": mseeds.PROFILES_ABOUT, "products": out})
    print(f"wrote {len(out)} profiles to {path.relative_to(ROOT)}")


# ─── check ───────────────────────────────────────────────────

CHECK_SYSTEM = """You check another model's grocery product profiles. For each product you get \
the catalog row (name, description, category, size, contains-tags) and the profile written for \
it. Answer each question yes or no:

- type_ok: is product_type what this product is, named as a recipe names the ingredient, \
without brand or size, and not merged with a different food (peanut butter is not butter, \
tomato paste is not canned tomatoes)?
- form_ok: is form its physical form as sold?
- diet_ok: does it suit exactly the diets listed (vegan, vegetarian, pescatarian)?
- tags_ok: do the catalog's contains-tags list everything it contains among dairy, egg, \
gluten, meat, nuts and soy? (This catalog tags fish and shellfish as meat.)

When any answer is no, say in note what is wrong and what it should be; otherwise note is "". \
Be strict but do not flag matters of taste."""

CHECK_TOOL = {
    "name": "submit_checks",
    "description": "One check per product.",
    "input_schema": {
        "type": "object", "required": ["checks"],
        "properties": {"checks": {"type": "array", "items": {
            "type": "object",
            "required": ["id", "type_ok", "form_ok", "diet_ok", "tags_ok", "note"],
            "properties": {
                "id": {"type": "integer"},
                "type_ok": {"type": "boolean"}, "form_ok": {"type": "boolean"},
                "diet_ok": {"type": "boolean"}, "tags_ok": {"type": "boolean"},
                "note": {"type": "string"},
            }}}},
    },
}

_CHECK_FIELDS = {"type_ok": "product_type", "form_ok": "form", "diet_ok": "diet",
                 "tags_ok": "dietary_tags"}


def cmd_check(g: Gemini, args) -> None:
    path = SEEDS / "product_profiles.json"
    data = json.loads(path.read_text())
    rows = {p["id"]: p for p in _products()}

    def key(prof: dict) -> str:
        # The profile as Gemini wrote it, before any fix by hand: a fix never re-asks.
        orig = dict(prof)
        for f in reversed(prof.get("fixes", [])):
            orig[f["field"]] = f["from"]
        body = json.dumps([PROMPT_VERSION, orig["hash"], orig["product_type"], orig["form"],
                           orig["diet"]], sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    todo = [p for p in data["products"] if g.get("check", key(p)) is None]
    print(f"check: {len(data['products']) - len(todo)} cached, {len(todo)} to ask")
    for i, batch in enumerate(_batches(todo, CHECK_BATCH)):
        if not batch:
            continue
        user = json.dumps({"products": [
            {**_row(rows[p["id"]]), "profile": {k: p[k] for k in
                                                ("product_type", "form", "variant", "diet")}}
            for p in batch]}, ensure_ascii=False)
        reply = g.call(f"check {i + 1}", CHECK_SYSTEM, user, CHECK_TOOL,
                       max_tokens=80 * len(batch) + 1000)
        got = {int(r["id"]): r for r in reply["input"].get("checks", [])}
        for p in batch:
            if p["id"] in got:
                g.put("check", key(p), got[p["id"]])
    flagged = 0
    for p in data["products"]:
        c = g.get("check", key(p))
        if c is None:
            continue
        p["flags"] = [{"field": fld, "by": "check", "note": c.get("note", "")}
                      for ok, fld in _CHECK_FIELDS.items() if c.get(ok) is False]
        flagged += bool(p["flags"])
    _write(path, data)
    print(f"{flagged} of {len(data['products'])} profiles flagged")


# ─── embed ───────────────────────────────────────────────────

def _ollama(args) -> emb.OllamaEmbedder:
    return emb.OllamaEmbedder(args.ollama_url, args.embedding_model, args.cache_dir / "ollama")


def cmd_embed(g: Gemini, args) -> None:
    data = json.loads((SEEDS / "product_profiles.json").read_text())
    e = _ollama(args)
    texts = [p["profile_text"] for p in data["products"]]
    vectors = e.embed(texts, "document", titles=[p["product_type"] for p in data["products"]])
    _write(SEEDS / "product_embeddings.json", {
        "v": 1, "model": args.embedding_model, "dims": emb.DIMS, "kind": "document",
        "prompt": emb.DOCUMENT_PROMPT.format(title="<product_type>") + "<profile_text>",
        "about": mseeds.EMBEDDINGS_ABOUT,
        "products": [{"id": p["id"], "hash": p["hash"],
                      "vector": [round(x, 5) for x in v]}
                     for p, v in zip(data["products"], vectors, strict=True)]})
    print(f"embedded {len(vectors)} profiles")


def cmd_gemini_vectors(g: Gemini, args) -> None:
    """seeds/gemini_vectors.json: Gemini embedding vectors of each product's identity string
    (product_type | form | variant, from its reviewed profile) and of each seed ingredient's
    (its labelled exact type, form and variant), for EMBEDDING_MODEL=gemini:<model>. No
    describer call: the seed names' identities come from their reviewed labels. Each text
    counts against Gemini's embedding quota (1,000 a day on the free tier)."""
    import os

    prof = json.loads((SEEDS / "product_profiles.json").read_text())["products"]
    types = json.loads((SEEDS / "ingredient_types.json").read_text())["ingredients"]
    model = args.gemini_embedding_model.split(":", 1)[-1]
    e = emb.GeminiEmbedder(model, os.environ.get("GEMINI_API_KEY", ""), max_wait_s=600.0)
    p_text = [mseeds.identity_text(p["product_type"], p["form"], p["variant"]) for p in prof]
    i_text = {k: mseeds.identity_text(i["exact_types"][0] if i["exact_types"] else i["name"],
                                      i["form"], i["variant"]) for k, i in types.items()}
    texts = list(dict.fromkeys(p_text + list(i_text.values())))
    vec: dict[str, list[float]] = {}
    vec.update(zip(texts, e.embed(texts), strict=True))
    _write(SEEDS / mseeds.GEMINI_VECTORS, {
        "v": 1, "model": model, "dims": emb.GEMINI_DIMS, "text": "identity",
        "about": mseeds.GEMINI_ABOUT,
        "products": [{"id": p["id"], "hash": p["hash"], "identity": t,
                      "vector": [round(x, 6) for x in vec[t]]}
                     for p, t in zip(prof, p_text, strict=True)],
        "ingredients": {k: {"identity": t, "vector": [round(x, 6) for x in vec[t]]}
                        for k, t in i_text.items()}})
    print(f"embedded {len(texts)} identity strings "
          f"({len(prof)} products, {len(i_text)} seed names)")


# ─── types ───────────────────────────────────────────────────

TYPES_SYSTEM = """You map recipe ingredient names to the product types a grocery catalog \
sells, the way a grocery platform labels query-product pairs (Exact, Substitute, Irrelevant). \
You get the catalog's product types, each with the products it holds, and a list of \
ingredient names from recipes. For each ingredient return:

- exact_types: the catalog types that ARE this ingredient, which a shopper buying for this \
recipe line accepts as the item itself. Several when a recipe's name covers several (an \
"onion" line accepts any onion type the catalog has). When the catalog has no such type, give \
the type's canonical name anyway (lower case), so it is known to be missing.
- substitute_types: catalog types a cook could use instead in this recipe line, best first, \
at most 4; [] when none works.
- unrelated_types: catalog types that share a word with the ingredient or look related but are \
a different food and must never be offered for it (for "butter": "peanut butter"); [] when \
none.
- form: the physical form the ingredient's name asks for ("dried" for "Dried Oregano", \
"canned" for "Canned Tuna", "fresh" for "Fresh Mint"), or "".
- variant: the variant the name asks for ("extra virgin", "salted", "boneless"), or "".

Use type names exactly as listed, except for an exact type the catalog lacks."""

TYPES_TOOL = {
    "name": "submit_ingredient_types",
    "description": "One mapping per ingredient name given.",
    "input_schema": {
        "type": "object", "required": ["ingredients"],
        "properties": {"ingredients": {"type": "array", "items": {
            "type": "object",
            "required": ["name", "exact_types", "substitute_types", "unrelated_types",
                         "form", "variant"],
            "properties": {
                "name": {"type": "string"},
                "exact_types": {"type": "array", "items": {"type": "string"}},
                "substitute_types": {"type": "array", "items": {"type": "string"}},
                "unrelated_types": {"type": "array", "items": {"type": "string"}},
                "form": {"type": "string"},
                "variant": {"type": "string"},
            }}}},
    },
}


def _type_catalog(profiles: list[dict]) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in profiles:
        out.setdefault(p["product_type"], []).append(p["name"])
    return dict(sorted(out.items()))


def cmd_types(g: Gemini, args) -> None:
    profiles = json.loads((SEEDS / "product_profiles.json").read_text())["products"]
    catalog = _type_catalog(profiles)
    cat_hash = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()[:16]
    names = mseeds.seed_ingredient_names()

    def key(name: str) -> str:
        return hashlib.sha256(f"{PROMPT_VERSION}|{cat_hash}|{name}".encode()).hexdigest()

    todo = [n for n in names if g.get("types", key(n)) is None]
    print(f"types: {len(names) - len(todo)} cached, {len(todo)} to ask "
          f"({len(catalog)} catalog types)")
    for i, batch in enumerate(_batches(todo, TYPES_BATCH)):
        if not batch:
            continue
        user = json.dumps({"catalog_types": catalog, "ingredients": batch}, ensure_ascii=False)
        reply = g.call(f"types {i + 1}", TYPES_SYSTEM, user, TYPES_TOOL,
                       max_tokens=150 * len(batch) + 1000)
        got = {r["name"]: r for r in reply["input"].get("ingredients", [])}
        for n in batch:
            r = got.get(n) or next((v for k, v in got.items() if k.lower() == n.lower()), None)
            if r is not None:
                g.put("types", key(n), r)
    e = _ollama(args)
    vectors = dict(zip(names, e.embed(names, "query"), strict=True))
    path = SEEDS / "ingredient_types.json"
    old = json.loads(path.read_text())["ingredients"] if path.exists() else {}
    out = {}
    for n in names:
        r = g.get("types", key(n))
        if r is None:
            continue
        k = mseeds.ingredient_key(n)
        entry = {"name": n,
                 "exact_types": _clean(r["exact_types"]),
                 "substitute_types": _clean(r["substitute_types"]),
                 "unrelated_types": _clean(r["unrelated_types"]),
                 "form": r["form"].lower().strip(), "variant": " ".join(r["variant"].lower()
                                                                        .split()),
                 "fixes": old.get(k, {}).get("fixes", [])}
        for f in entry["fixes"]:
            entry[f["field"]] = f["to"]
        entry["vector"] = [round(x, 5) for x in vectors[n]]
        out[k] = entry
    _write(path, {"v": 1, "model": MODEL, "embedding_model": args.embedding_model,
                  "dims": emb.DIMS, "kind": "query",
                  "prompt": emb.QUERY_PROMPT + "<name>", "about": mseeds.TYPES_ABOUT,
                  "catalog_types": cat_hash, "ingredients": out})
    unknown = sorted({t for v in out.values() for t in v["exact_types"]} - set(catalog))
    print(f"wrote {len(out)} ingredients; exact types the catalog lacks: {unknown}")


def _clean(types: list[str]) -> list[str]:
    return list(dict.fromkeys(" ".join(t.lower().split()) for t in types if t.strip()))


# ─── review page ─────────────────────────────────────────────

def cmd_review(g: Gemini, args) -> None:
    data = json.loads((SEEDS / "product_profiles.json").read_text())
    rows = {p["id"]: p for p in _products()}
    profs = sorted(data["products"], key=lambda p: (not p["flags"], not p["fixes"], p["id"]))
    n_flag = sum(1 for p in profs if p["flags"])
    n_fix = sum(1 for p in profs if p["fixes"])

    def cell(v) -> str:
        return html.escape(", ".join(v) if isinstance(v, list) else str(v or "—"))

    body = []
    for p in profs:
        r = rows.get(p["id"], {})
        flags = "".join(f'<div class="flag"><b>{html.escape(f["field"])}</b>: '
                        f'{html.escape(f["note"] or "flagged")}</div>' for f in p["flags"])
        fixes = "".join(f'<div class="fix"><b>{html.escape(f["field"])}</b>: '
                        f'{cell(f["from"])} → {cell(f["to"])}. {html.escape(f["why"])}</div>'
                        for f in p["fixes"])
        if p.get("review"):
            fixes += f'<div class="kept">{html.escape(p["review"])}</div>'
        cls = "flagged" if p["flags"] else "fixed" if p["fixes"] else ""
        body.append(
            f'<tr class="{cls}"><td class="num">{p["id"]}</td>'
            f'<td><div class="name">{html.escape(p["name"])}</div>'
            f'<div class="desc">{html.escape(r.get("description", ""))}</div></td>'
            f'<td class="type">{cell(p["product_type"])}</td>'
            f'<td>{cell(" › ".join(p["taxonomy"]))}</td><td>{cell(p["form"])}</td>'
            f'<td>{cell(p["variant"])}</td><td>{cell(p["diet"])}</td>'
            f'<td>{cell(r.get("dietary_tags", ""))}</td>'
            f'<td>{flags}{fixes}</td></tr>')
    types = json.loads((SEEDS / "ingredient_types.json").read_text())
    type_rows = []
    for e in types["ingredients"].values():
        for f in e.get("fixes", []):
            type_rows.append(
                f'<tr class="fixed"><td><div class="name">{html.escape(e["name"])}</div></td>'
                f'<td>{html.escape(f["field"])}</td><td>{cell(f["from"])}</td>'
                f'<td>{cell(f["to"])}</td><td>{html.escape(f["why"])}</td></tr>')
    page = REVIEW_PAGE.format(
        n=len(profs), n_flag=n_flag, n_fix=n_fix, model=html.escape(data["model"]),
        when=datetime.now(UTC).date().isoformat(), rows="\n".join(body),
        types_model=html.escape(types["model"]), n_types=len(types["ingredients"]),
        type_rows="\n".join(type_rows) or '<tr><td colspan="5">None.</td></tr>')
    Path(args.out).expanduser().write_text(page, encoding="utf-8")
    print(f"wrote {args.out}: {len(profs)} rows, {n_flag} flagged, {n_fix} fixed by hand")


REVIEW_PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Product profiles review</title>
<style>
:root {{ --bg:#fbfbf9; --fg:#1d1d1b; --muted:#66655f; --line:#e2e0d8; --head:#f1efe8;
  --flag:#fff1e0; --flag-fg:#8a4b00; --fix:#e8f3ec; --fix-fg:#1f6b3a; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#161614; --fg:#ecebe6; --muted:#a19f97;
  --line:#33322e; --head:#22211e; --flag:#3a2a14; --flag-fg:#f3c483; --fix:#15301f;
  --fix-fg:#8fd8a8; }} }}
body {{ background:var(--bg); color:var(--fg); margin:0; padding:24px 16px;
  font:14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }}
h1 {{ font-size:20px; margin:0 0 6px; }} h2 {{ font-size:16px; margin:28px 0 6px; }}
p {{ color:var(--muted); margin:4px 0 16px; max-width:70ch; }}
table.narrow {{ min-width:640px; }}
.wrap {{ overflow-x:auto; border:1px solid var(--line); border-radius:8px; }}
table {{ border-collapse:collapse; width:100%; min-width:980px; }}
th, td {{ text-align:left; vertical-align:top; padding:8px 10px;
  border-bottom:1px solid var(--line); }}
th {{ background:var(--head); position:sticky; top:0; font-weight:600; }}
.num {{ font-variant-numeric:tabular-nums; color:var(--muted); }}
.name {{ font-weight:600; }} .desc {{ color:var(--muted); font-size:12.5px; }}
.type {{ font-weight:600; }}
tr.flagged td:first-child {{ box-shadow:inset 3px 0 var(--flag-fg); }}
tr.fixed td:first-child {{ box-shadow:inset 3px 0 var(--fix-fg); }}
.flag {{ background:var(--flag); color:var(--flag-fg); border-radius:4px; padding:2px 6px;
  margin:0 0 4px; font-size:12.5px; }}
.fix {{ background:var(--fix); color:var(--fix-fg); border-radius:4px; padding:2px 6px;
  margin:0 0 4px; font-size:12.5px; }}
.kept {{ color:var(--muted); font-size:12.5px; margin:0 0 4px; }}
</style></head><body>
<h1>Product profiles review</h1>
<p>{n} products profiled by {model} ({when}); {n_flag} flagged by the separate check pass
(listed first, with the check's note and what review decided), {n_fix} fixed by hand (each fix
says what changed and why). Every row was reviewed by hand; a row with neither a flag nor a fix
was checked and left as written. The contains-tags are the catalog's own and are never changed
by a profile.</p>
<div class="wrap"><table>
<thead><tr><th>#</th><th>Product</th><th>Type</th><th>Taxonomy</th><th>Form</th>
<th>Variant</th><th>Diet</th><th>Contains</th><th>Flags and fixes</th></tr></thead>
<tbody>
{rows}
</tbody></table></div>
<h2>Ingredient mapping fixes</h2>
<p>The {n_types} ingredient names of the library and the demo starters were mapped to product
types by {types_model} (seeds/ingredient_types.json). Fixed by hand after review:</p>
<div class="wrap"><table class="narrow">
<thead><tr><th>Ingredient</th><th>Field</th><th>Was</th><th>Now</th><th>Why</th></tr></thead>
<tbody>
{type_rows}
</tbody></table></div>
</body></html>
"""


# ─── golden set ──────────────────────────────────────────────

JUDGE_SYSTEM = """You judge grocery search results for recipe ingredients, with the labels of \
Amazon's Shopping Queries dataset (ESCI). For each ingredient you get the products a matcher \
offered for it. Label every product:

- E (exact): it is the ingredient; a shopper buying for this recipe line accepts it as the \
item itself (another brand, size or plain variant still counts).
- S (substitute): it is not the ingredient, but a cook could use it instead in this line.
- C (complement): it goes with the ingredient (same dish or meal) but cannot replace it.
- I (irrelevant): none of these.

Judge what the product is, from its name and description, not how its words look."""

JUDGE_TOOL = {
    "name": "submit_labels",
    "description": "One label per (ingredient, product) given.",
    "input_schema": {
        "type": "object", "required": ["labels"],
        "properties": {"labels": {"type": "array", "items": {
            "type": "object", "required": ["ingredient", "id", "label"],
            "properties": {
                "ingredient": {"type": "string"},
                "id": {"type": "integer"},
                "label": {"type": "string", "enum": ["E", "S", "C", "I"]},
            }}}},
    },
}


def cmd_golden(g: Gemini, args) -> None:
    from evals import substitutes_eval

    rows = {p["id"]: p for p in _products()}
    pools = substitutes_eval.candidate_pools(depth=args.depth)

    def key(name: str, ids: list[int]) -> str:
        # The judge's model is part of the key: one golden set is labelled by one judge.
        body = json.dumps([PROMPT_VERSION, MODEL, name, [_row(rows[i]) for i in ids]],
                          sort_keys=True)
        return hashlib.sha256(body.encode()).hexdigest()

    todo = [(n, ids) for n, ids in pools.items() if g.get("judge", key(n, ids)) is None]
    print(f"golden: {len(pools) - len(todo)} cached, {len(todo)} to ask "
          f"({sum(len(v) for v in pools.values())} pairs)")
    for i, batch in enumerate(_batches(todo, GOLDEN_BATCH)):
        if not batch:
            continue
        user = json.dumps({"ingredients": [
            {"ingredient": n, "products": [{"id": r["id"], "name": r["name"],
                                            "description": r["description"]}
                                           for r in (_row(rows[i]) for i in ids)]}
            for n, ids in batch]}, ensure_ascii=False)
        reply = g.call(f"golden {i + 1}", JUDGE_SYSTEM, user, JUDGE_TOOL,
                       max_tokens=25 * sum(len(ids) for _, ids in batch) + 1000)
        got: dict[str, dict[int, str]] = {}
        for r in reply["input"].get("labels", []):
            got.setdefault(r["ingredient"].lower(), {})[int(r["id"])] = r["label"]
        for n, ids in batch:
            labels = got.get(n.lower(), {})
            if all(i in labels for i in ids):
                g.put("judge", key(n, ids), {str(i): labels[i] for i in ids})
            else:
                print(f"  {n}: {len([i for i in ids if i not in labels])} unlabelled; rerun")
    out = {}
    for n, ids in pools.items():
        labels = g.get("judge", key(n, ids))
        if labels is not None:
            out[n] = {"labels": {i: labels[i] for i in sorted(labels, key=int)}}
    _write(GOLDEN, {"v": 1, "judge": MODEL, "labels": "E exact, S substitute, C complement, "
                    "I irrelevant (ESCI)", "gains": substitutes_eval.GAINS,
                    "depth": args.depth, "about": substitutes_eval.GOLDEN_ABOUT,
                    "ingredients": out})
    print(f"wrote {len(out)} ingredients to {GOLDEN.relative_to(ROOT)}")


# ─── main ────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("step", choices=["profiles", "check", "embed", "types", "review", "golden",
                                     "gemini-vectors", "spent"])
    ap.add_argument("--cache-dir", type=Path,
                    default=Path("~/.cache/pantry-matching").expanduser())
    ap.add_argument("--budget", type=int, default=40, help="Gemini HTTP attempts allowed, "
                    "counted across every run that shares the cache dir")
    ap.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    ap.add_argument("--embedding-model", default="embeddinggemma")
    ap.add_argument("--gemini-embedding-model", default="gemini:gemini-embedding-001")
    ap.add_argument("--out", default="~/Downloads/pantry-design/product-profiles-review.html")
    ap.add_argument("--depth", type=int, default=20, help="golden: candidates per method")
    ap.add_argument("--model", default=MODEL, help="a Gemini flash model, gemini:<name>")
    ap.add_argument("--retries", type=int, default=2, help="tries after a 429 or 5xx, 90 s "
                    "apart")
    args = ap.parse_args(argv)
    MODEL = args.model
    args.cache_dir = args.cache_dir.expanduser()
    g = Gemini(args.cache_dir, args.budget, args.retries)
    if args.step == "spent":
        print(f"{g.spent()} of {args.budget} Gemini calls spent")
        return
    {"profiles": cmd_profiles, "check": cmd_check, "embed": cmd_embed, "types": cmd_types,
     "review": cmd_review, "golden": cmd_golden,
     "gemini-vectors": cmd_gemini_vectors}[args.step](g, args)
    print(f"Gemini calls spent so far: {g.spent()} of {args.budget}")


if __name__ == "__main__":
    main()
