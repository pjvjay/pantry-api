"""
Tiny persistence layer — SQLite by default, seeded from JSON on demand.

`python -m pantry_planner.db seed` (re)builds the DB from seeds/.
"""
from __future__ import annotations

import functools
import json
import sys
from collections.abc import Iterable
from pathlib import Path
from typing import NamedTuple

from sqlalchemy import Boolean, Column, Float, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Session

from .config import settings
from .models import Product, Recipe, RecipeIngredient

SEEDS_DIR = Path(__file__).resolve().parent.parent / "seeds"


class Base(DeclarativeBase):
    pass


class ProductRow(Base):
    __tablename__ = "products"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    description = Column(String, nullable=False, default="")
    price = Column(Float, nullable=False)
    category = Column(String, nullable=True)
    # 0002_product_attributes — mirrors pantry-db migrations
    subcategory = Column(String, nullable=False, default="")
    dietary_tags = Column(String, nullable=False, default="")
    unit_size = Column(String, nullable=False, default="")
    unit_qty = Column(Float, nullable=True)
    unit_uom = Column(String, nullable=False, default="")
    # 0003_stores_reviews_terms — mirrors pantry-db migrations
    brand = Column(String, nullable=False, default="")


class StoreRow(Base):
    __tablename__ = "stores"
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    lat = Column(Float, nullable=False)
    lon = Column(Float, nullable=False)
    address = Column(String, nullable=False, default="")


class StoreProductRow(Base):
    __tablename__ = "store_products"
    store_id = Column(Integer, primary_key=True)
    product_id = Column(Integer, primary_key=True)
    price = Column(Float, nullable=False)


class ReviewRow(Base):
    __tablename__ = "reviews"
    id = Column(Integer, primary_key=True)
    product_id = Column(Integer, nullable=False, index=True)
    rating = Column(Integer, nullable=False)
    comment = Column(String, nullable=False, default="")
    created_at = Column(String, nullable=False, default="")


class ProductTermRow(Base):
    __tablename__ = "product_terms"
    term = Column(String, primary_key=True)
    product_id = Column(Integer, primary_key=True)


class ProductOriginRow(Base):
    """0004_product_origins — mirrors pantry-db migrations.

    Cache of LLM-resolved countries of origin. Heuristic results are NOT
    cached (deterministic and free to recompute); only answers that cost
    an API call land here, so each product pays for at most one call ever.
    """
    __tablename__ = "product_origins"
    product_id = Column(Integer, primary_key=True)
    country = Column(String, nullable=False)
    # 0004 kept confidence numeric. Evidence carries high/medium/low, so the
    # two are mapped at the boundary (see origins._conf_label / _conf_float)
    # rather than migrating a column that older readers still use.
    confidence = Column(Float, nullable=False, default=0.0)
    source = Column(String, nullable=False, default="llm")
    reasoning = Column(String, nullable=False, default="")
    resolved_at = Column(String, nullable=False, default="")
    # 0005_origin_evidence
    status = Column(String, nullable=False, default="unknown")
    claim_type = Column(String, nullable=False, default="unknown")
    verbatim = Column(String, nullable=False, default="")
    ingredient_origin = Column(String, nullable=False, default="")
    manufactured_in = Column(String, nullable=False, default="")
    evidence_count = Column(Integer, nullable=False, default=0)


class ProductOriginEvidenceRow(Base):
    """0005_origin_evidence — many rows per product, one per observation.

    Rows are allowed to contradict each other; reconciliation happens at
    read time so a disagreement stays visible instead of being overwritten.
    """
    __tablename__ = "product_origin_evidence"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, nullable=False, index=True)
    source = Column(String, nullable=False)
    source_ref = Column(String, nullable=False, default="")
    claim_type = Column(String, nullable=False, default="unknown")
    verbatim = Column(String, nullable=False, default="")
    ingredient_origin = Column(String, nullable=False, default="")
    manufactured_in = Column(String, nullable=False, default="")
    confidence = Column(String, nullable=False, default="low")
    importer_only = Column(Boolean, nullable=False, default=False)
    note = Column(String, nullable=False, default="")
    observed_at = Column(String, nullable=False, default="")


class OriginSubmissionRow(Base):
    """0006_origin_submissions — a reviewed queue for agent-submitted claims.

    A pending claim is never evidence: the resolver reads only
    product_origin_evidence, so nothing here changes a plan until a reviewer
    approves. Approval COPIES the claim into evidence (source "agent-label")
    and records the new row's id in `evidence_id`; the submission stays, with
    who reviewed it, when and why. Rejections are kept so the same wording
    resubmitted returns the earlier verdict instead of re-entering the queue.
    Columns mirror the migration exactly (names, nullability, defaults).
    """
    __tablename__ = "origin_submissions"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, nullable=False, index=True)
    claim_type = Column(String, nullable=False)
    country = Column(String, nullable=False)
    ingredient_origin = Column(String, nullable=False, default="")
    manufactured_in = Column(String, nullable=False, default="")
    verbatim = Column(String, nullable=False)
    confidence = Column(String, nullable=False, default="low")
    importer_only = Column(Boolean, nullable=False, default=False)
    note = Column(String, nullable=False, default="")
    source_ref = Column(String, nullable=False, default="")
    submitted_by = Column(String, nullable=False, default="")
    submitted_at = Column(String, nullable=False, default="")
    status = Column(String, nullable=False, default="pending", index=True)
    reviewed_by = Column(String, nullable=False, default="")
    reviewed_at = Column(String, nullable=False, default="")
    review_note = Column(String, nullable=False, default="")
    evidence_id = Column(Integer, nullable=True)


class PriceObservationRow(Base):
    """0005_origin_evidence — timestamped readings scraped from store pages.

    Distinct from store_products (the seeded synthetic catalog): `branch` is
    the store the page actually showed, which is session state resolved from
    the client IP and cannot be chosen.
    """
    __tablename__ = "price_observations"
    id = Column(Integer, primary_key=True, autoincrement=True)
    product_id = Column(Integer, nullable=True, index=True)
    item_query = Column(String, nullable=False, default="")
    store = Column(String, nullable=False, default="")
    branch = Column(String, nullable=False, default="")
    product_name = Column(String, nullable=False, default="")
    size = Column(String, nullable=False, default="")
    price = Column(Float, nullable=True)
    price_text = Column(String, nullable=False, default="")
    unit_price = Column(String, nullable=False, default="")
    link = Column(String, nullable=False, default="")
    observed_at = Column(String, nullable=False, default="")
    run_id = Column(String, nullable=False, default="")


class RecipeRow(Base):
    __tablename__ = "recipes"
    slug = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    servings = Column(Integer, nullable=False, default=1)


class RecipeIngredientRow(Base):
    __tablename__ = "recipe_ingredients"
    recipe_slug = Column(String, primary_key=True)
    line_no = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=True)


class RecipeLineAmountRow(Base):
    """0007_recipe_line_amounts — how much of each library recipe line to buy.

    A table of its own rather than columns on recipe_ingredients, so an API
    that knows about amounts runs against a database that does not have them
    yet (load_line_amounts returns None) and the classic plan path, which
    reads recipe_ingredients only, is untouched. The library's amounts are
    demo house amounts (synthetic, labelled), not a cookbook's. quantity NULL
    means not stated, and then `note` says why. Columns mirror the migration
    exactly; the foreign key to recipe_ingredients (ON DELETE CASCADE) lives
    in SQL only, as for every other table here.
    """
    __tablename__ = "recipe_line_amounts"
    recipe_slug = Column(String, primary_key=True)
    line_no = Column(Integer, primary_key=True)
    quantity = Column(Float, nullable=True)
    unit = Column(String, nullable=False, default="", server_default="")
    note = Column(String, nullable=False, default="", server_default="")


class NutrientSourceRow(Base):
    """0008_nutrition — one row per reference dataset: its licence, the attribution line
    shown wherever one of its numbers is, and `edition`, which says which edition the values
    are believed to be and how sure that is. Columns mirror the migration exactly."""
    __tablename__ = "nutrient_sources"
    source = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    publisher = Column(String, nullable=False, default="", server_default="")
    edition = Column(String, nullable=False, default="", server_default="")
    licence = Column(String, nullable=False, default="", server_default="")
    licence_url = Column(String, nullable=False, default="", server_default="")
    attribution = Column(String, nullable=False, default="", server_default="")
    url = Column(String, nullable=False, default="", server_default="")
    retrieved_at = Column(String, nullable=False, default="", server_default="")


class NutrientFoodRow(Base):
    """0008_nutrition — one reference food: a generic food's published values, never a
    product's label. ref_id is '<source>:<food code>'; description is verbatim; state_note
    says what was measured ("raw, meat only"). The foreign key to nutrient_sources and the
    UNIQUE (source, source_food_id) live in SQL only."""
    __tablename__ = "nutrient_foods"
    ref_id = Column(String, primary_key=True)
    source = Column(String, nullable=False)
    source_food_id = Column(String, nullable=False)
    food_code = Column(String, nullable=False, default="", server_default="")
    description = Column(String, nullable=False)
    state_note = Column(String, nullable=False, default="", server_default="")


class NutrientAmountRow(Base):
    """0008_nutrition — a nutrient per 100 g of a reference food. A nutrient the source did
    not publish has no row: unknown, which code never reads as 0. The CHECKs (the eight
    nutrient names, per_100g >= 0) and the cascade live in SQL only."""
    __tablename__ = "nutrient_amounts"
    ref_id = Column(String, primary_key=True)
    nutrient = Column(String, primary_key=True)
    per_100g = Column(Float, nullable=False)
    source_code = Column(String, nullable=False, default="", server_default="")


class IngredientNutrientMapRow(Base):
    """0008_nutrition — ingredient key (units.tokens() of a line's name, joined by spaces)
    -> its reference food. match_kind 'none' (ref_id NULL) means reviewed and nothing fits;
    a key with no row has not been reviewed. Both stay unknown, for different reasons."""
    __tablename__ = "ingredient_nutrient_map"
    ingredient_key = Column(String, primary_key=True)
    ref_id = Column(String, nullable=True)
    match_kind = Column(String, nullable=False)
    note = Column(String, nullable=False, default="", server_default="")
    reviewed_at = Column(String, nullable=False, default="", server_default="")


class NutrientMeasureRow(Base):
    """0009_nutrient_measures — the source's own weight for a volume or a count of a food
    ("15ml" = 13.682 g of olive oil). The only way a millilitre or count line becomes grams:
    no density is ever assumed. volume_ml is set for volume measures only."""
    __tablename__ = "nutrient_measures"
    ref_id = Column(String, primary_key=True)
    measure = Column(String, primary_key=True)
    grams = Column(Float, nullable=False)
    volume_ml = Column(Float, nullable=True)
    verbatim = Column(String, nullable=False)
    source_ref = Column(String, nullable=False, default="", server_default="")


# ─── Engine / session helpers ────────────────────────────────

@functools.lru_cache(maxsize=8)
def _engine_for(url: str):
    return create_engine(url, echo=False, future=True)


def engine():
    """One Engine per DB URL.

    Previously every call built a fresh Engine, so every Session() was a new
    connect+auth handshake and a lingering backend on Postgres — invisible on
    SQLite, ~3x slower per query on Postgres, and ten idle backends after a
    single plan. Keyed by URL (not a singleton) because tests point DB_URL at
    a different SQLite file per module.
    """
    return _engine_for(settings().db_url)


def init_schema() -> None:
    Base.metadata.create_all(engine())


# ─── Load / save ─────────────────────────────────────────────

def load_recipe(slug: str) -> Recipe:
    with Session(engine()) as s:
        r = s.get(RecipeRow, slug)
        if not r:
            raise ValueError(f"Recipe not found: {slug!r}")
        ings = (
            s.query(RecipeIngredientRow)
            .filter_by(recipe_slug=slug)
            .order_by(RecipeIngredientRow.line_no)
            .all()
        )
        return Recipe(
            slug=r.slug,
            name=r.name,
            servings=r.servings,
            ingredients=[
                RecipeIngredient(line_no=i.line_no, name=i.name, category=i.category)
                for i in ings
            ],
        )


class LineAmount(NamedTuple):
    quantity: float | None
    unit: str
    note: str


def load_line_amounts(slugs: list[str]) -> dict[str, dict[int, LineAmount]] | None:
    """slug -> line_no -> its amount, for the recipes that have amounts. None when the
    recipe_line_amounts table is not deployed yet (pantry-db migration 0007): unknown,
    which a caller must not read as "no amounts stated"."""
    from sqlalchemy.exc import OperationalError, ProgrammingError

    if not slugs:
        return {}
    try:
        with Session(engine()) as s:
            rows = (s.query(RecipeLineAmountRow)
                    .filter(RecipeLineAmountRow.recipe_slug.in_(slugs))
                    .order_by(RecipeLineAmountRow.recipe_slug, RecipeLineAmountRow.line_no)
                    .all())
            out: dict[str, dict[int, LineAmount]] = {}
            for r in rows:
                out.setdefault(str(r.recipe_slug), {})[int(r.line_no)] = LineAmount(
                    None if r.quantity is None else float(r.quantity),
                    str(r.unit or ""), str(r.note or ""))
            return out
    except (OperationalError, ProgrammingError) as e:
        if _table_missing(e, RecipeLineAmountRow.__tablename__):
            return None
        raise


def _table_missing(exc: BaseException, table: str) -> bool:
    """True when a DB error means `table` does not exist: SQLite says "no such table",
    Postgres raises UndefinedTable ("relation ... does not exist")."""
    text = str(exc).lower()
    return table in text and ("no such table" in text or "does not exist" in text
                              or "undefinedtable" in text)


# ─── Nutrition reference data (0008, 0009) ──────────────────

NUTRIENT_KEYS = ("energy_kcal", "protein_g", "fat_g", "satfat_g", "carbohydrate_g",
                 "fibre_g", "sugars_g", "sodium_mg")
SOURCE_FIELDS = ("name", "publisher", "edition", "licence", "licence_url", "attribution",
                 "url", "retrieved_at")


class NutrientFood(NamedTuple):
    ref_id: str
    source: str
    source_food_id: str
    food_code: str
    description: str
    state_note: str
    per_100g: dict[str, float]          # only the nutrients the source published


class NutrientMeasure(NamedTuple):
    measure: str
    grams: float
    volume_ml: float | None
    verbatim: str
    source_ref: str


class NutrientMapEntry(NamedTuple):
    ref_id: str | None                  # None: match_kind 'none'
    match_kind: str
    note: str


class Reference(NamedTuple):
    """What load_reference read. measures_deployed False: 0008 is there but 0009 is not, so
    no volume or count line can be converted (said so per line, never assumed)."""
    sources: dict[str, dict]
    foods: dict[str, NutrientFood]
    measures: dict[str, list[NutrientMeasure]]
    map: dict[str, NutrientMapEntry]
    measures_deployed: bool = True


def load_reference(keys: Iterable[str] | None = None) -> Reference | None:
    """The nutrition reference for these ingredient keys (all keys when None): their map
    rows, the foods those name with their published nutrients and measures, and every
    source. None when the 0008 tables are not deployed: unknown, which a caller must say,
    never read as "no nutrients"."""
    from sqlalchemy.exc import OperationalError, ProgrammingError

    tables = [NutrientSourceRow, NutrientFoodRow, NutrientAmountRow, IngredientNutrientMapRow]
    try:
        with Session(engine()) as s:
            q = s.query(IngredientNutrientMapRow).order_by(IngredientNutrientMapRow.ingredient_key)
            wanted = None if keys is None else sorted(set(keys))
            if wanted is not None:
                q = q.filter(IngredientNutrientMapRow.ingredient_key.in_(wanted))
            # An empty key list skips the IN () query (a syntax error on Postgres) but still
            # reads the sources, which also proves the tables exist.
            map_rows = q.all() if wanted is None or wanted else []
            mapping = {str(r.ingredient_key): NutrientMapEntry(
                None if r.ref_id is None else str(r.ref_id), str(r.match_kind), str(r.note or ""))
                for r in map_rows}
            ref_ids = sorted({m.ref_id for m in mapping.values() if m.ref_id is not None})
            food_rows = (s.query(NutrientFoodRow).filter(NutrientFoodRow.ref_id.in_(ref_ids))
                         .order_by(NutrientFoodRow.ref_id).all()) if ref_ids else []
            amount_rows = (s.query(NutrientAmountRow)
                           .filter(NutrientAmountRow.ref_id.in_(ref_ids)).all()) if ref_ids else []
            sources = {str(r.source): {"source": str(r.source),
                                       **{k: str(getattr(r, k) or "") for k in SOURCE_FIELDS}}
                       for r in s.query(NutrientSourceRow).order_by(NutrientSourceRow.source)}
    except (OperationalError, ProgrammingError) as e:
        if any(_table_missing(e, t.__tablename__) for t in tables):
            return None
        raise
    per: dict[str, dict[str, float]] = {}
    for a in amount_rows:
        per.setdefault(str(a.ref_id), {})[str(a.nutrient)] = float(a.per_100g)
    foods = {str(f.ref_id): NutrientFood(
        str(f.ref_id), str(f.source), str(f.source_food_id), str(f.food_code or ""),
        str(f.description), str(f.state_note or ""),
        {n: per[str(f.ref_id)][n] for n in NUTRIENT_KEYS if n in per.get(str(f.ref_id), {})})
        for f in food_rows}
    measures, deployed = _load_measures(ref_ids)
    return Reference(sources=sources, foods=foods, measures=measures, map=mapping,
                     measures_deployed=deployed)


def _load_measures(ref_ids: list[str]) -> tuple[dict[str, list[NutrientMeasure]], bool]:
    """ref_id -> its measures, and False when 0009 is not deployed (its own transaction, so a
    missing table there does not cost the 0008 data)."""
    from sqlalchemy.exc import OperationalError, ProgrammingError

    if not ref_ids:
        return {}, True
    try:
        with Session(engine()) as s:
            rows = (s.query(NutrientMeasureRow).filter(NutrientMeasureRow.ref_id.in_(ref_ids))
                    .order_by(NutrientMeasureRow.ref_id, NutrientMeasureRow.measure).all())
    except (OperationalError, ProgrammingError) as e:
        if _table_missing(e, NutrientMeasureRow.__tablename__):
            return {}, False
        raise
    out: dict[str, list[NutrientMeasure]] = {}
    for m in rows:
        out.setdefault(str(m.ref_id), []).append(NutrientMeasure(
            str(m.measure), float(m.grams), None if m.volume_ml is None else float(m.volume_ml),
            str(m.verbatim), str(m.source_ref or "")))
    return out, True


def nutrient_rows(data: dict) -> dict[str, list[tuple]]:
    """seeds/nutrients.json as rows of the five tables, in file order (KEEP-IN-SYNC:
    pantry-db scripts/gen-seed-sql.py nutrient_rows). source_food_id is the food's own field
    when the file has one, else its food_code; source_code is the source's nutrient id. A
    nutrient missing from per_100g gets no row: unknown, never 0."""
    sources = [(s["source"], *(s.get(k) or "" for k in SOURCE_FIELDS)) for s in data["sources"]]
    foods, amounts = [], []
    for f in data["foods"]:
        foods.append((f["ref_id"], f["source"], str(f.get("source_food_id", f["food_code"])),
                      str(f.get("food_code", "")), f["description"], f.get("state_note") or ""))
        codes = f.get("source_codes") or {}
        for n in NUTRIENT_KEYS:
            if n in f["per_100g"]:
                c = codes.get(n) or {}
                code = c.get("nutrient_code", c.get("nutrient_name_id"))
                amounts.append((f["ref_id"], n, float(f["per_100g"][n]),
                                "" if code is None else str(code)))
    measures = [(m["ref_id"], m["measure"], float(m["grams"]),
                 None if m.get("volume_ml") is None else float(m["volume_ml"]),
                 m["verbatim"], m.get("source_ref") or "") for m in data.get("measures", [])]
    mapping = [(m["ingredient_key"], m.get("ref_id"), m["match_kind"], m.get("note") or "",
                m.get("reviewed_at") or "") for m in data["map"]]
    return {"sources": sources, "foods": foods, "amounts": amounts, "measures": measures,
            "map": mapping}


def load_all_recipes() -> list[Recipe]:
    with Session(engine()) as s:
        slugs = [r.slug for r in s.query(RecipeRow).order_by(RecipeRow.slug).all()]
    return [load_recipe(slug) for slug in slugs]


def load_all_products() -> list[Product]:
    with Session(engine()) as s:
        rows = s.query(ProductRow).all()
        return [
            Product(
                id=r.id,
                name=r.name,
                description=r.description,
                price=r.price,
                category=r.category,
                subcategory=r.subcategory or None,
                dietary_tags=r.dietary_tags or "",
                unit_size=r.unit_size or "",
                unit_qty=r.unit_qty,
                unit_uom=r.unit_uom or "",
                brand=r.brand or "",
            )
            for r in rows
        ]


def load_origin_evidence(product_ids: list[int] | None = None
                         ) -> dict[int, list[ProductOriginEvidenceRow]]:
    """All evidence rows, grouped by product. Contradictions are preserved."""
    with Session(engine()) as s:
        q = s.query(ProductOriginEvidenceRow)
        if product_ids is not None:
            if not product_ids:
                return {}
            q = q.filter(ProductOriginEvidenceRow.product_id.in_(product_ids))
        rows = q.order_by(ProductOriginEvidenceRow.id).all()
        s.expunge_all()
    out: dict[int, list[ProductOriginEvidenceRow]] = {}
    for r in rows:
        out.setdefault(int(r.product_id), []).append(r)
    return out


# An observation is identified by what it says and where it came from —
# not by when it was read. Re-ingesting the same file must be a no-op.
# importer_only and confidence are part of the identity: a corrected
# re-read of the same label ("that address was the importer after all")
# changes the answer and must not be discarded as a duplicate.
_EVIDENCE_KEY = ("product_id", "source", "source_ref", "claim_type",
                 "verbatim", "ingredient_origin", "manufactured_in",
                 "importer_only", "confidence")


def save_origin_evidence(records: list[dict]) -> int:
    """Append evidence rows, skipping exact duplicates. Returns rows written.

    Append-only by design: a later lookup disagreeing with an earlier one is
    a fact about the sources, not a correction to be applied silently. But an
    identical re-read is not new evidence — evidence_count is surfaced as
    corroboration, so duplicates would overstate how well-supported a
    provenance claim is.
    """
    if not records:
        return 0
    written = 0
    with Session(engine()) as s:
        existing = {
            tuple(getattr(r, k) for k in _EVIDENCE_KEY)
            for r in s.query(ProductOriginEvidenceRow).all()
        }
        for r in records:
            key = tuple(r.get(k) for k in _EVIDENCE_KEY)
            if key in existing:
                continue
            existing.add(key)
            s.add(ProductOriginEvidenceRow(**r))
            written += 1
        s.commit()
    return written


def load_resolved_origins(product_ids: list[int] | None = None
                          ) -> dict[int, ProductOriginRow]:
    """The per-product resolved summary rows."""
    with Session(engine()) as s:
        q = s.query(ProductOriginRow)
        if product_ids is not None:
            if not product_ids:
                return {}
            q = q.filter(ProductOriginRow.product_id.in_(product_ids))
        rows = q.all()
        s.expunge_all()
        return {int(r.product_id): r for r in rows}


def save_resolved_origins(origins: list[dict]) -> None:
    """Upsert resolved summaries keyed by product_id."""
    if not origins:
        return
    with Session(engine()) as s:
        for o in origins:
            row = s.get(ProductOriginRow, o["product_id"]) or ProductOriginRow(
                product_id=o["product_id"])
            for field, value in o.items():
                if field != "product_id":
                    setattr(row, field, value)
            s.add(row)
        s.commit()


def save_price_observations(observations: list[dict]) -> int:
    """Append scraped price readings. Returns how many were written."""
    if not observations:
        return 0
    with Session(engine()) as s:
        for o in observations:
            s.add(PriceObservationRow(**o))
        s.commit()
    return len(observations)


def load_price_observations(product_id: int | None = None,
                            limit: int = 200) -> list[PriceObservationRow]:
    with Session(engine()) as s:
        q = s.query(PriceObservationRow)
        if product_id is not None:
            q = q.filter(PriceObservationRow.product_id == product_id)
        rows = q.order_by(PriceObservationRow.id.desc()).limit(limit).all()
        s.expunge_all()
        return rows


# ─── Seed loader ─────────────────────────────────────────────

def seed_from_json() -> None:
    """Wipe the DB and load fresh data from seeds/*.json. Store prices,
    reviews, brands, and the product_terms inverted index are synthesized
    deterministically (storeseed.py — same algorithm as pantry-db's
    seed generator)."""
    from . import storeseed

    init_schema()
    with Session(engine()) as s:
        # Clear existing (amounts before the lines they belong to: in Postgres they cascade
        # from recipe_ingredients; nutrition children before the foods and sources they
        # point at)
        s.query(IngredientNutrientMapRow).delete()
        s.query(NutrientMeasureRow).delete()
        s.query(NutrientAmountRow).delete()
        s.query(NutrientFoodRow).delete()
        s.query(NutrientSourceRow).delete()
        s.query(RecipeLineAmountRow).delete()
        s.query(RecipeIngredientRow).delete()
        s.query(RecipeRow).delete()
        s.query(OriginSubmissionRow).delete()
        s.query(ProductOriginEvidenceRow).delete()
        s.query(PriceObservationRow).delete()
        s.query(ProductOriginRow).delete()
        s.query(ProductTermRow).delete()
        s.query(ReviewRow).delete()
        s.query(StoreProductRow).delete()
        s.query(StoreRow).delete()
        s.query(ProductRow).delete()

        # Products (+ synthetic store offers / reviews / terms per product)
        with (SEEDS_DIR / "products.json").open() as f:
            products = json.load(f)
        review_id = 0
        for p in products:
            brand = storeseed.brand_for(p)
            s.add(ProductRow(
                id=p["id"],
                name=p["name"],
                description=p.get("description", ""),
                price=float(p["price"]),
                category=p.get("category"),
                subcategory=p.get("subcategory", ""),
                dietary_tags=p.get("dietary_tags", ""),
                unit_size=p.get("unit_size", ""),
                unit_qty=p.get("unit_qty"),
                unit_uom=p.get("unit_uom", ""),
                brand=brand,
            ))
            for sid, *_ in storeseed.STORES:
                s.add(StoreProductRow(
                    store_id=sid, product_id=p["id"],
                    price=storeseed.store_price(sid, p["id"], float(p["price"]))))
            for rating, comment, created in storeseed.reviews_for(p, brand):
                review_id += 1
                s.add(ReviewRow(id=review_id, product_id=p["id"], rating=rating,
                                comment=comment, created_at=created))
            for term in storeseed.product_terms(p):
                s.add(ProductTermRow(term=term, product_id=p["id"]))

        # Stores
        for sid, name, lat, lon, address in storeseed.STORES:
            s.add(StoreRow(id=sid, name=name, lat=lat, lon=lon, address=address))

        # Recipes
        with (SEEDS_DIR / "recipes.json").open() as f:
            recipes = json.load(f)
        for r in recipes:
            s.add(RecipeRow(
                slug=r["slug"],
                name=r["name"],
                servings=r.get("servings", 1),
            ))
            for i, ing in enumerate(r["ingredients"], start=1):
                s.add(RecipeIngredientRow(
                    recipe_slug=r["slug"],
                    line_no=i,
                    name=ing["name"] if isinstance(ing, dict) else ing,
                    category=ing.get("category") if isinstance(ing, dict) else None,
                ))
                # A line with any of the three keys has an amount row; a bare line has none.
                if isinstance(ing, dict) and {"quantity", "unit", "note"} & ing.keys():
                    s.add(RecipeLineAmountRow(
                        recipe_slug=r["slug"], line_no=i,
                        quantity=None if ing.get("quantity") is None else float(ing["quantity"]),
                        unit=ing.get("unit") or "", note=ing.get("note") or ""))

        # Nutrition reference data (0008, 0009)
        with (SEEDS_DIR / "nutrients.json").open(encoding="utf-8") as f:
            rows = nutrient_rows(json.load(f))
        for row in rows["sources"]:
            fields = dict(zip(SOURCE_FIELDS, row[1:], strict=True))
            s.add(NutrientSourceRow(source=row[0], **fields))
        for ref_id, source, sfid, code, description, state_note in rows["foods"]:
            s.add(NutrientFoodRow(ref_id=ref_id, source=source, source_food_id=sfid,
                                  food_code=code, description=description, state_note=state_note))
        for ref_id, nutrient, per_100g, code in rows["amounts"]:
            s.add(NutrientAmountRow(ref_id=ref_id, nutrient=nutrient, per_100g=per_100g,
                                    source_code=code))
        for ref_id, measure, grams, volume_ml, verbatim, source_ref in rows["measures"]:
            s.add(NutrientMeasureRow(ref_id=ref_id, measure=measure, grams=grams,
                                     volume_ml=volume_ml, verbatim=verbatim,
                                     source_ref=source_ref))
        for key, ref_id, kind, note, reviewed_at in rows["map"]:
            s.add(IngredientNutrientMapRow(ingredient_key=key, ref_id=ref_id, match_kind=kind,
                                           note=note, reviewed_at=reviewed_at))
        s.commit()
    from .config import redact_db_url
    print(f"Seeded {len(products)} products, {len(recipes)} recipes into {redact_db_url(settings().db_url)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_from_json()
    else:
        print("Usage: python -m pantry_planner.db seed", file=sys.stderr)
        sys.exit(1)
