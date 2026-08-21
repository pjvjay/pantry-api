"""
Tiny persistence layer — SQLite by default, seeded from JSON on demand.

`python -m pantry_planner.db seed` (re)builds the DB from seeds/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

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


# ─── Engine / session helpers ────────────────────────────────

def engine():
    return create_engine(settings().db_url, echo=False, future=True)


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
        # Clear existing
        s.query(RecipeIngredientRow).delete()
        s.query(RecipeRow).delete()
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
        s.commit()
    from .config import redact_db_url
    print(f"Seeded {len(products)} products, {len(recipes)} recipes into {redact_db_url(settings().db_url)}")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "seed":
        seed_from_json()
    else:
        print("Usage: python -m pantry_planner.db seed", file=sys.stderr)
        sys.exit(1)
