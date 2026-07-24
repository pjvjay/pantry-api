"""
Tiny persistence layer — SQLite by default, seeded from JSON on demand.

`python -m pantry_planner.db seed` (re)builds the DB from seeds/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from sqlalchemy import Column, Float, Integer, String, create_engine
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
            )
            for r in rows
        ]


# ─── Seed loader ─────────────────────────────────────────────

def seed_from_json() -> None:
    """Wipe the DB and load fresh data from seeds/*.json."""
    init_schema()
    with Session(engine()) as s:
        # Clear existing
        s.query(RecipeIngredientRow).delete()
        s.query(RecipeRow).delete()
        s.query(ProductRow).delete()

        # Products
        with (SEEDS_DIR / "products.json").open() as f:
            products = json.load(f)
        for p in products:
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
            ))

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
