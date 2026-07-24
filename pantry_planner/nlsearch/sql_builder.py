"""TRANSLATION: compile the validated parse into parameterized SQL.

Every predicate comes from the fixed pattern menu (PRICE_CEILING,
DIETARY_EXCLUDE/REQUIRE, CATEGORY/SUBCATEGORY_SCOPE, INGREDIENT_MATCH,
SIZE_RANGE, VALUE_SORT). Values only ever travel as bound parameters —
the LLM's words never touch the SQL string.

One UNION ALL query covers the whole recipe: each ingredient gets a
labeled block with its own token/size predicates, sharing the constraint
WHERE. Portable across SQLite (local dev) and Postgres (deployment).
"""
from __future__ import annotations

from .schemas import Constraints, IngredientSpec
from .units import normalize_quantity, tokens

PRODUCT_COLS = ("id, name, description, price, category, subcategory, "
                "dietary_tags, unit_size, unit_qty, unit_uom")
SIZE_CAP_FACTOR = 6      # exclude packs > 6x the need (catering boxes)
PER_INGREDIENT_LIMIT = 8


def _constraint_where(c: Constraints, params: dict) -> list[str]:
    where: list[str] = ["1=1"]
    if c.max_item_price is not None:                       # PRICE_CEILING
        where.append("price <= :max_item_price")
        params["max_item_price"] = c.max_item_price
    for i, tag in enumerate(c.exclude_tags):               # DIETARY_EXCLUDE
        where.append(f"(',' || dietary_tags || ',') NOT LIKE :xtag{i}")
        params[f"xtag{i}"] = f"%,{tag},%"
    for i, tag in enumerate(c.require_tags):               # DIETARY_REQUIRE
        where.append(f"(',' || dietary_tags || ',') LIKE :rtag{i}")
        params[f"rtag{i}"] = f"%,{tag},%"

    def scope(col: str, values: list[str], *, exclude: bool, key: str) -> None:
        if values:                                         # CATEGORY/SUBCATEGORY_SCOPE
            keys = ",".join(f":{key}{i}" for i in range(len(values)))
            where.append(f"{col} {'NOT IN' if exclude else 'IN'} ({keys})")
            params.update({f"{key}{i}": v for i, v in enumerate(values)})

    scope("category", c.categories, exclude=False, key="cat")
    scope("category", c.exclude_categories, exclude=True, key="xcat")
    scope("subcategory", c.subcategories, exclude=False, key="sub")
    scope("subcategory", c.exclude_subcategories, exclude=True, key="xsub")
    return where


def _ingredient_block(n: int, ing: IngredientSpec, constraint_sql: str,
                      params: dict, *, with_tokens: bool = True,
                      with_size: bool = True) -> str:
    """One labeled block of the UNION ALL."""
    preds = [constraint_sql]

    if with_tokens:                                        # INGREDIENT_MATCH
        toks = tokens(ing.name)
        if ing.form:
            toks = [ing.form, *toks]                       # form is a required token
        for j, t in enumerate(toks):
            preds.append(f"(LOWER(name) LIKE :i{n}t{j} OR LOWER(description) LIKE :i{n}t{j})")
            params[f"i{n}t{j}"] = f"%{t}%"

    order = ["price ASC"]
    need = normalize_quantity(ing.quantity, ing.unit)
    if with_size and need is not None:                     # SIZE_RANGE
        qty, uom = need
        params[f"need{n}"], params[f"uom{n}"] = qty, uom
        preds.append(f"(unit_qty IS NULL OR unit_uom <> :uom{n} "
                     f"OR unit_qty <= :need{n} * {SIZE_CAP_FACTOR})")
        order = [
            f"CASE WHEN unit_qty IS NULL OR unit_uom <> :uom{n} THEN 1 ELSE 0 END ASC",
            f"CASE WHEN unit_qty < :need{n} THEN 1 ELSE 0 END ASC",   # covering packs first
            f"ABS(unit_qty - :need{n}) ASC",                          # closest fit
            "price ASC",
        ]

    return (f"SELECT {n} AS ingredient_no, sub{n}.* FROM ("
            f"SELECT {PRODUCT_COLS} FROM products WHERE {' AND '.join(preds)} "
            f"ORDER BY {', '.join(order)} LIMIT :lim) AS sub{n}")


def build_retrieval_sql(c: Constraints, ingredients: list[IngredientSpec],
                        per_ingredient_limit: int = PER_INGREDIENT_LIMIT) -> tuple[str, dict]:
    params: dict = {"lim": per_ingredient_limit}
    constraint_sql = " AND ".join(_constraint_where(c, params))
    blocks = [_ingredient_block(n, ing, constraint_sql, params)
              for n, ing in enumerate(ingredients)]
    return " UNION ALL ".join(blocks), params


def build_fallback_sql(c: Constraints, missed: list[tuple[int, IngredientSpec]],
                       subcat_parent: dict[str, str],
                       per_ingredient_limit: int = PER_INGREDIENT_LIMIT) -> tuple[str, dict]:
    """The widening ladder, one batched query. Per missed ingredient:
    rung 1 (relaxed tokens: no size cap, no form token) UNION
    rung 2 (scope fallback: subcategory OR its parent category).
    Rung precedence is applied in Python (rung-1 rows win)."""
    params: dict = {"lim": per_ingredient_limit}
    constraint_sql = " AND ".join(_constraint_where(c, params))
    blocks: list[str] = []
    for n, ing in missed:
        relaxed = IngredientSpec(**{**ing.model_dump(), "form": None})
        blocks.append(
            _ingredient_block(n, relaxed, constraint_sql, params, with_size=False)
            .replace(f"SELECT {n} AS ingredient_no", f"SELECT {n} AS ingredient_no, 1 AS rung", 1))
        hint = (ing.category_hint or "").lower()
        if hint:
            params[f"fb{n}s"], params[f"fb{n}c"] = hint, subcat_parent.get(hint, hint)
            blocks.append(
                f"SELECT {n} AS ingredient_no, 2 AS rung, sub{n}f.* FROM ("
                f"SELECT {PRODUCT_COLS} FROM products WHERE {constraint_sql} "
                f"AND (subcategory = :fb{n}s OR category = :fb{n}c) "
                # exact-subcategory matches outrank parent-category padding
                f"ORDER BY CASE WHEN subcategory = :fb{n}s THEN 0 ELSE 1 END ASC, "
                f"price ASC LIMIT :lim) AS sub{n}f")
    return " UNION ALL ".join(blocks), params


def inline_for_display(sql: str, params: dict) -> str:
    """Substitute bound params for the UI's SQL panel. DISPLAY ONLY —
    execution always uses the parameterized form."""
    out = sql
    for key in sorted(params, key=len, reverse=True):
        v = params[key]
        rendered = f"'{v}'" if isinstance(v, str) else str(v)
        out = out.replace(f":{key}", rendered)
    return out.replace(" UNION ALL ", "\nUNION ALL\n").replace(" FROM (", "\n  FROM (")
