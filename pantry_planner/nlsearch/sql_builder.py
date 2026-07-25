"""TRANSLATION: compile the validated parse into the plan's named templates.

Every template composes from the fixed pattern menu (PRICE_CEILING,
DIETARY_EXCLUDE/REQUIRE, CATEGORY/SUBCATEGORY_SCOPE, TOKEN_AND_MATCH,
SIZE_RANGE, DISTANCE_WITHIN). Values only ever travel as bound parameters —
the LLM's words never touch the SQL string.

Ingredient matching is a single pass for the whole recipe: the parsed
tokens go in as one VALUES rowset joined against the precomputed
`product_terms` inverted index (indexed key lookups, no LIKE scans);
token-AND via HAVING COUNT(DISTINCT term) = ntokens; per-product best
store via one window, per-ingredient LIMIT via a second. Cost is linear
in matched rows — 20 ingredients is the same single round trip as 4.
Portable across SQLite >= 3.25 (local dev) and Postgres (deployment).

Distance is equirectangular in km, kept as a SQUARED comparison so the
SQL needs no sqrt(): :coslat = 111*cos(lat0) is precomputed in Python.
"""
from __future__ import annotations

import math

from .schemas import Constraints, IngredientSpec
from .units import normalize_quantity, tokens

PRODUCT_COLS = ("p.id AS id, p.name AS name, p.description AS description, "
                "p.price AS price, p.category AS category, "
                "p.subcategory AS subcategory, p.dietary_tags AS dietary_tags, "
                "p.unit_size AS unit_size, p.unit_qty AS unit_qty, "
                "p.unit_uom AS unit_uom, p.brand AS brand")
SIZE_CAP_FACTOR = 6      # exclude packs > 6x the need (catering boxes)
PER_INGREDIENT_LIMIT = 8
THIN_POOL = 3            # below this, t4 fetches substitutes

DIST_EXPR = ("((:lat - s.lat) * 111.0) * ((:lat - s.lat) * 111.0) + "
             "((:lon - s.lon) * :coslat) * ((:lon - s.lon) * :coslat)")


def _location_params(params: dict, lat: float, lon: float) -> None:
    params["lat"], params["lon"] = lat, lon
    params["coslat"] = 111.0 * math.cos(math.radians(lat))


def _constraint_where(c: Constraints, params: dict) -> list[str]:
    """The shared pattern menu. Aliases: p = products, sp = store_products."""
    where: list[str] = ["1=1"]
    if c.max_item_price is not None:                       # PRICE_CEILING (offer price)
        where.append("sp.price <= :max_item_price")
        params["max_item_price"] = c.max_item_price
    for i, tag in enumerate(c.exclude_tags):               # DIETARY_EXCLUDE
        where.append(f"(',' || p.dietary_tags || ',') NOT LIKE :xtag{i}")
        params[f"xtag{i}"] = f"%,{tag},%"
    for i, tag in enumerate(c.require_tags):               # DIETARY_REQUIRE
        where.append(f"(',' || p.dietary_tags || ',') LIKE :rtag{i}")
        params[f"rtag{i}"] = f"%,{tag},%"

    def scope(col: str, values: list[str], *, exclude: bool, key: str) -> None:
        if values:                                         # CATEGORY/SUBCATEGORY_SCOPE
            keys = ",".join(f":{key}{i}" for i in range(len(values)))
            where.append(f"p.{col} {'NOT IN' if exclude else 'IN'} ({keys})")
            params.update({f"{key}{i}": v for i, v in enumerate(values)})

    scope("category", c.categories, exclude=False, key="cat")
    scope("category", c.exclude_categories, exclude=True, key="xcat")
    scope("subcategory", c.subcategories, exclude=False, key="sub")
    scope("subcategory", c.exclude_subcategories, exclude=True, key="xsub")
    return where


def _ingredient_terms(ing: IngredientSpec, *, relaxed: bool) -> list[str]:
    """TOKEN_AND_MATCH terms. The purchase form is a required token unless
    this ingredient is on the form-relaxed path (t1 said strict = 0)."""
    toks = tokens(ing.name)
    if ing.form and not relaxed:
        toks = [ing.form, *toks]
    return toks


# ─── t1: existence probe ─────────────────────────────────────

def build_existence_sql(ingredients: list[IngredientSpec]) -> tuple[str, dict]:
    """One batched probe: per ingredient, how many products match ALL its
    tokens (strict = incl. purchase form) and ALL its base tokens (form
    relaxed). Catalog-level on purpose — constraints don't apply here, so
    'not stocked at all' stays distinct from 'not within constraints'."""
    params: dict = {}
    term_rows: list[str] = []
    count_rows: list[str] = []
    for n, ing in enumerate(ingredients):
        base = tokens(ing.name)
        all_toks = ([ing.form, *base] if ing.form else base)
        for j, t in enumerate(all_toks):
            params[f"i{n}t{j}"] = t
            is_base = 0 if (ing.form and j == 0) else 1
            term_rows.append(f"({n}, :i{n}t{j}, {is_base})")
        count_rows.append(f"({n}, {len(base)}, {len(all_toks)})")
    if not term_rows:                        # fully tokenless recipe: no matches
        params["noterm"] = ""
        term_rows = ["(-1, :noterm, 1)"]
    sql = (
        f"WITH ing_terms(ing_no, term, base) AS (VALUES {', '.join(term_rows)}),\n"
        f" ing_counts(ing_no, n_base, n_all) AS (VALUES {', '.join(count_rows)}),\n"
        " m AS (\n"
        "   SELECT it.ing_no, pt.product_id,\n"
        "          COUNT(DISTINCT CASE WHEN it.base = 1 THEN it.term END) AS base_hits,\n"
        "          COUNT(DISTINCT it.term) AS all_hits\n"
        "   FROM ing_terms it JOIN product_terms pt ON pt.term = it.term\n"
        "   GROUP BY it.ing_no, pt.product_id)\n"
        "SELECT c.ing_no,\n"
        "       COALESCE(SUM(CASE WHEN m.all_hits = c.n_all THEN 1 ELSE 0 END), 0) AS strict_matches,\n"
        "       COALESCE(SUM(CASE WHEN m.base_hits = c.n_base THEN 1 ELSE 0 END), 0) AS relaxed_matches\n"
        "FROM ing_counts c LEFT JOIN m ON m.ing_no = c.ing_no\n"
        "GROUP BY c.ing_no ORDER BY c.ing_no"
    )
    return sql, params


# ─── t2: options (the single-pass retrieval) ─────────────────

def build_options_sql(c: Constraints, ingredients: list[IngredientSpec],
                      relaxed: set[int], lat: float, lon: float,
                      max_km: float | None = None,
                      per_ingredient_limit: int = PER_INGREDIENT_LIMIT) -> tuple[str, dict]:
    """All ingredients resolved in one pass: token VALUES -> product_terms
    join -> token-AND -> per-product cheapest in-range store (rn_store) ->
    per-ingredient size-fit/price ranking (rn) capped at :lim."""
    params: dict = {"lim": per_ingredient_limit}
    _location_params(params, lat, lon)

    term_rows: list[str] = []
    count_rows: list[str] = []
    need_rows: list[str] = []
    for n, ing in enumerate(ingredients):
        toks = _ingredient_terms(ing, relaxed=(n in relaxed))
        for j, t in enumerate(toks):
            params[f"i{n}t{j}"] = t
            term_rows.append(f"({n}, :i{n}t{j})")
        count_rows.append(f"({n}, {len(toks)})")
        need = normalize_quantity(ing.quantity, ing.unit)
        if need is not None:                               # SIZE_RANGE
            params[f"need{n}"], params[f"uom{n}"] = need
            need_rows.append(f"({n}, :need{n}, :uom{n})")

    where = _constraint_where(c, params)
    if max_km is not None:                                 # DISTANCE_WITHIN
        params["maxdist2"] = max_km * max_km
        where.append(f"{DIST_EXPR} <= :maxdist2")

    sized = bool(need_rows)
    need_cte = f",\n ing_need(ing_no, need, uom) AS (VALUES {', '.join(need_rows)})" if sized else ""
    need_join = "\n   LEFT JOIN ing_need nn ON nn.ing_no = m.ing_no" if sized else ""
    if sized:
        where.append("(nn.need IS NULL OR p.unit_qty IS NULL OR p.unit_uom <> nn.uom "
                     f"OR p.unit_qty <= nn.need * {SIZE_CAP_FACTOR})")
        rank_join = " LEFT JOIN ing_need nd ON nd.ing_no = b.ing_no"
        rank_order = (
            "CASE WHEN nd.need IS NULL OR b.unit_qty IS NULL OR b.unit_uom <> nd.uom "
            "THEN 1 ELSE 0 END ASC,\n"
            "      CASE WHEN nd.need IS NOT NULL AND b.unit_qty IS NOT NULL "
            "AND b.unit_uom = nd.uom AND b.unit_qty < nd.need THEN 1 ELSE 0 END ASC,\n"
            "      CASE WHEN nd.need IS NOT NULL AND b.unit_qty IS NOT NULL "
            "AND b.unit_uom = nd.uom THEN ABS(b.unit_qty - nd.need) ELSE 0 END ASC,\n"
            "      b.store_price ASC, b.id ASC")
    else:
        rank_join = ""
        rank_order = "b.store_price ASC, b.id ASC"

    sql = (
        f"WITH ing_terms(ing_no, term) AS (VALUES {', '.join(term_rows)}),\n"
        f" ing_counts(ing_no, ntok) AS (VALUES {', '.join(count_rows)})"
        f"{need_cte},\n"
        " matches AS (\n"
        "   SELECT it.ing_no, pt.product_id\n"
        "   FROM ing_terms it\n"
        "   JOIN product_terms pt ON pt.term = it.term\n"
        "   JOIN ing_counts c ON c.ing_no = it.ing_no\n"
        "   GROUP BY it.ing_no, pt.product_id, c.ntok\n"
        "   HAVING COUNT(DISTINCT it.term) = c.ntok),\n"
        " offers AS (\n"
        f"   SELECT m.ing_no AS ing_no, {PRODUCT_COLS},\n"
        "          sp.price AS store_price, s.name AS store_name,\n"
        f"          {DIST_EXPR} AS dist_km2,\n"
        "          ROW_NUMBER() OVER (PARTITION BY m.ing_no, p.id "
        "ORDER BY sp.price ASC, s.id ASC) AS rn_store\n"
        "   FROM matches m\n"
        "   JOIN products p ON p.id = m.product_id\n"
        "   JOIN store_products sp ON sp.product_id = p.id\n"
        "   JOIN stores s ON s.id = sp.store_id"
        f"{need_join}\n"
        f"   WHERE {' AND '.join(where)})\n"
        "SELECT * FROM (\n"
        "  SELECT b.*, ROW_NUMBER() OVER (PARTITION BY b.ing_no ORDER BY\n"
        f"      {rank_order}) AS rn\n"
        f"  FROM offers b{rank_join}\n"
        "  WHERE b.rn_store = 1) ranked\n"
        "WHERE rn <= :lim\n"
        "ORDER BY ing_no, rn"
    )
    return sql, params


# ─── t3: per-brand statistics for the retrieved pools ────────

def build_stats_sql(pool_ids: list[int]) -> tuple[str, dict]:
    """Price + review stats per pooled product. Aggregated in separate
    subqueries (joining stores x reviews directly would multiply rows and
    inflate COUNT); regrouped per ingredient/brand in Python."""
    params = {f"pid{i}": pid for i, pid in enumerate(pool_ids)}
    in_list = ", ".join(f":pid{i}" for i in range(len(pool_ids)))
    sql = (
        "SELECT p.id AS product_id, p.name AS name, p.brand AS brand,\n"
        "       ps.min_price AS min_price, ps.avg_price AS avg_price,\n"
        "       rv.avg_rating AS avg_rating, rv.review_count AS review_count\n"
        "FROM products p\n"
        "JOIN (SELECT product_id, MIN(price) AS min_price, AVG(price) AS avg_price\n"
        "      FROM store_products GROUP BY product_id) ps ON ps.product_id = p.id\n"
        "LEFT JOIN (SELECT product_id, AVG(rating) AS avg_rating, COUNT(*) AS review_count\n"
        "           FROM reviews GROUP BY product_id) rv ON rv.product_id = p.id\n"
        f"WHERE p.id IN ({in_list})\n"
        "ORDER BY p.id"
    )
    return sql, params


# ─── t4: substitute lookup for thin pools ────────────────────

def build_substitute_sql(c: Constraints, subcategory: str, exclude_ids: list[int],
                         lat: float, lon: float, max_km: float | None = None,
                         limit: int = THIN_POOL) -> tuple[str, dict]:
    """Same-subcategory alternatives at their cheapest in-range store.
    Constraints still apply — a no-dairy basket never gets a dairy sub."""
    params: dict = {"sub_subcat": subcategory, "sub_lim": limit}
    _location_params(params, lat, lon)
    where = _constraint_where(c, params)
    where.append("p.subcategory = :sub_subcat")
    if exclude_ids:
        keys = ",".join(f":xid{i}" for i in range(len(exclude_ids)))
        where.append(f"p.id NOT IN ({keys})")
        params.update({f"xid{i}": v for i, v in enumerate(exclude_ids)})
    if max_km is not None:
        params["maxdist2"] = max_km * max_km
        where.append(f"{DIST_EXPR} <= :maxdist2")
    sql = (
        "SELECT * FROM (\n"
        f"  SELECT {PRODUCT_COLS}, sp.price AS store_price, s.name AS store_name,\n"
        f"         {DIST_EXPR} AS dist_km2,\n"
        "         ROW_NUMBER() OVER (PARTITION BY p.id ORDER BY sp.price ASC, s.id ASC) AS rn_store\n"
        "  FROM products p\n"
        "  JOIN store_products sp ON sp.product_id = p.id\n"
        "  JOIN stores s ON s.id = sp.store_id\n"
        f"  WHERE {' AND '.join(where)}) x\n"
        "WHERE rn_store = 1 ORDER BY store_price ASC LIMIT :sub_lim"
    )
    return sql, params


# ─── abort helper: suggestions for a missing ingredient ──────

def build_suggestions_sql(category_hint: str, limit: int = 3) -> tuple[str, dict]:
    """Top-N cheapest products sharing the parse's category hint — shown in
    the missing_ingredients alert so the abort is actionable."""
    params = {"hint": category_hint, "sug_lim": limit}
    sql = ("SELECT name, price FROM products p "
           "WHERE p.subcategory = :hint OR p.category = :hint "
           "ORDER BY price ASC LIMIT :sug_lim")
    return sql, params


# The plan's template registry: QueryStep.template -> builder.
TEMPLATES = {
    "existence_probe": build_existence_sql,
    "options_single_pass": build_options_sql,
    "brand_stats": build_stats_sql,
    "substitute_lookup": build_substitute_sql,
    "missing_suggestions": build_suggestions_sql,
}


def inline_for_display(sql: str, params: dict) -> str:
    """Substitute bound params for the UI's SQL panel. DISPLAY ONLY —
    execution always uses the parameterized form."""
    out = sql
    for key in sorted(params, key=len, reverse=True):
        v = params[key]
        rendered = f"'{v}'" if isinstance(v, str) else str(v)
        out = out.replace(f":{key}", rendered)
    return out
