"""4A: deterministic split-trip optimizer.

Given the chosen basket and the full (store x product) price matrix, answer
"is splitting the shop across stores worth the driving?" exactly:

  * every non-empty store subset is enumerated (<= 2^N - 1; N is small),
  * each basket line is priced at its cheapest store in the subset,
  * travel is the OPTIMAL loop home -> stores -> home (brute-force TSP —
    trivial at these sizes) in equirectangular km x TRAVEL_COST_PER_KM,
  * the frontier (best option per number of stops) is returned, with the
    overall minimum flagged recommended.

No LLM anywhere — this is the pipeline's "the deterministic option is both
cheaper and better" step. Pure functions; the DB query lives in
sql_builder.build_price_matrix_sql and the flow action feeds rows in.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from itertools import combinations, permutations

from .models import TripItem, TripOption

MAX_ENUMERATED_STORES = 8      # 255 subsets x 8! routes worst case — still ms


@dataclass(frozen=True)
class StoreInfo:
    id: int
    name: str
    lat: float
    lon: float
    dist_km: float             # from home (straight-line)


def _km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    coslat = math.cos(math.radians(lat1))
    return math.hypot((lat1 - lat2) * 111.0, (lon1 - lon2) * 111.0 * coslat)


def _trip_km(home: tuple[float, float], stores: list[StoreInfo]) -> float:
    """Shortest loop home -> every store -> home. Exact: brute force over
    permutations (fine for the store counts we enumerate)."""
    if not stores:
        return 0.0
    best = math.inf
    for order in permutations(stores):
        d = _km(*home, order[0].lat, order[0].lon)
        for a, b in zip(order, order[1:]):
            d += _km(a.lat, a.lon, b.lat, b.lon)
        d += _km(order[-1].lat, order[-1].lon, *home)
        best = min(best, d)
    return best


def optimize_trips(matrix_rows: list[dict], basket: list[tuple[int, str]],
                   *, home_lat: float, home_lon: float,
                   cost_per_km: float) -> list[TripOption]:
    """`matrix_rows`: mappings from build_price_matrix_sql. `basket`:
    (product_id, product_name) per plan line — duplicates allowed and
    priced per line. Returns the per-stop-count frontier sorted by number
    of stops; exactly one option carries recommended=True and its per-item
    store assignment."""
    stores: dict[int, StoreInfo] = {}
    price: dict[int, dict[int, float]] = {}          # store_id -> product -> price
    for r in matrix_rows:
        sid = r["store_id"]
        if sid not in stores:
            stores[sid] = StoreInfo(
                id=sid, name=r["store_name"], lat=r["lat"], lon=r["lon"],
                dist_km=round(math.sqrt(max(r["dist_km2"], 0.0)), 1))
        price.setdefault(sid, {})[r["product_id"]] = r["price"]

    store_list = sorted(stores.values(), key=lambda s: s.id)[:MAX_ENUMERATED_STORES]
    if not store_list or not basket:
        return []

    home = (home_lat, home_lon)
    best_by_k: dict[int, tuple[float, TripOption, dict[int, StoreInfo]]] = {}
    for k in range(1, len(store_list) + 1):
        for subset in combinations(store_list, k):
            # cheapest in-subset store per basket line; subset infeasible
            # if any product is stocked nowhere in it
            assignment: dict[int, StoreInfo] = {}
            basket_cost = 0.0
            feasible = True
            for pid, _name in basket:
                offers = [(price[s.id][pid], s) for s in subset
                          if pid in price.get(s.id, {})]
                if not offers:
                    feasible = False
                    break
                p, s = min(offers, key=lambda t: (t[0], t[1].id))
                assignment[pid] = s
                basket_cost += p
            if not feasible:
                continue
            used = sorted({assignment[pid] for pid, _ in basket}, key=lambda s: s.id)
            travel_km = _trip_km(home, used)
            total = basket_cost + travel_km * cost_per_km
            option = TripOption(
                stores=[s.name for s in used],
                basket_cost=round(basket_cost, 2),
                travel_km=round(travel_km, 1),
                travel_cost=round(travel_km * cost_per_km, 2),
                total_cost=round(total, 2))
            k_used = len(used)                       # subset may collapse
            if k_used not in best_by_k or total < best_by_k[k_used][0]:
                best_by_k[k_used] = (total, option, assignment)

    if not best_by_k:
        return []
    options = [entry for _k, entry in sorted(best_by_k.items())]
    one_stop_total = options[0][0]
    best_total, best_option, best_assignment = min(options, key=lambda e: e[0])
    for _total, opt, _asg in options:
        opt.savings_vs_one_stop = round(one_stop_total - opt.total_cost, 2) + 0.0
    best_option.recommended = True
    best_option.items = [
        TripItem(product_id=pid, product_name=name,
                 store_name=best_assignment[pid].name,
                 price=price[best_assignment[pid].id][pid])
        for pid, name in basket
    ]
    return [opt for _t, opt, _a in options]
