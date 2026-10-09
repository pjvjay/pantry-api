"""plan_meals (MCP) and POST /mealplan/plan: counted dishes drafted into a meal plan.

Demo mode, no LLM. Expected values come from the seed files (the starters, the library), the
user's sentence and plain counting, and the schedule the returned draft gets from
/mealplan/schedule, never from the summary under test.
"""
from __future__ import annotations

import datetime as dt
import json

import pytest
from mcp.server.mcpserver.exceptions import ToolError

from tests.mealplan_fixtures import START, STARTERS, client, done_db, use_db

PIZZA, RICE, BIRYANI, SHAKE = (f"starter:{k}" for k in ("pepperoni_pizza", "chicken_fried_rice",
                                                         "chicken_biryani", "mango_milkshake"))
# The user's sentence as the hub's parse reads it: exact and plural matches are dishes; the
# alias match (briyani) is proposed, never placed.
EXAMPLE = {
    "dishes": [{"recipe": "Pepperoni Pizza", "count": 3},
               {"recipe": "Chicken Fried Rice", "count": 2},
               {"recipe": "mango milkshakes", "count": 7}],
    "proposed": [{"recipe_key": BIRYANI, "count": 3, "input": "chicken briyani",
                  "how": "alias"}],
    "days": 14, "start_date": START.isoformat(),
}
CAPACITY = {"breakfast": 1, "lunch": 1, "dinner": 1, "snack": 2}


@pytest.fixture(scope="module", autouse=True)
def seeded(tmp_path_factory):
    use_db(tmp_path_factory, "meals")
    yield
    done_db()


@pytest.fixture()
def server():
    from pantry_planner.mcp_server import server

    return server


async def plan(server, **args) -> dict:
    return (await server.call_tool("plan_meals", args)).structured_content


def ops(summary: dict, kind: str) -> list[dict]:
    return [o for o in summary["ops"] if o["op"] == kind]


# ─── The tool ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_plan_meals_is_a_read_only_plan_tool(server):
    tools = {t.name: t for t in await server.list_tools()}
    t = tools["plan_meals"]
    assert t.annotations.read_only_hint is True and t.annotations.idempotent_hint is False
    assert t.annotations.open_world_hint is False and t.output_schema
    props = t.input_schema["properties"]
    assert {"dishes", "days", "proposed", "current", "start_date", "my_recipe_docs",
            "lat", "lon", "max_km"} <= set(props)
    assert not {"approve", "approved", "trips"} & set(props)   # it has no approve capability


@pytest.mark.asyncio
async def test_the_users_sentence_places_exact_and_plural_dishes_only(server):
    out = await plan(server, **EXAMPLE)
    s = out["summary"]
    assert s["kind"] == "mealplan" and s["days"] == 14
    assert s["start_date"] == START.isoformat() and s["household_servings"] == 2
    assert [(a["recipe_key"], a["count"], a["placed"], a["how"]) for a in s["added"]] == [
        (PIZZA, 3, 3, "exact"), (RICE, 2, 2, "exact"), (SHAKE, 7, 7, "plural")]
    # count = one meal occasion: 3 + 2 + 7 meals, each at the household's servings
    by_key = {}
    for m in s["meals"]:
        by_key[m["recipe_key"]] = by_key.get(m["recipe_key"], 0) + 1
        assert m["new"] is True
    assert by_key == {PIZZA: 3, RICE: 2, SHAKE: 7}
    assert all(m["servings"] is None for m in out["draft"]["meals"])
    assert {m["slot"] for m in s["meals"] if m["recipe_key"] == SHAKE} == {"snack"}
    # the alias match is a proposal with its question, and never placed anywhere
    [p] = s["proposals"]
    assert (p["recipe_key"], p["count"], p["how"]) == (BIRYANI, 3, "alias")
    assert p["question"] == "chicken briyani → Chicken Biryani (demo starter)?"
    assert p["op"] == {"op": "add_recipe", "recipe_key": BIRYANI, "title": "Chicken Biryani",
                       "slot": "dinner", "count": 3, "spread": True,
                       "ref": {"key": BIRYANI, "starter": "chicken_biryani"}}
    assert BIRYANI not in json.dumps(s["meals"]) + json.dumps(s["ops"])
    assert BIRYANI not in out["draft"]["recipes"]
    assert s["unmatched"] == [] and s["unplaced"] == []


@pytest.mark.asyncio
async def test_ops_place_each_new_meal_once_inside_the_window(server):
    s = (await plan(server, **EXAMPLE))["summary"]
    assert s["ops"][0] == {"op": "set_window", "start_date": START.isoformat(), "days": 14}
    assert [o["recipe_key"] for o in ops(s, "add_recipe")] == [PIZZA, RICE, SHAKE]
    assert [o["count"] for o in ops(s, "add_recipe")] == [3, 2, 7]
    placed = ops(s, "place")
    assert len(placed) == 12
    cells: dict[tuple[str, str], int] = {}
    for o in placed:
        d = dt.date.fromisoformat(o["date"])
        assert 0 <= (d - START).days < 14
        cells[(o["date"], o["slot"])] = cells.get((o["date"], o["slot"]), 0) + 1
    assert all(n <= CAPACITY[slot] for (_, slot), n in cells.items())
    assert not [o for o in s["ops"] if "approve" in o["op"]]
    assert any("only the shopper approves" in n for n in s["notes"])


@pytest.mark.asyncio
async def test_the_draft_schedules_to_the_summarys_trips(server):
    """The returned draft is a MealPlanDraft /mealplan/schedule takes, and its recommended
    strategy is the summary's: the same trip dates and totals."""
    out = await plan(server, **EXAMPLE)
    r = client().post("/mealplan/schedule", json=out["draft"])
    assert r.status_code == 200, r.text
    sched = r.json()
    rec = next(st for st in sched["strategies"] if st["recommended"])
    s = out["summary"]
    assert s["strategy"] == rec["name"] and s["total_cost"] == rec["total_cost"]
    assert [(t["date"], t["total_cost"]) for t in s["trips"]] == [
        (t["date"], t["total_cost"]) for t in rec["trips"]]
    assert [m["id"] for m in sched["meals"] if m["date"] is None] == []
    assert out["full"] is None


@pytest.mark.asyncio
async def test_nutrition_is_one_line_with_the_demo_badge(server):
    s = (await plan(server, **EXAMPLE))["summary"]
    # the starters' amounts are demo house amounts (seeds/mealplan_starters.json)
    assert {ln["amount_basis"] for st in STARTERS.values() for ln in st["lines"]} == {
        "demo_house_amounts"}
    assert "(demo amounts)" in s["nutrition"] and "kcal" in s["nutrition"]
    assert "\n" not in s["nutrition"]


@pytest.mark.asyncio
async def test_a_misspelt_or_fuzzy_title_from_the_model_is_only_proposed(server):
    s = (await plan(server, dishes=[{"recipe": "chicken briyani", "count": 3},
                                    {"recipe": "pepperoni piza", "count": 2}],
                    start_date=START.isoformat()))["summary"]
    assert s["added"] == [] and s["meals"] == [] and ops(s, "place") == []
    assert [(p["recipe_key"], p["how"], p["count"]) for p in s["proposals"]] == [
        (BIRYANI, "alias", 3), (PIZZA, "fuzzy", 2)]
    assert s["proposals"][1]["question"] == "pepperoni piza → Pepperoni Pizza (demo starter)?"


@pytest.mark.asyncio
async def test_unknown_names_are_listed_never_guessed(server):
    s = (await plan(server, dishes=[{"recipe": "dragon fruit stew", "count": 2},
                                    {"recipe": "pizza", "count": 1},
                                    {"recipe": "Pepperoni Pizza", "count": 1}],
                    start_date=START.isoformat()))["summary"]
    assert [a["recipe_key"] for a in s["added"]] == [PIZZA]
    stew, pizza = s["unmatched"]
    assert stew["input"] == "dragon fruit stew" and stew["candidates"] == []
    # "pizza" is in one title only: offered as what it could be, not chosen
    assert pizza["candidates"] == [{"recipe_key": PIZZA, "title": "Pepperoni Pizza",
                                    "label": "demo starter"}]


@pytest.mark.asyncio
async def test_slugs_and_keys_name_recipes_and_counts_add_up(server):
    s = (await plan(server, dishes=[{"recipe": "tomato_penne", "count": 1},
                                    {"recipe": "lib:tomato_penne", "count": 1},
                                    {"recipe": "mango_milkshake", "count": 2}],
                    start_date=START.isoformat()))["summary"]
    assert [(a["recipe_key"], a["count"], a["how"]) for a in s["added"]] == [
        ("lib:tomato_penne", 2, "key"), (SHAKE, 2, "key")]
    # a week holds 2 dinners and 2 snacks, so a new plan is 7 days
    assert s["days"] == 7
    assert ops(s, "add_recipe")[0]["ref"] == {"key": "lib:tomato_penne", "slug": "tomato_penne"}


def _current(**extra) -> dict:
    penne = "lib:tomato_penne"
    return {"start_date": START.isoformat(), "days": 7, "rev": 5,
            "recipes": [{"key": penne, "title": "Tomato Penne", "kind": "library"}],
            "meals": [{"id": f"{penne}#1", "recipe_key": penne, "date": START.isoformat(),
                       "slot": "dinner"},
                      {"id": f"{penne}#2", "recipe_key": penne,
                       "date": (START + dt.timedelta(days=1)).isoformat(), "slot": "dinner"}],
            **extra}


@pytest.mark.asyncio
async def test_a_current_plan_keeps_its_meals_and_ops_are_relative_to_it(server):
    s = (await plan(server, dishes=[{"recipe": "Pepperoni Pizza", "count": 3},
                                    {"recipe": "tomato penne", "count": 1}],
                    current=_current()))["summary"]
    assert s["base_rev"] == 5 and s["days"] == 7 and s["start_date"] == START.isoformat()
    old = [(m["date"], m["recipe_key"]) for m in s["meals"] if not m["new"]]
    assert old == [(START.isoformat(), "lib:tomato_penne"),
                   ((START + dt.timedelta(days=1)).isoformat(), "lib:tomato_penne")]
    assert ops(s, "set_window") == []                  # the plan's window stands
    assert ops(s, "add_meals") == [{"op": "add_meals", "recipe_key": "lib:tomato_penne",
                                    "title": "Tomato Penne", "count": 1}]
    assert [o["recipe_key"] for o in ops(s, "add_recipe")] == [PIZZA]
    taken = {(START + dt.timedelta(days=i)).isoformat() for i in (0, 1)}
    assert len(ops(s, "place")) == 4 and not {o["date"] for o in ops(s, "place")} & taken


@pytest.mark.asyncio
async def test_a_current_plan_is_lengthened_never_shortened(server):
    longer = (await plan(server, dishes=[{"recipe": "Pepperoni Pizza", "count": 1}],
                         days=14, current=_current()))["summary"]
    assert ops(longer, "set_window") == [{"op": "set_window", "start_date": START.isoformat(),
                                          "days": 14}]
    shorter = (await plan(server, dishes=[{"recipe": "Pepperoni Pizza", "count": 1}],
                          days=3, current=_current()))["summary"]
    assert shorter["days"] == 7 and ops(shorter, "set_window") == []
    assert any("not shortened" in n for n in shorter["notes"])


@pytest.mark.asyncio
async def test_approved_trips_are_never_touched_but_flagged(server):
    out = await plan(server, dishes=[{"recipe": "Pepperoni Pizza", "count": 1}],
                     current=_current(approved_trips=[{"date": START.isoformat()}]))
    assert out["draft"]["trips"] == []
    assert "approved trip" in out["summary"]["warnings"][0]


@pytest.mark.asyncio
async def test_a_dish_the_assistant_writes_is_planned_as_written(server):
    s_out = await plan(server, dishes=[{"recipe": "Lemon Rice", "count": 2, "servings": 2,
                                        "lines": ["300 g basmati rice", "- 1 lemon", ""]}],
                       start_date=START.isoformat())
    s = s_out["summary"]
    [a] = s["added"]
    assert a["how"] == "written" and a["recipe_key"].startswith("asst:")
    assert a["placed"] == 2
    doc = s_out["draft"]["recipes"][a["recipe_key"]]["ref"]["doc"]
    assert [(ln["name"], ln["quantity"], ln["unit"], ln["amount_basis"]) for ln in doc["lines"]] \
        == [("basmati rice", 300.0, "g", "written_by_assistant"),
            ("lemon", 1.0, "each", "written_by_assistant")]
    assert doc["servings"] == 2 and doc["source"]["method"] == "agent_written"
    unstated = (await plan(server, dishes=[{"recipe": "Lemon Rice", "count": 1,
                                            "lines": ["300 g basmati rice"]}],
                           start_date=START.isoformat()))["summary"]
    assert any("how many it serves" in w for w in unstated["warnings"])


@pytest.mark.asyncio
async def test_the_shoppers_own_recipe_is_matched_by_title(server):
    doc = {"key": "my:dal", "title": "Grandma Dal", "servings": 4, "servings_stated": True,
           "lines": [{"line_no": 1, "text": "400 g red lentils", "name": "red lentils",
                      "quantity": 400, "unit": "g", "amount_basis": "parsed_from_your_paste"}],
           "source": {"kind": "pasted", "method": "paste"}}
    s = (await plan(server, dishes=[{"recipe": "grandma dal", "count": 2}],
                    my_recipe_docs=[doc], start_date=START.isoformat()))["summary"]
    assert [(a["recipe_key"], a["label"], a["placed"]) for a in s["added"]] == [
        ("my:dal", "my recipe", 2)]
    assert ops(s, "add_recipe")[0]["ref"]["doc"]["title"] == "Grandma Dal"


@pytest.mark.asyncio
async def test_verbose_attaches_the_schedule(server):
    out = await plan(server, dishes=[{"recipe": "Pepperoni Pizza", "count": 1}],
                     start_date=START.isoformat(), verbose=True)
    assert out["full"]["start_date"] == START.isoformat() and out["full"]["strategies"]


@pytest.mark.asyncio
@pytest.mark.parametrize(("args", "match"), [
    ({}, "No dishes"),
    ({"dishes": [{"recipe": "Pepperoni Pizza", "count": 28},
                 {"recipe": "Chicken Fried Rice", "count": 28},
                 {"recipe": "Mango Milkshake", "count": 1}]}, "holds 56 meals"),
    ({"dishes": [{"recipe": "Pepperoni Pizza", "count": 1}],
      "current": _current(meals=[{"id": "a", "recipe_key": "lib:tomato_penne",
                                  "date": START.isoformat(), "slot": "dinner"},
                                 {"id": "b", "recipe_key": "lib:tomato_penne",
                                  "date": START.isoformat(), "slot": "dinner"}])},
     "slot_capacity"),
    ({"dishes": [{"recipe": "x", "count": 29}]}, "less than or equal to 28"),
])
async def test_what_it_refuses(server, args, match):
    with pytest.raises(ToolError, match=match):
        await server.call_tool("plan_meals", args)


# ─── REST ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_rest_twin_answers_as_the_tool_does(server):
    tool = await plan(server, **EXAMPLE)
    r = client().post("/mealplan/plan", json=EXAMPLE)
    assert r.status_code == 200, r.text
    assert r.json()["summary"] == tool["summary"]


def test_the_rest_twin_is_rate_limited_and_refuses_clearly():
    c = client()
    body = {"dishes": [{"recipe": "Pepperoni Pizza", "count": 1}],
            "start_date": START.isoformat()}
    assert c.post("/mealplan/plan", json={}).json()["detail"]["error"] == "plan_meals"
    codes = [c.post("/mealplan/plan", json=body).status_code for _ in range(4)]
    assert codes[:2] == [200, 200] and codes[-1] == 429
