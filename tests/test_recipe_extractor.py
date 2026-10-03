"""The recipe-shopper Agent Skill: its extractor and its SKILL.md.

skills/recipe-shopper/scripts/extract_recipe.py runs on the user's machine,
so the pantry server never fetches a URL. Every page here is SYNTHETIC
HTML written in this file (no real recipe page is committed); each test
states the exact lines the page holds and asserts the extractor returns
those, not merely something.

The SKILL.md guard is why the skill lives next to the server: every tool,
parameter, prompt and result field SKILL.md tells Claude to use must exist
in pantry_planner.mcp_server's definitions, so renaming one fails here
instead of silently breaking the skill. No network, no LLM.
"""
from __future__ import annotations

import importlib.util
import io
import json
import re
import subprocess
import sys
import zlib
from pathlib import Path

import pytest

SKILL_DIR = Path(__file__).resolve().parents[1] / "skills" / "recipe-shopper"
SCRIPT = SKILL_DIR / "scripts" / "extract_recipe.py"
SKILL_MD = SKILL_DIR / "SKILL.md"


def _load_extractor():
    spec = importlib.util.spec_from_file_location("extract_recipe", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load_extractor()


def _run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """The script as Claude runs it: a separate python3 process."""
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin,
                          capture_output=True, text=True, timeout=30)


def _page(tmp_path: Path, body: str, name: str = "page.html") -> Path:
    path = tmp_path / name
    path.write_text(f"<!doctype html><html><head><title>Site title | Blog</title>"
                    f"</head><body>{body}</body></html>", encoding="utf-8")
    return path


def _ld(obj) -> str:
    return f'<script type="application/ld+json">{json.dumps(obj)}</script>'


# ─── JSON-LD ─────────────────────────────────────────────────

def test_plain_jsonld_recipe_end_to_end(tmp_path):
    page = _page(tmp_path, _ld({
        "@context": "https://schema.org", "@type": "Recipe",
        "name": "Garlic Noodles", "recipeYield": "4 servings",
        "recipeIngredient": ["8 oz spaghetti", "6 cloves garlic, minced",
                             "2 tbsp butter"]}))
    out = _run(str(page))
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert data == {
        "source": str(page), "name": "Garlic Noodles", "yield": "4 servings",
        "servings": 4,
        "ingredients": ["8 oz spaghetti", "6 cloves garlic, minced", "2 tbsp butter"],
        "recipe_text": ("Garlic Noodles (serves 4)\n- 8 oz spaghetti\n"
                        "- 6 cloves garlic, minced\n- 2 tbsp butter"),
        "omitted": []}
    assert out.stderr == ""


def test_graph_with_yield_list_picks_the_recipe_and_the_worded_yield(tmp_path):
    """The WordPress recipe-plugin shape: one @graph holding the site, the
    page, the author and the recipe."""
    page = _page(tmp_path, _ld({"@context": "https://schema.org", "@graph": [
        {"@type": "Organization", "name": "Some Food Blog"},
        {"@type": "WebPage", "name": "Mala Chicken - Some Food Blog"},
        {"@type": "Person", "name": "A. Cook"},
        {"@type": "Recipe", "name": "Mala Chicken", "recipeYield": ["4", "4 servings"],
         "author": {"@type": "Person", "name": "A. Cook"},
         "recipeIngredient": ["1 lb boneless skinless chicken thigh",
                              "1 tbsp Shaoxing wine", "1 tbsp light soy sauce"]}]}))
    data = json.loads(_run(str(page)).stdout)
    assert (data["name"], data["yield"], data["servings"]) == (
        "Mala Chicken", "4 servings", 4)
    assert data["ingredients"] == ["1 lb boneless skinless chicken thigh",
                                   "1 tbsp Shaoxing wine", "1 tbsp light soy sauce"]


def test_type_list_top_level_array_and_a_broken_block_are_handled(tmp_path):
    """@type as a list, the recipe inside a top-level array, an unrelated
    block before it, a block that is not valid JSON, and a recipe nested
    under a WebPage's mainEntity after it: the first recipe with
    ingredients wins, the broken block is skipped."""
    broken = '<script type="application/ld+json">{"@type": "Recipe", "name": </script>'
    page = _page(tmp_path, broken
                 + _ld({"@type": "BreadcrumbList", "itemListElement": []})
                 + _ld([{"@type": "ImageObject", "url": "x.jpg"},
                        {"@type": ["Recipe", "NewsArticle"], "name": "Dal Tadka",
                         "recipeYield": 6,
                         "recipeIngredient": ["1 cup red lentils", "1 tsp cumin seeds"]}])
                 + _ld({"@type": "WebPage", "mainEntity": {
                     "@type": "https://schema.org/Recipe", "name": "Second",
                     "recipeIngredient": ["1 egg"]}}))
    data = json.loads(_run(str(page)).stdout)
    assert (data["name"], data["yield"], data["servings"]) == ("Dal Tadka", "6", 6)
    assert data["ingredients"] == ["1 cup red lentils", "1 tsp cumin seeds"]

    # the nested one is found when it is the only recipe
    alone = ex.extract(_ld({"@type": "WebPage", "mainEntity": {
        "@type": "https://schema.org/Recipe", "name": "Second",
        "recipeIngredient": ["1 egg"]}}))
    assert alone == {"name": "Second", "yield": "", "ingredients": ["1 egg"]}


def test_entities_tags_and_stray_whitespace_are_cleaned(tmp_path):
    raw = ('<script type="application/ld+json">\n'
           '{"@type": "Recipe", "name": "Tom&#039;s  Chili &amp;amp; Beans",\n'
           ' "recipeYield": "Serves 4-6",\n'
           ' "recipeIngredient": ["2 &frac12; cups\\n   kidney beans ",\n'
           '   "1&nbsp;tbsp <strong>chili powder</strong>",\n'
           '   "salt &amp;amp; pepper, to taste",\n'
           '   "   ", "1 can (14&nbsp;oz) diced tomatoes"]}\n'
           '</script>')
    data = json.loads(_run(str(_page(tmp_path, raw))).stdout)
    assert data["name"] == "Tom's Chili & Beans"
    assert data["ingredients"] == ["2 ½ cups kidney beans", "1 tbsp chili powder",
                                   "salt & pepper, to taste", "1 can (14 oz) diced tomatoes"]
    assert data["servings"] == 4                 # "Serves 4-6": the lower bound


def test_html_comment_wrapped_jsonld_and_raw_newlines_in_strings():
    page = ('<script type="application/ld+json; charset=utf-8"><!--\n'
            '{"@type": "Recipe", "name": "Two\nLines", "recipeIngredient": "1 cup rice"}\n'
            '--></script>')
    assert ex.extract(page) == {"name": "Two Lines", "yield": "",
                                "ingredients": ["1 cup rice"]}


# ─── microdata fallback ──────────────────────────────────────

def test_microdata_fallback_scopes_names_and_unclosed_list_items(tmp_path):
    """No JSON-LD. The recipe's name is its own itemprop, not the nested
    author's; <li> end tags are omitted, as HTML allows; a meta carries
    the yield."""
    page = _page(tmp_path, """
      <div itemscope itemtype="http://schema.org/Recipe">
        <h1 itemprop="name">Black Bean Tacos</h1>
        <p>By <span itemprop="author" itemscope itemtype="http://schema.org/Person">
          <span itemprop="name">Jane Doe</span></span>
        <meta itemprop="recipeYield" content="8 tacos">
        <ul>
          <li itemprop="recipeIngredient">1 can <b>black beans</b>, drained
          <li itemprop="recipeIngredient">8 corn tortillas
          <li itemprop="recipeIngredient">1 lime
        </ul>
        <script>var recipeIngredient = "not an ingredient";</script>
      </div>""")
    data = json.loads(_run(str(page)).stdout)
    assert data["name"] == "Black Bean Tacos"
    assert data["yield"] == "8 tacos"
    assert data["servings"] is None              # 8 tacos is not 8 servings
    assert data["ingredients"] == ["1 can black beans, drained", "8 corn tortillas",
                                   "1 lime"]
    assert data["recipe_text"].splitlines()[0] == "Black Bean Tacos (makes 8 tacos)"


def test_jsonld_wins_over_microdata_and_microdata_fills_in_when_jsonld_is_empty():
    micro = ('<div itemscope itemtype="https://schema.org/Recipe">'
             '<span itemprop="name">Micro</span>'
             '<span itemprop="recipeIngredient">2 eggs</span></div>')
    both = _ld({"@type": "Recipe", "name": "LD", "recipeIngredient": ["1 cup flour"]}) + micro
    assert ex.extract(both)["name"] == "LD"
    empty_ld = _ld({"@type": "Recipe", "name": "LD", "recipeIngredient": []}) + micro
    assert ex.extract(empty_ld) == {"name": "Micro", "yield": "", "ingredients": ["2 eggs"]}


def test_loose_microdata_without_a_recipe_scope_takes_the_page_title():
    page = ('<html><head><title>Pancakes</title></head><body>'
            '<li itemprop="recipeIngredient">1 cup flour</li>'
            '<li itemprop="recipeIngredient">1 egg</li></body></html>')
    assert ex.extract(page) == {"name": "Pancakes", "yield": "",
                                "ingredients": ["1 cup flour", "1 egg"]}


# ─── no recipe, bad input ────────────────────────────────────

def test_no_recipe_exits_2_with_a_clear_message(tmp_path):
    page = _page(tmp_path, "<h1>Just a moment...</h1>"
                 + _ld({"@type": "Article", "name": "Not a recipe"})
                 + "<script>document.write('<li itemprop=recipeIngredient>x</li>')</script>")
    out = _run(str(page))
    assert out.returncode == 2
    assert out.stdout == ""
    assert "no recipe found" in out.stderr and str(page) in out.stderr


def test_unreadable_inputs_exit_1_without_touching_the_network(tmp_path):
    missing = _run(str(tmp_path / "nope.html"))
    assert missing.returncode == 1 and "could not read" in missing.stderr
    scheme = _run("file:///etc/passwd")
    assert scheme.returncode == 1 and "unsupported URL scheme" in scheme.stderr
    assert scheme.stdout == ""
    usage = _run()
    assert usage.returncode == 1 and usage.stderr.startswith("usage: extract_recipe.py")


def test_stdin_input():
    page = _ld({"@type": "Recipe", "name": "Toast", "recipeIngredient": ["2 slices bread"]})
    out = _run("-", stdin=page)
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert (data["source"], data["ingredients"]) == ("stdin", ["2 slices bread"])


# ─── recipe_text ─────────────────────────────────────────────

def test_recipe_text_stays_within_the_tool_limit_and_names_what_it_left_out():
    lines = [f"{i} cups ingredient number {i} " + "x" * 80 for i in range(120)]
    text, omitted = ex.build_recipe_text("Huge", "2", lines)
    assert len(text) <= 8000 < len(text) + len(f"\n- {omitted[0]}")
    kept = text.splitlines()[1:]
    assert [ln[2:] for ln in kept] + omitted == lines      # order kept, nothing lost
    assert text.splitlines()[0] == "Huge (serves 2)"
    assert ex.MAX_RECIPE_TEXT == 8000


@pytest.mark.parametrize("yield_text,servings", [
    ("4", 4), ("4 servings", 4), ("Serves 4-6", 4), ("6 people", 6), ("2 portions", 2),
    ("24 cookies", None), ("1 loaf", None), ("", None)])
def test_servings_only_when_the_yield_counts_servings(yield_text, servings):
    assert ex.servings_from(yield_text) == servings


# ─── fetching (urlopen faked: no network) ────────────────────

class _FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, url: str, headers: dict[str, str]):
        super().__init__(body)
        self._url = url
        from email.message import Message
        self.headers = Message()
        for k, v in headers.items():
            self.headers[k] = v

    def geturl(self) -> str:
        return self._url


def test_fetch_sends_a_browser_user_agent_honours_the_timeout_and_decodes_gzip(monkeypatch):
    html = _ld({"@type": "Recipe", "name": "Crème brûlée",
                "recipeIngredient": ["4 egg yolks"]}).encode("utf-8")
    gz = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
    body = gz.compress(html) + gz.flush()
    seen = {}

    def fake_urlopen(req, timeout):
        seen["ua"], seen["timeout"], seen["url"] = req.get_header("User-agent"), timeout, \
            req.full_url
        return _FakeResponse(body, "https://example.com/final/", {
            "Content-Encoding": "gzip", "Content-Type": "text/html; charset=utf-8"})

    monkeypatch.setattr(ex.urllib.request, "urlopen", fake_urlopen)
    source, page = ex.fetch("https://example.com/start")
    assert seen == {"ua": ex.USER_AGENT, "timeout": 20, "url": "https://example.com/start"}
    assert "Mozilla/5.0" in ex.USER_AGENT
    assert source == "https://example.com/final/"          # after redirects
    assert ex.extract(page)["name"] == "Crème brûlée"


def test_fetch_caps_the_page_at_5_mb(monkeypatch, capsys):
    big = b"<p>" + b"a" * (6 * 1024 * 1024)
    monkeypatch.setattr(ex.urllib.request, "urlopen",
                        lambda req, timeout: _FakeResponse(big, req.full_url, {}))
    _, page = ex.fetch("https://example.com/big")
    assert len(page) == 5 * 1024 * 1024
    assert "only the first 5 MB" in capsys.readouterr().err


def test_http_errors_exit_1(monkeypatch, capsys):
    import urllib.error

    def refuse(req, timeout):
        raise urllib.error.HTTPError(req.full_url, 403, "Forbidden", None, None)

    monkeypatch.setattr(ex.urllib.request, "urlopen", refuse)
    assert ex.main(["https://example.com/blocked"]) == 1
    err = capsys.readouterr().err
    assert "HTTP 403 Forbidden" in err


def test_the_extractor_never_executes_or_imports_anything_from_the_page():
    """Text parsing only: the script has no eval/exec, no subprocess, and
    uses the standard library alone."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert not re.search(r"(?<![.\w])(eval|exec|compile|__import__)\(", src)
    imported = set(re.findall(r"^(?:import|from) ([a-z_.]+)", src, re.MULTILINE))
    assert imported <= {"__future__", "html", "html.parser", "http.client", "json", "re",
                        "sys", "urllib.error", "urllib.request", "zlib", "typing"}
    assert all(m.split(".")[0] in sys.stdlib_module_names for m in imported)


# ─── SKILL.md guard ──────────────────────────────────────────

def _frontmatter_and_body() -> tuple[dict, str]:
    import yaml

    text = SKILL_MD.read_text(encoding="utf-8")
    m = re.match(r"^---\n(.*?)\n---\n(.*)$", text, re.DOTALL)
    assert m, "SKILL.md must start with a --- YAML frontmatter block"
    meta = yaml.safe_load(m.group(1))
    assert isinstance(meta, dict)
    return meta, m.group(2)


def test_skill_frontmatter_parses_and_names_the_skill():
    meta, body = _frontmatter_and_body()
    assert meta["name"] == "recipe-shopper" == SKILL_DIR.name
    assert re.fullmatch(r"[a-z0-9-]{1,64}", meta["name"])
    desc = meta["description"]
    assert isinstance(desc, str) and 0 < len(desc) <= 1024
    for when in ("recipe link", "pastes a recipe", "cheapest", "nearby"):
        assert when in desc, when                # says WHEN to use it
    assert body.strip()


def _schema_names(schema) -> set[str]:
    """Every property name and enum/const value anywhere in a JSON schema."""
    names: set[str] = set()
    if isinstance(schema, dict):
        names |= set(schema.get("properties", {}))
        names |= {v for v in schema.get("enum", []) if isinstance(v, str)}
        if isinstance(schema.get("const"), str):
            names.add(schema["const"])
        for v in schema.values():
            names |= _schema_names(v)
    elif isinstance(schema, list):
        for v in schema:
            names |= _schema_names(v)
    return names


@pytest.mark.asyncio
async def test_every_tool_parameter_and_field_skill_md_names_exists_on_the_server():
    from pantry_planner.mcp_server import server
    from pantry_planner.nlsearch.plan import GateCode

    tools = {t.name: t for t in await server.list_tools()}
    prompts = {p.name for p in await server.list_prompts()}
    plan = tools["plan_from_text"]
    params = plan.input_schema["properties"]
    _, body = _frontmatter_and_body()
    flat = " ".join(body.split())

    # 1. what the task requires SKILL.md to use is there, and on the server
    for name in ("plan_from_text", "allow_partial", "max_km", "lat", "lon", "recipe_text"):
        assert f"`{name}`" in body, name
    assert {"allow_partial", "max_km", "lat", "lon", "recipe_text"} <= set(params)
    assert params["allow_partial"]["default"] is False      # hence "always true"

    # 2. the example call is a valid argument object for plan_from_text
    calls = [json.loads(b) for b in re.findall(r"```json\n(.*?)\n```", body, re.DOTALL)]
    assert calls, "SKILL.md shows the plan_from_text call as a ```json block"
    for call in calls:
        assert set(call) <= set(params), set(call) - set(params)
        assert call["allow_partial"] is True

    # 3. the limits SKILL.md states are the server's limits
    max_km = next(s for s in params["max_km"]["anyOf"] if s.get("type") == "number")
    assert "`max_km` (0.5-100)" in flat
    assert (max_km["minimum"], max_km["maximum"]) == (0.5, 100)
    assert "At most 8000 characters" in flat
    assert params["recipe_text"]["maxLength"] == 8000 == ex.MAX_RECIPE_TEXT

    # 4. every identifier in inline code resolves to something real
    printed = _run("-", stdin=_ld({"@type": "Recipe", "name": "x",
                                   "recipeIngredient": ["1 egg"]}))
    extractor_keys = set(json.loads(printed.stdout))      # what the script really prints
    assert {"name", "yield", "servings", "ingredients", "recipe_text"} <= extractor_keys
    known = (set(tools) | prompts | set(params) | _schema_names(plan.output_schema)
             | {c.value for c in GateCode} | extractor_keys
             | {"python3", "true", "false", "null"})
    prose = re.sub(r"```.*?```", "", body, flags=re.DOTALL)
    unknown = []
    for span in re.findall(r"`([^`\n]+)`", prose):
        span = re.sub(r"^mcp__[a-z0-9_-]+?__", "", span)   # a client's tool prefix
        if not re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)*", span):
            continue                                      # commands, URLs, values
        unknown += [seg for seg in span.split(".") if seg not in known]
    assert unknown == [], unknown
    assert "pantry-plan-from-text" in body               # the ContextForge spelling


def test_skill_md_points_at_the_script_that_exists():
    _, body = _frontmatter_and_body()
    assert "python3 scripts/extract_recipe.py" in body
    assert SCRIPT.is_file()
    assert SCRIPT.read_text(encoding="utf-8").startswith("#!/usr/bin/env python3")
