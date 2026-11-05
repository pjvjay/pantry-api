"""The recipe-shopper extractor against whole pages, including a 3 MB one.

tests/fixtures/recipe_pages/ holds SYNTHETIC pages, each written for these
tests to imitate a markup shape recipe sites publish: a WordPress SEO @graph
beside a recipe-card plugin, microdata with amounts in their own spans, a
WebPage's mainEntity with a sponsored link inside a line, a windows-1252
page with CDATA-wrapped JSON-LD, the legacy one-string "ingredients", and a
bot challenge with no recipe. None is copied from a real site.
expected.json says, by hand, what each page holds, and the tests assert
exactly that.

The large-page tests build a page over 3 MB whose JSON-LD comes last, as on
pages that print the recipe schema after the comments and ad scripts. It is
under MAX_BYTES, so a file, stdin, a plain fetch and a gzip fetch must all
find its recipe. A page whose JSON-LD starts past MAX_BYTES must not, and
the warning must say why. demo-hub's link import takes MAX_BYTES and
extract() from this script and runs extract() in worker threads, so these
are the pages it will read. No network, no LLM.
"""
from __future__ import annotations

import importlib.util
import io
import json
import subprocess
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor
from email.message import Message
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "skills" / "recipe-shopper" / "scripts" / "extract_recipe.py"
PAGES = Path(__file__).resolve().parent / "fixtures" / "recipe_pages"
EXPECTED = json.loads((PAGES / "expected.json").read_text(encoding="utf-8"))["pages"]
THREE_MB = 3 * 1024 * 1024


def _load_extractor():
    spec = importlib.util.spec_from_file_location("extract_recipe_fixtures", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ex = _load_extractor()


def _run(*args: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    """The script as the skill runs it: a separate python3 process."""
    return subprocess.run([sys.executable, str(SCRIPT), *args], input=stdin,
                          capture_output=True, text=True, timeout=60)


def _read(name: str) -> dict | None:
    """What extract() finds in a fixture read as a file, plus the serving
    count the CLI prints beside it."""
    _, page = ex.read_input(str(PAGES / name))
    found = ex.extract(page)
    return None if found is None else {**found, "servings": ex.servings_from(found["yield"])}


# ─── the fixture pages ───────────────────────────────────────

def test_every_fixture_page_is_listed_and_labelled_synthetic():
    pages = sorted(p.name for p in PAGES.glob("*.html"))
    assert pages == sorted(EXPECTED)
    for name in pages:
        head = (PAGES / name).read_bytes()[:400].decode("ascii")
        assert "SYNTHETIC test page" in head, name


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_each_fixture_page_yields_exactly_the_recipe_it_holds(name):
    assert _read(name) == EXPECTED[name]


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_the_cli_reads_each_fixture_page_as_the_skill_runs_it(name):
    out = _run(str(PAGES / name))
    want = EXPECTED[name]
    if want is None:
        assert (out.returncode, out.stdout) == (2, "")
        assert "no recipe found" in out.stderr
        return
    assert out.returncode == 0, out.stderr
    data = json.loads(out.stdout)
    assert {k: data[k] for k in ("name", "yield", "servings", "ingredients")} == \
        {k: want[k] for k in ("name", "yield", "servings", "ingredients")}
    assert data["recipe_text"].splitlines()[1:] == [f"- {ln}" for ln in want["ingredients"]]


def test_extract_gives_the_same_answers_from_many_threads_at_once():
    """The hub runs extract() in worker threads, so two imports can overlap.
    Each call builds its own parser, and nothing it writes is shared."""
    names = sorted(EXPECTED) * 4
    pages = {name: ex.read_input(str(PAGES / name))[1] for name in EXPECTED}
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda n: ex.extract(pages[n]), names))
    for name, found in zip(names, results, strict=True):
        want = EXPECTED[name]
        assert (found is None) == (want is None), name
        if found is not None:
            assert found["ingredients"] == want["ingredients"], name


# ─── a 3 MB page with its recipe at the end ──────────────────

LARGE_RECIPE = {
    "@context": "https://schema.org", "@type": "Recipe",
    "name": "Sheet-Pan Lemon Chicken Thighs", "recipeYield": ["6", "6 servings"],
    "recipeIngredient": ["8 bone-in chicken thighs", "1 lb baby potatoes, halved",
                         "1 red onion, cut in wedges", "3 tablespoons olive oil",
                         "1 teaspoon dried oregano", "1 lemon, sliced"]}
LARGE_WANT = {"method": "jsonld", "name": "Sheet-Pan Lemon Chicken Thighs",
              "yield": "6 servings", "ingredients": LARGE_RECIPE["recipeIngredient"]}


def _large_page(size: int) -> str:
    """At least `size` characters of page (all ASCII, so as many bytes),
    then LARGE_RECIPE's JSON-LD and the closing tags. Before it come a style
    block, an icon sprite, reader comments and ad scripts whose text looks
    like microdata: script bodies are only ever read as text, so none of it
    is a recipe. Built here, deterministically, so no multi-megabyte file is
    committed."""
    head = ('<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<title>Sheet-pan dinner - Example Kitchen</title><style>'
            + "".join(f".c{n}{{margin:0 0 {n % 4}em}}" for n in range(300))
            + '</style></head><body><svg style="display:none">'
            + "".join(f'<symbol id="i{n}"><path d="M0 0h24v24H0z"/></symbol>'
                      for n in range(100))
            + "</svg><main><h1>Sheet-pan dinner</h1><section class=\"comments\">")
    parts = [head]
    total = len(head)
    n = 0
    while total < size:
        block = (f'<div class="comment" id="c{n}"><p class="meta"><span class="author">'
                 f'Reader {n}</span> <time datetime="2026-01-01">Jan 1</time></p>'
                 f'<p>Made this on a Tuesday &amp; it was gone by Wednesday &mdash; '
                 f'thank you!</p><ul class="actions"><li><a href="#reply-{n}">Reply</a>'
                 f'</li></ul></div>\n')
        if n % 40 == 0:
            block += (f'<script>window.ad{n} = {{"html": "<li itemprop=\\"recipeIngredient'
                      f'\\">1 free sample</li>", "pad": "{"x" * 4000}"}};</script>\n')
        parts.append(block)
        total += len(block)
        n += 1
    parts.append(f'</section></main><script type="application/ld+json">'
                 f'{json.dumps(LARGE_RECIPE)}</script></body></html>')
    return "".join(parts)


@pytest.fixture(scope="module")
def large_page() -> str:
    page = _large_page(THREE_MB)
    assert page.index('type="application/ld+json"') > THREE_MB    # the recipe comes last
    assert THREE_MB < len(page.encode("utf-8")) < ex.MAX_BYTES
    return page


class _Response(io.BytesIO):
    """What urlopen returns, for a fetch with no network."""

    def __init__(self, body: bytes, url: str, headers: dict[str, str]):
        super().__init__(body)
        self._url = url
        self.headers = Message()
        for k, v in headers.items():
            self.headers[k] = v

    def geturl(self) -> str:
        return self._url


def test_a_3_mb_page_with_its_json_ld_at_the_end_is_read_from_a_file_and_stdin(
        tmp_path, large_page):
    path = tmp_path / "large.html"
    path.write_text(large_page, encoding="utf-8")
    for args, stdin in (((str(path),), None), (("-",), large_page)):
        out = _run(*args, stdin=stdin)
        assert out.returncode == 0, out.stderr
        assert out.stderr == ""                  # nothing was cut
        data = json.loads(out.stdout)
        assert (data["name"], data["yield"], data["servings"], data["ingredients"]) == (
            LARGE_WANT["name"], LARGE_WANT["yield"], 6, LARGE_WANT["ingredients"])


@pytest.mark.parametrize("encoding", ["", "gzip"])
def test_a_3_mb_page_is_fetched_whole_plain_or_gzipped(monkeypatch, capsys, large_page,
                                                       encoding):
    """The gzip case also checks the decompression cap: it stops output at
    MAX_BYTES, which a 3 MB page is under."""
    body = large_page.encode("utf-8")
    headers = {"Content-Type": "text/html; charset=utf-8"}
    if encoding:
        gz = zlib.compressobj(wbits=16 + zlib.MAX_WBITS)
        body = gz.compress(body) + gz.flush()
        headers["Content-Encoding"] = encoding
        assert len(body) < len(large_page) // 10   # what arrives is small; what it holds is not
    monkeypatch.setattr(ex, "_urlopen",
                        lambda req, timeout: _Response(body, req.full_url, headers))
    _, page = ex.fetch("https://example.com/sheet-pan-chicken")
    assert page == large_page
    assert ex.extract(page) == LARGE_WANT
    assert capsys.readouterr().err == ""


def test_a_recipe_that_starts_past_max_bytes_is_never_read_and_the_warning_says_why(
        monkeypatch, capsys):
    """MAX_BYTES is where reading stops, so JSON-LD that starts after it is
    never seen. The script warns that the page was cut, then exits 2 as for
    any page with no recipe. That is why the hub refuses a page over
    MAX_BYTES with 413 rather than reading part of it."""
    page = _large_page(ex.MAX_BYTES + 64 * 1024)
    assert page.index('type="application/ld+json"') > ex.MAX_BYTES
    monkeypatch.setattr(ex, "_urlopen", lambda req, timeout: _Response(
        page.encode("utf-8"), req.full_url, {"Content-Type": "text/html; charset=utf-8"}))
    assert ex.main(["https://example.com/huge"]) == ex.EXIT_NO_RECIPE
    err = capsys.readouterr().err
    mb = ex.MAX_BYTES // (1024 * 1024)
    assert f"page is larger than {mb} MB; only the first {mb} MB were read" in err
    assert "no recipe found in https://example.com/huge" in err
