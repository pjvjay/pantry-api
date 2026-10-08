#!/usr/bin/env python3
"""Pull the schema.org Recipe out of a recipe page, client-side.

    python3 extract_recipe.py https://example.com/some-recipe/
    python3 extract_recipe.py saved-page.html
    curl -s https://example.com/some-recipe/ | python3 extract_recipe.py -

Prints one JSON object:

    {"source": ..., "name": ..., "yield": ..., "servings": ...,
     "ingredients": [...], "recipe_text": ..., "omitted": [...]}

`ingredients` are the recipe's ingredient lines verbatim (entities decoded,
whitespace collapsed). `recipe_text` is those lines ready for the pantry
plan_from_text tool: "<name> (serves N)" then one "- <line>" per
ingredient, at most 8000 characters; `omitted` lists any line that did not
fit (normally empty).

Where the data comes from, in order: every <script type="application/ld+json">
block (a Recipe at the top level, inside a top-level array, inside "@graph",
or nested under another object such as a WebPage's mainEntity; "@type" may
be a string or a list), then microdata (itemprop="recipeIngredient" inside an
itemtype=".../Recipe" scope). The page is parsed as text only: no script on
it is ever executed, and no other URL it names is fetched.

Fetching: http(s) only, a redirect included (a redirect to ftp:, file: or
any other scheme is refused, never followed); at most 5 MB; and the whole
fetch gives up once 20 seconds have passed since it started (checked
between reads, and no single read waits longer than that either), so a
server that drips bytes cannot stall it.

Exit codes: 0 found; 1 the input could not be read (network, HTTP status,
timeout, missing file, unsupported scheme or redirect); 2 the input was
read but holds no recipe (common on pages behind a bot challenge, or sites
that publish no structured data) — then read the page another way and copy
the ingredient lines verbatim.

Python 3 standard library only.
"""
from __future__ import annotations

import html
import http.client
import json
import re
import sys
import time
import urllib.error
import urllib.request
import zlib
from html.parser import HTMLParser
from typing import Any

MAX_BYTES = 5 * 1024 * 1024          # read at most 5 MB of page
TIMEOUT_S = 20                       # the whole fetch, not one read
CHUNK = 64 * 1024
MAX_RECIPE_TEXT = 8000               # plan_from_text's recipe_text limit
USER_AGENT = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
              "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36")
EXIT_FETCH, EXIT_NO_RECIPE = 1, 2

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "param", "source", "track", "wbr"}
_BREAK_TAG = re.compile(r"<(?:br|/?p|/?div|/?li)\b[^>]*>", re.IGNORECASE)
_TAG = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")


class FetchError(Exception):
    """The input could not be read."""


# ─── text clean-up ───────────────────────────────────────────

def clean(value: Any) -> str:
    """Decode HTML entities (twice-encoded ones too), drop stray tags,
    collapse whitespace (non-breaking spaces included)."""
    if value is None:
        return ""
    text = str(value)
    for _ in range(3):                       # "&amp;#8217;" -> "&#8217;" -> "'"
        decoded = html.unescape(text)
        if decoded == text:
            break
        text = decoded
    text = _TAG.sub("", _BREAK_TAG.sub(" ", text))   # "<b>1 lb</b>," keeps its comma
    return _WS.sub(" ", text.replace("\u00a0", " ")).strip()


def _first_text(value: Any) -> str:
    """A schema.org text value: a string, a number, a list of them, or an
    object carrying name/text/@value."""
    if isinstance(value, list):
        for v in value:
            t = _first_text(v)
            if t:
                return t
        return ""
    if isinstance(value, dict):
        for key in ("name", "text", "@value"):
            if key in value:
                return _first_text(value[key])
        return ""
    if isinstance(value, bool):
        return ""
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return clean(value)


def _yield_text(value: Any) -> str:
    """recipeYield is often a list like ["4", "4 servings"]: prefer the
    entry that says what the number counts."""
    if isinstance(value, list):
        texts = [t for t in (_first_text(v) for v in value) if t]
        worded = [t for t in texts if re.search(r"[A-Za-z]", t)]
        return (worded or texts or [""])[0]
    return _first_text(value)


def servings_from(yield_text: str) -> int | None:
    """The serving count when the yield is one: "4", "4 servings",
    "Serves 4-6", "6 people". None for "24 cookies" or "1 loaf"."""
    m = re.search(r"\d+", yield_text)
    if not m:
        return None
    if re.fullmatch(r"\s*\d+\s*", yield_text) or re.search(
            r"serv|people|persons?|portions?|plates?", yield_text, re.IGNORECASE):
        return int(m.group())
    return None


_LINE_BREAK = re.compile(r"\r\n|\r|\n|<br\s*/?>", re.IGNORECASE)


def _ingredient_lines(value: Any) -> list[str]:
    """recipeIngredient lines. A list holds one ingredient per item (a line
    break inside an item is just wrapping). schema.org also allows a single
    Text value, and the legacy "ingredients" often is one string with a line
    per ingredient: that string is split at its line breaks, never collapsed
    into one line."""
    if isinstance(value, list):
        items = value
    elif value is None:
        items = []
    elif isinstance(value, dict):
        items = [value]
    else:
        items = _LINE_BREAK.split(str(value))
    lines = []
    for item in items:
        text = _first_text(item) if isinstance(item, dict) else clean(item)
        if text:
            lines.append(text)
    return lines


# ─── JSON-LD ─────────────────────────────────────────────────

def _is_recipe(node: dict) -> bool:
    types = node.get("@type")
    types = types if isinstance(types, list) else [types]
    for t in types:
        if isinstance(t, str) and re.split(r"[/:#]", t.strip())[-1].lower() == "recipe":
            return True
    return False


def _walk(node: Any, found: list[dict]) -> None:
    """Every Recipe object anywhere in a JSON-LD document, document order."""
    if isinstance(node, list):
        for item in node:
            _walk(item, found)
    elif isinstance(node, dict):
        if _is_recipe(node):
            found.append(node)
        for key, value in node.items():
            if key != "@context":
                _walk(value, found)


def _load_jsonld(block: str) -> Any:
    text = block.strip()
    # tolerate the wrappers some CMSes still emit around script bodies
    text = re.sub(r"^\s*(<!--|//\s*<!\[CDATA\[|<!\[CDATA\[)", "", text)
    text = re.sub(r"(-->|//\s*\]\]>|\]\]>)\s*$", "", text).strip()
    if not text:
        return None
    try:
        return json.loads(text, strict=False)    # strict=False: raw newlines in strings
    except json.JSONDecodeError:
        return None


def recipes_from_jsonld(blocks: list[str]) -> list[dict]:
    found: list[dict] = []
    for block in blocks:
        doc = _load_jsonld(block)
        if doc is not None:
            _walk(doc, found)
    out = []
    for node in found:
        ingredients = _ingredient_lines(node.get("recipeIngredient", node.get("ingredients")))
        out.append({"name": _first_text(node.get("name")) or _first_text(node.get("headline")),
                    "yield": _yield_text(node.get("recipeYield", node.get("yield"))),
                    "ingredients": ingredients, "method": "jsonld"})
    return out


# ─── HTML: JSON-LD blocks and microdata ──────────────────────

class _Scope:
    def __init__(self, itemtype: str):
        self.itemtype = itemtype
        self.props: dict[str, list[str]] = {}

    @property
    def is_recipe(self) -> bool:
        return any(re.split(r"[/:#]", t)[-1].lower() == "recipe" for t in self.itemtype.split())


# A capture collects the text of one itemprop element for its scope.
_Capture = tuple[_Scope, str, list[str]]
# Elements whose end tag HTML lets authors leave out: a new one closes the
# open one, or "<li>a<li>b" would read as one ingredient "a b".
_IMPLIED_END = {"li": {"ul", "ol"}, "p": {"div", "section", "article", "ul", "ol"}}
_URL_VALUED = {"link", "img", "source", "audio", "video", "embed", "iframe", "object",
               "track", "area"}


class _PageParser(HTMLParser):
    """Collects ld+json script bodies and microdata itemprops. Script and
    style bodies are only ever read as text."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.jsonld: list[str] = []
        root = _Scope("")
        self.scopes: list[_Scope] = [root]      # every scope, document order
        self.title = ""
        self._title_state = 0                   # 0 before, 1 inside, 2 after the first <title>
        self._script: list[str] | None = None   # an ld+json body being read
        self._raw_depth = 0                     # inside a script/style element
        self._stack: list[tuple[str, _Scope | None, list[_Capture]]] = []
        self._scope_stack: list[_Scope] = [root]

    @staticmethod
    def _attr_value(tag: str, attrs: dict[str, str]) -> str | None:
        """The value an itemprop element carries in an attribute, or None
        when its value is its text content."""
        if "content" in attrs:
            return attrs["content"]
        if tag in ("data", "meter") and "value" in attrs:
            return attrs["value"]
        if tag in _URL_VALUED:
            return ""                            # a URL, never recipe text
        return None

    def handle_starttag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        attrs = {k.lower(): (v or "") for k, v in attrs_list}
        if tag in ("script", "style"):
            self._raw_depth += 1
            if tag == "script" and attrs.get("type", "").split(";")[0].strip().lower() \
                    == "application/ld+json":
                self._script = []
            return
        if tag == "title" and self._title_state == 0:
            self._title_state = 1
        if tag in _IMPLIED_END:
            self._close_implied(tag)
        # an element's own itemprop belongs to the ENCLOSING scope
        props = (attrs.get("itemprop") or "").split()
        scope = self._scope_stack[-1]
        captures: list[_Capture] = []
        if props and "itemscope" not in attrs:
            value = self._attr_value(tag, attrs)
            if value is not None:
                for prop in props:
                    scope.props.setdefault(prop, []).append(clean(value))
            elif tag not in _VOID:
                captures = [(scope, prop, []) for prop in props]
        if tag in _VOID:
            return
        new_scope = None
        if "itemscope" in attrs:
            new_scope = _Scope(attrs.get("itemtype", ""))
            self.scopes.append(new_scope)
            self._scope_stack.append(new_scope)
        self._stack.append((tag, new_scope, captures))

    def handle_startendtag(self, tag: str, attrs_list: list[tuple[str, str | None]]) -> None:
        if tag in ("script", "style"):
            return                               # <script/> has no body to read
        self.handle_starttag(tag, attrs_list)
        if tag not in _VOID:
            self.handle_endtag(tag)              # <span itemprop="x"/> holds no text

    def _close_implied(self, tag: str) -> None:
        for i in range(len(self._stack) - 1, -1, -1):
            open_tag = self._stack[i][0]
            if open_tag in _IMPLIED_END[tag]:
                return                           # a new list/block: nothing to close
            if open_tag == tag:
                self._pop_to(i)
                return

    def _pop_to(self, index: int) -> None:
        """Close every open element from the top of the stack down to `index`."""
        while len(self._stack) > index:
            _, scope, captures = self._stack.pop()
            for owner, prop, parts in captures:
                text = clean("".join(parts))
                if text:
                    owner.props.setdefault(prop, []).append(text)
            if scope is not None and len(self._scope_stack) > 1:
                self._scope_stack.pop()

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._raw_depth = max(0, self._raw_depth - 1)
            if tag == "script" and self._script is not None:
                self.jsonld.append("".join(self._script))
                self._script = None
            return
        if tag == "title" and self._title_state == 1:
            self._title_state = 2
        for i in range(len(self._stack) - 1, -1, -1):
            if self._stack[i][0] == tag:
                self._pop_to(i)
                return                           # no match: a stray end tag

    def handle_data(self, data: str) -> None:
        if self._raw_depth:
            if self._script is not None:
                self._script.append(data)
            return
        if self._title_state == 1:
            self.title += data
        for _, _, captures in self._stack:
            for _, _, parts in captures:
                parts.append(data)

    def close(self) -> None:
        super().close()
        if self._script is not None:             # an unterminated ld+json block
            self.jsonld.append("".join(self._script))
            self._script = None
        self._pop_to(0)                          # flush elements left open


def recipes_from_microdata(parser: _PageParser) -> list[dict]:
    out = []
    for scope in parser.scopes:
        if not scope.is_recipe:
            continue
        lines = scope.props.get("recipeIngredient") or scope.props.get("ingredients") or []
        out.append({"name": (scope.props.get("name") or [""])[0],
                    "yield": (scope.props.get("recipeYield") or [""])[0],
                    "ingredients": [ln for ln in lines if ln], "method": "microdata"})
    if not any(r["ingredients"] for r in out):
        # itemprop="recipeIngredient" with no typed Recipe scope around it
        loose = [ln for s in parser.scopes for ln in s.props.get("recipeIngredient", []) if ln]
        if loose:
            out.append({"name": clean(parser.title), "yield": "", "ingredients": loose,
                        "method": "microdata"})
    return out


def extract(page: str) -> dict | None:
    """The first recipe with ingredient lines: JSON-LD first, then
    microdata. None when the page has neither.

    Returns {"name", "yield", "ingredients", "method"}. `method` is "jsonld"
    or "microdata", whichever markup held the recipe: an importer records it
    as the recipe's source, and only this function knows which one it read."""
    parser = _PageParser()
    parser.feed(page)
    parser.close()
    for recipe in recipes_from_jsonld(parser.jsonld) + recipes_from_microdata(parser):
        if recipe["ingredients"]:
            if not recipe["name"]:
                recipe["name"] = clean(parser.title)
            return recipe
    return None


# ─── recipe_text for plan_from_text ──────────────────────────

def build_recipe_text(name: str, yield_text: str, ingredients: list[str],
                      limit: int = MAX_RECIPE_TEXT) -> tuple[str, list[str]]:
    servings = servings_from(yield_text)
    title = name or "Recipe"
    if servings:
        header = f"{title} (serves {servings})"
    elif yield_text:
        header = f"{title} (makes {yield_text})"
    else:
        header = title
    text = header[:limit]
    for i, line in enumerate(ingredients):
        entry = f"\n- {line}"
        if len(text) + len(entry) > limit:
            return text, ingredients[i:]         # the rest, in recipe order
        text += entry
    return text, []


# ─── input ───────────────────────────────────────────────────

def _decode(body: bytes, charset: str | None) -> str:
    if not charset:
        head = body[:4096].decode("ascii", errors="ignore")
        m = re.search(r"""<meta[^>]+charset=["']?([A-Za-z0-9_.:-]+)""", head, re.IGNORECASE)
        charset = m.group(1) if m else "utf-8"
    try:
        return body.decode(charset, errors="replace")
    except LookupError:
        return body.decode("utf-8", errors="replace")


def _decompress(body: bytes, encoding: str) -> bytes:
    """Undo gzip/deflate transfer compression, capped at MAX_BYTES of output."""
    encoding = encoding.lower().strip()
    try:
        if encoding in ("", "identity"):
            return body
        if encoding in ("gzip", "x-gzip"):
            return zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(body, MAX_BYTES)
        if encoding == "deflate":
            try:
                return zlib.decompressobj().decompress(body, MAX_BYTES)
            except zlib.error:                   # raw deflate, no zlib header
                return zlib.decompressobj(-zlib.MAX_WBITS).decompress(body, MAX_BYTES)
    except zlib.error as e:
        raise FetchError(f"could not decompress the {encoding} response: {e}") from e
    raise FetchError(f"unsupported Content-Encoding {encoding!r}")


_HTTP = re.compile(r"^https?://", re.IGNORECASE)


class _HttpOnlyRedirects(urllib.request.HTTPRedirectHandler):
    """Follow a redirect only to another http(s) URL. urllib's own handler
    also follows ftp://, which let a recipe page send this script to an
    arbitrary FTP host (internal ones included) and parse what it served."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not _HTTP.match(newurl):
            raise FetchError(f"refused a redirect from {req.full_url} to {newurl}: "
                             "only http(s) URLs are fetched")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_HttpOnlyRedirects)


def _urlopen(req: urllib.request.Request, timeout: float):
    """The one network call (tests replace it): http(s) redirects only."""
    return _OPENER.open(req, timeout=timeout)


def _read_body(resp, url: str, deadline: float) -> bytes:
    """At most MAX_BYTES + 1 bytes, in whatever pieces arrive, giving up at
    the deadline. read1 returns as soon as any data is there, so a server
    sending a byte at a time is caught between reads."""
    read = getattr(resp, "read1", None) or resp.read
    chunks: list[bytes] = []
    size = 0
    while size <= MAX_BYTES:
        if time.monotonic() > deadline:
            raise FetchError(f"timed out after {TIMEOUT_S}s fetching {url}")
        chunk = read(min(CHUNK, MAX_BYTES + 1 - size))
        if not chunk:
            break
        chunks.append(chunk)
        size += len(chunk)
    return b"".join(chunks)


def fetch(url: str) -> tuple[str, str]:
    """(final URL after redirects, page text). Redirects are followed to
    http(s) URLs only, and the whole fetch is bounded by TIMEOUT_S."""
    req = urllib.request.Request(url, headers={
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Accept-Encoding": "gzip, deflate",
    })
    deadline = time.monotonic() + TIMEOUT_S
    try:
        with _urlopen(req, timeout=TIMEOUT_S) as resp:
            final_url = resp.geturl()
            if not _HTTP.match(final_url):
                raise FetchError(f"refused {final_url}: only http(s) URLs are fetched")
            body = _read_body(resp, url, deadline)
            encoding = resp.headers.get("Content-Encoding", "")
            charset = resp.headers.get_content_charset()
    except urllib.error.HTTPError as e:
        raise FetchError(f"HTTP {e.code} {e.reason} for {url}") from e
    except urllib.error.URLError as e:
        raise FetchError(f"could not reach {url}: {e.reason}") from e
    except TimeoutError as e:
        raise FetchError(f"timed out after {TIMEOUT_S}s fetching {url}") from e
    except (OSError, http.client.HTTPException, ValueError) as e:
        raise FetchError(f"could not fetch {url}: {e}") from e
    if len(body) > MAX_BYTES:
        print(f"warning: page is larger than {MAX_BYTES // (1024 * 1024)} MB; "
              "only the first 5 MB were read", file=sys.stderr)
        body = body[:MAX_BYTES]
    return final_url, _decode(_decompress(body, encoding), charset)


def read_input(arg: str) -> tuple[str, str]:
    """(source label, page text) for a URL, a file path or '-'."""
    if arg == "-":
        data = sys.stdin.buffer.read(MAX_BYTES + 1)[:MAX_BYTES]
        return "stdin", _decode(data, None)
    if re.match(r"^https?://", arg, re.IGNORECASE):
        return fetch(arg)
    if re.match(r"^[A-Za-z][A-Za-z0-9+.-]*://", arg):
        raise FetchError(f"unsupported URL scheme in {arg!r}: give an http(s) URL, "
                         "a file path or '-'")
    try:
        with open(arg, "rb") as f:
            data = f.read(MAX_BYTES + 1)[:MAX_BYTES]
    except OSError as e:
        raise FetchError(f"could not read {arg}: {e.strerror or e}") from e
    return arg, _decode(data, None)


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0] in ("-h", "--help"):
        print("usage: extract_recipe.py <url | file | ->", file=sys.stderr)
        return 0 if argv and argv[0] in ("-h", "--help") else EXIT_FETCH
    try:
        source, page = read_input(argv[0])
    except FetchError as e:
        print(f"error: {e}", file=sys.stderr)
        return EXIT_FETCH
    recipe = extract(page)
    if recipe is None:
        print(f"no recipe found in {source}: no schema.org Recipe with ingredients in its "
              "JSON-LD or microdata. Read the page another way and copy the ingredient "
              "lines verbatim.", file=sys.stderr)
        return EXIT_NO_RECIPE
    recipe_text, omitted = build_recipe_text(recipe["name"], recipe["yield"],
                                             recipe["ingredients"])
    if omitted:
        print(f"warning: recipe_text holds {len(recipe['ingredients']) - len(omitted)} of "
              f"{len(recipe['ingredients'])} ingredient lines ({MAX_RECIPE_TEXT}-char limit)",
              file=sys.stderr)
    json.dump({"source": source, "name": recipe["name"], "yield": recipe["yield"],
               "servings": servings_from(recipe["yield"]),
               "ingredients": recipe["ingredients"],
               "recipe_text": recipe_text, "omitted": omitted},
              sys.stdout, ensure_ascii=False, indent=2)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
