"""Quick add: "3 Pepperoni Pizza + 2 Chicken Fried Rice + 3 chicken briyani + 7 mango
milkshakes in 2 weeks" read into counted recipe selections, with no LLM.

1. The period ("in 2 weeks", "a fortnight", "10 days") is read and cut out. The rest is split
   on '+', commas, semicolons, new lines, and 'and' when a count follows it, so "peanut
   butter and jelly sandwich" stays whole.
2. Each item's count: digits, number words, "2x", "2 x", "2 ×", "×3" or "x3" (leading or
   trailing). No count means one, flagged count_stated false. A slot word ("for breakfast",
   "snacks") becomes the slot hint.
3. Names are normalised: NFKD, accents stripped, case-folded, punctuation dropped, stopwords
   (of, the, a, an, and, with) dropped. Each word is singularised by rule: -ies -> y, -oes ->
   o, -es after s/x/z/ch/sh, a final -s on words over 3 letters, plus an exceptions list.
4. Matching, against the library, the demo starters and the shopper's own recipes, level by
   level; the first level with a match wins:
   exact (the same word set), plural (the same after singularising), alias (a recipe's whole
   alternative names, or a title word's listed misspellings), fuzzy (every title word matched
   by its own input word within Damerau-Levenshtein distance 2 for words of 6+ letters, 1 for
   4-5, none for shorter, and no input word left over).
5. More than one recipe at the winning level (and, for fuzzy, at the smallest distance) is
   ambiguous: no match, the candidates listed. With no match at any level, the recipes whose
   title contains every input word are offered as candidates; two or more of them is
   ambiguous too.

Only exact and plural matches may be accepted without asking. alias and fuzzy come back with
needs_confirmation, and the console asks "chicken briyani -> Chicken Biryani (demo starter)?".
"""
from __future__ import annotations

import functools
import itertools
import json
import re
import unicodedata
from dataclasses import dataclass, field

from ..db import SEEDS_DIR

ALIASES_FILE = SEEDS_DIR / "recipe_aliases.json"

STOPWORDS = frozenset({"of", "the", "a", "an", "and", "with"})
NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
    "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14,
    "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20, "a": 1, "an": 1,
}
# Words the suffix rules get wrong, and words that end in s but are singular.
SINGULAR_EXCEPTIONS = {
    "cookies": "cookie", "pies": "pie", "brownies": "brownie", "smoothies": "smoothie",
    "leaves": "leaf", "loaves": "loaf", "halves": "half", "hummus": "hummus",
    "couscous": "couscous", "asparagus": "asparagus", "molasses": "molasses",
    "citrus": "citrus", "swiss": "swiss", "shoes": "shoe", "series": "series",
}
SLOT_WORDS = {"breakfast": "breakfast", "breakfasts": "breakfast", "lunch": "lunch",
              "lunches": "lunch", "dinner": "dinner", "dinners": "dinner",
              "supper": "dinner", "snack": "snack", "snacks": "snack"}
MAX_COUNT = 28
MAX_DAYS = 14

_NUM = r"(?:\d{1,3}|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")"
_PERIOD = re.compile(
    r"(?:\b(?:in|for|over|across|within)\s+(?:the\s+)?(?:next\s+)?(?:a\s+)?fortnight\b"
    r"|\b(?:a\s+)?fortnight\b"
    r"|\b(?:in|for|over|across|within)\s+(?:the\s+)?(?:next\s+)?(?P<n1>" + _NUM + r")\s+"
    r"(?P<u1>weeks?|days?)\b"
    r"|\b(?P<n2>" + _NUM + r")\s+(?P<u2>weeks?|days?)\s*$)",
    re.IGNORECASE)
_SPLIT = re.compile(r"\s*(?:[+,;\n]|\band\b(?=\s*(?:\d|[x×]\s*\d|(?:"
                    + "|".join(w for w in NUMBER_WORDS if w not in {"a", "an"})
                    + r")\b)))\s*", re.IGNORECASE)
_LEAD = re.compile(r"^(?P<n>\d{1,3})\s*[x×](?=\s|[^\W\d_])\s*|^(?P<w>" + _NUM
                   + r")\s+(?:[x×]\s+)?|^[x×]\s*(?P<m>\d{1,3})\s+", re.IGNORECASE)
_TRAIL = re.compile(r"\s+(?:[x×]\s*(?P<n>\d{1,3})|(?P<m>\d{1,3})\s*[x×])$", re.IGNORECASE)
_SLOT = re.compile(r"\s*\b(?:for|as|at)?\s*\b(?P<s>" + "|".join(SLOT_WORDS) + r")\b\s*",
                   re.IGNORECASE)


def _number(s: str) -> int:
    return int(s) if s.isdigit() else NUMBER_WORDS[s.lower()]


# ─── Words ───────────────────────────────────────────────────

def singular(word: str) -> str:
    if word in SINGULAR_EXCEPTIONS:
        return SINGULAR_EXCEPTIONS[word]
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith("oes") and len(word) > 4:
        return word[:-2]
    if word.endswith("es") and len(word) > 4 and word[:-2].endswith(("s", "x", "z", "ch", "sh")):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        return word[:-1]
    return word


def words(name: str) -> list[str]:
    """NFKD, accents stripped, case-folded, punctuation dropped, stopwords out."""
    plain = "".join(c for c in unicodedata.normalize("NFKD", name)
                    if not unicodedata.combining(c)).casefold()
    return [w for w in re.split(r"[^0-9a-z]+", plain) if w and w not in STOPWORDS]


def damerau_levenshtein(a: str, b: str) -> int:
    """The unrestricted Damerau-Levenshtein distance (insertions, deletions, substitutions
    and transpositions of adjacent characters, a substring editable more than once):
    Lowrance and Wagner's algorithm."""
    inf = len(a) + len(b)
    d = [[0] * (len(b) + 2) for _ in range(len(a) + 2)]
    d[0][0] = inf
    for i in range(len(a) + 1):
        d[i + 1][0], d[i + 1][1] = inf, i
    for j in range(len(b) + 1):
        d[0][j + 1], d[1][j + 1] = inf, j
    last: dict[str, int] = {}
    for i in range(1, len(a) + 1):
        db = 0
        for j in range(1, len(b) + 1):
            i1, j1 = last.get(b[j - 1], 0), db
            cost = 0 if a[i - 1] == b[j - 1] else 1
            if cost == 0:
                db = j
            d[i + 1][j + 1] = min(d[i][j] + cost, d[i + 1][j] + 1, d[i][j + 1] + 1,
                                  d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1))
        last[a[i - 1]] = i
    return d[len(a) + 1][len(b) + 1]


def allowed_distance(word: str) -> int:
    """2 for a title word of 6 or more letters, 1 for 4-5, 0 for shorter."""
    return 2 if len(word) >= 6 else 1 if len(word) >= 4 else 0


# ─── Recipes to match ────────────────────────────────────────

@dataclass(frozen=True)
class Candidate:
    key: str
    title: str
    kind: str                 # library | starter | my
    slot: str
    label: str
    aliases: tuple[str, ...] = ()
    token_aliases: dict = field(default_factory=dict, hash=False, compare=False)


@functools.lru_cache(maxsize=1)
def aliases_file() -> dict:
    return json.loads(ALIASES_FILE.read_text(encoding="utf-8"))


def candidates(library: list[tuple[str, str]], starters: list[dict],
               mine: list[dict]) -> list[Candidate]:
    """library: (slug, name); starters: mealplan_starters.json entries; mine: the shopper's
    own [{key, title, aliases?}]. Library first, then starters, then the shopper's."""
    common = aliases_file()["token_aliases"]
    by_slug = aliases_file()["recipes"]
    out = [Candidate(key=f"lib:{slug}", title=name, kind="library", slot="dinner",
                     label="library", aliases=tuple(by_slug.get(slug, [])),
                     token_aliases=common) for slug, name in library]
    for s in starters:
        out.append(Candidate(key=s["doc_key"], title=s["title"], kind="starter", slot=s["slot"],
                             label="demo starter", aliases=tuple(s.get("aliases", [])),
                             token_aliases={**common, **s.get("token_aliases", {})}))
    for r in mine:
        out.append(Candidate(key=r["key"], title=r["title"], kind="my",
                             slot=r.get("slot") or "dinner", label="my recipe",
                             aliases=tuple(r.get("aliases") or []), token_aliases=common))
    return out


# ─── Matching ────────────────────────────────────────────────

@dataclass
class Match:
    candidate: Candidate
    how: str
    distance: int = 0


def _fuzzy(title: list[str], given: list[str]) -> int | None:
    """The smallest summed distance pairing each title word with its own input word within
    its allowance, using every input word; None when there is no such pairing."""
    if len(title) != len(given) or not title or len(title) > 7:
        return None
    best = None
    for perm in itertools.permutations(given):
        total = 0
        for t, g in zip(title, perm, strict=True):
            dist = damerau_levenshtein(t, g)
            if dist > allowed_distance(t):
                break
            total += dist
        else:
            best = total if best is None else min(best, total)
    return best


def _unalias(given: list[str], token_aliases: dict) -> list[str]:
    back = {singular(v.casefold()): k for k, vs in token_aliases.items() for v in vs}
    return [back.get(w, w) for w in given]


def match_name(name: str, cands: list[Candidate]) -> tuple[list[Match], str]:
    """(matches at the winning level, the level) or ([], '')."""
    given = words(name)
    if not given:
        return [], ""
    g_set, g_sing = set(given), {singular(w) for w in given}
    levels = [
        ("exact", lambda c: set(words(c.title)) == g_set),
        ("plural", lambda c: {singular(w) for w in words(c.title)} == g_sing),
        ("alias", lambda c: any({singular(w) for w in words(a)} == g_sing for a in c.aliases)
         or {singular(w) for w in _unalias([singular(w) for w in given], c.token_aliases)}
         == {singular(w) for w in words(c.title)}),
    ]
    for how, test in levels:
        found = [Match(c, how) for c in cands if test(c)]
        if found:
            return found, how
    fuzzy = []
    for c in cands:
        dist = _fuzzy([singular(w) for w in words(c.title)], [singular(w) for w in given])
        if dist is not None:
            fuzzy.append(Match(c, "fuzzy", dist))
    if fuzzy:
        low = min(m.distance for m in fuzzy)
        return [m for m in fuzzy if m.distance == low], "fuzzy"
    return [], ""


def contains_all(name: str, cands: list[Candidate]) -> list[Candidate]:
    """Recipes whose title has every input word (singularised): offered, never chosen."""
    g = {singular(w) for w in words(name)}
    if not g:
        return []
    return [c for c in cands if g <= {singular(w) for w in words(c.title)}]


# ─── Parsing ─────────────────────────────────────────────────

def read_period(text: str) -> tuple[int | None, str]:
    """(days, text without the period phrase)."""
    m = _PERIOD.search(text)
    if not m:
        return None, text
    if m.group("n1") or m.group("n2"):
        n = _number(m.group("n1") or m.group("n2"))
        unit = (m.group("u1") or m.group("u2")).lower()
        days = n * 7 if unit.startswith("week") else n
    else:
        days = 14
    return days, (text[:m.start()] + " " + text[m.end():]).strip()


def read_item(raw: str) -> tuple[str, int, bool, str | None]:
    """(name, count, count_stated, slot_hint) for one item."""
    s = raw.strip()
    count, stated = 1, False
    m = _LEAD.match(s)
    if m:
        count, stated = _number(m.group("n") or m.group("w") or m.group("m")), True
        s = s[m.end():]
    else:
        m = _TRAIL.search(s)
        if m:
            count, stated = int(m.group("n") or m.group("m")), True
            s = s[:m.start()]
    slot = None
    m = _SLOT.search(s)
    if m and words(s[:m.start()] + " " + s[m.end():]):
        slot = SLOT_WORDS[m.group("s").lower()]
        s = (s[:m.start()] + " " + s[m.end():]).strip()
    # "a fortnight of grilled cheese" leaves "of grilled cheese": the name starts at a word.
    s = re.sub(r"^(?:(?:of|the)\s+)+", "", s.strip(), flags=re.IGNORECASE)
    return s.strip(), count, stated, slot


def _plural_slot(slot: str, n: int) -> str:
    if n == 1:
        return slot
    return {"lunch": "lunches"}.get(slot, slot + "s")


def parse(text: str, cands: list[Candidate], household_servings: int = 2) -> dict:
    """The selection-parse response (see the module docstring)."""
    period, rest = read_period(text)
    warnings: list[str] = []
    if period is not None and not 1 <= period <= MAX_DAYS:
        warnings.append(f"the period read is {period} days; a plan covers 1 to {MAX_DAYS} days")
    selections, unmatched = [], []
    for raw in _SPLIT.split(rest):
        if not raw.strip():
            continue
        name, count, stated, slot_hint = read_item(raw)
        if not words(name):
            continue
        if count > MAX_COUNT:
            warnings.append(f"{raw.strip()!r}: {count} meals of one recipe is more than "
                            f"{MAX_COUNT}")
        found, _how = match_name(name, cands)
        matched = found[0] if len(found) == 1 else None
        if matched is not None:
            listed = [matched.candidate]
            status = "matched"
        elif found:
            listed = [m.candidate for m in found]
            status = "ambiguous"
        else:
            listed = contains_all(name, cands)
            status = "ambiguous" if len(listed) > 1 else "unmatched"
        meaning = None
        if matched is not None:
            slot = slot_hint or matched.candidate.slot
            meaning = (f"{count} × {matched.candidate.title} = {count} "
                       f"{_plural_slot(slot, count)} for {household_servings} people")
        selections.append({
            "input": raw.strip(), "name": name, "count": count, "count_stated": stated,
            "slot_hint": slot_hint, "status": status,
            "matched_as": None if matched is None else {
                "recipe_key": matched.candidate.key, "title": matched.candidate.title,
                "kind": matched.candidate.kind, "label": matched.candidate.label,
                "slot": matched.candidate.slot, "how": matched.how,
                "distance": matched.distance},
            "needs_confirmation": matched is None or matched.how in {"alias", "fuzzy"},
            "candidates": [{"recipe_key": c.key, "title": c.title, "kind": c.kind,
                            "label": c.label} for c in listed if matched is None],
            "meaning": meaning,
        })
        if status == "unmatched":
            unmatched.append(name)
    return {"selections": selections, "unmatched": unmatched, "period_days": period,
            "warnings": warnings}
