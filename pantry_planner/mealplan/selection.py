"""Quick add: "3 Pepperoni Pizza + 2 Chicken Fried Rice + 3 chicken briyani + 7 mango
milkshakes in 2 weeks" read into counted recipe selections, with no LLM.

1. The period ("in 2 weeks", "a fortnight", "10 days", or a leading "10 days of") is read
   and cut out. The rest is split on '+', commas, semicolons, new lines, and 'and' when a
   count follows it, so "peanut butter and jelly sandwich" stays whole.
2. Each item's count: digits, number words, "2x", "2 x", "2 ×", "×3" or "x3" (leading or
   trailing). No count means one, flagged count_stated false. A slot word after for, as or
   at ("for breakfast") becomes the slot hint. A bare one ("snacks") may be part of the
   title ("Breakfast Burrito", "Dinner Rolls"), so the name is matched with it first, and
   it becomes the hint only when that finds nothing.
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

The route is public and has no LLM, so its work is bounded: at most MAX_ITEMS dishes are read
(a warning names the rest), a dish name of more than MAX_ITEM_WORDS words is not matched, runs
of spaces are read as one, and the fuzzy level is a minimum-cost assignment over the title x
input word-distance matrix (at most 7 x 7), each distance computed once per request, only up
to the word's allowance, and within a request-wide MAX_FUZZY_ROWS.
"""
from __future__ import annotations

import functools
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
# Dishes read from one Quick add. A plan holds 12 recipes (models.MAX_RECIPES); the rest is
# room for a dish typed twice or a name that matches nothing.
MAX_ITEMS = 20
MAX_ITEM_WORDS = 12       # a longer "dish name" is not matched
MAX_FUZZY_WORDS = 7       # fuzzy matching pairs at most 7 title words with 7 input words
# Work one request may spend on fuzzy matching, in rows of the banded distance (at most 5
# cells each), each word pair counted once. Twenty typed dishes against fifty recipes of real
# words use a small part of it; past it there is no fuzzy matching, and a warning says so.
MAX_FUZZY_ROWS = 50_000

_NUM = r"(?:\d{1,3}|" + "|".join(sorted(NUMBER_WORDS, key=len, reverse=True)) + r")"
_PERIOD = re.compile(
    r"(?:\b(?:in|for|over|across|within)\s+(?:the\s+)?(?:next\s+)?(?:a\s+)?fortnight\b"
    r"|\b(?:a\s+)?fortnight\b"
    r"|\b(?:in|for|over|across|within)\s+(?:the\s+)?(?:next\s+)?(?P<n1>" + _NUM + r")\s+"
    r"(?P<u1>weeks?|days?)\b"
    r"|\b(?P<n2>" + _NUM + r")\s+(?P<u2>weeks?|days?)\s*$"
    r"|^\s*(?P<n3>" + _NUM + r")\s+(?P<u3>weeks?|days?)\s+of\b)",
    re.IGNORECASE)
_BLANKS = re.compile(r"[^\S\n]+")
_SPLIT = re.compile(r"\s*(?:[+,;\n]|\band\b(?=\s*(?:\d|[x×]\s*\d|(?:"
                    + "|".join(w for w in NUMBER_WORDS if w not in {"a", "an"})
                    + r")\b)))\s*", re.IGNORECASE)
_LEAD = re.compile(r"^(?P<n>\d{1,3})\s*[x×](?=\s|[^\W\d_])\s*|^(?P<w>" + _NUM
                   + r")\s+(?:[x×]\s+)?|^[x×]\s*(?P<m>\d{1,3})\s+", re.IGNORECASE)
_TRAIL = re.compile(r"\s+(?:[x×]\s*(?P<n>\d{1,3})|(?P<m>\d{1,3})\s*[x×])$", re.IGNORECASE)
_SLOT_AFTER = re.compile(r"\s*\b(?:for|as|at)\s+(?:(?:an?|the)\s+)?(?P<s>"
                         + "|".join(SLOT_WORDS) + r")\b\s*", re.IGNORECASE)
_SLOT_BARE = re.compile(r"\s*\b(?P<s>" + "|".join(SLOT_WORDS) + r")\b\s*", re.IGNORECASE)


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


def within_distance(a: str, b: str, k: int) -> int | None:
    """damerau_levenshtein(a, b) when it is at most k, else None, in O(k x len) time.

    A common prefix and suffix cost nothing and are cut first (the distance's recurrence
    pairs equal end letters at no cost, read from either end). The rest is the same algorithm
    on the diagonal band |i - j| <= k: a prefix pair off the band is more than k apart (the
    lengths alone differ by more), so every cell holding k or less is computed exactly and
    anything off the band reads as more than k. It stops at the first row with no cell
    within k, since a row's smallest distance never falls in later rows."""
    if a == b:
        return 0
    n, m = len(a), len(b)
    if k <= 0 or abs(n - m) > k:
        return None
    p, top = 0, min(n, m)
    while p < top and a[p] == b[p]:
        p += 1
    s = 0
    while s < top - p and a[n - 1 - s] == b[m - 1 - s]:
        s += 1
    a, b, n, m = a[p:n - s], b[p:m - s], n - p - s, m - p - s
    if not n or not m:
        return n + m
    big = k + 1
    d = [[big] * (m + 2) for _ in range(n + 2)]
    for i in range(min(n, k) + 1):
        d[i + 1][1] = i
    for j in range(min(m, k) + 1):
        d[1][j + 1] = j
    last: dict[str, int] = {}
    for i in range(1, n + 1):
        ai, above, row = a[i - 1], d[i], d[i + 1]
        db, row_min = 0, big
        for j in range(max(1, i - k), min(m, i + k) + 1):
            i1, j1 = last.get(b[j - 1], 0), db
            if ai == b[j - 1]:
                v, db = above[j], j
            else:
                v = above[j] + 1
            v = min(v, row[j] + 1, above[j + 1] + 1)
            if i1 and j1:            # a transposition, with letters between dropped or added
                v = min(v, d[i1][j1] + (i - i1 - 1) + 1 + (j - j1 - 1))
            row[j + 1] = v
            row_min = min(row_min, v)
        if row_min > k:
            return None
        last[ai] = i
    v = d[n + 1][m + 1]
    return v if v <= k else None


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
    seen = {c.key for c in out}
    for r in mine:
        if r["key"] in seen:           # the same recipe sent twice is one candidate
            continue
        seen.add(r["key"])
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


def _fuzzy(title: list[str] | tuple[str, ...], given: list[str] | tuple[str, ...],
           dist=None) -> int | None:
    """The smallest summed distance pairing each title word with its own input word within
    its allowance, using every input word; None when there is no such pairing.

    A minimum-cost assignment over the title x input distance matrix: title word by title
    word, the cheapest total for each set of input words used so far (at most 7 x 2^7
    steps), stopping at a title word no unused input word is close enough to. `dist(t, g)`
    is the pair's distance within allowed_distance(t), or None."""
    n = len(title)
    if n != len(given) or not n or n > MAX_FUZZY_WORDS:
        return None
    if dist is None:
        def dist(t: str, g: str) -> int | None:
            return within_distance(t, g, allowed_distance(t))
    best = {0: 0}                       # input words used (a bit set) -> smallest total
    for t in title:
        row = [dist(t, g) for g in given]
        nxt: dict[int, int] = {}
        for used, total in best.items():
            for j, dj in enumerate(row):
                if dj is None or used >> j & 1:
                    continue
                key = used | 1 << j
                if key not in nxt or total + dj < nxt[key]:
                    nxt[key] = total + dj
        if not nxt:
            return None
        best = nxt
    return best[(1 << n) - 1]


def _key(words_: list[str]) -> frozenset[str]:
    return frozenset(singular(w) for w in words_)


class Matcher:
    """Names matched against one request's candidates. Each candidate's words are read once,
    and each fuzzy pairing and word distance is computed once per request, so the same
    title sent many times, or the same dish typed twice, costs one. Fuzzy matching spends at
    most MAX_FUZZY_ROWS; the name it runs out on, and every later one, gets no fuzzy match
    (`out_of_work` turns true), never one chosen from the recipes compared so far."""

    def __init__(self, cands: list[Candidate]):
        self.cands = cands
        backs: dict[int, dict[str, str]] = {}
        self._titles = []
        for c in cands:
            ws = words(c.title)
            back = backs.get(id(c.token_aliases))
            if back is None:
                back = backs[id(c.token_aliases)] = {
                    singular(v.casefold()): k for k, vs in c.token_aliases.items() for v in vs}
            self._titles.append((frozenset(ws), _key(ws), tuple(singular(w) for w in ws),
                                 tuple(_key(words(a)) for a in c.aliases), back))
        self._fuzzy: dict[tuple, int | None] = {}
        self._dist: dict[tuple[str, str], int | None] = {}
        self._rows_left = MAX_FUZZY_ROWS
        self.out_of_work = False

    def _distance(self, t: str, g: str) -> int | None:
        if (t, g) in self._dist:
            return self._dist[t, g]
        k = allowed_distance(t)
        if t == g:
            dist = 0
        elif k == 0 or abs(len(t) - len(g)) > k:
            dist = None
        elif self._rows_left < min(len(t), len(g)):
            self.out_of_work = True
            return None
        else:
            self._rows_left -= min(len(t), len(g))
            dist = within_distance(t, g, k)
        self._dist[t, g] = dist
        return dist

    def match(self, name: str) -> tuple[list[Match], str]:
        """(matches at the winning level, the level) or ([], '')."""
        given = words(name)
        if not given or len(given) > MAX_ITEM_WORDS:
            return [], ""
        g_set, g_sing = frozenset(given), _key(given)
        g_list = tuple(singular(w) for w in given)
        levels = [
            ("exact", lambda tw: tw[0] == g_set),
            ("plural", lambda tw: tw[1] == g_sing),
            ("alias", lambda tw: any(a == g_sing for a in tw[3])
             or frozenset(singular(tw[4].get(w, w)) for w in g_list) == tw[1]),
        ]
        for how, test in levels:
            found = [Match(c, how) for c, tw in zip(self.cands, self._titles, strict=True)
                     if test(tw)]
            if found:
                return found, how
        fuzzy = []
        for c, tw in zip(self.cands, self._titles, strict=True):
            if self.out_of_work:
                return [], ""
            key = (tw[2], g_list)
            if key not in self._fuzzy:
                self._fuzzy[key] = _fuzzy(tw[2], g_list, self._distance)
            if self._fuzzy[key] is not None:
                fuzzy.append(Match(c, "fuzzy", self._fuzzy[key]))
        if self.out_of_work:
            return [], ""
        if fuzzy:
            low = min(m.distance for m in fuzzy)
            return [m for m in fuzzy if m.distance == low], "fuzzy"
        return [], ""

    def contains_all(self, name: str) -> list[Candidate]:
        """Recipes whose title has every input word (singularised): offered, never chosen."""
        given = words(name)
        if not given or len(given) > MAX_ITEM_WORDS:
            return []
        g = _key(given)
        return [c for c, tw in zip(self.cands, self._titles, strict=True) if g <= tw[1]]


def match_name(name: str, cands: list[Candidate]) -> tuple[list[Match], str]:
    """(matches at the winning level, the level) or ([], '')."""
    return Matcher(cands).match(name)


def contains_all(name: str, cands: list[Candidate]) -> list[Candidate]:
    """Recipes whose title has every input word (singularised): offered, never chosen."""
    return Matcher(cands).contains_all(name)


# ─── Parsing ─────────────────────────────────────────────────

def read_period(text: str) -> tuple[int | None, str]:
    """(days, text without the period phrase)."""
    m = _PERIOD.search(text)
    if not m:
        return None, text
    if m.group("n1") or m.group("n2") or m.group("n3"):
        n = _number(m.group("n1") or m.group("n2") or m.group("n3"))
        unit = (m.group("u1") or m.group("u2") or m.group("u3")).lower()
        days = n * 7 if unit.startswith("week") else n
    else:
        days = 14
    return days, (text[:m.start()] + " " + text[m.end():]).strip()


def _lead_words(s: str) -> str:
    # "a fortnight of grilled cheese" leaves "of grilled cheese": the name starts at a word.
    return re.sub(r"^(?:(?:of|the)\s+)+", "", s.strip(), flags=re.IGNORECASE).strip()


def read_item(raw: str) -> tuple[str, int, bool, str | None, str | None]:
    """(name, count, count_stated, slot_hint, kept) for one item. kept is the name with a
    bare slot word left in ("breakfast burrito"), for parse() to try first; None when the
    slot word followed for, as or at, or there was none."""
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
    slot, kept = None, None
    for pattern in (_SLOT_AFTER, _SLOT_BARE):
        m = pattern.search(s)
        if m and words(s[:m.start()] + " " + s[m.end():]):
            if pattern is _SLOT_BARE:
                kept = _lead_words(s)
            slot = SLOT_WORDS[m.group("s").lower()]
            s = (s[:m.start()] + " " + s[m.end():]).strip()
            break
    return _lead_words(s), count, stated, slot, kept


def _plural_slot(slot: str, n: int) -> str:
    if n == 1:
        return slot
    return {"lunch": "lunches"}.get(slot, slot + "s")


def parse(text: str, cands: list[Candidate], household_servings: int = 2) -> dict:
    """The selection-parse response (see the module docstring)."""
    # A run of spaces or tabs reads as one space (new lines still split dishes): the
    # patterns below then never backtrack over a long run.
    text = _BLANKS.sub(" ", text)
    period, rest = read_period(text)
    warnings: list[str] = []
    if period is not None and not 1 <= period <= MAX_DAYS:
        warnings.append(f"the period read is {period} days; a plan covers 1 to {MAX_DAYS} days")
    matcher = Matcher(cands)
    selections, unmatched, left_out = [], [], []
    for raw in _SPLIT.split(rest):
        if not raw.strip():
            continue
        name, count, stated, slot_hint, kept = read_item(raw)
        if not words(name):
            continue
        if len(selections) == MAX_ITEMS:
            left_out.append(raw.strip())
            continue
        if count > MAX_COUNT:
            warnings.append(f"{raw.strip()!r}: {count} meals of one recipe is more than "
                            f"{MAX_COUNT}")
        if len(words(name)) > MAX_ITEM_WORDS:
            warnings.append(f"{raw.strip()!r}: a dish name of more than {MAX_ITEM_WORDS} "
                            "words is not matched")
        spent = matcher.out_of_work
        found = matcher.match(kept)[0] if kept is not None else []
        if found:
            name, slot_hint = kept, None          # the slot word is part of the title
        else:
            found, _how = matcher.match(name)
        if matcher.out_of_work and not spent:
            warnings.append(f"from {raw.strip()!r} on, names were matched exactly, as plurals "
                            "or by alias only: there were too many to compare for spelling "
                            "slips")
        matched = found[0] if len(found) == 1 else None
        if matched is not None:
            listed = [matched.candidate]
            status = "matched"
        elif found:
            listed = [m.candidate for m in found]
            status = "ambiguous"
        else:
            listed = matcher.contains_all(name)
            status = "ambiguous" if len(listed) > 1 else "unmatched"
        meaning = None
        if matched is not None:
            slot = slot_hint or matched.candidate.slot
            meaning = (f"{count} × {matched.candidate.title} = {count} "
                       f"{_plural_slot(slot, count)} for {household_servings} "
                       f"{'person' if household_servings == 1 else 'people'}")
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
    if left_out:
        warnings.append(f"Quick add reads {MAX_ITEMS} dishes at a time; {len(left_out)} more "
                        f"were not read, from {left_out[0]!r}")
    return {"selections": selections, "unmatched": unmatched, "period_days": period,
            "warnings": warnings}
