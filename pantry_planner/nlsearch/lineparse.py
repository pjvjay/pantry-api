"""Deterministic ingredient-line parsing: "2 cloves garlic, minced" -> name,
form, quantity, unit and prep, with no LLM.

Lifted out of demomode.parse_recipe so three callers share one reading of a
line: the demo-mode recipe parse, POST /recipes/parse-lines (a pasted or
imported ingredient list the shopper reviews before planning) and the meal
planner's written dishes. demomode.parse_recipe still decides which lines of
a whole recipe are ingredient lines (the "- " bullets) and reads the title,
servings and shopping notes; this module only reads one line. The lift
changed no output: tests/fixtures/lineparse_golden.json was recorded from
demomode before it, and test_lineparse checks both against it.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# A leading amount: "2", "1.5", "1/4", "1 1/2", "1½", "¾", and a range's
# lower end ("1 to 3", "1-3") — then an optional unit. Units normalize_quantity
# knows become a size need; the rest (inch, pinch, bunch, ...) only count.
_NUM = r"(?:\d+(?:\.\d+)?(?:\s+\d+/\d+|\s*[½¼¾⅓⅔⅛])?|\d+/\d+|[½¼¾⅓⅔⅛])"
_QTY = re.compile(
    rf"^({_NUM})(?:\s*(?:-|–|to)\s*{_NUM})?\s*"
    r"(g|grams?|kg|kilograms?|ml|millilit(?:er|re)s?|l|lit(?:er|re)s?|lbs?|pounds?"
    r"|oz|ounces?|cups?|cans?|cloves?|dozen|tbsps?|tablespoons?|tsps?|teaspoons?"
    r"|inch(?:es)?|thumbs?|pinch(?:es)?|bunch(?:es)?|sprigs?|slices?|pieces?|heads?"
    r"|stalks?|handfuls?)?\.?(?=\s|$)\s*",
    re.IGNORECASE)
_FRACTIONS = {"½": 0.5, "¼": 0.25, "¾": 0.75, "⅓": 1 / 3, "⅔": 2 / 3, "⅛": 0.125}
_FORMS = {"canned", "frozen", "dried", "ground", "smoked", "pickled", "powdered"}
# What the COOK does, kept out of the name as the live parser keeps it out
# ("1 cup chopped cilantro" buys cilantro). Purchase descriptors — boneless,
# skinless, whole, unsalted, smoked, low-sodium — stay in the name, as the
# live parser keeps them: they decide which product to buy. "crushed" is not
# here: Crushed Tomatoes and Crushed Red Pepper Flakes are products.
_PREP = {"chopped", "sliced", "minced", "diced", "shredded", "grated", "mashed",
         "peeled", "cubed", "julienned", "finely", "thinly", "roughly", "freshly",
         "fine"}
# "5 garlic cloves", "2 thyme sprigs": the count word comes after the name
_TRAILING_COUNT = {"clove", "cloves", "sprig", "sprigs", "stalk", "stalks"}
_BULLETS = "-*• "


@dataclass(frozen=True)
class ParsedLine:
    """One ingredient line as the planner reads it. `name` is the product to
    buy without its purchase form; `form` is that form ("canned", "ground")
    or None; `prep` is what the cook does plus the text after the line's
    first comma, or None."""
    name: str
    form: str | None
    quantity: float | None
    unit: str | None
    prep: str | None

    @property
    def doc_name(self) -> str:
        """The name a shopper reviews: the form written back in front ("ground
        beef", "frozen peas"), except a can's, which the unit already says
        ("1 can crushed tomatoes" reads crushed tomatoes, unit can)."""
        if self.form and not (self.form == "canned" and self.unit == "can"):
            return f"{self.form} {self.name}"
        return self.name


def _amount(text: str) -> float:
    """"1 1/2" -> 1.5, "1½" -> 1.5, "3/4" -> 0.75, "2" -> 2.0."""
    total = 0.0
    for part in re.findall(r"\d+/\d+|\d+(?:\.\d+)?|[½¼¾⅓⅔⅛]", text):
        if part in _FRACTIONS:
            total += _FRACTIONS[part]
        elif "/" in part:
            num, den = part.split("/")
            total += int(num) / int(den) if int(den) else 0.0
        else:
            total += float(part)
    return round(total, 4)


def _strip_notes(item: str) -> tuple[str, str]:
    """(item without parenthetical notes, the text after its first comma):
    "garlic cloves (, thinly sliced)" -> ("garlic cloves", "");
    "onion, finely chopped" -> ("onion", "finely chopped")."""
    prev = None
    while prev != item:                          # innermost groups first
        prev, item = item, re.sub(r"\([^()]*\)", " ", item)
    item = item.replace("(", " ").replace(")", " ")
    head, _, rest = item.partition(",")
    return " ".join(head.split()), " ".join(rest.split())


def parse_line(line: str) -> ParsedLine:
    """Read one ingredient line, with or without its list bullet.

    Consistent with the live parser on what to buy: parenthetical notes and
    the text after a comma are dropped from the name, prep words go to
    `prep`, a leading purchase form goes to `form`, and purchase descriptors
    stay in the name. A line with no amount has quantity and unit None."""
    raw = line.strip().lstrip(_BULLETS).strip()
    item, after_comma = _strip_notes(raw)
    qty = unit = None
    if qm := _QTY.match(item):
        qty = _amount(qm.group(1))
        unit = (qm.group(2) or "each").lower().rstrip(".")
        item = item[qm.end():]
        if not item.strip() and qm.group(2):     # "2 cloves": the unit IS the item
            item, unit = qm.group(2), "each"
    words = [w for w in item.lower().split() if w != "of"]
    prep = [w for w in words if w in _PREP] + ([after_comma] if after_comma else [])
    words = [w for w in words if w not in _PREP]
    if unit == "each" and len(words) > 1 and words[-1] in _TRAILING_COUNT:
        unit = words.pop()
    form = None
    if words and words[0] in _FORMS:
        form = words.pop(0)
    if unit and unit.startswith("can"):
        form, unit = "canned", "can"
    name = " ".join(words).strip() or item.strip() or raw
    return ParsedLine(name=name, form=form, quantity=qty, unit=unit,
                      prep=" ".join(prep) or None)


def parse_servings(text: str) -> int | None:
    """"Serves 4" anywhere in the text -> 4; None when it says nothing.
    This is the demo parser's rule, kept as narrow as it was."""
    m = re.search(r"serves\s+(\d+)", text, re.IGNORECASE)
    return int(m.group(1)) if m else None


_YIELD = (re.compile(r"serves\s+(\d+)", re.IGNORECASE),
          re.compile(r"(\d+)\s+(?:servings?|portions?|people)\b", re.IGNORECASE),
          re.compile(r"^\s*(\d+)\s*$"))


def servings_from_yield(text: str) -> int | None:
    """A recipe's stated yield as a serving count: "Serves 4", "4 servings",
    "Makes 6 portions", or a bare "4" (what schema.org recipeYield often
    holds). None when nothing says how many it serves; never a guessed 1."""
    for pattern in _YIELD:
        if m := pattern.search(text or ""):
            n = int(m.group(1))
            return n if n > 0 else None
    return None
