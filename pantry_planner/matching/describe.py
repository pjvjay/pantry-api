"""What an ingredient name no seed knows IS, from a low-cost model, before it is matched.

A bare name ("cooking onion", "caster sugar", "jeera", "nam pla") is ambiguous, regional or
unusual; embedding it alone leaves the matcher to guess from a few words. The describer asks a
model (DESCRIBER_MODEL, a model spec such as "gemini:gemini-3.1-flash-lite") to identify each
name the way the catalog files its products, in fields that say only what identifies an item:

  item           the item's own plain name ("thyme", "brown rice", "lamb shoulder")
  catalog_type   the nearest catalog type, from the catalog's own type list, and
  relation       what the item is to it: same, variety (the same food in another colour, size,
                 cut, grade, brand or packaging), substitute (a different food that could stand
                 in) or none. product_type is the catalog type when the relation is same or
                 variety, else the item's own name (so it can never be read as a catalog type);
                 a stand-in is kept as `substitute`, offered as working instead
  form           one of the forms the catalog's products have ("fresh", "ground", "canned"…)
  variant        the variant the name asks for ("salted", "extra virgin"), or ""

Never flavour or culinary use: prose like "aromatic base for savoury dishes" fits onion, garlic
and shallots alike and pulls them together. No list of look-alikes or confusable pairs either:
the prompt asks for the item's own name before the catalog choice, and the matcher checks the
filing by meaning. The matcher (matcher.py) takes product_type as the Same item when the
catalog has it, compares form and variant as fields, and embeds the identity string of the
item's own words (seeds.identity_text) to check that filing against the type's centroid and
for what works instead.

Answers are cached (matching/cache.py: the database, with an in-process memo) by the
describer's version, the model, the catalog's type list and the normalized name, so a name is
described once; a failure is not cached. A model that cannot be reached, or a day's allowance
of calls that is spent, raises DescriberUnavailableError, and the matcher falls back to
embedding the bare name.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from dataclasses import dataclass

from . import cache, seeds

PROMPT_VERSION = 2
BATCH = 30
MAX_TOKENS = 4096
# The model call's settings, measured on 2026-11-09 with gemini-3.1-flash-lite over 30 names,
# three runs each: at the model's default temperature the answers swung between runs (7, 6 and
# 0 of 10 unstocked names filed under a neighbouring catalog type); at temperature 0 with the
# configured reasoning effort ("low") the same answer every run but only 3 of 20 stocked names
# filed under their type; at temperature 0 without a reasoning effort the same answer every run,
# 17 of 20 stocked names right, 1 of 10 unstocked filed. So the describer fixes both, whatever
# GEMINI_REASONING_EFFORT says for the planner's calls.
TEMPERATURE = 0.0
REASONING_EFFORT = ""


class DescriberUnavailableError(RuntimeError):
    """The describer's model cannot be reached, or is turned off; the message says why."""


@dataclass(frozen=True)
class Identity:
    """What a name is: `item`, its own plain name ("thyme", "brown rice"); `product_type`, the
    catalog type that is the same food when the describer found one, else the item's own name
    (so it can never be read as a catalog type); the form and variant the name asks for. `text`
    is the identity string that is embedded (seeds.identity_text), built from the item's own
    words, not the catalog type, so a filing can be checked against that type's centroid."""
    name: str
    item: str
    product_type: str
    form: str
    variant: str
    substitute: str = ""        # the catalog type the describer called a stand-in, or ""

    @property
    def text(self) -> str:
        return seeds.identity_text(self.item, self.form, self.variant)


TOOL = {
    "name": "submit_identities",
    "description": "One identity per ingredient name given, in the same order.",
    "input_schema": {
        "type": "object", "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "required": ["name", "item", "catalog_type", "relation", "form", "variant"],
            "properties": {
                "name": {"type": "string"},
                "item": {"type": "string"},
                "catalog_type": {"type": "string"},
                "relation": {"type": "string",
                             "description": "same, variety, substitute or none"},
                "form": {"type": "string"},
                "variant": {"type": "string"},
            }}}},
    },
}


def system_prompt(types: list[str], forms: list[str]) -> str:
    """Prompt version 2: the item's own name first; then the nearest catalog type and how the
    item relates to it, with "substitute" as an answer of its own. Forced to choose between
    filing a name under a type and nothing, the model filed brown rice under white rice and
    pecorino under parmesan (5 of 10 unstocked names, 2026-11-09); given the stand-in answer,
    1 of 10, at a cost of 1 stocked name in 53 and 2 in 100. The examples are foods none of the
    evaluation sets hold."""
    return (
        "You identify recipe ingredient names precisely, as a grocery catalog would file them. "
        "For each name return only what identifies the item, never flavour or how it is used:\n"
        "- item: the item's own plain name in lower case, without form, size or variant words "
        "(\"thyme\", \"brown rice\", \"onion\").\n"
        "- catalog_type: the catalog type from the list below, exactly as written, that is "
        "nearest to this item, or \"\" when none is close.\n"
        "- relation: how the item relates to that catalog type: \"same\" (the very same "
        "product), \"variety\" (the same food in another colour, size, cut, grade, brand or "
        "packaging, e.g. medium eggs and eggs, small red onions and red onion), \"substitute\" "
        "(a different food or product that could stand in for it, e.g. sweet potato and russet "
        "potatoes, ground lamb and ground beef), or \"none\". A shopper who asked for the item "
        "and was handed the catalog type: would they say it is what they asked for (same or "
        "variety) or a stand-in (substitute)? When unsure, say substitute.\n"
        f"- form: one of {json.dumps(forms)}, or \"\" when the name implies none.\n"
        "- variant: the specific variant the name asks for (\"salted\", \"extra virgin\", "
        "\"boneless skinless\", \"low sodium\"), or \"\".\n"
        f"Catalog types: {json.dumps(types)}")


def normalize(name: str) -> str:
    """The cache key's name: Unicode-normalized, lower case, single spaces, no edge punctuation."""
    text = unicodedata.normalize("NFKC", name).lower()
    return re.sub(r"\s+", " ", text).strip(" \t,.;:-")


def _catalog() -> tuple[list[str], list[str], str]:
    s = seeds.load()
    types = sorted(s.types())
    forms = sorted({p.form for p in s.profiles.values() if p.form})
    fingerprint = hashlib.sha256(json.dumps([types, forms]).encode()).hexdigest()[:16]
    return types, forms, fingerprint


class Describer:
    """The model call and its in-process cache (keyed as the module docstring says)."""

    def __init__(self, model: str):
        self.model = model
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def key(self, name: str, fingerprint: str) -> str:
        return cache.identity_key(PROMPT_VERSION, self.model, fingerprint, normalize(name))

    @property
    def cached(self) -> int:
        return len(cache.identities.memo)

    def describe(self, names: list[str]) -> dict[str, Identity]:
        """name -> Identity for every name: from the cache (matching/cache.py), else one model
        call per BATCH names, within the day's allowance of describer calls."""
        from .. import llm

        types, forms, fp = _catalog()
        keys = {n: self.key(n, fp) for n in dict.fromkeys(names)}
        found = cache.identities.get_many(keys.values())
        todo = [n for n, k in keys.items() if k not in found]
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            with cache.identities.lock(keys[chunk[0]]):      # one flight per first name
                found.update(cache.identities.get_many(keys[n] for n in chunk))
                chunk = [n for n in chunk if keys[n] not in found]
                if not chunk:
                    continue
                try:
                    cache.budget().spend("describe_calls", 1)
                except cache.BudgetExceededError as e:
                    raise DescriberUnavailableError(str(e)) from e
                try:
                    result = llm.forced_tool_call(
                        model=self.model, system=system_prompt(types, forms),
                        messages=[{"role": "user", "content": json.dumps({"names": chunk})}],
                        tool=TOOL, max_tokens=MAX_TOKENS, gemini_temperature=TEMPERATURE,
                        gemini_reasoning_effort=REASONING_EFFORT)
                except llm.LLMError as e:
                    raise DescriberUnavailableError(f"{self.model} is unavailable: {e}") from e
                self.calls += 1
                self.tokens_in += result.input_tokens
                self.tokens_out += result.output_tokens
                got = {normalize(it.get("name", "")): it for it in result.input.get("items", [])
                       if isinstance(it, dict)}
                fresh: dict[str, dict] = {}
                for n in chunk:
                    it = got.get(normalize(n))
                    if not it or not str(it.get("item", "")).strip():
                        continue            # not answered: asked again next time
                    ident = _fields(n, it, types, forms)
                    fresh[keys[n]] = {
                        "name": n, "normalized": normalize(n),
                        "product_type": ident.product_type, "form": ident.form,
                        "variant": ident.variant,
                        "look_alikes": (ident.substitute,) if ident.substitute else (),
                        "identity": ident.text, "model": self.model,
                        "prompt_version": PROMPT_VERSION, "fingerprint": fp,
                        "status": "auto"}
                cache.identities.put_many(fresh)
                found.update(fresh)
        return {n: _from_row(n, v) for n in names if (v := found.get(keys[n])) is not None}


def _field(it: dict, key: str) -> str:
    """One answer field, cleaned: lower case, single spaces, and cut at the point where the
    model's function-call output now and then runs the next field into this one's value
    ("white fish fillets,form:" was seen once in 400 answers)."""
    value = re.split(r",\s*[a-z_]+:", str(it.get(key, "")))[0]
    return " ".join(value.split()).lower()


def _fields(name: str, it: dict, types: list[str], forms: list[str]) -> Identity:
    """The model's answer, validated: the catalog type counts only when it is a type exactly
    and the relation is same or variety (else the item's own name stands, and a stand-in is
    kept as `substitute`); form must be one of the catalog's forms (else ""); variant never
    repeats the form."""
    item, ctype, relation = _field(it, "item"), _field(it, "catalog_type"), _field(it, "relation")
    form, variant = _field(it, "form"), _field(it, "variant")
    known = ctype in types
    return Identity(name=name, item=item,
                    product_type=ctype if known and relation in ("same", "variety") else item,
                    form=form if form in forms else "",
                    variant="" if variant == form else variant,
                    substitute=ctype if known and relation == "substitute" else "")


def _from_row(name: str, v: dict) -> Identity:
    """A cached row: the item's own name is the first part of its identity string, and the
    stand-in the describer named is the look_alikes column's one entry."""
    return Identity(name=name, item=v["identity"].split(" | ", 1)[0],
                    product_type=v["product_type"], form=v["form"], variant=v["variant"],
                    substitute=next(iter(v["look_alikes"]), ""))


_describer: Describer | None = None
_installed = False
_lock = threading.Lock()


def get_describer() -> Describer | None:
    """The describer request time uses: one installed with set_describer, else the settings'
    DESCRIBER_MODEL (None when it is empty)."""
    global _describer, _installed
    with _lock:
        if not _installed:
            from ..config import settings

            model = settings().describer_model
            _describer = Describer(model) if model else None
            _installed = True
        return _describer


def set_describer(describer: Describer | None) -> None:
    """Install `describer` (None: names no seed knows are embedded as they are). Test seam."""
    global _describer, _installed
    with _lock:
        _describer, _installed = describer, True


def reset_describer() -> None:
    global _describer, _installed
    with _lock:
        _describer, _installed = None, False
