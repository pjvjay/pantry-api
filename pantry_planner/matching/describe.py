"""What an ingredient name no seed knows IS, from a low-cost model, before it is matched.

A bare name ("cooking onion", "caster sugar", "jeera", "nam pla") is ambiguous, regional or
unusual; embedding it alone leaves the matcher to guess from a few words. The describer asks a
model (DESCRIBER_MODEL, a model spec such as "gemini:gemini-3.1-flash-lite") to identify each
name the way the catalog files its products, in fields that say only what identifies an item:

  product_type   the catalog type that IS the item, chosen from the catalog's own type list;
                 when none is the item itself, the item's own plain name ("dill", "lamb
                 shoulder"), so it can never be read as a catalog type
  form           one of the forms the catalog's products have ("fresh", "ground", "canned"…)
  variant        the variant the name asks for ("salted", "extra virgin"), or ""
  look_alikes    catalog types a shopper might confuse it with: kept for review only, never a
                 rule (the matcher does not read it)

Never flavour or culinary use: prose like "aromatic base for savoury dishes" fits onion, garlic
and shallots alike and pulls them together. The matcher (matcher.py) takes product_type as the
Same item when the catalog has it, compares form and variant as fields, and embeds the identity
string (seeds.identity_text) for what works instead.

Answers are cached by the describer's version, the model, the catalog's type list and the
normalized name, so a name is described once; a failure is not cached. A model that cannot be
reached raises DescriberUnavailableError, and the matcher falls back to embedding the bare name.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
import unicodedata
from dataclasses import dataclass

from . import seeds

PROMPT_VERSION = 1
BATCH = 30
MAX_TOKENS = 4096


class DescriberUnavailableError(RuntimeError):
    """The describer's model cannot be reached, or is turned off; the message says why."""


@dataclass(frozen=True)
class Identity:
    name: str
    product_type: str
    form: str
    variant: str
    look_alikes: tuple[str, ...]

    @property
    def text(self) -> str:
        return seeds.identity_text(self.product_type, self.form, self.variant)


TOOL = {
    "name": "submit_identities",
    "description": "One identity per ingredient name given, in the same order.",
    "input_schema": {
        "type": "object", "required": ["items"],
        "properties": {"items": {"type": "array", "items": {
            "type": "object",
            "required": ["name", "product_type", "form", "variant", "look_alikes"],
            "properties": {
                "name": {"type": "string"},
                "product_type": {"type": "string"},
                "form": {"type": "string"},
                "variant": {"type": "string"},
                "look_alikes": {"type": "array", "items": {"type": "string"}},
            }}}},
    },
}


def system_prompt(types: list[str], forms: list[str]) -> str:
    return (
        "You identify recipe ingredient names precisely, as a grocery catalog would file them. "
        "For each name return only what identifies the item, never flavour or how it is used:\n"
        "- product_type: the catalog type that IS this item, chosen from the list below, exactly "
        "as written; when none is the item itself (only similar ones), give the item's own plain "
        "name in lower case.\n"
        f"- form: one of {json.dumps(forms)}, or \"\" when the name implies none.\n"
        "- variant: the specific variant the name asks for (\"salted\", \"extra virgin\", "
        "\"boneless skinless\", \"low sodium\"), or \"\".\n"
        "- look_alikes: up to 3 catalog types a shopper might confuse with it but that are a "
        "different item; [] when none.\n"
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
        self._memo: dict[str, Identity] = {}
        self._lock = threading.Lock()
        self.calls = 0
        self.tokens_in = 0
        self.tokens_out = 0

    def key(self, name: str, fingerprint: str) -> str:
        raw = f"{PROMPT_VERSION}|{self.model}|{fingerprint}|{normalize(name)}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def describe(self, names: list[str]) -> dict[str, Identity]:
        """name -> Identity for every name (one model call per BATCH names not cached)."""
        from .. import llm

        types, forms, fp = _catalog()
        with self._lock:
            todo = list(dict.fromkeys(n for n in names if self.key(n, fp) not in self._memo))
        for i in range(0, len(todo), BATCH):
            chunk = todo[i:i + BATCH]
            try:
                result = llm.forced_tool_call(
                    model=self.model, system=system_prompt(types, forms),
                    messages=[{"role": "user", "content": json.dumps({"names": chunk})}],
                    tool=TOOL, max_tokens=MAX_TOKENS)
            except llm.LLMError as e:
                raise DescriberUnavailableError(f"{self.model} is unavailable: {e}") from e
            self.calls += 1
            self.tokens_in += result.input_tokens
            self.tokens_out += result.output_tokens
            got = {normalize(it.get("name", "")): it for it in result.input.get("items", [])
                   if isinstance(it, dict)}
            with self._lock:
                for n in chunk:
                    it = got.get(normalize(n))
                    if not it or not str(it.get("product_type", "")).strip():
                        continue            # not answered: asked again next time
                    self._memo[self.key(n, fp)] = Identity(
                        name=n, product_type=str(it["product_type"]).strip().lower(),
                        form=str(it.get("form", "")).strip().lower(),
                        variant=str(it.get("variant", "")).strip().lower(),
                        look_alikes=tuple(str(x).strip().lower() for x in it.get("look_alikes", [])
                                          if str(x).strip())[:3])
        with self._lock:
            return {n: self._memo[self.key(n, fp)] for n in names
                    if self.key(n, fp) in self._memo}


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
