"""A stand-in for EmbeddingGemma: tests never reach Ollama.

FakeEmbedder answers from a table of text -> vector built from the committed seeds (a product
type's centroid, so a name maps where the test says), and any other text with a vector far
from every product type (the negated mean of the centroids), which the matcher reads as not
stocked. It records what it was asked, so a test can tell the disk cache from a request.
DownEmbedder is an embedder that cannot be reached.
"""
from __future__ import annotations

import math

from pantry_planner.matching import embed, seeds


def unit(v: list[float]) -> list[float]:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v]


def far_vector() -> list[float]:
    cents = list(seeds.load().centroids.values())
    return unit([-sum(col) / len(cents) for col in zip(*cents, strict=True)])


class FakeEmbedder:
    name = "fake-embedder"

    def __init__(self, table: dict[str, list[float]] | None = None):
        self.table = {k.lower(): v for k, v in (table or {}).items()}
        self.asked: list[tuple[str, str]] = []

    @classmethod
    def mapping(cls, **names_to_types: str) -> FakeEmbedder:
        """FakeEmbedder.mapping(**{"unsalted butter": "butter"}): each name embeds as the
        centroid of the product type it names."""
        cents = seeds.load().centroids
        return cls({name: list(cents[t]) for name, t in names_to_types.items()})

    def embed(self, texts, kind, titles=None):
        self.asked += [(t, kind) for t in texts]
        return [list(self.table.get(t.lower(), far_vector())) for t in texts]

    def ping(self):
        return True, ""


class DownEmbedder:
    name = "down-embedder"

    def __init__(self):
        self.asked = 0

    def embed(self, texts, kind, titles=None):
        self.asked += 1
        raise embed.EmbedderUnavailableError("down-embedder is unreachable (ConnectError: test)")

    def ping(self):
        return False, "down-embedder is unreachable (ConnectError: test)"
