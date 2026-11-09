"""Fill the matching cache before shoppers import recipes.

Describes every name in seeds/common_ingredients.json (and any extra files given) with
DESCRIBER_MODEL and embeds each identity string with EMBEDDING_MODEL, into the database the
settings name (pantry-db 0010_matching_cache), through the same code request time uses
(matching/describe.py, matching/embed.py). A name already cached costs nothing, so the script
is safe to rerun and to stop; it stops by itself when the day's allowance of model requests
(MATCHING_DAILY_EMBED_TEXTS, MATCHING_DAILY_DESCRIBE_CALLS) is spent, and a rerun the next day
carries on. Gemini's free tier allows 1,000 embedding texts a day, so the whole list takes a
day or two.

  DB_URL=... GEMINI_API_KEY=... DESCRIBER_MODEL=gemini:gemini-3.1-flash-lite \\
  EMBEDDING_MODEL=gemini:gemini-embedding-001 python scripts/prefill_matching.py [names.json ...]

Prints what it described and embedded, what the cache already had, and the day's allowance
left. --dry-run lists the names that would be asked for, and asks for nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pantry_planner.matching import cache, describe, embed, seeds  # noqa: E402

SEEDS = Path(__file__).resolve().parents[1] / "seeds"
BATCH = describe.BATCH


def names_from(paths: list[Path]) -> list[str]:
    out: list[str] = []
    for p in paths:
        data = json.loads(p.read_text(encoding="utf-8"))
        out += data["names"] if isinstance(data, dict) else data
    known = {i.name.lower() for i in seeds.load().ingredients.values()}
    return [n for n in dict.fromkeys(n.strip() for n in out if n.strip()) if n.lower() not in known]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("files", nargs="*", type=Path, default=[SEEDS / "common_ingredients.json"])
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)
    names = names_from(args.files or [SEEDS / "common_ingredients.json"])
    if args.dry_run:
        print("\n".join(names))
        print(f"{len(names)} names", file=sys.stderr)
        return 0
    d = describe.get_describer()
    e = embed.get_embedder()
    if d is None or e is None:
        sys.exit("DESCRIBER_MODEL and EMBEDDING_MODEL (gemini:...) must both be set")
    e.max_wait_s = 600.0    # a script waits for the minute's allowance; the app does not

    described = embedded = 0
    stopped = ""
    for i in range(0, len(names), BATCH):
        chunk = names[i:i + BATCH]
        try:
            before = d.calls
            idents = d.describe(chunk)
            described += d.calls - before
            texts = sorted({x.text for x in idents.values()})
            cached = cache.embeddings.get_many(cache.embedding_key(
                getattr(e, "model", ""), embed.GEMINI_DIMS, t) for t in texts)
            todo = [t for t in texts if cache.embedding_key(
                getattr(e, "model", ""), embed.GEMINI_DIMS, t) not in cached]
            if todo:
                e.embed(todo)
                embedded += len(todo)
        except (describe.DescriberUnavailableError, embed.EmbedderUnavailableError) as exc:
            stopped = str(exc)
            break
        print(f"{min(i + BATCH, len(names))}/{len(names)} names; identities asked this run: "
              f"{described} calls, texts embedded: {embedded}", flush=True)
    st = cache.state()
    print(json.dumps({"names": len(names), "describe_calls_this_run": described,
                      "texts_embedded_this_run": embedded, "cache": st["identities"],
                      "embeddings": st["embeddings"], "budget": st["budget"],
                      "stopped": stopped or None}, indent=1))
    return 0 if not stopped else 2


if __name__ == "__main__":
    raise SystemExit(main())
