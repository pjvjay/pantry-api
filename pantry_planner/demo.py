"""
CLI entrypoint — run one recipe end-to-end from the terminal.

Usage:
    python -m pantry_planner.demo <recipe_slug>
"""
from __future__ import annotations

import json
import sys

from . import flow


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python -m pantry_planner.demo <recipe_slug>", file=sys.stderr)
        print("Example: python -m pantry_planner.demo pbj_sandwich", file=sys.stderr)
        return 1

    slug = sys.argv[1]
    plan = flow.run(slug)
    print(json.dumps(plan.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
