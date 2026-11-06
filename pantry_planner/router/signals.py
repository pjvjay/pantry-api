"""Escalation signals the code already has: where the selector's pick sits in the matcher's
order (flow's match_order: line_no -> [(product_id, group)], for lines matched by meaning).

A model's own confidence is a weak signal: confidence a model states in words runs high and
predicts its failures little better than chance (Xiong et al., "Can LLMs Express Their
Uncertainty?", ICLR 2024). The matcher's groups are not the model's opinion, so a pick that
leaves them is worth a second look whatever confidence came with it:

- off_list: the pick is not among the line's candidates at all;
- left_same_item: the line has a Same item candidate and the pick is not one of them.

A line matched by words, or with no order, gives no signal.
"""
from __future__ import annotations

from ..models import SelectorResult


def matcher_flags(result: SelectorResult,
                  order: dict[int, list[tuple[int, str]]] | None) -> dict[int, str]:
    """line_no -> the signal (off_list, left_same_item) for each selection that has one."""
    flags: dict[int, str] = {}
    for s in result.selections:
        ranked = (order or {}).get(s.line_no)
        if not ranked:
            continue
        groups = {pid: group for pid, group in ranked}
        if s.product_id not in groups:
            flags[s.line_no] = "off_list"
        elif groups[s.product_id] != "same_item" and "same_item" in groups.values():
            flags[s.line_no] = "left_same_item"
    return flags
