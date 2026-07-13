# Router Comparison — cascade vs three-phase

Head-to-head over the same golden set. Both strategies ran through
the full pipeline; per-recipe stats below, aggregates at the bottom.

## Per-recipe results

| Recipe | Cascade model(s) | Cascade $ | Cascade time | Cascade acc | 3-phase model | 3-phase $ | 3-phase time | 3-phase acc |
|---|---|---|---|---|---|---|---|---|
| Peanut Butter & Jelly Sandwich | `haiku-4-5-20251001 → +sonnet` | $0.0169 | 6.5s | 100% | `haiku-4-5-20251001` | $0.0086 | 7.8s | 100% |
| Grilled Cheese Sandwich | `haiku-4-5-20251001` | $0.0047 | 2.4s | 100% | `haiku-4-5-20251001` | $0.0075 | 5.0s | 100% |
| Spaghetti Bolognese | `haiku-4-5-20251001` | $0.0060 | 3.1s | 100% | `haiku-4-5-20251001` | $0.0095 | 7.3s | 100% |
| Simple Chicken Curry | `haiku-4-5-20251001` | $0.0066 | 3.8s | 100% | `haiku-4-5-20251001` | $0.0109 | 8.9s | 100% |
| Vegetable Stir Fry | `haiku-4-5-20251001` | $0.0053 | 2.8s | 100% | `haiku-4-5-20251001` | $0.0083 | 5.4s | 100% |
| **TOTAL** | — | **$0.0395** | **18.6s** | **100%** | — | **$0.0448** | **34.3s** | **100%** |

## Winner per axis

| Axis | Winner | Cascade | 3-phase | Delta |
|---|---|---|---|---|
| Cost      | 🟢 cascade | $0.0395 | $0.0448 | $0.0053 |
| Latency   | 🟢 cascade | 18.6s | 34.3s | 15.7s |
| Accuracy  | ≈ | 100% | 100% | 0.0% |

## Takeaway

**On this dataset, cascade wins.** Same or better accuracy (100% vs 100%) at 88% of the cost and 54% of the latency.

Three-phase would earn its keep on datasets with more variance — when Haiku fails often enough that the ~$0.001 classifier call prevents a $0.02 Sonnet escalation. On this golden set, Haiku handles most cases first-try, so cascade's opportunistic escalation is strictly cheaper than pre-classifying every request.

_Generated 2026-07-12 23:35:34 UTC_