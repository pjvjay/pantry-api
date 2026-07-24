"""Natural-language recipe → SQL retrieval (constrained NL2SQL).

Pipeline (NL2SQL_Handbook taxonomy):
  PRE-PROCESSING   units.py (normalization), vocab.py (schema linking)
  TRANSLATION      query_parser.py (multi-shot semantic parse)
                   + sql_builder.py (pattern-menu compilation)
  POST-PROCESSING  query_parser.validate_parsed (clamping),
                   retrieve.py (execution-guided widening)
"""
from .retrieve import RetrievalResult, UnparseableRecipe, retrieve  # noqa: F401
