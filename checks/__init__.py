"""
Check modules package.

Each module implements one category of the vector-quality testing framework.
All checks are written against the VectorDBConnector interface, making
them database-agnostic.

Modules:
  - vector_integrity: dimension consistency and zero/NaN/Inf checks
  - neighbor_consistency: adjacent-chunk consistency checks
  - duplicate_detection: Category 3 — cosine-neighbor candidates for duplicate review
  - metadata_completeness: Category 4 — required field and value distribution checks
  - auto_pairs: Generate metadata-based candidate pairs (Phase 1)
  - llm_review: LLM-assisted pair review (Phase 2)
  - pair_runner: Run neighbor checks from reviewed Excel (Phase 3)
"""
