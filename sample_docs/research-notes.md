# Retrieval research notes
Semantic retrieval represents passages and queries as dense vectors. Similarity can reveal related ideas even when the wording differs. Keyword retrieval is useful for precise names, identifiers, and technical terms. Hybrid retrieval combines the rankings from both systems.

Evaluation should use queries with labeled relevant documents. Measure recall at k and reciprocal rank. Keep evaluation questions separate from development examples when tuning the system. A tiny synthetic sample is a smoke test, not evidence of performance on real documents.
