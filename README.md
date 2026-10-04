# Passage — document search with evidence

Find relevant passages in PDFs, Markdown, and text files. Search by meaning, keywords, or a combination, and trace every result to its filename and page.

## Quick start

Python 3.10 or later (tested with 3.12).

```bash
python -m venv .venv
source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

The app opens in your browser. The bundled sample library lets you start immediately. Try **“How can I recover access if I lose my phone?”**, then upload your own documents. Turn off the sample library to search only your files.

Semantic and hybrid search download `sentence-transformers/all-MiniLM-L6-v2` on first use. Internet is required for that initial download; inference then runs locally. Keyword mode needs no model. No paid API key is required.

## My spin on the original

The [Sentence Transformers semantic search example](https://github.com/huggingface/sentence-transformers/blob/main/examples/sentence_transformer/applications/semantic-search/semantic_search.py) demonstrates embedding a short sentence corpus and ranking it by similarity. This project implements an original document workflow around that approach:

- PDF/TXT/Markdown ingestion and overlapping, page-aware passages.
- BM25 keyword baseline and hybrid reciprocal rank fusion.
- Source filters, quoted evidence, and page references.
- Side-by-side keyword comparison and downloadable evidence JSON.
- A synthetic benchmark, retrieval tests, and GitHub Actions checks.

This is a new application using Sentence Transformers as a dependency; it is not a fork of the full upstream library, and it does not claim to train a new model. No upstream source files are copied. See [ATTRIBUTION.md](ATTRIBUTION.md).

## How retrieval works

Text is split into 140-word passages with 30-word overlap, without crossing page boundaries. MiniLM embeds each passage; normalized dot products give cosine similarity. BM25 ranks exact terms. Hybrid search adds `1 / (60 + rank)` from each ranking. The document filter applies before fusion. Raw scores across modes are not comparable and are not confidence probabilities.

## Evaluation

```bash
pip install -r requirements-dev.txt
python -m pytest -q
python evaluate.py --semantic
```

The benchmark contains six authored questions over three synthetic documents. It reports document-level recall@3, MRR@3, and top-1 accuracy. This is a reproducible smoke benchmark, not a general accuracy claim. For a serious portfolio extension, add at least 50 labeled questions from a real, openly licensed corpus, separate development and held-out sets, and compare retrieval quality and latency.

## Privacy and limits

Run locally for private files. Upload contents and embeddings are kept in Streamlit session memory and are not deliberately written to disk or sent to an external inference service. The shared model resource is cached across sessions; document indexes are session-specific. Hugging Face receives model-download requests. A hosted deployment sends uploads to its host, so it has a different privacy boundary.

Supports up to 20 uploaded files, 10 MB per file, 300 pages per PDF, and 3,000 passages total. This is a small-library demo, not a hardened multi-user document platform. Scanned PDFs need OCR; encrypted PDFs are rejected. PDF extraction can alter reading order. English is the primary target language. The app retrieves passages rather than generating answers, and weak semantic matches can still appear.


