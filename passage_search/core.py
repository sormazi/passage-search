from dataclasses import dataclass, asdict
from io import BytesIO
from pathlib import Path
import math
import re
from collections import Counter


@dataclass(frozen=True)
class Passage:
    source: str
    page: int
    text: str
    number: int


def chunk_text(text, size=140, overlap=30):
    if size <= 0 or not 0 <= overlap < size:
        raise ValueError('Overlap must be smaller than a positive chunk size.')
    words = text.split()
    for start in range(0, len(words), size - overlap):
        yield ' '.join(words[start:start + size])
        if start + size >= len(words):
            break


def parse_document(name, data):
    suffix = Path(name).suffix.lower()
    if suffix == '.pdf':
        from pypdf import PdfReader
        reader = PdfReader(BytesIO(data))
        if reader.is_encrypted:
            raise ValueError('Password-protected PDFs are not supported.')
        if len(reader.pages) > 300:
            raise ValueError('Please upload a PDF with at most 300 pages.')
        pages = [p.extract_text() or '' for p in reader.pages]
    elif suffix in {'.txt', '.md'}:
        pages = [data.decode('utf-8-sig')]
    else:
        raise ValueError('Choose a PDF, TXT, or Markdown file.')
    passages = []
    for page, text in enumerate(pages, 1):
        for chunk in chunk_text(text):
            passages.append(Passage(name, page, chunk, len(passages) + 1))
    if not passages:
        raise ValueError('No readable text found. Scanned PDFs need OCR first.')
    return passages


def tokens(text):
    return re.findall(r"\b\w+\b", text.lower())


class KeywordIndex:
    """BM25 baseline; scores are ranking signals, not probabilities."""
    def __init__(self, passages):
        self.passages = passages
        self.counts = [Counter(tokens(p.text)) for p in passages]
        self.lengths = [sum(c.values()) for c in self.counts]
        self.average = sum(self.lengths) / max(len(passages), 1) or 1
        self.df = Counter(t for c in self.counts for t in c)

    def scores(self, query):
        result = []
        for c, length in zip(self.counts, self.lengths):
            score = 0.0
            for term in set(tokens(query)):
                freq = c[term]
                if freq:
                    idf = math.log(1 + (len(self.counts) - self.df[term] + .5) / (self.df[term] + .5))
                    score += idf * freq * 2.5 / (freq + 1.5 * (.25 + .75 * length / self.average))
            result.append(score)
        return result


class SearchIndex:
    def __init__(self, passages, model=None):
        self.passages = passages
        self.keyword = KeywordIndex(passages)
        self.model = model
        self.embeddings = None
        if model is not None:
            self.embeddings = model.encode_document([p.text for p in passages], normalize_embeddings=True)

    def search(self, query, mode='Hybrid', limit=5, source=None):
        if not query.strip():
            return []
        if mode not in {'Keyword', 'Semantic', 'Hybrid'}:
            raise ValueError('Unknown search mode.')
        lexical = self.keyword.scores(query)
        eligible = [i for i, p in enumerate(self.passages) if source is None or p.source == source]
        semantic = None
        if mode != 'Keyword':
            if self.model is None:
                raise ValueError('Load the semantic model first.')
            import numpy as np
            q = self.model.encode_query([query], normalize_embeddings=True)[0]
            semantic = np.asarray(self.embeddings) @ q
        if mode == 'Keyword':
            scores = lexical
            eligible = [i for i in eligible if lexical[i] > 0]
        elif mode == 'Semantic':
            scores = semantic
        else:
            # Reciprocal rank fusion avoids comparing incompatible raw scores.
            scores = [0.0] * len(self.passages)
            for values, indices in [(lexical, [i for i in eligible if lexical[i] > 0]), (semantic, eligible)]:
                ranked = sorted(indices, key=lambda i: (-float(values[i]), i))
                for rank, i in enumerate(ranked, 1):
                    scores[i] += 1 / (60 + rank)
        ordered = sorted(eligible, key=lambda i: (-float(scores[i]), i))[:max(0, limit)]
        return [{**asdict(self.passages[i]), 'score': round(float(scores[i]), 5)} for i in ordered]
