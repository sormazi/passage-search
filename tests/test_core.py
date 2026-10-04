import numpy as np
import pytest
from io import BytesIO
from pypdf import PdfWriter
from passage_search.core import Passage, SearchIndex, chunk_text, parse_document


def test_chunk_coverage_and_overlap():
    words = [str(i) for i in range(311)]
    chunks = list(chunk_text(' '.join(words)))
    assert chunks[0].split()[-30:] == chunks[1].split()[:30]
    assert set(' '.join(chunks).split()) == set(words)
    assert all(len(c.split()) <= 140 for c in chunks)
    with pytest.raises(ValueError):
        list(chunk_text('text', 10, 10))


def test_text_and_unsupported_input():
    assert parse_document('notes.md', b'\xef\xbb\xbfhello')[0].text == 'hello'
    for name, data in [('empty.txt', b''), ('image.png', b'bytes'), ('bad.txt', b'\xff')]:
        with pytest.raises((ValueError, UnicodeDecodeError)):
            parse_document(name, data)


def test_blank_pdf_and_encrypted_pdf():
    writer = PdfWriter()
    writer.add_blank_page(width=100, height=100)
    buf = BytesIO(); writer.write(buf)
    with pytest.raises(ValueError, match='No readable text'):
        parse_document('blank.pdf', buf.getvalue())
    writer.encrypt('secret')
    buf = BytesIO(); writer.write(buf)
    with pytest.raises(ValueError, match='Password-protected'):
        parse_document('locked.pdf', buf.getvalue())


def test_keyword_filter_and_no_matches():
    passages = [Passage('a', 2, 'backup code recovery', 1), Passage('b', 1, 'travel expense receipts', 1)]
    index = SearchIndex(passages)
    assert index.search('backup', 'Keyword')[0]['page'] == 2
    assert index.search('backup', 'Keyword', source='b') == []
    assert index.search('zzzz', 'Keyword') == []
    assert index.search('  ', 'Keyword') == []
    with pytest.raises(ValueError, match='Load'):
        index.search('backup', 'Semantic')


class FakeModel:
    def encode_document(self, texts, **kwargs):
        return np.array([[1., 0.], [0., 1.]])
    def encode_query(self, queries, **kwargs):
        return np.array([[1., 0.]])


def test_semantic_and_fusion_filtered_rankings():
    index = SearchIndex([Passage('a', 1, 'recovery', 1), Passage('b', 2, 'expense', 1)], FakeModel())
    assert index.search('forgot login', 'Semantic')[0]['source'] == 'a'
    assert index.search('recovery', 'Hybrid')[0]['source'] == 'a'
    assert index.search('recovery', 'Hybrid', source='b')[0]['source'] == 'b'
    assert index.search('recovery', 'Hybrid', limit=0) == []
