import hashlib
import json
from pathlib import Path
import streamlit as st
from passage_search.core import parse_document, SearchIndex

st.set_page_config(page_title='Passage · Find the evidence', page_icon='🔎', layout='wide')
st.markdown('''<style>
.stApp {background: #f5f4f0;} h1 {letter-spacing:-2px;}
[data-testid="stSidebar"] {background:#e8ece7;}
[data-testid="stMetric"] {background:white;padding:18px;border-radius:12px;}
</style>''', unsafe_allow_html=True)
MODEL = 'sentence-transformers/all-MiniLM-L6-v2'

@st.cache_resource
def load_model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL)

st.caption('PASSAGE / YOUR DOCUMENTS, WITH EVIDENCE')
st.title('Find the passage you need.')
st.write('Search by meaning, exact terms, or both. Every match leads back to its source.')
with st.sidebar:
    st.header('Your library')
    uploads = st.file_uploader('Add documents', type=['pdf', 'txt', 'md'], accept_multiple_files=True)
    demo = st.toggle('Use sample library', value=True)
    st.caption('Up to 20 files, 10 MB each, and 300 pages per PDF. Scanned PDFs require OCR.')
    st.divider()
    mode = st.radio('Search approach', ['Hybrid', 'Semantic', 'Keyword'])
    st.caption('Hybrid combines meaning and BM25 keyword rankings. Keyword works without a model download.')
    top_k = st.slider('Number of matches', 1, 10, 5)
    st.caption('Documents stay in this running app session. The model downloads once from Hugging Face; search runs locally afterward. For private documents, run on your computer.')

files = []
if demo:
    files.extend((p.name, p.read_bytes()) for p in sorted((Path(__file__).parent / 'sample_docs').glob('*.md')))
if uploads:
    if len(uploads) > 20:
        st.error('Choose at most 20 files.'); st.stop()
    for f in uploads:
        if f.size > 10 * 1024 * 1024:
            st.error(f'{f.name} exceeds 10 MB.'); st.stop()
        files.append((f.name, f.getvalue()))
if not files:
    st.info('Upload documents or turn on the sample library to begin.'); st.stop()
if len({name for name, _ in files}) != len(files):
    st.error('Each document needs a unique filename. Rename duplicates before uploading.'); st.stop()
key = hashlib.sha256(b''.join(n.encode() + hashlib.sha256(d).digest() for n, d in files)).hexdigest()
if st.session_state.get('library_key') != key:
    passages, errors = [], []
    for name, data in files:
        try:
            passages.extend(parse_document(name, data))
        except Exception as exc:
            errors.append(f'{name}: {exc}')
    st.session_state.update(library_key=key, passages=passages, errors=errors, index=None)
for error in st.session_state.errors:
    st.warning(error)
passages = st.session_state.passages
if not passages:
    st.stop()
if len(passages) > 3000:
    st.error('This library is too large. Please reduce it to 3,000 passages.'); st.stop()
a,b,c = st.columns(3)
a.metric('Documents', len({p.source for p in passages}))
b.metric('Searchable passages', len(passages))
c.metric('Retrieval', mode)
source = st.selectbox('Search within', ['All documents'] + sorted({p.source for p in passages}))
with st.form('search'):
    query = st.text_input('What are you looking for?', placeholder='How can I recover access if I lose my phone?')
    submitted = st.form_submit_button('Find evidence', type='primary')
if submitted:
    if not query.strip():
        st.warning('Enter a question or a few search terms.'); st.stop()
    try:
        index = st.session_state.index
        if index is None or (mode != 'Keyword' and index.model is None):
            with st.spinner('Preparing your library… First semantic search downloads the model.'):
                index = SearchIndex(passages, None if mode == 'Keyword' else load_model())
                st.session_state.index = index
        selected = None if source == 'All documents' else source
        results = index.search(query, mode, top_k, selected)
        st.subheader('Evidence')
        st.caption('Rankings indicate relevance, not factual correctness or confidence. Semantic search can return weak matches; check the quoted text.')
        if not results:
            st.info('No keyword matches. Try different terms or semantic search.')
        for rank, hit in enumerate(results, 1):
            with st.container(border=True):
                st.markdown(f'**{rank}. {hit["source"]}**')
                st.caption(f'Page {hit["page"]} · Passage {hit["number"]} · Ranking score {hit["score"]:.4f}')
                st.text(hit['text'])
        if mode != 'Keyword':
            with st.expander('Compare with keyword search'):
                baseline = index.search(query, 'Keyword', top_k, selected)
                if not baseline:
                    st.write('No exact keyword matches. The semantic results above may still find related ideas.')
                for hit in baseline:
                    st.write(f'{hit["source"]} · Page {hit["page"]}')
                    st.text(hit['text'])
        st.download_button('Export evidence as JSON', json.dumps({'query':query, 'mode':mode, 'results':results}, indent=2), 'evidence.json', 'application/json')
    except Exception as exc:
        st.error(f'Search could not finish: {exc}')
        st.info('If model download is unavailable, switch to Keyword and search again.')
