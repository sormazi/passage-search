"""Small synthetic smoke benchmark. Run: python evaluate.py [--semantic]."""
import argparse
import json
from pathlib import Path
from passage_search.core import parse_document, SearchIndex

QUESTIONS = [
    ('How can I recover access if I lose my phone?', 'account-guide.md'),
    ('When must I submit travel expense receipts?', 'team-handbook.md'),
    ('How do we measure the quality of document retrieval?', 'research-notes.md'),
    ('backup codes', 'account-guide.md'),
    ('work remotely', 'team-handbook.md'),
    ('dense vectors', 'research-notes.md'),
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--semantic', action='store_true')
    args = parser.parse_args()
    passages = []
    for p in sorted((Path(__file__).parent / 'sample_docs').glob('*.md')):
        passages.extend(parse_document(p.name, p.read_bytes()))
    model = None
    if args.semantic:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    index = SearchIndex(passages, model)
    output = {'note':'Six synthetic queries on three short documents; smoke test only.', 'queries':len(QUESTIONS), 'metrics':{}}
    for mode in (['Keyword', 'Semantic', 'Hybrid'] if model is not None else ['Keyword']):
        reciprocal, recall, top1 = [], [], []
        for query, expected in QUESTIONS:
            hits = index.search(query, mode, 3)
            rank = next((r for r,h in enumerate(hits,1) if h['source'] == expected), None)
            reciprocal.append(1/rank if rank else 0)
            recall.append(int(rank is not None)); top1.append(int(rank == 1))
        output['metrics'][mode] = {'MRR@3':sum(reciprocal)/len(QUESTIONS), 'Recall@3':sum(recall)/len(QUESTIONS), 'Top1 accuracy':sum(top1)/len(QUESTIONS)}
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
