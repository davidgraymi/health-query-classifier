from rank_bm25 import BM25Okapi
from .utils import tokenize

class BM25Index:
    def __init__(self, docs):
        self.docs = docs
        self.corpus_tokens = [tokenize(d.text) for d in docs]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, k: int = 50):
        q = tokenize(query)
        scores = self.bm25.get_scores(q)
        top = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return [(self.docs[i], float(scores[i])) for i in top]
