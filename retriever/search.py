from .index_bm25 import BM25Index
from .index_dense import DenseIndex
from .rrf import rrf
try:
    from .rerank import CrossEncoderReranker
except Exception:
    CrossEncoderReranker = None
from .ingest import load_jsonl

class Retriever:
    def __init__(self, corpora_config, use_reranker=False):
        self.corpora = {}
        docs_all = []
        for name, cfg in corpora_config.items():
            docs = load_jsonl(cfg["path"], tuple(cfg.get("text_fields", ("question","answer"))))
            self.corpora[name] = docs
            docs_all.extend(docs)
        self.bm25 = BM25Index(docs_all)
        self.dense = DenseIndex(docs_all)
        self.reranker = CrossEncoderReranker() if (use_reranker and CrossEncoderReranker) else None

    def retrieve(self, query, k=10, for_ui=True):
        bm = self.bm25.search(query, k=100)
        de = self.dense.search(query, k=100)
        fused = rrf([bm, de], k=max(k, 20))
        if self.reranker:
            reranked = self.reranker.rerank(query, [d for d, _ in fused])[:k]
            results = [(d, float(s)) for d, s in reranked]
        else:
            results = fused[:k]
        if not for_ui:
            return results
        return [{
            "id": d.id,
            "title": d.title,
            "snippet": d.text[:300] + ("..." if len(d.text) > 300 else ""),
            "score": s,
            "meta": d.meta
        } for d, s in results]
