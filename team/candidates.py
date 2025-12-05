from typing import Dict, List
from retriever import Retriever
from retriever.rrf import rrf
from team.interfaces import Candidate
from pathlib import Path

def _default_corpora_config() -> Dict[str, dict]:
    return {
        "medical_qa": {"path":"data/corpora/medical_qa.jsonl",
                       "text_fields":["question","answer","title"]},
        "miriad":     {"path":"data/corpora/miriad_text.jsonl",
                       "text_fields":["question","answer","title"]},
        "unidoc":     {"path":"data/corpora/unidoc_qa.jsonl",
                       "text_fields":["question","answer","title"]},
    }

def _available(cfg: Dict[str, dict]) -> Dict[str, dict]:
    return {k:v for k,v in cfg.items() if Path(v["path"]).exists()}

def get_candidates(
    query: str,
    retriever: Retriever,
    k_retrieve: int = 50,
) -> List[Candidate]:
    """
    Returns top-N fused candidates with component scores (bm25, dense, rrf).
    """
    r = retriever

    # get separate result lists (doc, score)
    bm = r.bm25.search(query, k=max(k_retrieve, 100))
    de = r.dense.search(query, k=max(k_retrieve, 100))

    # maps for score lookup
    bm_map = {d.id: float(s) for d, s in bm}
    de_map = {d.id: float(s) for d, s in de}

    # fuse and pick candidate set
    fused = rrf([bm, de], k=max(k_retrieve, 50))

    # compute RRF per candidate using rank positions
    K = 60
    bm_rank = {d.id:i for i,(d,_) in enumerate(bm)}
    de_rank = {d.id:i for i,(d,_) in enumerate(de)}

    out: List[Candidate] = []
    for doc, _ in fused[:k_retrieve]:
        rrf_score = 0.0
        if doc.id in bm_rank:
            rrf_score += 1.0 / (K + bm_rank[doc.id] + 1)
        if doc.id in de_rank:
            rrf_score += 1.0 / (K + de_rank[doc.id] + 1)
        out.append(Candidate(
            id=doc.id,
            title=doc.title or "",
            text=doc.text,
            meta=doc.meta or {},
            bm25=bm_map.get(doc.id, 0.0),
            dense=de_map.get(doc.id, 0.0),
            rrf=rrf_score,
        ))
    # baseline order: RRF
    out.sort(key=lambda c: c.rrf, reverse=True)
    return out


#how to call/run below for everyone
# from team.candidates import get_candidates

# q = "worst headache of my life with fever and stiff neck"
# cands = get_candidates(q, k_retrieve=60)  # returns List[Candidate]
# for c in cands[:3]:
#     print(c.id, c.bm25, c.dense, c.rrf, c.title)
    