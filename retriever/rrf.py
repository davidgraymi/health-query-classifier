from collections import defaultdict

def rrf(rank_lists, k=10, K=60):
    scores = defaultdict(float)
    id2doc = {}
    for rl in rank_lists:
        for r, (doc, _) in enumerate(rl):
            id2doc[doc.id] = doc
            scores[doc.id] += 1.0 / (K + r + 1)
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return [(id2doc[i], s) for i, s in ranked]
