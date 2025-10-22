# retriever/index_dense.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

# Optional FAISS (fallback to NumPy if not available)
try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

class DenseIndex:
    def __init__(self, docs, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=16):  # conservative batch size on macOS
        self.docs = docs

        # Force CPU + keep threads low to avoid macOS segfaults
        torch.set_num_threads(1)
        self.model = SentenceTransformer(model_name, device="cpu")

        texts = [d.text for d in docs]
        embs = []

        with torch.inference_mode():
            for part in _chunks(texts, batch_size):
                part_emb = self.model.encode(
                    part,
                    batch_size=batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
                embs.append(part_emb.astype(np.float32))

        self.emb = np.vstack(embs).astype(np.float32)

        if _HAS_FAISS:
            d = self.emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.emb)
        else:
            self.index = None  # NumPy top-k fallback

    def search(self, query: str, k: int = 50):
        qv = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype(np.float32)[0]

        if self.index is not None:
            D, I = self.index.search(qv.reshape(1, -1), k)
            return [(self.docs[int(i)], float(D[0][j])) for j, i in enumerate(I[0])]

        # NumPy fallback (cosine/IP since vectors normalized)
        sims = self.emb @ qv
        if k >= len(sims):
            order = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, kth=k-1)[:k]
            order = idx[np.argsort(-sims[idx])]
        return [(self.docs[int(i)], float(sims[int(i)])) for i in order]
