# retriever/index_dense.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import numpy as np
import pickle
import torch
from pathlib import Path
from sentence_transformers import SentenceTransformer
from classifier.utils import DEVICE 

try:
    import faiss  # type: ignore
    _HAS_FAISS = True
except Exception:
    _HAS_FAISS = False

def _chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def _compute_cache_key(docs, model_name):
    """Compute a hash key for caching based on documents and model."""
    # Create a hash from document IDs/texts and model name
    doc_ids = "".join([d.id for d in docs])
    content = f"{model_name}:{doc_ids}"
    return hashlib.md5(content.encode()).hexdigest()

class DenseIndex:
    def __init__(self, docs, model_name="sentence-transformers/all-MiniLM-L6-v2",
                 batch_size=64, embedding_model=None, cache_dir=".cache/embeddings"):  
        self.docs = docs

        torch.set_num_threads(1)
        if embedding_model:
            self.model = embedding_model
            # Use the device from the provided model
            self.device = self.model.device
            # Get model name from the model object
            actual_model_name = getattr(self.model, 'model_card_data', {}).get('base_model', model_name)
            if hasattr(self.model, '_model_card_vars') and 'model_id' in self.model._model_card_vars:
                actual_model_name = self.model._model_card_vars['model_id']
        else:
            self.model = SentenceTransformer(model_name, device=DEVICE)
            self.device = DEVICE
            actual_model_name = model_name

        # Try to load from cache
        cache_key = _compute_cache_key(docs, actual_model_name)
        cache_path = Path(cache_dir) / f"{cache_key}.pkl"
        
        if cache_path.exists():
            print(f"Loading embeddings from cache: {cache_path}")
            try:
                with open(cache_path, 'rb') as f:
                    self.emb = pickle.load(f)
                print(f"✓ Loaded {len(self.emb)} cached embeddings")
            except Exception as e:
                print(f"Cache load failed: {e}, recomputing...")
                self.emb = self._compute_embeddings(docs, batch_size, cache_path)
        else:
            print(f"No cache found, computing embeddings...")
            self.emb = self._compute_embeddings(docs, batch_size, cache_path)

        if _HAS_FAISS:
            d = self.emb.shape[1]
            self.index = faiss.IndexFlatIP(d)
            self.index.add(self.emb)
        else:
            self.index = None  # NumPy top-k fallback

    def _compute_embeddings(self, docs, batch_size, cache_path):
        """Compute embeddings and save to cache."""
        texts = [d.text for d in docs]
        embs = []
        
        # Partial cache logic
        partial_cache_path = cache_path.parent / f"{cache_path.stem}.partial.pkl"
        start_index = 0
        
        if partial_cache_path.exists():
            print(f"Found partial cache: {partial_cache_path}")
            try:
                with open(partial_cache_path, 'rb') as f:
                    embs = pickle.load(f)
                start_index = sum(len(e) for e in embs)
                print(f"Resuming from doc {start_index}/{len(texts)}")
            except Exception as e:
                print(f"Failed to load partial cache: {e}, starting over...")
                embs = []
                start_index = 0

        texts_to_process = texts[start_index:]
        start_batch = len(embs)
        total_batches = (len(texts) + batch_size - 1) // batch_size
        
        if texts_to_process:
            print(f"Indexing remaining {len(texts_to_process)} documents (batches {start_batch+1}-{total_batches}) on {self.device}...")
    
            with torch.inference_mode():
                total_processed = start_index
                for i, part in enumerate(_chunks(texts_to_process, batch_size), 1):
                    part_emb = self.model.encode(
                        part,
                        batch_size=batch_size,
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                        device=self.device,
                    )
                    embs.append(part_emb.astype(np.float32))
                    total_processed += len(part)
                    
                    current_batch = start_batch + i
                    if current_batch % 10 == 0 or total_processed == len(texts):
                        print(f"  Processed {current_batch}/{total_batches} batches ({total_processed}/{len(texts)} docs)")
                        # Save partial
                        with open(partial_cache_path, 'wb') as f:
                            pickle.dump(embs, f)
                        print(f"  Saved partial cache to {partial_cache_path}")

        emb = np.vstack(embs).astype(np.float32)
        
        # Save to cache
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(emb, f)
        print(f"✓ Saved embeddings to cache: {cache_path}")
        
        # Cleanup partial cache
        if partial_cache_path.exists():
            partial_cache_path.unlink()
        
        return emb

    def search(self, query: str, k: int = 50):
        qv = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
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
