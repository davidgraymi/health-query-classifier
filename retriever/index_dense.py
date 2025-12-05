# retriever/index_dense.py
import os
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import hashlib
import threading
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
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        
        # Thread safety
        self.lock = threading.Lock()
        self.ready_count = 0
        self.emb_batches = [] # List of numpy arrays for fallback
        
        torch.set_num_threads(1)
        if embedding_model:
            self.model = embedding_model
            self.device = self.model.device
            actual_model_name = getattr(self.model, 'model_card_data', {}).get('base_model', model_name)
            if hasattr(self.model, '_model_card_vars') and 'model_id' in self.model._model_card_vars:
                actual_model_name = self.model._model_card_vars['model_id']
        else:
            self.model = SentenceTransformer(model_name, device=DEVICE)
            self.device = DEVICE
            actual_model_name = model_name

        self.cache_key = _compute_cache_key(docs, actual_model_name)
        self.cache_path = Path(cache_dir) / f"{self.cache_key}.pkl"

        # Initialize index structure
        if _HAS_FAISS:
            # We need to know dimension to init FAISS. 
            # We'll init it when the first batch arrives or if we load full cache.
            self.index = None 
        else:
            self.index = None

        # Start background ingestion
        self.ingest_thread = threading.Thread(target=self._ingest_embeddings, daemon=True)
        self.ingest_thread.start()

    def _generate_embeddings(self):
        """Yields batches of embeddings from cache or computation."""
        texts = [d.text for d in self.docs]
        
        # 1. Try full cache first
        if self.cache_path.exists():
            print(f"Loading embeddings from cache: {self.cache_path}")
            try:
                with open(self.cache_path, 'rb') as f:
                    full_emb = pickle.load(f)
                print(f"✓ Loaded {len(full_emb)} cached embeddings")
                # Yield as a single large batch
                yield full_emb
                return
            except Exception as e:
                print(f"Cache load failed: {e}, recomputing...")

        # 2. Partial cache logic
        partial_cache_path = self.cache_path.parent / f"{self.cache_path.stem}.partial.pkl"
        start_index = 0
        existing_embs = []

        if partial_cache_path.exists():
            try:
                with open(partial_cache_path, 'rb') as f:
                    existing_embs = pickle.load(f)
                
                # Yield existing chunks
                # We assume existing_embs is a list of batches from previous run
                # But wait, previous implementation saved list of batches.
                # Let's verify if it saved list of batches or vstacked array.
                # Previous impl: pickle.dump(embs, f) where embs is list of arrays.
                
                for batch in existing_embs:
                    yield batch
                
                start_index = sum(len(e) for e in existing_embs)
            except Exception as e:
                existing_embs = []
                start_index = 0

        # 3. Compute remaining
        texts_to_process = texts[start_index:]
        if not texts_to_process:
            return
        
        # We need to keep track of all embs (existing + new) to save partial/full cache
        # But `existing_embs` might be large.
        # We will append new batches to `existing_embs` locally to save partials.
        
        with torch.inference_mode():
            total_processed = start_index
            total_batches = (len(texts) + self.batch_size - 1) // self.batch_size
            start_batch = len(existing_embs)

            for i, part in enumerate(_chunks(texts_to_process, self.batch_size), 1):
                part_emb = self.model.encode(
                    part,
                    batch_size=self.batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                    device=self.device,
                )
                batch_emb = part_emb.astype(np.float32)
                yield batch_emb
                
                existing_embs.append(batch_emb)
                total_processed += len(part)

                # Save partial
                with open(partial_cache_path, 'wb') as f:
                    pickle.dump(existing_embs, f)

    def _ingest_embeddings(self):
        """Background thread to ingest embeddings from generator."""
        all_embs = []
        
        for batch_emb in self._generate_embeddings():
            with self.lock:
                if _HAS_FAISS:
                    if self.index is None:
                        d = batch_emb.shape[1]
                        self.index = faiss.IndexFlatIP(d)
                    self.index.add(batch_emb)
                
                # We also keep track for fallback or saving
                self.emb_batches.append(batch_emb)
                self.ready_count += len(batch_emb)
                
            all_embs.append(batch_emb)

        # Finalize
        full_emb = np.vstack(all_embs).astype(np.float32)
        
        # Save full cache
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.cache_path, 'wb') as f:
            pickle.dump(full_emb, f)
        print(f"✓ Saved embeddings to cache: {self.cache_path}")
        
        # Cleanup partial
        partial_cache_path = self.cache_path.parent / f"{self.cache_path.stem}.partial.pkl"
        if partial_cache_path.exists():
            partial_cache_path.unlink()

    def search(self, query: str, k: int = 50):
        qv = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            device=self.device,
        ).astype(np.float32)[0]

        with self.lock:
            current_count = self.ready_count
            if current_count == 0:
                print("Warning: Index not yet initialized, returning empty results.")
                return []
            
            # If we have partial data, we search it.
            if _HAS_FAISS and self.index is not None:
                # FAISS index is updated incrementally
                D, I = self.index.search(qv.reshape(1, -1), min(k, current_count))
                return [(self.docs[int(i)], float(D[0][j])) for j, i in enumerate(I[0]) if i != -1]
            
            # NumPy fallback
            # We might have multiple batches, need to stack them for search
            # Optimization: cache the stacked version if it hasn't changed? 
            # For now, just stack what we have.
            curr_emb = np.vstack(self.emb_batches)
            
        sims = curr_emb @ qv
        effective_k = min(k, len(sims))
        
        if effective_k >= len(sims):
            order = np.argsort(-sims)
        else:
            idx = np.argpartition(-sims, kth=effective_k-1)[:effective_k]
            order = idx[np.argsort(-sims[idx])]
            
        return [(self.docs[int(i)], float(sims[int(i)])) for i in order]

    def get_progress(self):
        """Returns (current_count, total_count) of indexed documents."""
        with self.lock:
            return self.ready_count, len(self.docs)
