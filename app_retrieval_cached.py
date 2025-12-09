"""
Medical Q&A UI - BM25 + Dense Retrieval Models WITH DISK CACHING
This version caches the indexes to disk for fast startup (30 seconds vs 5-8 minutes!)
"""

import gradio as gr
from typing import Dict, List
from pathlib import Path
import pickle
import hashlib
import json
from retriever.index_bm25 import BM25Index
from retriever.index_dense import DenseIndex
from retriever.ingest import load_jsonl
from retriever.rrf import rrf
from team.interfaces import Candidate

# Cache directory
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

print("=" * 70)
print(" Medical Document Retrieval System (CACHED VERSION)")
print(" Using BM25 + Dense Embeddings + RRF Fusion")
print(" With disk caching for fast startup!")
print("=" * 70)


def _default_corpora_config() -> Dict[str, dict]:
    return {
        "medical_qa": {"path": "data/corpora/medical_qa.jsonl",
                       "text_fields": ["question", "answer", "title"]},
        "miriad": {"path": "data/corpora/miriad_text.jsonl",
                   "text_fields": ["question", "answer", "title"]},
        "unidoc": {"path": "data/corpora/unidoc_qa.jsonl",
                   "text_fields": ["question", "answer", "title"]},
    }


def _available(cfg: Dict[str, dict]) -> Dict[str, dict]:
    return {k: v for k, v in cfg.items() if Path(v["path"]).exists()}


def _get_cache_key(corpora_config: Dict[str, dict]) -> str:
    """Generate a unique cache key based on corpora config"""
    config_str = json.dumps(corpora_config, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()


class CachedRetriever:
    """Retriever with disk caching for BM25 and Dense indexes"""
    
    def __init__(self, corpora_config: Dict[str, dict], use_reranker: bool = False):
        self.corpora_config = corpora_config
        self.use_reranker = use_reranker
        self.cache_key = _get_cache_key(corpora_config)
        
        # Cache file paths
        self.bm25_cache = CACHE_DIR / f"bm25_{self.cache_key}.pkl"
        self.dense_cache = CACHE_DIR / f"dense_{self.cache_key}.pkl"
        self.docs_cache = CACHE_DIR / f"docs_{self.cache_key}.pkl"
        
        # Load or build indexes
        self.docs_all = self._load_or_build_docs()
        self.bm25 = self._load_or_build_bm25()
        self.dense = self._load_or_build_dense()
        
    def _load_or_build_docs(self) -> List:
        """Load documents from cache or build from scratch"""
        if self.docs_cache.exists():
            print(f"Loading documents from cache... ({self.docs_cache.name})")
            try:
                with open(self.docs_cache, 'rb') as f:
                    docs_all = pickle.load(f)
                print(f"  ✓ Loaded {len(docs_all)} documents from cache")
                return docs_all
            except Exception as e:
                print(f"  ✗ Cache load failed: {e}")
                print("  → Rebuilding documents...")
        
        print("Building documents from corpora files...")
        docs_all = []
        for name, cfg in self.corpora_config.items():
            print(f"  Loading {name}...")
            docs = load_jsonl(cfg["path"], tuple(cfg.get("text_fields", ("question", "answer"))))
            docs_all.extend(docs)
        
        # Save to cache
        print(f"Saving documents to cache... ({len(docs_all)} docs)")
        with open(self.docs_cache, 'wb') as f:
            pickle.dump(docs_all, f)
        
        return docs_all
    
    def _load_or_build_bm25(self) -> BM25Index:
        """Load BM25 index from cache or build from scratch"""
        if self.bm25_cache.exists():
            print(f"Loading BM25 index from cache... ({self.bm25_cache.name})")
            try:
                with open(self.bm25_cache, 'rb') as f:
                    bm25_index = pickle.load(f)
                print(f"  ✓ BM25 index loaded from cache")
                return bm25_index
            except Exception as e:
                print(f"  ✗ Cache load failed: {e}")
                print("  → Rebuilding BM25 index...")
        
        print("Building BM25 index from scratch...")
        bm25_index = BM25Index(self.docs_all)
        
        # Save to cache
        print(f"Saving BM25 index to cache...")
        with open(self.bm25_cache, 'wb') as f:
            pickle.dump(bm25_index, f)
        
        return bm25_index
    
    def _load_or_build_dense(self) -> DenseIndex:
        """Load Dense index from cache or build from scratch"""
        if self.dense_cache.exists():
            print(f"Loading Dense index from cache... ({self.dense_cache.name})")
            try:
                with open(self.dense_cache, 'rb') as f:
                    dense_index = pickle.load(f)
                print(f"  ✓ Dense index loaded from cache")
                return dense_index
            except Exception as e:
                print(f"  ✗ Cache load failed: {e}")
                print("  → Rebuilding Dense index...")
        
        print("Building Dense index from scratch (this takes 5-8 minutes)...")
        dense_index = DenseIndex(self.docs_all)
        
        # Save to cache
        print(f"Saving Dense index to cache...")
        with open(self.dense_cache, 'wb') as f:
            pickle.dump(dense_index, f)
        
        return dense_index


# Initialize cached retriever (fast if cached, slow first time)
print("\nInitializing retrieval system...")
cfg = _available(_default_corpora_config())
if not cfg:
    raise RuntimeError("No corpora files found in data/corpora. Build them first.")

retriever = CachedRetriever(corpora_config=cfg, use_reranker=False)

print("\n✓ Retrieval system ready!")
print(f"  Total documents indexed: {len(retriever.docs_all):,}")
print("=" * 70)


def get_candidates_cached(query: str, k_retrieve: int = 50) -> List[Candidate]:
    """
    Returns top-N fused candidates with component scores (bm25, dense, rrf).
    Uses the cached retriever for fast queries.
    """
    # Get separate result lists (doc, score)
    bm = retriever.bm25.search(query, k=max(k_retrieve, 100))
    de = retriever.dense.search(query, k=max(k_retrieve, 100))

    # Maps for score lookup
    bm_map = {d.id: float(s) for d, s in bm}
    de_map = {d.id: float(s) for d, s in de}

    # Fuse and pick candidate set
    fused = rrf([bm, de], k=max(k_retrieve, 50))

    # Compute RRF per candidate using rank positions
    K = 60
    bm_rank = {d.id: i for i, (d, _) in enumerate(bm)}
    de_rank = {d.id: i for i, (d, _) in enumerate(de)}

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
    # Baseline order: RRF
    out.sort(key=lambda c: c.rrf, reverse=True)
    return out


def retrieve_documents(query, num_results=5):
    """Retrieve relevant medical documents using your team's models"""
    if not query or not query.strip():
        return """
        <div style="padding: 20px; background-color: #e7f3ff; border-radius: 10px; border-left: 5px solid #2196f3;">
            <h3 style="margin-top: 0; color: #0d47a1;">How to Use</h3>
            <p style="margin: 0; color: #1565c0;">Enter a medical query and we'll find relevant documents using BM25 + Dense retrieval with RRF fusion.</p>
            <p style="margin: 8px 0 0 0; color: #1565c0;"><strong>Example:</strong> "headache with blurred vision" or "symptoms of diabetes"</p>
        </div>
        """
    
    try:
        # Use cached retrieval system (fast!)
        hits = get_candidates_cached(query=query, k_retrieve=num_results)
        
        if not hits:
            return """
            <div style="padding: 20px; background-color: #fff3cd; border-radius: 10px; border-left: 5px solid #ffc107;">
                <h3 style="margin-top: 0; color: #856404;">No Results Found</h3>
                <p style="margin: 0; color: #856404;">Try rephrasing your query or using different medical terms.</p>
            </div>
            """
        
        # Build results HTML
        result_html = f"""
        <div style="padding: 15px; background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); border-radius: 10px; margin-bottom: 20px; border-left: 5px solid #28a745;">
            <h3 style="margin-top: 0; color: #155724;">Found {len(hits)} Relevant Medical Documents</h3>
            <p style="margin: 0;"><strong>Retrieved using:</strong> BM25 + Dense Embeddings + RRF Fusion (CACHED)</p>
        </div>
        """
        
        for i, hit in enumerate(hits, 1):
            title = hit.title if hit.title and hit.title.strip() else None
            source = hit.meta.get('source', 'Unknown') if hit.meta else 'Unknown'
            
            # Check if we have separate question/answer fields in metadata
            question = hit.meta.get('question', '') if hit.meta else ''
            answer = hit.meta.get('answer', '') if hit.meta else ''
            
            # If we have separate Q&A, format them nicely
            if question and answer:
                content_html = f"""
                    <div style="margin-bottom: 12px;">
                        <strong style="color: #1976d2;">Question:</strong>
                        <p style="margin: 5px 0 0 0; line-height: 1.6; color: #424242;">{question}</p>
                    </div>
                    <div>
                        <strong style="color: #388e3c;">Answer:</strong>
                        <p style="margin: 5px 0 0 0; line-height: 1.6; color: #424242;">{answer[:500] + ("..." if len(answer) > 500 else "")}</p>
                    </div>
                """
            else:
                # Fallback to combined text
                text = hit.text[:500] + ("..." if len(hit.text) > 500 else "")
                content_html = f'<p style="margin: 0; line-height: 1.7; color: #34495e;">{text}</p>'
            
            # Display relevance scores
            bm25_score = hit.bm25
            dense_score = hit.dense
            rrf_score = hit.rrf
            
            # Build title HTML only if title exists
            title_html = f'<h4 style="margin: 0 0 15px 0; color: #2c3e50;">{title}</h4>' if title else ''
            
            result_html += f"""
            <div style="border: 2px solid #dee2e6; padding: 20px; margin: 20px 0; border-radius: 10px; background-color: white; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 15px; margin: -20px -20px 20px -20px; border-radius: 8px 8px 0 0;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <h4 style="margin: 0; color: white;">Document #{i}</h4>
                        <span style="background-color: rgba(255,255,255,0.2); padding: 4px 12px; border-radius: 12px; font-size: 0.85em; color: white;">
                            {source}
                        </span>
                    </div>
                </div>
                
                <div style="margin-bottom: 15px;">
                    {title_html}
                    {content_html}
                </div>
                
                <div style="padding-top: 12px; border-top: 1px solid #e9ecef;">
                    <div style="display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px;">
                        <div style="background-color: #e3f2fd; padding: 8px; border-radius: 5px; text-align: center;">
                            <div style="font-size: 0.75em; color: #1976d2; font-weight: bold;">BM25</div>
                            <div style="font-size: 1.1em; color: #0d47a1;">{bm25_score:.4f}</div>
                        </div>
                        <div style="background-color: #f3e5f5; padding: 8px; border-radius: 5px; text-align: center;">
                            <div style="font-size: 0.75em; color: #7b1fa2; font-weight: bold;">Dense</div>
                            <div style="font-size: 1.1em; color: #4a148c;">{dense_score:.4f}</div>
                        </div>
                        <div style="background-color: #e8f5e9; padding: 8px; border-radius: 5px; text-align: center;">
                            <div style="font-size: 0.75em; color: #388e3c; font-weight: bold;">RRF Fusion</div>
                            <div style="font-size: 1.1em; color: #1b5e20;">{rrf_score:.4f}</div>
                        </div>
                    </div>
                </div>
            </div>
            """
        
        return result_html
        
    except Exception as e:
        return f"""
        <div style="padding: 20px; background-color: #f8d7da; border-radius: 10px; border-left: 5px solid #dc3545;">
            <h3 style="margin-top: 0; color: #721c24;">Error</h3>
            <p style="margin: 0; color: #721c24;">{str(e)}</p>
        </div>
        """


# Create Gradio interface
with gr.Blocks(title="Medical Document Retrieval (Cached)") as demo:
    gr.Markdown("""
    # Medical Document Retrieval System (CACHED VERSION)
    
    **Models:**
    - BM25 Index (keyword-based retrieval)
    - Dense Embeddings (embeddinggemma-300m-medical)
    - RRF Fusion (combines both approaches)
    
    ### Features:
    - Searches across 10,000+ medical documents
    - Shows relevance scores from each model component
    - Returns the most relevant medical information
    """)
    
    with gr.Row():
        with gr.Column():
            query_input = gr.Textbox(
                label="Enter your medical query",
                placeholder="Example: headache with blurred vision",
                lines=2
            )
            num_results = gr.Slider(
                minimum=1,
                maximum=10,
                value=5,
                step=1,
                label="Number of results to retrieve"
            )
            submit_btn = gr.Button("Retrieve Documents", variant="primary", size="lg")
        
    output_html = gr.HTML(label="Search Results")
    
    submit_btn.click(
        fn=retrieve_documents,
        inputs=[query_input, num_results],
        outputs=output_html
    )
    
    gr.Examples(
        examples=[
            "headache with blurred vision",
            "symptoms of diabetes",
            "chest pain when exercising",
            "treatment for high blood pressure",
            "causes of chronic fatigue",
        ],
        inputs=query_input,
        label="Try these example queries:"
    )
    
    gr.Markdown("""
    ---
    ### Technical Details
    - **BM25**: Statistical keyword matching (TF-IDF based)
    - **Dense**: Semantic search using transformer embeddings
    - **RRF Fusion**: Reciprocal Rank Fusion combines both methods
    - **Caching**: Indexes saved to disk in `cache/` folder for fast reloading
    
    *Note: First launch builds and caches indexes (5-8 min). After that, startup takes only ~30 seconds!*
    """)

print("\nOpening web interface...")
print("   Local access: http://127.0.0.1:7863")
print("   Public link will be generated...")
print("=" * 70)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7863, share=True)
