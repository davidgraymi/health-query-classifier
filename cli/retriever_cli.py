# cli/retriever_cli.py
import argparse, json, os, sys
from pathlib import Path

# allow running from repo root without installing as a package
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from retriever import Retriever 

def default_corpora_config():
    return {
        "medical_qa": {
            "path": "data/corpora/medical_qa.jsonl",
            "text_fields": ["question", "answer", "title"],
        },
        "miriad": {
            "path": "data/corpora/miriad_text.jsonl",
            "text_fields": ["text", "title"],
        },
        "pubmed": {
            "path": "data/corpora/pubmed_abstracts.jsonl",
            "text_fields": ["title", "text"],
        },
        "unidoc": {
            "path": "data/corpora/unidoc_qa.jsonl",
            "text_fields": ["question", "answer", "title"],
        },
    }

def filter_config(cfg, include):
    if not include:
        return cfg
    keep = set(x.strip() for x in include.split(",") if x.strip())
    return {k: v for k, v in cfg.items() if k in keep}

def check_files(cfg):
    missing = [v["path"] for v in cfg.values() if not Path(v["path"]).exists()]
    if missing:
        print("✗ Missing corpora files:\n  - " + "\n  - ".join(missing))
        print("Run:  python adapters/build_corpora.py")
        sys.exit(1)

def main():
    ap = argparse.ArgumentParser(
        description="Hybrid retrieval (BM25 + Dense + RRF, optional re-rank)"
    )
    ap.add_argument("query", type=str, help="Your search query")
    ap.add_argument("--k", type=int, default=10, help="Number of results to return")
    ap.add_argument(
        "--rerank", action="store_true",
        help="Use cross-encoder reranker (slower, usually better)"
    )
    ap.add_argument(
        "--corpora",
        type=str,
        default="",
        help="Comma-separated subset to search (e.g., 'medical_qa,pubmed'). "
             "Leave blank to search all.",
    )
    args = ap.parse_args()

    cfg = default_corpora_config()
    cfg = filter_config(cfg, args.corpora)
    check_files(cfg)

    retr = Retriever(corpora_config=cfg, use_reranker=args.rerank)
    hits = retr.retrieve(args.query, k=args.k, for_ui=True)
    print(json.dumps(hits, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
