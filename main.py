import argparse
import json
import os
import sys

from dataclasses import asdict
from pathlib import Path
from team.candidates import get_candidates


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
    args = ap.parse_args()

    hits = get_candidates(
        query=args.query,
        k_retrieve=args.k,
        use_reranker=args.rerank,
    )

    serializable = [asdict(hit) for hit in hits]
    print(json.dumps(serializable, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
