import argparse
import json
import os
import readline
import sys

from dataclasses import asdict
from pathlib import Path
from team.candidates import get_candidates


EXIT_COMMANDS = ["exit", "quit"]

PROMPT = "Query> "


def main(k: int, use_reranker: bool) -> None:
    print(f"(Ctrl-D or 'quit' to exit)\n")

    while True:
        try:
            query = input(PROMPT).strip()
            if not query or query.lower() in EXIT_COMMANDS:
                break

            hits = get_candidates(
                query=query,
                k_retrieve=k,
                use_reranker=use_reranker,
            )

            print(f"Found {len(hits)} matching documents\n")

            if not hits:
                print("No results found.\n")

                continue

            for i, hit in enumerate(hits, 1):
                serializable = asdict(hit)
                print(json.dumps(serializable, indent=2, ensure_ascii=False))

        except EOFError:
            print("\nBye!")

            break

        except KeyboardInterrupt:
            print("\nBye!")

            break


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Hybrid retrieval (BM25 + Dense + RRF, optional re-rank)"
    )
    ap.add_argument("--k", type=int, default=10, help="Number of results to return")
    ap.add_argument(
        "--rerank", action="store_true",
        help="Use cross-encoder reranker (slower, usually better)"
    )
    args = ap.parse_args()

    main(k=args.k, use_reranker=args.rerank)
