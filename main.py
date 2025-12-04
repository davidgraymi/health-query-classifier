import argparse
import json
from dataclasses import asdict

from pipeline import HealthQueryPipeline

EXIT_COMMANDS = ["exit", "quit"]
PROMPT = "\nQuery> "

def main(pipeline: HealthQueryPipeline, k: int) -> None:
    print(f"(Ctrl-D or 'quit' to exit)")

    while True:
        try:
            query = input(PROMPT).strip()
            if not query or query.lower() in EXIT_COMMANDS:
                break

            # Use the pipeline to get results
            result = pipeline.predict(query, k=k)
            
            classification = result["classification"]
            prediction = classification["prediction"]
            
            print(f"\nTriaging query as {prediction}")
            print(f"\nConfidence:")
            for cat, prob in classification["probabilities"].items():
                percent = prob * 100
                print(f"  {cat}: {percent:3.2f}%")
            print()

            if "medical" == prediction:
                hits = result["retrieval"]
                print(f"Found {len(hits)} matching medical documents\n")

                if not hits:
                    print("No medical documents found.\n")
                    continue

                for i, hit in enumerate(hits, 1):
                    # hit is already a dict from the pipeline
                    print(json.dumps(hit, indent=2, ensure_ascii=False))
            else:
                print(f"TODO: handle queries of type {prediction}")
                continue

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

    # Initialize pipeline
    pipeline = HealthQueryPipeline(use_reranker=args.rerank)
    pipeline.initialize()

    main(pipeline, k=args.k)
