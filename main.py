import argparse
import json

from classifier.head import ClassifierHead
from classifier.infer import classifier_init, predict_query
from classifier.utils import CATEGORIES
from dataclasses import asdict
from sentence_transformers import SentenceTransformer
from team.candidates import get_candidates


EXIT_COMMANDS = ["exit", "quit"]

PROMPT = "\nQuery> "


def main(k: int, use_reranker: bool, embedding_model: SentenceTransformer, classifier: ClassifierHead) -> None:
    print(f"(Ctrl-D or 'quit' to exit)")

    while True:
        try:
            query = input(PROMPT).strip()
            if not query or query.lower() in EXIT_COMMANDS:
                break

            classification = predict_query(
                text=[query],
                embedding_model=embedding_model,
                classifier_head=classifier,
            )

            predictions = classification["prediction"]

            print(f"\nTriaging query as {predictions[0]}")
            print(f"\nConfidence:")
            for i, prob in enumerate(classification['probabilities']):
                cat = CATEGORIES[i]
                percent = prob * 100
                print(f"  {cat}: {percent:3.2f}%")
            print()

            if "medical" in predictions:
                hits = get_candidates(
                    query=query,
                    k_retrieve=k,
                    use_reranker=use_reranker,
                )

                print(f"Found {len(hits)} matching medical documents\n")

                if not hits:
                    print("No medical documents found.\n")

                    continue

                for i, hit in enumerate(hits, 1):
                    serializable = asdict(hit)
                    print(json.dumps(serializable, indent=2, ensure_ascii=False))
            else:
                print(f"TODO: handle queries of type {predictions}")

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

    embedding_model, classifier = classifier_init()

    main(
        k=args.k,
        use_reranker=args.rerank,
        embedding_model=embedding_model,
        classifier=classifier,
    )
