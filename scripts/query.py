#! /usr/bin/env python3

import json
import readline
import sys

from pyserini.search.lucene import LuceneSearcher


def main():
    index_dir = sys.argv[1] if len(sys.argv) > 1 else "indexes/pubmed"

    searcher = LuceneSearcher(index_dir)

    print(f"Loaded {searcher.num_docs} documents from {index_dir}")
    print(f"(Ctrl-D or 'quit' to exit)\n")

    while True:
        try:
            query = input("PubMed> ").strip()
            if not query or query.lower() in ['quit', 'exit']:
                break

            hits = searcher.search(query, k=10)

            print(f"{len(hits)}/{searcher.num_docs} matching documents found\n")

            if not hits:
                print("No results found.\n")

                continue

            for i, hit in enumerate(hits, 1):
                doc = searcher.doc(hit.docid)

                raw = json.loads(doc.raw())

                title = raw.get('title', '')
                contents = raw.get('contents', '')

                abstract = contents[len(title):] if contents.startswith(title) else contents

                print(f"{i}. PMID {hit.docid} \"{title}\" (score: {hit.score:.4f})")
                print(f"   {abstract[:120]}...\n")

        except EOFError:
            print("\nBye!")

            break

        except KeyboardInterrupt:
            print("\nBye!")

            break


if __name__ == "__main__":
    main()
