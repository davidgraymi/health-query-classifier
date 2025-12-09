import json, jsonlines, pathlib
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset
from math import ceil
from pubmed import download_pubmed

OUT = pathlib.Path("data/corpora")
OUT.mkdir(parents=True, exist_ok=True)

PUBMED_ARTICLES_PER_XML_FILE = 30000

def write_jsonl(path, rows):
    print(f"Writing {len(rows)} records to {path}")
    with jsonlines.open(path, "w") as out:
        out.write_all(rows)
    print(f"Finished writing {path}")

# 1) LasseRegin medical Q&A
def build_lasseregin():
    print("Starting LasseRegin build...")
    import urllib.request
    url = "https://raw.githubusercontent.com/LasseRegin/medical-question-answer-data/master/icliniqQAs.json"

    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        print(f"Failed to download LasseRegin data: {e}")
        return

    rows = []
    for i, r in enumerate(tqdm(data, desc="LasseRegin", leave=False)):
        rows.append({
            "id": f"icliniq:{i}",
            "title": r.get("title",""),
            "question": r.get("question",""),
            "answer": r.get("answer",""),
            "source": "icliniq"
        })
    write_jsonl(OUT / "medical_qa.jsonl", rows)
    print("Completed LasseRegin build.")

# 2) MIRIAD-4.4M-split
def build_miriad(sample_size=200_000):
    print(f"Starting MIRIAD build (sample_size={sample_size})...")
    try:
        ds = load_dataset("miriad/miriad-4.4M", num_proc=4, split="train")

        ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    except Exception as e:
        print(f"Failed to load MIRIAD dataset: {e}")
        return

    rows = []
    for i, ex in enumerate(tqdm(ds, desc="miriad", leave=False)):
        rows.append({
            "id": f"miriad:{i}",
            "title": ex.get("paper_title",""),
            "question": ex.get("question", ""),
            "answer": ex.get("passage_text", ""),
            "year": ex.get("year",""),
            "specialty": ex.get("specialty",""),

        })
    write_jsonl(OUT / "miriad_text.jsonl", rows)
    print("Completed MIRIAD build.")

# 3) PubMed abstracts
def build_pubmed(max_records=500_000):
    num_files = int(ceil(max_records / PUBMED_ARTICLES_PER_XML_FILE))
    print(f"Starting PubMed build (num_files={num_files}, max_records={max_records})...")

    download_pubmed(OUT / "pubmed.jsonl", num_files)
    print("Completed PubMed build.")

# 4) UniDoc-Bench (QA)
def build_unidoc(max_items=1000):
    print(f"Starting UniDoc build (max_items={max_items})...")
    try:
        ds = load_dataset("Salesforce/UniDoc-Bench", split="healthcare")
    except Exception as e:
        print(f"Failed to load UniDoc dataset: {e}")
        return

    rows = []
    for i, ex in enumerate(tqdm(ds, desc="unidoc", leave=False)):
        q = ex.get("question","") or ex.get("query","")
        a = ex.get("answer","") or ""
        pdf = ex.get("pdf_path") or ex.get("document_path") or ""
        domain = ex.get("domain","")
        rows.append({
            "id": f"unidoc:{i}",
            "title": f"{domain} PDF",
            "question": q,
            "answer": a,
            "pdf_path": pdf
        })
        if i+1 >= max_items:
            break
    write_jsonl(OUT / "unidoc_qa.jsonl", rows)
    print("Completed UniDoc build.")

def main():
    print("Starting parallel corpora build...")
    # Define tasks
    tasks = [
        (build_lasseregin, []),
        (build_miriad, [1000]),
        (build_pubmed, [500_000]),

        (build_unidoc, [1000])
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(func, *args) for func, args in tasks]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"A task failed: {e}")

    print("✅ All corpora built successfully in data/corpora/")

if __name__ == "__main__":
    main()
