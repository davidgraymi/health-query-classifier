import json, jsonlines, pathlib
import logging
import concurrent.futures
from tqdm import tqdm
from datasets import load_dataset

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
logger = logging.getLogger(__name__)

OUT = pathlib.Path("data/corpora")
OUT.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    logger.info(f"Writing {len(rows)} records to {path}")
    with jsonlines.open(path, "w") as out:
        out.write_all(rows)
    logger.info(f"Finished writing {path}")

# 1) LasseRegin medical Q&A
def build_lasseregin():
    logger.info("Starting LasseRegin build...")
    import urllib.request
    url = "https://raw.githubusercontent.com/LasseRegin/medical-question-answer-data/master/icliniqQAs.json"
    
    try:
        with urllib.request.urlopen(url) as response:
            data = json.loads(response.read().decode("utf-8"))
    except Exception as e:
        logger.error(f"Failed to download LasseRegin data: {e}")
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
    logger.info("Completed LasseRegin build.")

# 2) MIRIAD-4.4M-split
def build_miriad(sample_size=200_000):
    logger.info(f"Starting MIRIAD build (sample_size={sample_size})...")
    try:
        ds = load_dataset("tomaarsen/miriad-4.4M-split", split="test", num_proc=4)
        ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    except Exception as e:
        logger.error(f"Failed to load MIRIAD dataset: {e}")
        return

    rows = []
    for i, ex in enumerate(tqdm(ds, desc="miriad", leave=False)):
        rows.append({
            "id": f"miriad:{i}",
            "title": ex.get("title",""),
            "question": ex.get("question", ""),
            "answer": ex.get("passage_text", ""),
        })
    write_jsonl(OUT / "miriad_text.jsonl", rows)
    logger.info("Completed MIRIAD build.")

# 3) PubMed abstracts
def build_pubmed(max_records=500_000):
    logger.info(f"Starting PubMed build (max_records={max_records})...")
    try:
        ds = load_dataset("ncbi/pubmed")
    except Exception as e:
        logger.error(f"Failed to load PubMed dataset: {e}")
        return

    rows, n = [], 0
    for ex in tqdm(ds, desc="pubmed", leave=False):
        title = (ex.get("Title") or "").strip()
        abstract = (ex.get("Abstract") or "").strip()
        if not (title or abstract):
            continue
        rows.append({
            "id": f"pubmed:{ex.get('PMID','')}",
            "title": title,
            "text": (title + "\n\n" + abstract).strip(),
            "journal": ex.get("JournalTitle",""),
            "year": ex.get("Year","")
        })
        n += 1
        if n >= max_records:
            break
    write_jsonl(OUT / "pubmed_abstracts.jsonl", rows)
    logger.info("Completed PubMed build.")

# 4) UniDoc-Bench (QA)
def build_unidoc(max_items=1000):
    logger.info(f"Starting UniDoc build (max_items={max_items})...")
    try:
        ds = load_dataset("Salesforce/UniDoc-Bench", split="test")
    except Exception as e:
        logger.error(f"Failed to load UniDoc dataset: {e}")
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
    logger.info("Completed UniDoc build.")

def main():
    logger.info("Starting parallel corpora build...")
    
    # Define tasks
    tasks = [
        # (build_lasseregin, []),
        (build_miriad, [200_000]),
        # (build_pubmed, [500_000]),
        # (build_unidoc, [1000])
    ]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [executor.submit(func, *args) for func, args in tasks]
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logger.error(f"A task failed: {e}")

    logger.info("✅ All corpora built successfully in data/corpora/")

if __name__ == "__main__":
    main()
