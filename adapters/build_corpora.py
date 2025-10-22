import os, json, jsonlines, pathlib
from tqdm import tqdm
from datasets import load_dataset

OUT = pathlib.Path("data/corpora")
OUT.mkdir(parents=True, exist_ok=True)

def write_jsonl(path, rows):
    with jsonlines.open(path, "w") as out:
        for r in rows:
            out.write(r)

# 1) LasseRegin medical Q&A 
def build_lasseregin():
    import urllib.request
    url = "https://raw.githubusercontent.com/LasseRegin/medical-question-answer-data/master/icliniqQAs.json"
    data = json.loads(urllib.request.urlopen(url).read().decode("utf-8"))
    rows = []
    for i, r in enumerate(data):
        rows.append({
            "id": f"icliniq:{i}",
            "title": r.get("title",""),
            "question": r.get("question",""),
            "answer": r.get("answer",""),
            "source": "icliniq"
        })
    write_jsonl(OUT / "medical_qa.jsonl", rows)

# 2) MIRIAD-4.4M-split 
def build_miriad(sample_size=200_000):
    ds = load_dataset("tomaarsen/miriad-4.4M-split", split="train")
    ds = ds.shuffle(seed=42).select(range(min(sample_size, len(ds))))
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="miriad")):
        text = ex.get("text") or ex.get("content") or ""
        if not text:
            continue
        rows.append({
            "id": f"miriad:{i}",
            "title": ex.get("title",""),
            "text": text
        })
    write_jsonl(OUT / "miriad_text.jsonl", rows)

# 3) PubMed abstracts DONT MATTER ANYMORE
def build_pubmed(max_records=500_000):
    ds = load_dataset("ncbi/pubmed", split="train")
    rows, n = [], 0
    for ex in tqdm(ds, desc="pubmed"):
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

# 4) UniDoc-Bench (QA)
def build_unidoc(max_items=1000):
    ds = load_dataset("Salesforce/UniDoc-Bench", split="train")
    rows = []
    for i, ex in enumerate(tqdm(ds, desc="unidoc")):
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

if __name__ == "__main__":
    build_lasseregin()
    build_miriad(sample_size=200_000)
    #build_pubmed(max_records=500_000)
    build_unidoc(max_items=1000)
    print("✅ Wrote corpora to data/corpora/")
