import json, pathlib
from .data_schemas import Doc

def load_jsonl(path: str, text_fields=("question","answer")):
    p = pathlib.Path(path)
    docs = []
    with p.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            # Collect fields; allow either "text" or joined fields
            if "text" in row and row["text"]:
                combined = row["text"]
            else:
                combined = " ".join([row.get(tf, "") for tf in text_fields]).strip()
            title = row.get("title") or row.get("category") or ""
            docs.append(Doc(
                id=str(row.get("id", f"{p.stem}:{i}")),
                text=combined,
                title=title,
                meta=row
            ))
    return docs
