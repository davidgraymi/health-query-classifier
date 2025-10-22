from dataclasses import dataclass
from typing import Any, Dict

@dataclass
class Candidate:
    id: str
    title: str
    text: str
    meta: Dict[str, Any]
    bm25: float
    dense: float
    rrf: float
