from dataclasses import dataclass

@dataclass
class Doc:
    id: str
    text: str
    title: str | None = None
    meta: dict | None = None
