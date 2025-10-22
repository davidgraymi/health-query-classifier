import re

# Simple non-word splitter (keeps letters/numbers, splits on punctuation/whitespace)
_WS = re.compile(r"\W+", flags=re.UNICODE)

def tokenize(s: str) -> list[str]:
    """
    Lowercase + split on non-word chars. Returns [] for None/empty.
    Used by BM25 to build the tokenized corpus and query.
    """
    if not s:
        return []
    return [t for t in _WS.split(s.lower()) if t]


