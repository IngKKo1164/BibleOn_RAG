"""Wire-format helpers shared by retrieval, generation and local tooling."""
from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from functools import lru_cache


class BibleProviderError(RuntimeError):
    def __init__(self, code: str, status: int = 502):
        super().__init__(code)
        self.code = code
        self.status = status


def json_text(value) -> str:
    # Match JSON.stringify's compact Unicode payload, without permitting NaN.
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def sha256(value: str | bytes) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def text_length(value: str) -> int:
    """Preserve UTF-16 character budgets used by the previous implementation."""
    return len(value.encode("utf-16-le", errors="surrogatepass")) // 2


def is_integer(value) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value == math.floor(value)


def unique(values: Iterable) -> list:
    return list(dict.fromkeys(values))


def unique_verses(verses: Iterable[dict]) -> list[dict]:
    return list({v["id"]: v for v in verses}.values())


@lru_cache(maxsize=32768)
def canonical_id_sort_key(value: str):
    # Canonical chunk/source IDs are ASCII. Preserve the previous ICU ordering:
    # punctuation (including ':') precedes digits, so 1: comes before 19:.
    # This is a tie-breaker, not numeric chapter ordering.
    alphabet = " _-,;:!?.'\"()[]{}@*/\\&#%`^+<=>|~$0123456789abcdefghijklmnopqrstuvwxyz"
    rank = {char: i for i, char in enumerate(alphabet)}
    return tuple(rank.get(c.lower(), 1000+ord(c)) for c in value), tuple(c.isupper() for c in value)


def object_schema(properties: dict) -> dict:
    return {"type": "object", "additionalProperties": False,
            "properties": properties, "required": list(properties)}


def compact_verses(verses: list[dict]) -> list[dict]:
    return [{k: v[k] for k in ("id", "reference", "text")} for v in verses]


def compact_previous(previous: dict | None) -> dict | None:
    return {"question": previous["question"], "passages": compact_verses(previous["passages"])} if previous else None
