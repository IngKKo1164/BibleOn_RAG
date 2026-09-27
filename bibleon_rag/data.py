"""Read inert private assets; never evaluate JavaScript or download a corpus."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from .common import sha256

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = ROOT / "data" / "runtime"
ASSET_FILES = {"corpus.json.gz", "headings.json", "vectors.i8", "index.json", "lexical.json.gz", "study.json.gz"}


def read_json(path: Path):
    raw = path.read_bytes()
    if path.suffix == ".gz":
        raw = gzip.decompress(raw)
    return json.loads(raw)


def verify_assets(directory: Path = DEFAULT_DATA, manifest: Path | None = None) -> list[dict]:
    manifest = manifest or ROOT / "python-runtime-assets.json"
    records = read_json(manifest)
    if (not isinstance(records, list) or len(records) != len(ASSET_FILES)
            or any(not isinstance(row, dict) for row in records)
            or {row.get("file") for row in records} != ASSET_FILES):
        raise ValueError("INCOMPLETE_RUNTIME_ASSETS")
    for row in records:
        name = row["file"]
        if Path(name).name != name or name in (".", ".."):
            raise ValueError("INVALID_ASSET_PATH")
        path = directory / name
        if path.is_symlink() or sha256(path.read_bytes()) != row["sha256"]:
            raise ValueError(f"RUNTIME_ASSET_MISMATCH:{name}")
    return records


def load_corpus(directory: Path = DEFAULT_DATA) -> dict:
    return read_json(directory / "corpus.json.gz")


def load_headings(directory: Path = DEFAULT_DATA) -> dict:
    record = read_json(directory / "headings.json")
    from .common import json_text
    if sha256(json_text(record["boundaries"])) != record["metadata"]["boundarySha256"]:
        raise ValueError("HEADING_BOUNDARY_MISMATCH")
    return record


def load_study_data(directory: Path = DEFAULT_DATA) -> dict:
    return read_json(directory / "study.json.gz")
