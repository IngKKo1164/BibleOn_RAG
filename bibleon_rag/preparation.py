"""Local corpus/heading/index preparation. New embeddings require --live."""
from __future__ import annotations

import gzip
import json
import math
import os
from pathlib import Path

from .common import json_text, sha256, text_length
from .config import load_env
from .data import ASSET_FILES, DEFAULT_DATA, ROOT, load_headings, read_json, verify_assets
from .passages import CHUNK_VERSION, compile_heading_boundaries, create_bible_chunks
from .retrieval import BibleRetriever, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, lexical_fingerprint, quantize_vector
from .transport import ProviderTransport, embed_texts


def atomic_write(path: Path, content: bytes):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+".tmp")
    temporary.write_bytes(content)
    os.replace(temporary, path)


def update_manifest(directory=DEFAULT_DATA, manifest=None):
    files = ["corpus.json.gz", "headings.json", "vectors.i8", "index.json", "lexical.json.gz", "study.json.gz"]
    records = [{"file": name, "sha256": sha256((directory/name).read_bytes())} for name in files if (directory/name).is_file()]
    # A custom data directory owns its own manifest; never mutate another workspace.
    manifest = manifest or (ROOT/"python-runtime-assets.json" if directory.resolve() == DEFAULT_DATA.resolve() else directory/"python-runtime-assets.json")
    if len(records) == len(ASSET_FILES):
        atomic_write(manifest, (json.dumps(records, ensure_ascii=False, indent=2)+"\n").encode("utf-8"))
    return records


def restore_runtime_data(source: Path, target=DEFAULT_DATA, manifest=None):
    source, target = source.resolve(), target.resolve()
    records = verify_assets(source, manifest)
    planned = []
    for row in records:
        from_path, to_path = source/row["file"], target/row["file"]
        if to_path.is_symlink():
            raise ValueError("INVALID_ASSET_PATH")
        content = from_path.read_bytes()
        if to_path.exists() and to_path.read_bytes() != content:
            raise ValueError(f"REFUSING_TO_OVERWRITE_MODIFIED_DATA:{row['file']}")
        planned.append((to_path, content))
    for path, content in planned:
        if not path.exists():
            atomic_write(path, content)
    return {"restored": len(planned), "networkUsed": False, "gitExcluded": True}


def build_heading_boundaries(source: Path, directory=DEFAULT_DATA):
    files = sorted(source.glob("*.json"))
    documents = [read_json(path) for path in files]
    corpus_raw = gzip.decompress((directory/"corpus.json.gz").read_bytes())
    result = compile_heading_boundaries(json.loads(corpus_raw), documents)
    metadata = {"version": "rnksv-heading-coordinates-v1", "corpusSha256": sha256(corpus_raw),
                "sourceSha256": sha256("\n".join(path.name+":"+sha256(path.read_bytes()) for path in files)),
                "boundarySha256": sha256(json_text(result["boundaries"])), "chapterBoundary": True,
                "textTranslation": "KRV", "headingLabelsIncluded": False}
    atomic_write(directory/"headings.json", json_text({"metadata": metadata, "boundaries": result["boundaries"]}).encode("utf-8"))
    atomic_write(directory.parent/"private/derived/heading-chunks/boundary-audit.json", json_text({"metadata": metadata, **result["audit"]}).encode("utf-8"))
    update_manifest(directory)
    return {"metadata": metadata, "headings": result["audit"]["headings"], "distinctHeadingStarts": result["audit"]["distinctHeadingStarts"],
            "differingChapters": len(result["audit"]["differences"]), "unalignedHeadings": 0}


def pack_corpus(source: Path, directory=DEFAULT_DATA):
    raw = source.read_bytes()
    corpus = json.loads(raw)
    if not isinstance(corpus, dict) or not corpus:
        raise ValueError("INVALID_CORPUS")
    for book in corpus.values():
        if not isinstance(book.get("name"), str) or not isinstance(book.get("chapters"), dict):
            raise ValueError("INVALID_CORPUS")
        for c, verses in book["chapters"].items():
            if not c.isdigit() or int(c) < 1 or not isinstance(verses, dict):
                raise ValueError("INVALID_CORPUS")
            if any(not v.isdigit() or int(v) < 1 or not isinstance(t, str) or not t.strip() for v, t in verses.items()):
                raise ValueError("INVALID_CORPUS")
    atomic_write(directory/"corpus.json.gz", gzip.compress(raw, mtime=0))
    update_manifest(directory)
    return {"books": len(corpus), "corpusSha256": sha256(raw), "networkUsed": False}


async def build_index(*, live=False, directory=DEFAULT_DATA, cache=None, fetcher=None):
    corpus_raw = gzip.decompress((directory/"corpus.json.gz").read_bytes())
    corpus, headings = json.loads(corpus_raw), load_headings(directory)
    if sha256(corpus_raw) != headings["metadata"]["corpusSha256"]:
        raise ValueError("HEADING_CORPUS_MISMATCH")
    chunks = create_bible_chunks(corpus, headings["boundaries"])
    cache = cache or ROOT/"artifacts/rag-heading-v2/embeddings"
    cache.mkdir(parents=True, exist_ok=True)
    def cache_key(c):
        return sha256(f"{EMBEDDING_MODEL}:{EMBEDDING_DIMENSIONS}:{c['text']}")
    def cached(c):
        path = cache/(cache_key(c)+".json")
        if not path.is_file():
            return None
        row = read_json(path)
        vector = row.get("vector")
        if (row.get("textHash") != sha256(c["text"]) or row.get("model") != EMBEDDING_MODEL or not isinstance(vector, list)
                or len(vector) != 512 or any(type(n) not in (int, float) or not math.isfinite(n) for n in vector) or not any(vector)):
            raise ValueError("INVALID_CACHED_DOCUMENT_VECTOR")
        return vector
    vectors = {c["id"]: cached(c) for c in chunks}
    pending = [c for c in chunks if vectors[c["id"]] is None]
    plan = {"version": CHUNK_VERSION, "chunkCount": len(chunks), "sectionCount": len({c["sectionId"] for c in chunks}),
            "cachedDocuments": len(chunks)-len(pending), "newDocuments": len(pending),
            "newDocumentCharacters": sum(text_length(c["text"]) for c in pending), "model": EMBEDDING_MODEL, "dimensions": 512,
            "providerPayload": "KRV text and book/chapter label only; no RNKSV heading text or full text",
            "boundarySha256": headings["metadata"]["boundarySha256"]}
    if pending and not live:
        return {**plan, "complete": False, "networkCalls": 0, "requiresLive": True}
    transport, tokens, calls = None, 0, 0
    try:
        if pending:
            env = load_env()
            if not env.get("QWEN_API_KEY", "").strip():
                raise ValueError("RESEARCH_QWEN_API_KEY_REQUIRED")
            if fetcher is None:
                transport = ProviderTransport(env)
                fetcher = transport.request
            for i in range(0, len(pending), 10):
                batch = pending[i:i+10]
                response = await embed_texts([c["text"] for c in batch], env=env, fetcher=fetcher)
                calls += 1; tokens += response["inputTokens"]
                for c, vector in zip(batch, response["vectors"], strict=True):
                    vectors[c["id"]] = vector
                    atomic_write(cache/(cache_key(c)+".json"), json_text({"model": EMBEDDING_MODEL, "textHash": sha256(c["text"]), "vector": vector, "origin": "new-document-embedding"}).encode("utf-8"))
        raw_vectors = b"".join(quantize_vector(vectors[c["id"]]).tobytes() for c in chunks)
        lexical = BibleRetriever(corpus, chunks=chunks).serialize_lexical()
        metadata = {"version": CHUNK_VERSION, "model": EMBEDDING_MODEL, "dimensions": 512, "count": len(chunks),
                    "corpusSha256": sha256(corpus_raw), "chunkSha256": sha256("\n".join(c["id"]+"\n"+c["text"] for c in chunks)),
                    "boundarySha256": headings["metadata"]["boundarySha256"], "vectorSha256": sha256(raw_vectors),
                    "lexicalSha256": lexical_fingerprint(lexical), "compression": "gzip-json-and-raw-int8",
                    "quantization": "int8-per-vector; cosine-normalized at query time",
                    "source": "KRV text segmented by RNKSV heading coordinates; no heading labels, RNKSV text, generated questions, answers, or user records."}
        atomic_write(directory/"vectors.i8", raw_vectors)
        atomic_write(directory/"lexical.json.gz", gzip.compress(json_text(lexical).encode("utf-8"), mtime=0))
        atomic_write(directory/"index.json", json_text(metadata).encode("utf-8"))
        update_manifest(directory)
        return {**plan, "complete": True, "networkCalls": calls, "inputTokens": tokens}
    finally:
        if transport is not None:
            await transport.aclose()
