"""One bounded REPAIR-inspired search correction; not a trained PSR/SAR model."""
from __future__ import annotations

from .common import BibleProviderError, is_integer, object_schema, text_length, unique

REPAIR_LIMITS = {"requirements": 6, "queries": 2, "newCandidates": 48, "rerankCandidates": 48}


def repair_enabled(env):
    return env.get("BIBLE_RETRIEVAL_REPAIR") != "off"


def repair_requirements_schema(parts, candidate_id):
    return {"type": "array", "maxItems": 6, "items": object_schema({
        "partId": {"type": "string", "enum": [p["partId"] for p in parts]},
        "need": {"type": "string", "minLength": 1, "maxLength": 120},
        "importance": {"type": "string", "enum": ["essential", "supplementary"]},
        "status": {"type": "string", "enum": ["supported", "refuted", "missing", "conflict"]},
        "anchors": {"type": "array", "maxItems": 2, "items": object_schema({
            "chunkId": candidate_id, "verse": {"type": "integer", "minimum": 1},
            "quote": {"type": "string", "minLength": 2, "maxLength": 120}})},
        "query": {"type": "string", "maxLength": 160}})}


def assess_repair(plan, requirements, parts, retrieval):
    def invalid():
        raise BibleProviderError("AI_REPAIR_ASSESSMENT_INVALID")
    if not isinstance(requirements, list) or len(requirements) > 6:
        invalid()
    chunks = {h["chunk"]["id"]: h["chunk"] for h in retrieval["hits"]}
    part_ids, seen = {p["partId"] for p in parts}, set()
    for r in requirements:
        if (not isinstance(r, dict) or r.get("partId") not in part_ids
                or not isinstance(r.get("need"), str) or not r["need"].strip() or text_length(r["need"]) > 120
                or r.get("importance") not in ("essential", "supplementary")
                or r.get("status") not in ("supported", "refuted", "missing", "conflict")
                or not isinstance(r.get("query"), str) or text_length(r["query"]) > 160
                or not isinstance(r.get("anchors"), list) or len(r["anchors"]) > 2):
            invalid()
        key = r["partId"] + ":" + r["need"].strip()
        if key in seen:
            invalid()
        seen.add(key)
        supported = r["status"] in ("supported", "refuted")
        if (not r["anchors"] or r["query"] != "") if supported else bool(r["anchors"]):
            invalid()
        for a in r["anchors"]:
            if not isinstance(a, dict):
                invalid()
            text = next((v["text"] for v in chunks.get(a.get("chunkId"), {}).get("verses", []) if v["verse"] == a.get("verse")), None)
            coverage = next((p["chunkIds"] for p in plan["coverage"] if p["partId"] == r["partId"]), [])
            if (a.get("chunkId") not in coverage or a.get("verse") not in plan["focus"].get(a.get("chunkId"), [])
                    or not isinstance(a.get("quote"), str) or len(a["quote"].strip()) < 2 or text_length(a["quote"]) > 120
                    or not text or a["quote"] not in text):
                invalid()
    if plan["route"] in ("bible", "ambiguous"):
        if any(not any(r["partId"] == p["partId"] for r in requirements) for p in parts):
            invalid()
    elif requirements:
        invalid()
    unresolved = [r for r in requirements if r["status"] in ("missing", "conflict")]
    coverage = []
    for p in plan["coverage"]:
        rows = [r for r in requirements if r["partId"] == p["partId"]]
        blocked = any(r["importance"] == "essential" and r["status"] in ("missing", "conflict") for r in rows)
        accepted = {a["chunkId"] for r in rows for a in r["anchors"]}
        coverage.append({**p, "chunkIds": [] if blocked else [i for i in p["chunkIds"] if i in accepted]})
    chunk_ids = unique(i for p in coverage for i in p["chunkIds"])
    return {**plan, "coverage": coverage, "chunkIds": chunk_ids,
            "focus": {i: ns for i, ns in plan["focus"].items() if i in chunk_ids},
            "requirements": requirements, "unresolved": unresolved,
            "missingParts": [p for p in coverage if not p["chunkIds"] or any(r["partId"] == p["partId"] for r in unresolved)]}


def repair_queries(plan, parts, *, expand=True):
    targets = sorted(plan.get("unresolved", []), key=lambda r: -((4 if r["importance"] == "essential" else 0) + (2 if r["status"] == "conflict" else 1)))
    seen, queries = set(), []
    for r in targets:
        part = next((p for p in parts if p["partId"] == r["partId"]), None)
        if part is None:
            continue
        hint = r["query"].strip() if expand else ""
        query = part["question"] + "\n검색 단서: " + hint if hint else part["question"]
        if query in seen:
            continue
        seen.add(query)
        queries.append({"partId": r["partId"], "need": r["need"], "query": query, "expanded": bool(hint)})
        if len(queries) == 2:
            break
    if not queries:
        queries = [{"partId": p["partId"], "query": p["question"], "expanded": False} for p in plan["missingParts"][:2]]
    return queries


def recovery_rerank_pool(retrieval, plan, branches):
    by_id = {h["chunk"]["id"]: h for h in retrieval["hits"]}
    keep = {}
    def add(key):
        if key in by_id and len(keep) < 48:
            keep[key] = None
    for key in plan["chunkIds"] + plan["rankedIds"][:12]:
        add(key)
    for i in range(12):
        for rows in branches:
            if i < len(rows):
                add(rows[i]["chunk"]["id"])
    for h in retrieval["hits"]:
        add(h["chunk"]["id"])
    return {**retrieval, "hits": [by_id[i] for i in keep], "candidates": [c for c in retrieval["candidates"] if c["id"] in keep]}
