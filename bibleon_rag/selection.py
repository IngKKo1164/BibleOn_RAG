"""Budgeting, compact candidate aliases, reranking and balanced evidence excerpts."""
from __future__ import annotations

import re
from dataclasses import dataclass

from .common import BibleProviderError, is_integer, json_text, object_schema, text_length, unique
from .discovery import lexical_hints
from .passages import chapter_at, text_at, verse_at
from .repair import repair_requirements_schema
from .retrieval import RAG_LIMITS, candidate_excerpt, retrieval_variants

SCREEN_LIMITS = {"thresholdCharacters": 18000, "previewCharacters": 160, "perPart": 12, "total": 24}
INPUT_BUDGETS = {"standard": 128, "complex": 160}


def first_pass_limit(question, parts=()):
    return 160 if len(parts) > 1 or re.search(r"비교|차이|관계|모순|각각|원어|어원|배경|역사", question) else 128


def budget_candidate_pool(pool, question, parts=()):
    limit = first_pass_limit(question, parts)
    if len(pool["hits"]) <= limit:
        return pool
    keep = {h["chunk"]["id"] for h in pool["hits"][:limit]}
    keep.update(h["chunk"]["id"] for h in pool["hits"] if any(p["path"] == "explicit" for p in h.get("paths", [])))
    return {**pool, "hits": [h for h in pool["hits"] if h["chunk"]["id"] in keep],
            "candidates": [c for c in pool["candidates"] if c["id"] in keep]}


def compact_answer_input(value):
    evidence = value.get("evidence")
    if not isinstance(evidence, list) or text_length(json_text(value)) < 3200:
        return {"input": value, "compact": False}
    if not all(isinstance(v.get("id"), str) and isinstance(v.get("text"), str) for v in evidence):
        return {"input": value, "compact": False}
    result = {**value, "evidence": [[v["id"], v["text"]] for v in evidence]}
    if "questionParts" in value and "partEvidenceIds" in value:
        result.pop("questionCoverage", None)
    if "sectionContext" in value:
        fields = ("bookId", "chapter", "start", "end", "startReference", "endReference", "complete", "omittedVerses")
        result["sectionContext"] = [{k: s[k] for k in fields if k in s} for s in value["sectionContext"]]
    for key in ("history", "sectionContext", "crossReferences", "studySources", "contextVerseIds"):
        if isinstance(result.get(key), list) and not result[key]:
            del result[key]
    if result.get("previousEvidence", False) is None:
        del result["previousEvidence"]
    return {"input": result, "compact": True}


@dataclass
class CandidateWire:
    candidates: list
    ids: list
    canonical: dict
    aliases: dict

    def decode(self, key):
        if not isinstance(key, str) or key not in self.canonical:
            raise BibleProviderError("AI_SELECTION_INVALID")
        return self.canonical[key]


def candidate_wire(pool, *, preview=False, question="", needs=(), preview_characters=160):
    canonical, aliases, candidates = {}, {}, []
    query = question + " " + lexical_hints(question)
    from .evidence_policy import preview_budgets, need_preview
    budgets = preview_budgets(pool, preview_characters) if preview and needs else {}
    for i, hit in enumerate(pool["hits"]):
        c, key = hit["chunk"], f"C{i+1}"
        canonical[key], aliases[c["id"]] = c["id"], key
        p = candidate_excerpt(c, query, max_characters=preview_characters, queries=retrieval_variants(query)) if preview else None
        if preview and needs:
            p = need_preview(c, question, needs, budgets[c["id"]])
        if p is not None and not p["excerpt"] and not needs:
            v = c["verses"][0]
            p = {"excerpt": f"{v['verse']} {v['text']}", "complete": len(c["verses"]) == 1}
        candidates.append([key, c["reference"], 0 if p and not p["complete"] else 1,
                           p["excerpt"] if p is not None else "\n".join(f"{v['verse']} {v['text']}" for v in c["verses"])])
    return CandidateWire(candidates, list(canonical), canonical, aliases)


def screening_schema(parts, ids):
    return {**object_schema({"byPart": object_schema({p["partId"]: {"type": "array", "maxItems": 12,
            "items": {"$ref": "#/$defs/candidateId"}} for p in parts})}),
            "$defs": {"candidateId": {"type": "string", "enum": ids}}}


def _ids_valid(value, maximum):
    return isinstance(value, list) and len(value) <= maximum and all(isinstance(i, str) for i in value) and len(set(value)) == len(value)


def decode_screening(raw, parts, wire, *, recover_duplicates=False, balanced=False):
    if not isinstance(raw, dict) or not isinstance(raw.get("byPart"), dict) or len(raw["byPart"]) != len(parts):
        raise BibleProviderError("AI_SELECTION_INVALID")
    selected, groups = [], []
    for p in parts:
        ids = raw["byPart"].get(p["partId"])
        if recover_duplicates and isinstance(ids, list) and len(ids) <= 12 and all(isinstance(i, str) for i in ids):
            ids = unique(ids)
        if not _ids_valid(ids, 12):
            raise BibleProviderError("AI_SELECTION_INVALID")
        decoded = [wire.decode(i) for i in ids]
        selected.extend(decoded); groups.append(decoded)
    if balanced:
        return unique(row[i] for i in range(12) for row in groups if i < len(row))[:24]
    selected = unique(selected)
    if len(selected) > 24:
        raise BibleProviderError("AI_SELECTION_INVALID")
    return selected


def decode_reranking(raw, wire, *, recover_duplicates=False):
    if (not isinstance(raw, dict) or not isinstance(raw.get("rankedIds"), list)
            or not isinstance(raw.get("evidenceByPart"), dict) or not isinstance(raw.get("focus"), list)):
        raise BibleProviderError("AI_SELECTION_INVALID")
    try:
        result = {**raw, "rankedIds": [wire.decode(i) for i in raw["rankedIds"]],
                  "evidenceByPart": {k: [wire.decode(i) for i in v] for k, v in raw["evidenceByPart"].items() if isinstance(v, list)},
                  "focus": [{**f, "chunkId": wire.decode(f.get("chunkId"))} for f in raw["focus"]]}
        if len(result["evidenceByPart"]) != len(raw["evidenceByPart"]):
            raise BibleProviderError("AI_SELECTION_INVALID")
        if recover_duplicates:
            if len(result["rankedIds"]) <= 12:
                result["rankedIds"] = unique(result["rankedIds"])
            result["evidenceByPart"] = {k: unique(v) if len(v) <= 3 else v for k, v in result["evidenceByPart"].items()}
            for f in result["focus"]:
                values = f.get("verses")
                if isinstance(values, list) and len(values) <= 4 and all(is_integer(n) for n in values):
                    f["verses"] = unique(values)
        if isinstance(raw.get("requirements"), list):
            result["requirements"] = [{**r, "anchors": [{**a, "chunkId": wire.decode(a.get("chunkId"))} for a in r["anchors"]]
                                       if isinstance(r.get("anchors"), list) else None} for r in raw["requirements"]]
        return result
    except (TypeError, AttributeError) as error:
        raise BibleProviderError("AI_SELECTION_INVALID") from error


def reranking_schema(parts, ids, *, repair=False, strict_hints=False, focus_verses=None):
    item = {"$ref": "#/$defs/candidateId"}
    props = {"route": {"type": "string", "enum": ["bible", "ambiguous", "outOfScope", "greeting", "safety"]},
             "rankedIds": {"type": "array", "maxItems": 12, "items": item},
             "evidenceByPart": object_schema({p["partId"]: {"type": "array", "maxItems": 3, "items": item} for p in parts}),
             "focus": {"type": "array", "maxItems": 6, "items": object_schema({"chunkId": item,
                       "verses": {"type": "array", "minItems": 1, "maxItems": 4, "items": {"type": "integer", "minimum": 1}}})},
             "studyTerms": {"type": "array", "maxItems": 3, "items": {"type": "string", "maxLength": 48}}}
    if strict_hints:
        props["studyTerms"]["items"]["pattern"] = "^[a-zA-Z][a-zA-Z '-]{0,47}$"
    if focus_verses:
        if set(focus_verses) != set(ids):
            raise ValueError("FOCUS_SCHEMA_CANDIDATE_MISMATCH")
        choices = []
        for key in ids:
            numbers = sorted(set(focus_verses[key]))
            if not numbers or any(not is_integer(n) or n < 1 for n in numbers):
                raise ValueError("FOCUS_SCHEMA_VERSES_INVALID")
            # Continuous ranges avoid repeating thousands of verse enum values.
            number_schema = {"type": "integer", **({"minimum": numbers[0], "maximum": numbers[-1]}
                if numbers[-1]-numbers[0]+1 == len(numbers) else {"enum": numbers})}
            choices.append(object_schema({"chunkId": {"type": "string", "enum": [key]},
                "verses": {"type": "array", "minItems": 1, "maxItems": 4, "items": number_schema}}))
        props["focus"]["items"] = {"anyOf": choices}
    if repair:
        props["requirements"] = repair_requirements_schema(parts, item)
    return {**object_schema(props), "$defs": {"candidateId": {"type": "string", **({"enum": ids} if ids else {})}}}


def validate_reranking(raw, parts, retrieval):
    def invalid(reason="SELECTION_STRUCTURE"):
        error = BibleProviderError("AI_SELECTION_INVALID")
        error.diagnostic = reason
        raise error
    chunks = {h["chunk"]["id"]: h["chunk"] for h in retrieval["hits"]}
    if (not isinstance(raw, dict) or raw.get("route") not in ("bible", "ambiguous", "outOfScope", "greeting", "safety")
            or not _ids_valid(raw.get("rankedIds"), 12) or any(i not in chunks for i in raw["rankedIds"])
            or not isinstance(raw.get("evidenceByPart"), dict) or len(raw["evidenceByPart"]) != len(parts)
            or not isinstance(raw.get("focus"), list) or len(raw["focus"]) > 6
            or not isinstance(raw.get("studyTerms"), list) or len(raw["studyTerms"]) > 3):
        invalid()
    selected, coverage = [], []
    for p in parts:
        ids = raw["evidenceByPart"].get(p["partId"])
        if not _ids_valid(ids, 3) or any(i not in raw["rankedIds"] for i in ids):
            invalid("SELECTION_MEMBERSHIP")
        selected.extend(ids); coverage.append({**p, "chunkIds": ids})
    selected = unique(selected)
    if len(selected) > 6:
        invalid()
    focus = {}
    for f in raw["focus"]:
        if (not isinstance(f, dict) or f.get("chunkId") not in raw["rankedIds"]
                or not isinstance(f.get("verses"), list) or not 1 <= len(f["verses"]) <= 4
                or any(not is_integer(n) for n in f["verses"]) or len(set(f["verses"])) != len(f["verses"])
                or any(not any(v["verse"] == n for v in chunks[f["chunkId"]]["verses"]) for n in f["verses"])):
            invalid("SELECTION_FOCUS_COORDINATES")
        if f["chunkId"] in selected:
            focus[f["chunkId"]] = unique(focus.get(f["chunkId"], []) + f["verses"])
    if len(focus) != len(selected) or (raw["route"] != "bible" and (selected or raw["rankedIds"])):
        invalid("SELECTION_FOCUS_MISSING")
    # Optional English dictionary hints are not scripture evidence. Discard
    # malformed hints, while still failing closed on all passage coordinates.
    terms = [t for t in raw["studyTerms"] if isinstance(t, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z '-]{0,47}", t)]
    return {"route": raw["route"], "chunkIds": selected, "coverage": coverage, "focus": focus,
            "rankedIds": raw["rankedIds"], "studyTerms": terms,
            "missingParts": [p for p in coverage if not p["chunkIds"]] if raw["route"] == "bible" else []}


def focused_evidence(corpus, retrieval, plan, passage=(), limits=None, *, boundary_context=0,
                     required_focus=True):
    limits = limits or RAG_LIMITS
    by_id = {h["chunk"]["id"]: h["chunk"] for h in retrieval["hits"]}
    selected, queues, sections = {}, [], []
    chars = 0
    def add(v, required=False):
        nonlocal chars
        row = verse_at(corpus, v)
        if row["id"] in selected:
            return True
        if len(selected) >= limits["evidenceVerses"] or chars + text_length(row["text"]) > limits["evidenceCharacters"]:
            if required:
                raise ValueError("REQUIRED_EVIDENCE_LIMIT")
            return False
        selected[row["id"]] = row
        chars += text_length(row["text"])
        return True
    for v in passage:
        add(v, True)
    for key in plan["chunkIds"]:
        c, focus = by_id.get(key), plan["focus"].get(key)
        if not c or not focus:
            raise ValueError("UNRETRIEVED_FOCUS")
        for n in focus:
            if not any(v["verse"] == n for v in c["verses"]):
                raise ValueError("INVALID_FOCUS_VERSE")
            add({"bookId": c["bookId"], "chapter": c["chapter"], "verse": n}, required_focus)
        start, end = c.get("sectionStart", c["start"]), c.get("sectionEnd", c["end"])
        queue = [{"bookId": c["bookId"], "chapter": c["chapter"], "verse": v} for v in range(start, end+1) if text_at(corpus, c["bookId"], c["chapter"], v)]
        queue.sort(key=lambda v: (min(abs(v["verse"]-n) for n in focus), v["verse"]))
        queues.append(queue)
        sections.append({"id": c.get("sectionId", c["id"]), "bookId": c["bookId"], "chapter": c["chapter"], "start": start, "end": end, "selectedChunkId": key})
    while any(queues):
        for queue in queues:
            if queue:
                add(queue.pop(0))
    # Heading boundaries can separate an event from its immediately preceding
    # location/time. Add real neighbours only after the selected section, within
    # the same total budget; never change chunks, focus anchors or retrieval scores.
    if boundary_context:
        if not is_integer(boundary_context) or not 0 < boundary_context <= 2:
            raise ValueError("BOUNDARY_CONTEXT_LIMIT")
        book_rows = {}
        for s in sections:
            b = s["bookId"]
            if b not in book_rows:
                book_rows[b] = [(int(c), int(v)) for c, rows in sorted(corpus[b]["chapters"].items(), key=lambda x: int(x[0]))
                                for v in sorted(rows, key=int)]
            positions = book_rows[b]
            left = positions.index((s["chapter"], s["start"]))
            right = positions.index((s["chapter"], s["end"]))
            neighbours = positions[max(0, left-boundary_context):left] + positions[right+1:right+1+boundary_context]
            s["boundaryContextIds"] = []
            for c, v in neighbours:
                ref = {"bookId": b, "chapter": c, "verse": v}
                if add(ref):
                    s["boundaryContextIds"].append(f"{b}:{c}:{v}")
    book_order = {b: i for i, b in enumerate(corpus)}
    evidence = sorted(selected.values(), key=lambda v: (book_order[v["bookId"]], v["chapter"], v["verse"]))
    context = []
    for s in sections:
        chunk = by_id[s["selectedChunkId"]]
        present = [v for v in evidence if v["bookId"] == s["bookId"] and v["chapter"] == s["chapter"] and s["start"] <= v["verse"] <= s["end"]]
        count = sum(s["start"] <= int(v) <= s["end"] for v in chapter_at(corpus, s["bookId"], s["chapter"]))
        context.append({**s, "sectionId": s["id"],
                        "primaryVerseIds": [v["id"] for v in present if chunk["start"] <= v["verse"] <= chunk["end"]],
                        "contextVerseIds": [v["id"] for v in present if not chunk["start"] <= v["verse"] <= chunk["end"]],
                        "complete": len(present) == count, "omittedVerses": count-len(present)})
    anchors = [f"{by_id[i]['bookId']}:{by_id[i]['chapter']}:{n}" for i in plan["chunkIds"] for n in plan["focus"][i]]
    return {"evidence": evidence, "sections": context,
            "anchorIds": [a for a in anchors if a in selected],
            "omittedFocusIds": [a for a in anchors if a not in selected]}
