"""Broad branch union before reranking, without generated question indexing."""
from __future__ import annotations

import re

from .common import canonical_id_sort_key, text_length, unique
from .passages import chapter_at, parse_bible_references, text_at
from .retrieval import (BibleRetriever, load_bible_retriever, rag_terms, retrieval_query,
                        retrieval_scope, retrieval_variants)
from .transport import embed_texts

DISCOVERY_LIMITS = {"branch": 100, "candidates": 420, "characters": 760000, "selected": 6, "final": 12}
HINTS = [(r"기도|간구", "기도 간구 빌매 구하"), (r"빌려|대여", "빌리 빌려"), (r"도끼", "도끼 부"),
         (r"맹인|시각\s*장애|보지\s*못", "소경 눈 맹인"), (r"피부병|나병", "문둥병"),
         (r"잘라|절단|자르", "찍어 버리 빼어"), (r"회복", "돌이키 곤경 회복"), (r"치료|낫게|고치", "고치 낫 나으"),
         (r"자녀|아이들", "자녀 자식"), (r"정직", "정직 진실"), (r"화해", "화목 화평"), (r"도움|돕", "도우 구제"),
         (r"걱정|불안", "염려 근심"), (r"돈|재산", "재물 은 돈"), (r"병사|군인", "군사"),
         (r"죽음|사망", "죽 사망"), (r"부활|다시\s*살", "살리 부활"), (r"노예", "종 노예"),
         (r"감옥", "옥 갇히"), (r"세금", "세리 세금 세")]


def lexical_hints(question):
    return " ".join(unique(t for p, t in HINTS if re.search(p, question)))


def _window_view(index, corpus):
    if index.window_view is not None:
        return index.window_view
    parents, inverse, docs = {}, {}, []
    for i, c in enumerate(index.chunks):
        parents[c.get("sectionId", c["id"])] = {**c, "start": c.get("sectionStart", c["start"]), "end": c.get("sectionEnd", c["end"])}
        for v in c["verses"]:
            inverse.setdefault(f"{c['bookId']}:{c['chapter']}:{v['verse']}", []).append(i)
    for p in parents.values():
        for v in range(p["start"], p["end"] + 1):
            if not text_at(corpus, p["bookId"], p["chapter"], v):
                continue
            verses = []
            for n in range(max(p["start"], v-2), min(p["end"], v+2) + 1):
                text = text_at(corpus, p["bookId"], p["chapter"], n)
                if text:
                    verses.append({"verse": n, "text": text})
            docs.append({**p, "id": f"{p['bookId']}:{p['chapter']}:{v}", "verses": verses,
                         "text": "\n".join(x["text"] for x in verses)})
    index.window_view = (BibleRetriever(corpus, chunks=docs), docs, inverse)
    return index.window_view


def lexical_view_branches(index, corpus, question, *, retry=False, scope_question=None):
    scope = retrieval_scope(corpus, question, coordinated_books=True)
    books = set(scope["bookIds"] if scope_question is None else retrieval_scope(corpus, scope_question, coordinated_books=True)["bookIds"])
    query = scope["contentQuery"] or question
    hint, limit = lexical_hints(query), 32 if retry else 16
    augmented = f"{query} {hint}" if hint else query
    eligible = lambda i: not books or index.chunks[i]["bookId"] in books
    branches = []
    if hint:
        branches.append(("modern-vocabulary", [r for r in index.lexical(augmented) if eligible(r["index"])][:limit]))
    view, docs, inverse = _window_view(index, corpus)
    seen, rows = set(), []
    for r in view.lexical(augmented):
        for i in inverse.get(docs[r["index"]]["id"], []):
            if i not in seen and eligible(i):
                seen.add(i); rows.append({"index": i, "score": r["score"]})
        if len(rows) >= limit:
            break
    branches.append(("five-verse-window", rows[:limit]))
    if retry:
        important = " ".join(w for w in unique(t[2:] for t in rag_terms(query) if t.startswith("w:")) if len(w) > 1)
        if important:
            branches.append(("missing-part", [r for r in index.focused(f"{important} {hint}") if eligible(r["index"])][:16]))
    return branches


def discover_candidates(index, corpus, question, *, vector=None, refs=None, extra_branches=(),
                        prepared_query=None, retry=False, views=True, scope_question=None):
    query = retrieval_query(question) if prepared_query is None else prepared_query
    scope = retrieval_scope(corpus, query if scope_question is None else scope_question, coordinated_books=True)
    lists = index.rankings(query, vector=vector, refs=refs or [], book_scope=scope["bookIds"])
    branches = [("lexical", lists["lex"]), ("focused", lists["focus"]), ("dense", lists["dense"])]
    if views:
        branches += lexical_view_branches(index, corpus, query, retry=retry, scope_question=scope_question)
    branches += list(extra_branches)
    found = {}
    def add(i, path, rank):
        c = index.chunks[i]
        if scope["bookIds"] and c["bookId"] not in scope["bookIds"]:
            return
        hit = found.setdefault(c["id"], {"chunk": c, "score": 0, "paths": []})
        hit["score"] += 1/(10+rank+1)
        hit["paths"].append({"path": path, "rank": rank+1})
    for ref in refs or []:
        for i, c in enumerate(index.chunks):
            if c["bookId"] == ref["bookId"] and c["chapter"] == ref["chapter"] and c["start"] <= ref["verse"] <= c["end"]:
                add(i, "explicit", 0)
    for path, rows in branches:
        for n, r in enumerate(rows[:100]):
            add(r["index"], path, n)
    for n, variant in enumerate(retrieval_variants(query)):
        s = retrieval_scope(corpus, variant, coordinated_books=True)
        books = set(scope["bookIds"] if scope_question is not None else s["bookIds"] or scope["bookIds"])
        rows = [r for r in index.lexical(s["contentQuery"] or variant) if not books or index.chunks[r["index"]]["bookId"] in books][:8]
        for i, r in enumerate(rows):
            add(r["index"], f"clause-{n}", i)
    hits = sorted(found.values(), key=lambda h: (-h["score"], canonical_id_sort_key(h["chunk"]["id"])))
    if len(hits) > 420:
        raise ValueError("DISCOVERY_POOL_LIMIT")
    candidates = [{"id": h["chunk"]["id"], "reference": h["chunk"]["reference"],
                   "excerpt": "\n".join(f"{v['verse']} {v['text']}" for v in h["chunk"]["verses"]), "complete": True} for h in hits]
    if sum(text_length(c["excerpt"]) for c in candidates) > 760000:
        raise ValueError("DISCOVERY_CHARACTER_LIMIT")
    return {"hits": hits, "candidates": candidates, "query": query, "bookScope": scope["bookIds"]}


def independent_study_branch(index, study_index, question):
    from .study import search_independent_study, study_query_terms
    if not study_index or not re.search(r"배경|역사|당시|문화|원어|어원|주석|사전|교리|신학|\b[GH]\d{1,5}\b", question, re.I | re.A):
        return []
    matches = search_independent_study(study_index, study_query_terms(question), question)[:4]
    seen, rows = set(), []
    for match in matches:
        ids = {study_index.data["verseIds"][n] for n in match["row"]["refs"]}
        per_source = 0
        for i, c in enumerate(index.chunks):
            if i in seen or not any(f"{c['bookId']}:{c['chapter']}:{v['verse']}" in ids for v in c["verses"]):
                continue
            seen.add(i); rows.append({"index": i, "score": match["score"]}); per_source += 1
            if per_source >= 8:
                break
    return rows[:24]


async def retrieve_broad_bible(*, corpus, question, env, fetcher, previous_evidence=None,
                              passage=None, retriever=None, extra_branches=(), study_index=None):
    index = retriever if retriever is not None else load_bible_retriever(corpus)
    query = retrieval_query(question, previous_evidence)
    vector, input_tokens, degraded = None, 0, False
    try:
        response = await embed_texts([query], env=env, fetcher=fetcher, kind="query")
        vector, input_tokens = response["vectors"][0], response["inputTokens"]
    except Exception:
        # asyncio cancellation inherits BaseException, so it never degrades to lexical search.
        degraded = True
    followup = previous_evidence.get("passages", []) if previous_evidence and query != retrieval_query(question) else []
    refs = parse_bible_references(corpus, question)["refs"] + (passage or []) + followup
    independent = independent_study_branch(index, study_index, query)
    result = discover_candidates(index, corpus, query, vector=vector, refs=refs,
                                 extra_branches=[*extra_branches, ("independent-study-reference", independent)], prepared_query=query)
    return {**result, "index": index, "vector": vector, "inputTokens": input_tokens, "embeddingCalls": 1, "degraded": degraded}
