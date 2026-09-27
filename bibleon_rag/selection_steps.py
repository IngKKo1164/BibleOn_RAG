"""Graph nodes for discovery, candidate budgeting and one recovery pass."""
from .common import text_length, unique
from .discovery import DISCOVERY_LIMITS, discover_candidates, retrieve_broad_bible
from .prompt_text import RERANK_PROMPT, REPAIR_PROMPT, SCREEN_PROMPT
from .repair import assess_repair, recovery_rerank_pool, repair_enabled, repair_queries
from .selection import (SCREEN_LIMITS, budget_candidate_pool, candidate_wire, decode_reranking,
                        decode_screening, focused_evidence, reranking_schema, screening_schema, validate_reranking)
from .study import expand_one_hop
from .transport import embed_texts


async def retrieve_candidates(state, context):
    retrieval = await retrieve_broad_bible(corpus=context.corpus, question=context.question,
        previous_evidence=context.previous_evidence, passage=context.passage, env=context.env,
        fetcher=context.fetcher, retriever=context.retriever, study_index=context.get_study_index())
    context.observe({"stage": "retrieve", "result": {"candidateCount": len(retrieval["hits"]),
        "candidateIds": [h["chunk"]["id"] for h in retrieval["hits"]],
        "characters": sum(text_length(c["excerpt"]) for c in retrieval["candidates"]),
        "degraded": retrieval["degraded"], "queryVariants": 1}, "usage": {"embeddingInputTokens": retrieval["inputTokens"]}})
    return {"retrieval": retrieval}


def budget_candidates(state, context):
    retrieval = state["retrieval"]
    pool = budget_candidate_pool(retrieval, context.question, context.question_parts)
    context.observe({"stage": "candidate-budget", "result": {"broad": len(retrieval["hits"]),
        "firstPass": len(pool["hits"]), "deferred": len(retrieval["hits"])-len(pool["hits"])}})
    return {"screenedPool": pool, "needsScreening": sum(text_length(c["excerpt"]) for c in pool["candidates"]) > SCREEN_LIMITS["thresholdCharacters"]}


async def screen_candidates(state, context):
    retrieval = state["retrieval"]
    wire = candidate_wire(state["screenedPool"], preview=True, question=retrieval.get("query") or context.question)
    payload = {"question": context.question, "questionParts": context.question_parts, "candidates": wire.candidates}
    if context.previous_evidence:
        payload["previousQuestion"] = context.previous_evidence["question"]
    response = await context.call("screen", SCREEN_PROMPT, payload, screening_schema(context.question_parts, wire.ids))
    ids = decode_screening(response["result"], context.question_parts, wire)
    keep = set(ids)
    pool = {**retrieval, "hits": [h for h in retrieval["hits"] if h["chunk"]["id"] in keep],
            "candidates": [c for c in retrieval["candidates"] if c["id"] in keep]} if ids else retrieval
    context.observe({"stage": "screening", "result": {"candidateIds": [h["chunk"]["id"] for h in pool["hits"]],
                     "candidateCount": len(pool["hits"]), "previewCharacters": sum(text_length(c[3]) for c in wire.candidates), "fallback": not ids}})
    return {"screenedPool": pool}


async def rerank_pool(pool, recovery, context):
    repair = repair_enabled(context.env)
    wire = candidate_wire(pool)
    payload = {"question": context.question, "questionParts": context.question_parts, "candidates": wire.candidates}
    if recovery:
        payload.update(missingParts=[p["partId"] for p in recovery["missingParts"]],
            previouslySelected=[{"partId": p["partId"], "chunkIds": [wire.aliases[i] for i in p["chunkIds"] if i in wire.aliases]} for p in recovery["coverage"]])
        if repair and "unresolved" in recovery:
            payload["unresolvedRequirements"] = [{k: r[k] for k in ("partId", "need", "status")} for r in recovery["unresolved"]]
    if context.previous_evidence:
        payload["previousQuestion"] = context.previous_evidence["question"]
    response = await context.call("rerank", RERANK_PROMPT + ("\n"+REPAIR_PROMPT if repair else ""), payload,
                                  reranking_schema(context.question_parts, wire.ids, repair=repair))
    decoded = decode_reranking(response["result"], wire)
    plan = validate_reranking(decoded, context.question_parts, pool)
    return assess_repair(plan, decoded.get("requirements"), context.question_parts, pool) if repair else plan


async def rerank_candidates(state, context):
    pool, retrieval = state["screenedPool"], state["retrieval"]
    plan = await rerank_pool(pool, None, context)
    recovery = ((plan["route"] == "bible" or (repair_enabled(context.env) and plan["route"] == "ambiguous")) and bool(plan["missingParts"])) or (pool is not retrieval and plan["route"] in ("ambiguous", "outOfScope"))
    return {"plan": plan, "recoveryCount": 0, "needsRecovery": recovery}


async def recover_candidates(state, context):
    retrieval, plan = state["retrieval"], state["plan"]
    repair = repair_enabled(context.env)
    if plan["route"] != "bible":
        plan = {**plan, "missingParts": context.question_parts}
    queries = repair_queries(plan, context.question_parts, expand=context.env.get("BIBLE_REPAIR_QUERY_EXPANSION") != "off") if repair else [
        {"query": p["question"], "partId": p["partId"], "expanded": False} for p in plan["missingParts"]]
    expanded = [q for q in queries if q["expanded"]]
    vectors, embedding_tokens, embedding_calls, degraded = [], 0, 0, False
    if expanded:
        context.check_deadline(); embedding_calls = 1
        try:
            response = await embed_texts([q["query"] for q in expanded], env=context.env, fetcher=context.fetcher, kind="query")
            vectors, embedding_tokens = response["vectors"], response["inputTokens"]
        except Exception:
            context.check_deadline(); degraded = True
    additional, branches = [], []
    for q in queries:
        context.check_deadline()
        vector = vectors[expanded.index(q)] if q["expanded"] and vectors else None
        more = discover_candidates(retrieval["index"], context.corpus, q["query"], retry=True, vector=vector,
                                   scope_question=retrieval.get("query") or context.question)
        rows = more["hits"][:24]
        branches.append(rows); additional.extend(rows)
    study_index = context.get_study_index()
    if study_index is not None and plan["chunkIds"]:
        primary = focused_evidence(context.corpus, retrieval, plan, limits={"evidenceCharacters": 6000, "evidenceVerses": 96})["evidence"]
        linked = expand_one_hop(context.corpus, primary, context.question, study_index)["linked"]
        neighbors = []
        for v in linked:
            c = next((c for c in retrieval["index"].chunks if c["bookId"] == v["bookId"] and c["chapter"] == v["chapter"] and c["start"] <= v["verse"] <= c["end"]), None)
            if c is not None:
                neighbors.append({"chunk": c, "score": 0, "paths": [{"path": "one-hop-recovery", "rank": 1}]})
        branches.append(neighbors); additional.extend(neighbors)
    known, added = {h["chunk"]["id"] for h in retrieval["hits"]}, []
    cap = DISCOVERY_LIMITS["candidates"] + (48 if repair else 0)
    for h in additional:
        if h["chunk"]["id"] in known:
            continue
        if len(added) >= 48 or len(known) >= cap:
            break
        known.add(h["chunk"]["id"]); added.append(h)
    retrieval = {**retrieval, "hits": retrieval["hits"]+added,
        "candidates": retrieval["candidates"] + [{"id": h["chunk"]["id"], "reference": h["chunk"]["reference"],
            "complete": True, "excerpt": "\n".join(f"{v['verse']} {v['text']}" for v in h["chunk"]["verses"])} for h in added],
        "inputTokens": retrieval["inputTokens"]+embedding_tokens, "embeddingCalls": (retrieval.get("embeddingCalls") or 1)+embedding_calls,
        "degraded": retrieval["degraded"] or degraded}
    if sum(text_length(c["excerpt"]) for c in retrieval["candidates"]) > DISCOVERY_LIMITS["characters"]:
        raise ValueError("DISCOVERY_CHARACTER_LIMIT")
    pool = recovery_rerank_pool(retrieval, plan, branches) if repair else retrieval
    context.observe({"stage": "retrieval-recovery", "result": {"attempt": 1, "method": "repair-inspired-v1" if repair else "legacy",
        "missingParts": [p["partId"] for p in plan["missingParts"]],
        "targets": [{k: q[k] for k in ("partId", "need", "expanded") if k in q} for q in queries], "added": len(added),
        "candidateIds": [h["chunk"]["id"] for h in retrieval["hits"]], "rerankIds": [h["chunk"]["id"] for h in pool["hits"]],
        "embeddingCalls": embedding_calls, "embeddingDegraded": degraded}, "usage": {"embeddingInputTokens": embedding_tokens}})
    return {"retrieval": retrieval, "plan": plan, "recoveryCount": 1, "recoveryPool": pool}


async def rerank_recovered(state, context):
    retrieval, plan = state["retrieval"], state["plan"]
    recovered = await rerank_pool(state.get("recoveryPool") or retrieval, plan, context)
    if repair_enabled(context.env):
        return {"plan": recovered}
    if recovered["route"] == "bible":
        coverage = [p if p["chunkIds"] else next(c for c in plan["coverage"] if c["partId"] == p["partId"]) for p in recovered["coverage"]]
        chosen = []
        for i in range(3):
            for part in coverage:
                if i < len(part["chunkIds"]):
                    key = part["chunkIds"][i]
                    if key not in chosen and len(chosen) < 6:
                        chosen.append(key)
        focus = [{"chunkId": key, "verses": recovered["focus"].get(key) or plan["focus"].get(key)} for key in chosen]
        plan = validate_reranking({"route": "bible", "rankedIds": unique(chosen+recovered["rankedIds"]+plan["rankedIds"])[:12],
            "evidenceByPart": {p["partId"]: [key for key in p["chunkIds"] if key in chosen] for p in coverage}, "focus": focus,
            "studyTerms": unique(recovered["studyTerms"]+plan["studyTerms"])[:3]}, context.question_parts, retrieval)
    else:
        plan = recovered
    return {"plan": plan}


def select_evidence(state, context):
    retrieval, plan, count = state["retrieval"], state["plan"], state.get("recoveryCount", 0)
    report = {"chunkIds": plan["chunkIds"], "rankedIds": plan["rankedIds"], "parts": len(plan["coverage"]),
              "missingParts": [p["partId"] for p in plan["missingParts"]], "recoveryCount": count}
    if "requirements" in plan:
        report["requirements"] = [{k: r[k] for k in ("partId", "need", "status", "importance")} for r in plan["requirements"]]
    context.observe({"stage": "selection", "result": report})
    if plan["route"] != "bible" or not plan["chunkIds"]:
        return {"selected": {"plan": plan, "retrieval": retrieval, "recoveryCount": count}}
    selected = focused_evidence(context.corpus, retrieval, plan, context.passage)
    context.observe({"stage": "evidence", "result": {"verses": len(selected["evidence"]), "anchorIds": selected["anchorIds"],
        "retainedVerseIds": [v["id"] for v in selected["evidence"]], "incompleteSections": sum(not s["complete"] for s in selected["sections"])}})
    return {"selected": {"plan": plan, "retrieval": retrieval, **selected, "recoveryCount": count}}
