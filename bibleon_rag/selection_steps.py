"""Graph nodes for discovery, candidate budgeting and one recovery pass."""
from .common import BibleProviderError, text_length, unique
from .observability import snapshot
from .passages import verse_at
from .discovery import DISCOVERY_LIMITS, discover_candidates, retrieve_broad_bible
from .prompt_text import RERANK_PROMPT, REPAIR_PROMPT, SCREEN_PROMPT
from .repair import assess_repair, recovery_rerank_pool, repair_enabled, repair_queries
from .selection import (SCREEN_LIMITS, budget_candidate_pool, candidate_wire, decode_reranking,
                        decode_screening, focused_evidence, reranking_schema, screening_schema, validate_reranking)
from .study import expand_one_hop
from .transport import embed_texts
from .evidence_policy import (enabled, need_queries, balanced_pool, protected_candidates, bounded_need_selection,
                              need_parts, merge_need_selection, expand_discovery)
from .selection import first_pass_limit
from .evidence_policy import reliability
from .latency import latency_mode
from .config import discovery_seed_budget, study_discovery_mode, focus_schema_mode, bilingual_background


async def retrieve_candidates(state, context):
    contextual = {}
    if state["conversation"] is not None:
        contextual = {"contextual_query": state["searchQuestion"], "context_refs": [
            verse_at(context.corpus, ref) for ref in state["conversation"]["evidenceRefs"]] if state["contextPlan"]["usesContext"] else []}
    if state["intentEnabled"]:
        contextual["contextual_query"] = state["searchQuestion"]
        # A model-created query may mention a book. Only the actual user query
        # may narrow retrieval; an interpretive anchor is not a restriction.
        contextual["scope_question"] = "" if state["scope"]["direct"] and (state["scope"]["crossrefs"] or enabled(state, context, 1)) else state["question"]
        contextual["original_query"] = state["contextPlan"]["query"] if state["contextPlan"] else state["question"]
    if enabled(state, context, 2):
        contextual["need_searches"] = need_queries(state["intentPlan"]["needs"], contextual["original_query"])
    if (state['intentEnabled'] and study_discovery_mode(context.env)=='planned'
            and bilingual_background(context.env)):
        contextual['study_queries'] = state['intentPlan'].get('backgroundQueries',[])
    retrieval = await retrieve_broad_bible(corpus=context.corpus, question=state["question"],
        previous_evidence=state["previousEvidence"], passage=state["passage"], env=context.env,
        fetcher=context.fetcher, retriever=context.get_retriever(), study_index=context.get_study_index(state["studyEnabled"]), **contextual)
    context.observe({"stage": "retrieve", "result": {"candidateCount": len(retrieval["hits"]),
        "candidateIds": [h["chunk"]["id"] for h in retrieval["hits"]],
        "characters": sum(text_length(c["excerpt"]) for c in retrieval["candidates"]),
        "degraded": retrieval["degraded"], "queryVariants": 1}, "usage": {"embeddingInputTokens": retrieval["inputTokens"]}})
    if enabled(state, context, 5):
        snapshot(context, "retrieved_raw", pool=retrieval)
        retrieval = expand_discovery(retrieval, context.corpus, state["scope"]["primary"], state["question"], context.get_study_index(state["studyEnabled"]),
                                     seed_budget=discovery_seed_budget(context.env))
    snapshot(context, "retrieved", pool=retrieval)
    return {"retrieval": retrieval}


def budget_candidates(state, context):
    from .profiles import profile_for
    if profile_for(context.env) is not None:
        from .research_steps import budget_profile_candidates
        return budget_profile_candidates(state, context)
    retrieval = state["retrieval"]
    if state.get('contextFirst') and not retrieval['degraded']:
        from .interpretation import expanded_review_context
        update = expanded_review_context(state, context)
        snapshot(context, 'context_review', verses=update['evidence'])
        context.observe({'stage':'selection-policy','mode':'review-first','merged':False})
        return {**update, 'contextFirst':True,'needsScreening':False,'fusedSelection':False}
    pool = (balanced_pool(retrieval, first_pass_limit(state["question"], state["questionParts"])) if enabled(state, context, 3)
            else budget_candidate_pool(retrieval, state["question"], state["questionParts"]))
    snapshot(context, "budgeted", pool=pool)
    context.observe({"stage": "candidate-budget", "result": {"broad": len(retrieval["hits"]),
        "firstPass": len(pool["hits"]), "deferred": len(retrieval["hits"])-len(pool["hits"])}})
    screening = sum(text_length(c["excerpt"]) for c in pool["candidates"]) > SCREEN_LIMITS["thresholdCharacters"]
    from .latency import conditional_fusion
    mode = latency_mode(state, context)
    fused = (screening and enabled(state, context, 3) and context.env.get('BIBLE_SET_SELECTION') != 'on'
             and (mode in {'fused', 'selection', 'strict'} or mode == 'conditional' and conditional_fusion(state, context)))
    context.observe({'stage': 'selection-policy', 'mode': mode, 'merged': fused})
    return {"screenedPool": pool, "needsScreening": screening and not fused, "fusedSelection": fused,
            **({'contextFirst':False} if 'contextFirst' in state else {})}


async def screen_candidates(state, context):
    retrieval = state["retrieval"]
    wire = candidate_wire(state["screenedPool"], preview=True, question=retrieval.get("query") or state["question"],
                          needs=state["intentPlan"]["needs"] if enabled(state, context, 3) else ())
    parts = need_parts(state, context) if enabled(state, context, 3) else state["questionParts"]
    snapshot(context, "screen_preview", pool=state["screenedPool"], wire=wire)
    payload = {"question": state["question"], "questionParts": parts, "candidates": wire.candidates}
    if state["intentPlan"]:
        payload["evidenceNeeds"] = state["intentPlan"]["needs"]
    if state["previousEvidence"]:
        payload["previousQuestion"] = state["previousEvidence"]["question"]
    response = await context.call(state, "screen", SCREEN_PROMPT, payload, screening_schema(parts, wire.ids))
    try:
        ids = decode_screening(response["result"], parts, wire,
                               recover_duplicates=state["intentEnabled"], balanced=enabled(state, context, 3))
    except BibleProviderError as error:
        if not state["intentEnabled"] or error.code != "AI_SELECTION_INVALID":
            raise
        context.observe({"stage": "diagnostic", "component": "screen", "code": "AI_SELECTION_INVALID"})
        ids = []
    fallback = not ids
    if fallback and state["intentEnabled"]:
        hits = state["screenedPool"]["hits"]
        anchors = [h["chunk"]["id"] for h in hits if any(p["path"] == "explicit" for p in h.get("paths", []))]
        ids = unique(anchors + [h["chunk"]["id"] for h in hits])[:SCREEN_LIMITS["total"]]
    if enabled(state, context, 3):
        ids = unique(protected_candidates(state["screenedPool"], 1) + ids)[:SCREEN_LIMITS["total"]]
    keep = set(ids)
    pool = {**retrieval, "hits": [h for h in retrieval["hits"] if h["chunk"]["id"] in keep],
            "candidates": [c for c in retrieval["candidates"] if c["id"] in keep]} if ids else (state["screenedPool"] if state["intentEnabled"] else retrieval)
    context.observe({"stage": "screening", "result": {"candidateIds": [h["chunk"]["id"] for h in pool["hits"]],
                     "candidateCount": len(pool["hits"]), "previewCharacters": sum(text_length(c[3]) for c in wire.candidates), "fallback": fallback}})
    snapshot(context, "screened", pool=pool)
    return {"screenedPool": pool}


async def rerank_pool(state, pool, recovery, context, *, preview=False):
    repair = repair_enabled(context.env) and not state["intentEnabled"]
    wire = candidate_wire(pool, preview=preview, question=state["question"],
                          needs=state["intentPlan"]['needs'] if preview else (), preview_characters=320)
    parts = need_parts(state, context) if enabled(state, context, 3) else state["questionParts"]
    attempt = 1 if recovery is not None else 0
    snapshot(context, "rerank_input", pool=pool, wire=wire, attempt=attempt)
    payload = {"question": state["question"], "questionParts": parts, "candidates": wire.candidates}
    if state["intentPlan"]:
        payload["evidenceNeeds"] = state["intentPlan"]["needs"]
    if reliability(state, context, 2):
        payload["retrievalBranches"] = {wire.aliases[h["chunk"]["id"]]: unique(
            p["path"] for p in h.get("paths", []) if p["path"].startswith("need-"))
            for h in pool["hits"] if any(p["path"].startswith("need-") for p in h.get("paths", []))}
    if recovery:
        payload.update(missingParts=[p["partId"] for p in recovery["missingParts"]],
            previouslySelected=[{"partId": p["partId"], "chunkIds": [wire.aliases[i] for i in p["chunkIds"] if i in wire.aliases]} for p in recovery["coverage"]])
        if repair and "unresolved" in recovery:
            payload["unresolvedRequirements"] = [{k: r[k] for k in ("partId", "need", "status")} for r in recovery["unresolved"]]
    if state["previousEvidence"]:
        payload["previousQuestion"] = state["previousEvidence"]["question"]
    focus_verses = {wire.aliases[h["chunk"]["id"]]: [v["verse"] for v in h["chunk"]["verses"]] for h in pool["hits"]}
    if preview:
        focus_verses = {row[0]: [int(line.split()[0]) for line in row[3].splitlines()] for row in wire.candidates}
        # Only actually visible complete verses may be chosen. No hidden focus.
        if any(not numbers for numbers in focus_verses.values()):
            raise ValueError('EMPTY_SELECTION_PREVIEW')
    setwise = enabled(state, context, 3) and context.env.get("BIBLE_SET_SELECTION") == "on"
    if setwise:
        from .evidence_sets import SET_PROMPT, set_schema, select_sets
        response = await context.call(state, "rerank", SET_PROMPT, payload, set_schema(parts, wire.ids, focus_verses))
        decoded = select_sets(response["result"], parts, wire, pool, state["intentPlan"]["needs"], preference_first=reliability(state, context, 2))
    else:
        bound_focus = state['intentEnabled'] and focus_schema_mode(context.env) == 'bound'
        selection_rule = ("\n각 필수 need의 주장·조건·이유를 본문에서 확인해 evidenceByPart에 완전한 근거 묶음으로 남긴다. 질문에 나온 대표 구절만 반복 선택하지 말고 서로 다른 요구를 설명하는 보완 본문을 보존한다. 비교·실패와 회복·원인과 결과는 양쪽 장면이 필요하다. retrievalBranches는 검색 경로 표시일 뿐 정답이나 관련성의 보장이 아니다. 선택한 모든 후보는 focus에도 실제 절 번호를 넣는다." if reliability(state, context, 2) else "")
        if reliability(state, context, 5):
            selection_rule += ('\nproofKind=event/sequence의 요구는 실제 행동이 기록된 장면을 고른다. 이전 시점의 예고·다짐·약속만으로 이후 사건을 대신하지 않는다. '
                               'sequence는 앞 장면과 실제 뒤 장면을 각각 보존하고, 집단 전체에 주어진 일반 임무와 특정 개인에게 다시 맡긴 일도 구분한다. '
                               'text_scope는 본문 자체와 추정 이유의 차이를 설명할 자료를 남긴다. 원인이 명시되지 않았다고 해당 본문까지 제외하지 않는다.')
        if preview:
            selection_rule += "\n후보는 전체 본문 또는 완전한 절 단위의 발췌다. complete=0이면 표시된 절 외의 내용을 추측하지 않는다. focus는 실제 보이는 절 번호만 고른다. 후보 선별과 재정렬을 함께 수행하며 선택 뒤 전체 문맥과 별도 충분성 검토가 이어진다."
        schema = reranking_schema(parts, wire.ids, repair=repair, strict_hints=state["intentEnabled"],
                focus_verses=focus_verses if state["intentEnabled"] else None)
        if state['intentEnabled'] and focus_schema_mode(context.env)=='compact':
            from .latency import compact_coordinate_schema
            schema=compact_coordinate_schema(schema,focus_verses)
        # Flat coordinate pairs preserve the *same* admissible coordinates.
        # Keep the old schema if the provider's global enum budget would be
        # exceeded; never silently drop candidates/verses to fit that budget.
        refs_mode=focus_schema_mode(context.env)=='refs' and len(wire.ids)+sum(map(len,focus_verses.values()))<=1000
        compact_focus = (preview and latency_mode(state, context) == 'selection') or refs_mode
        if compact_focus:
            from .latency import selection_schema, decode_selection_focus
            schema = selection_schema(schema, focus_verses)
            selection_rule += '\n이번 출력에서는 focus 객체 대신 동일한 근거 좌표를 focusRefs에 ["C1:3","C1:4","C2:9"]처럼 쓴다. 각 값은 후보 별칭:실제 절 번호다. 선택한 후보마다 최소 1개, 최대 4개이며 전체 후보는 최대 6개다. 실제 표시된 절만 선택하고 다른 내용이나 번호를 만들지 않는다.'
        base_prompt = RERANK_PROMPT
        if bound_focus:
            from .latency import bound_selection_schema
            from .prompt_text import RERANK_CRITERIA
            schema = bound_selection_schema(schema, focus_verses, parts)
            base_prompt = RERANK_CRITERIA + ('\nroute는 bible, ambiguous(지시 대상 식별 불가), outOfScope, greeting, safety다. '
                '별도 rankedIds/evidenceByPart/focus 목록은 쓰지 않는다. selections에 관련도순으로 최대 6개 후보를 한 번씩 고른다. '
                '각 항목의 chunkId는 실제 후보, verses는 그 후보 안에서 보이는 핵심 절 번호 1~4개, '
                'partIds는 그 근거가 실제로 필요한 questionParts 번호 목록이다. 한 후보가 여러 요구를 지원할 수 있다. '
                '한 요구에는 최대 3개 후보만 배정하며 비교 양쪽·실제 장면을 보존한다. '
                '관련 후보가 없으면 selections=[]이며 bible 외 경로도 []다. studyTerms는 필요한 영어 사전 표제어 최대 3개다. '
                '표제어는 사실이나 원어 대응의 증명이 아니다. 답변·해설·숨겨진 추론을 출력하지 않는다.')
            selection_rule = selection_rule.replace('evidenceByPart', 'selections의 partIds').replace('focus', 'selections의 verses')
        response = await context.call(state, "rerank", base_prompt + selection_rule + ("\n"+REPAIR_PROMPT if repair else "") + ("\nquestionParts는 독립된 근거 요구다. 필수 요구마다 실제 근거를 확보하며 전체 선택은 6개 이하로 한다. 한쪽 대상만으로 비교 전체를 충족했다고 하지 않는다." if enabled(state, context, 3) else ""), payload,
            schema)
        if bound_focus:
            from .latency import decode_bound_selection
            bound = decode_bound_selection(response['result'], focus_verses, parts, recover_duplicates=True)
            omitted = bound.pop('budgetOmissions', [])
            if omitted:
                state['selectionBudgetAudits'] = state.get('selectionBudgetAudits', []) + [
                    {**row, 'chunkId': wire.decode(row['chunkId']), 'attempt': attempt} for row in omitted]
                context.observe({'stage': 'selection-budget', 'omittedAssignments': len(omitted)})
            response = {**response, 'result': bound}
        if compact_focus:
            response = {**response, 'result': decode_selection_focus(response['result'], focus_verses)}
        decoded = decode_reranking(response["result"], wire, recover_duplicates=state["intentEnabled"])
    if enabled(state, context, 3) and not setwise:
        bounded = bounded_need_selection(state, decoded, context)
        if bounded["evidenceByPart"] != decoded["evidenceByPart"]:
            context.observe({"stage": "diagnostic", "component": "rerank", "code": "SELECTION_FOCUS_MISSING"})
        decoded = bounded
    snapshot(context, "rerank_ranked", pool=pool, ids=decoded["rankedIds"], attempt=attempt)
    snapshot(context, "rerank_selected", pool=pool,
             ids=unique(i for values in decoded["evidenceByPart"].values() for i in values), attempt=attempt)
    try:
        plan = validate_reranking(decoded, parts, pool)
        if enabled(state, context, 3):
            plan = merge_need_selection(state, plan, context)
    except Exception as error:
        context.observe({"stage": "diagnostic", "component": "rerank", "code": getattr(error, "diagnostic", "AI_SELECTION_INVALID")})
        raise
    snapshot(context, "validated_selection", pool=pool, ids=plan["chunkIds"], attempt=attempt)
    if repair:
        plan = assess_repair(plan, decoded.get("requirements"), state["questionParts"], pool)
        snapshot(context, "repair_selection", pool=pool, ids=plan["chunkIds"], attempt=attempt)
    return plan


async def rerank_candidates(state, context):
    pool, retrieval = state["screenedPool"], state["retrieval"]
    try:
        plan = await rerank_pool(state, pool, None, context, preview=state.get('fusedSelection', False))
    except BibleProviderError as error:
        if (error.code not in {'AI_SELECTION_INVALID','AI_INCOMPLETE'} or not state['intentEnabled']
                or context.env.get('BIBLE_EXPLANATION_POLICY') != 'explicit'):
            raise
        # Do not accept unranked IDs or invent missing focus coordinates. The
        # real review can still use the anchor, then request bounded recovery.
        context.observe({'stage':'diagnostic','component':'rerank','code':error.code})
        plan = {'route':'bible','chunkIds':[],'rankedIds':[],'focus':{},'studyTerms':[],
                'coverage':[{'partId':p['partId'],'chunkIds':[]} for p in state['questionParts']],
                'missingParts':state['questionParts']}
    narrowed = {h['chunk']['id'] for h in pool['hits']} != {h['chunk']['id'] for h in retrieval['hits']}
    recovery = ((plan["route"] == "bible" or (repair_enabled(context.env) and plan["route"] == "ambiguous")) and bool(plan["missingParts"])) or (narrowed and plan["route"] in ("ambiguous", "outOfScope"))
    return {"plan": plan, "recoveryCount": 0, "needsRecovery": recovery and not state["intentEnabled"]}


async def recover_candidates(state, context):
    retrieval, plan = state["retrieval"], state["plan"]
    repair = repair_enabled(context.env)
    if plan["route"] != "bible":
        plan = {**plan, "missingParts": state["questionParts"]}
    queries = repair_queries(plan, state["questionParts"], expand=context.env.get("BIBLE_REPAIR_QUERY_EXPANSION") != "off") if repair else [
        {"query": p["question"], "partId": p["partId"], "expanded": False} for p in plan["missingParts"]]
    if state["intentEnabled"]:
        from .gap_recovery import gap_queries
        queries = gap_queries(state, context)
    if state["contextPlan"] and state["contextPlan"]["usesContext"]:
        # Reuse the same bounded hints; no second interpretation/model call.
        hints = state["searchQuestion"][len(state["question"]):]
        queries = [{**q, "query": q["query"] + hints} for q in queries]
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
                                   scope_question=("" if state["scope"]["direct"] and state["scope"]["crossrefs"] else state["question"])
                                       if state["intentEnabled"] else retrieval.get("query") or state["question"])
        rows = more["hits"][:24]
        branches.append(rows); additional.extend(rows)
    study_index = context.get_study_index(state["studyEnabled"])
    if study_index is not None and plan["chunkIds"]:
        primary = focused_evidence(context.corpus, retrieval, plan, limits={"evidenceCharacters": 6000, "evidenceVerses": 96})["evidence"]
        linked = expand_one_hop(context.corpus, primary, state["question"], study_index)["linked"]
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
    context.observe({"stage": "retrieval-recovery", "result": {"attempt": 1,
        "method": "gap-feedback-v2" if state["intentEnabled"] else "repair-inspired-v1" if repair else "legacy",
        "missingParts": [p["partId"] for p in plan["missingParts"]],
        "targets": [{k: q[k] for k in ("partId", "need", "expanded") if k in q} for q in queries], "added": len(added),
        "candidateIds": [h["chunk"]["id"] for h in retrieval["hits"]], "rerankIds": [h["chunk"]["id"] for h in pool["hits"]],
        "embeddingCalls": embedding_calls, "embeddingDegraded": degraded}, "usage": {"embeddingInputTokens": embedding_tokens}})
    snapshot(context, "recovered", pool=retrieval, attempt=1)
    return {"retrieval": retrieval, "plan": plan, "recoveryCount": 1, "recoveryPool": pool}


async def rerank_recovered(state, context):
    retrieval, plan = state["retrieval"], state["plan"]
    from .profiles import profile_for
    profile = profile_for(context.env)
    if profile is not None:
        from .research_steps import cap_candidate_pool
        state['recoveryPool'] = cap_candidate_pool(state.get('recoveryPool') or retrieval, profile.candidates)
    try:
        recovered = await rerank_pool(state, state.get("recoveryPool") or retrieval, plan, context)
    except BibleProviderError as error:
        if not state["intentEnabled"] or error.code != "AI_SELECTION_INVALID":
            raise
        # A malformed supplemental selection cannot erase validated first-pass
        # evidence. The following review still marks unresolved needs missing.
        context.observe({"stage": "diagnostic", "component": "rerank", "code": "AI_SELECTION_INVALID"})
        return {"plan": plan, "needsRecovery": False}
    if state["intentEnabled"]:
        from .gap_recovery import merge_plans
        return {"plan": merge_plans(plan, recovered), "needsRecovery": False}
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
            "studyTerms": unique(recovered["studyTerms"]+plan["studyTerms"])[:3]}, state["questionParts"], retrieval)
    else:
        plan = recovered
    return {"plan": plan}


def select_evidence(state, context):
    retrieval, plan, count = state["retrieval"], state["plan"], state.get("recoveryCount", 0)
    snapshot(context, "final_selection", pool=retrieval, ids=plan["chunkIds"], attempt=count)
    report = {"chunkIds": plan["chunkIds"], "rankedIds": plan["rankedIds"], "parts": len(plan["coverage"]),
              "missingParts": [p["partId"] for p in plan["missingParts"]], "recoveryCount": count}
    if "requirements" in plan:
        report["requirements"] = [{k: r[k] for k in ("partId", "need", "status", "importance")} for r in plan["requirements"]]
    context.observe({"stage": "selection", "result": report})
    if plan["route"] != "bible" or not plan["chunkIds"]:
        return {"selected": {"plan": plan, "retrieval": retrieval, "recoveryCount": count}}
    from .profiles import profile_for
    profile = profile_for(context.env)
    from .profiles import execution_profile
    profile = execution_profile(state, context.env)
    limits = ({'evidenceVerses':profile.evidence_verses,'evidenceCharacters':profile.evidence_characters}
              if profile is not None else None)
    selected = focused_evidence(context.corpus, retrieval, plan, state["passage"], limits=limits,
                                boundary_context=2 if state["intentEnabled"] else 0,
                                required_focus=profile is None or profile.adaptive)
    snapshot(context, "excerpt", verses=selected["evidence"], attempt=count)
    context.observe({"stage": "evidence", "result": {"verses": len(selected["evidence"]), "anchorIds": selected["anchorIds"],
        "retainedVerseIds": [v["id"] for v in selected["evidence"]], "incompleteSections": sum(not s["complete"] for s in selected["sections"])}})
    return {"selected": {"plan": plan, "retrieval": retrieval, **selected, "recoveryCount": count}}
