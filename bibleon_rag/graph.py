"""The actual 19-node LangGraph; no manual/JavaScript execution fallback."""
from __future__ import annotations

import inspect
import os
import time
from langgraph.runtime import Runtime

from . import answer_steps as answer
from . import selection_steps as selection
from .common import text_length, unique_verses
from .observability import snapshot
from .passages import parse_bible_references
from .runtime import BibleRuntime, create_execution, execution_resources, invoke_graph
from .graph_state import GraphState, commit_node, completed_result, decode, validate_state
from .interpretation import plan_request, assess_evidence
from .evidence_policy import enabled, reliability
from .config import bilingual_background
from .study import expand_one_hop, retrieve_study_sources
from .sufficiency import missing_evidence_reply
from .validation import fallback, unresolved_referent

GRAPH_RECURSION_LIMIT = 32


def assert_local_tracing():
    flags = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_VERBOSE")
    if (any(os.getenv(k) and os.environ[k].lower() not in ("false", "0") for k in flags)
            or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise RuntimeError("LANGGRAPH_EXTERNAL_TRACING_DISABLED")
    from langsmith.run_helpers import get_current_run_tree
    if get_current_run_tree() is not None:
        raise RuntimeError("LANGGRAPH_EXTERNAL_TRACING_DISABLED")


def route_request(state, context):
    from .profiles import automatic_routing
    if automatic_routing(context.env):
        # Coordinates remain locally validated, but never decide semantic intent.
        return {'lane': 'plan'}
    if state["scope"]["invalid"]:
        return {"result": {**fallback("uncertain", state["usage"]),
            "text": "입력한 성경의 장과 절을 찾지 못했습니다. 책 이름과 장·절 번호를 확인해 주세요."}}
    if not state["intentEnabled"] and state["contextPlan"] and state["contextPlan"]["mode"] == "social":
        return {"lane": "conversation", "evidence": []}
    if not state["intentEnabled"] and state["contextPlan"] and state["contextPlan"]["mode"] == "unresolved":
        return {"result": fallback("uncertain", state["usage"])}
    if not state["intentEnabled"] and not (state["contextPlan"] and state["contextPlan"]["usesContext"]) and unresolved_referent(state["question"], state["previousEvidence"], state["passage"]):
        return {"result": fallback("uncertain", state["usage"])}
    scope = state["scope"]
    direct = {**scope, "exact": True, "refs": scope["primary"]} if scope["lookup"] else parse_bible_references(context.corpus, state["question"])
    if direct["exact"] and direct["invalid"]:
        return {"result": fallback("uncertain", state["usage"])}
    if direct["exact"] and direct["refs"] and (scope["lookup"] or (len(direct["refs"]) <= 8 and sum(text_length(v["text"]) for v in direct["refs"]) <= 1800)):
        snapshot(context, "direct", verses=direct["refs"])
        return {"result": {"kind": "answer", "text": "\n\n".join(f"**{v['reference']}**\n\n{v['text']}" for v in direct["refs"]),
                           "citations": direct["refs"], "contextSummary": "", "outOfScope": False, "needsClarification": False, "usage": state["usage"]}}
    if state["intentEnabled"]:
        return {"lane": "plan"}
    context.get_study_index(state["studyEnabled"])
    if scope["direct"]:
        snapshot(context, "direct", verses=scope["primary"]+scope["context"])
    return {"lane": "direct", "primaryEvidence": scope["primary"], "evidence": scope["primary"]+scope["context"],
            "sectionContext": scope.get("sections", [])} if scope["direct"] else {"lane": "retrieve"}


async def retrieve(state, context):
    context.observe({"stage": "retrieve", "event": "start"})
    update = await selection.retrieve_candidates(state, context)
    state["usage"].update(embeddingInputTokens=update["retrieval"]["inputTokens"], embeddingCalls=update["retrieval"]["embeddingCalls"])
    return update


async def recover(state, context):
    if "retrieval" not in state:
        state.update(await retrieve(state, context))
    if "plan" not in state:
        state["plan"] = {"route": "bible", "chunkIds": [], "rankedIds": [], "focus": {},
            "coverage": [{"partId": p["partId"], "chunkIds": []} for p in state["questionParts"]],
            "missingParts": state["questionParts"], "studyTerms": state.get("studyTerms", [])}
    update = await selection.recover_candidates(state, context)
    state["usage"].update(embeddingInputTokens=update["retrieval"]["inputTokens"], embeddingCalls=update["retrieval"]["embeddingCalls"])
    return update


def evidence(state, context):
    selected = selection.select_evidence(state, context)["selected"]
    if state["intentEnabled"]:
        # Selected source passages can never evict the user's anchor or its local context.
        base = state["scope"]["primary"]+state["scope"]["context"]
        # A second selection supplements the previous evidence; it cannot erase it.
        previous = state.get("evidence", []) if state.get("recoveryCount") else []
        protected = {a["evidenceId"] for c in state.get("evidenceAssessment", []) for a in c["supports"]}
        base += [v for v in previous if v["id"] in protected]
        combined = unique_verses(base)
        ids, chars = {v["id"] for v in combined}, sum(len(v["text"]) for v in combined)
        for v in selected.get("evidence", [])+previous:
            if v["id"] not in ids and len(combined) < 96 and chars+len(v["text"]) <= 12000:
                combined.append(v)
                ids.add(v["id"])
                chars += len(v["text"])
        return {"evidence": combined, "primaryEvidence": combined,
                "sectionContext": state.get("sectionContext", state["scope"].get("sections", []))+selected.get("sections", []),
                "selectionCoverage": selected["plan"].get("coverage", []),
                "studyTerms": list(dict.fromkeys(state.get("studyTerms", [])+selected["plan"].get("studyTerms", [])))[:3]}
    if selected["plan"]["route"] != "bible":
        return {"result": fallback("uncertain" if selected["plan"]["route"] == "ambiguous" else selected["plan"]["route"], state["usage"])}
    if not selected.get("evidence"):
        return {"result": missing_evidence_reply(state["usage"])}
    return {"evidence": selected["evidence"], "primaryEvidence": selected["evidence"], "sectionContext": selected["sections"],
            "selectionCoverage": selected["plan"]["coverage"],
            "studyTerms": list(dict.fromkeys(state.get("studyTerms", [])+selected["plan"]["studyTerms"]))[:3]}


def _enrich_sources(state, context):
    translated_titles = (context.env.get('BIBLE_SUPERSCRIPTIONS') == 'web'
                         and state['intentEnabled'] and state['evidenceMode'] == 'adaptive')
    if not state["studyEnabled"] and not translated_titles:
        return {}
    context.observe({"stage": "study-retrieve", "event": "start"})
    try:
        index = context.get_study_index(state["studyEnabled"] or translated_titles)
        context.check_deadline()
        if index is None:
            raise ValueError("STUDY_UNAVAILABLE")
        roots = state["scope"]["primary"] or state["primaryEvidence"]
        if context.env.get('BIBLE_REFERENCE_ROOTS') == 'provided' and state['intentEnabled']:
            from .study import provided_reference_roots
            roots = provided_reference_roots(state['scope']['primary'], state['evidence'], state['question'], index)
        expanded = expand_one_hop(context.corpus, roots, state["question"], index) if state["studyEnabled"] and state["scope"]["crossrefs"] else {"roots": [], "linked": [], "relations": []}
        sources = retrieve_study_sources(state["primaryEvidence"], state.get("studyTerms", []), state["question"], index,
            independent=True, enhanced=state["intentEnabled"],
            multilingual_queries=(state["intentPlan"] or {}).get("backgroundQueries", [])
                if bilingual_background(context.env) else (),
            balanced=reliability(state, context, 3),
            anchor_ids=[v['id'] for v in state['scope']['primary']]
                if state['intentEnabled'] and context.env.get('BIBLE_BACKGROUND_SEARCH') == 'anchored' else ()) if state["studyEnabled"] and state["scope"]["sources"] else []
        if translated_titles:
            from .study import add_superscriptions
            # Explicitly selected chapters get their own real metadata. For
            # implicit questions, only a planned historical need requests it.
            title_roots = state['scope']['primary'] or (state['primaryEvidence']
                if any(n['kind'] == 'history' for n in (state['intentPlan'] or {}).get('needs', [])) else [])
            sources = add_superscriptions(title_roots, sources, index)
            if not state['studyEnabled'] and not sources:
                return {}
        if state["intentEnabled"] and state.get("recoveryCount"):
            from .gap_recovery import merge_sources
            sources = merge_sources(state, sources)
        evidence = unique_verses(state["evidence"]+expanded["linked"])
        if context.env.get('BIBLE_REFERENCE_ROOTS') == 'provided' and state['intentEnabled']:
            evidence = list(state['evidence'])
            seen, chars = {v['id'] for v in evidence}, sum(len(v['text']) for v in evidence)
            for v in expanded['linked']:
                if v['id'] not in seen and len(evidence) < 96 and chars + len(v['text']) <= 12000:
                    evidence.append(v); seen.add(v['id']); chars += len(v['text'])
            expanded['relations'] = [r for r in expanded['relations'] if r['to'] in seen]
        if enabled(state, context, 4):
            snapshot(context, "study_selected", verses=evidence, sources=sources,
                     attempt=state.get("recoveryCount", 0))
        context.observe({"stage": "study-retrieve", "result": {"primaryVerses": len(state["primaryEvidence"]), "roots": expanded["roots"],
            "linkedVerses": [v["id"] for v in expanded["linked"]], "relations": expanded["relations"], "sources": sources}})
        return {"evidence": evidence, "crossReferences": expanded["relations"], "studySources": sources, "studyLoaded": True}
    except Exception as error:
        context.check_deadline()
        context.observe({"stage": "study-unavailable", "result": {"degraded": True}})
        context.observe({"stage": "study-retrieve"})
        from .profiles import profile_for
        if profile_for(context.env) is not None:
            diagnostic = {'stage': 'study', 'code': type(error).__name__}
            return {'sourceDiagnostics': state.get('sourceDiagnostics', []) + [diagnostic]}
        return {}


def enrich(state, context):
    update = _enrich_sources(state, context)
    from .profiles import profile_for
    if profile_for(context.env) is not None:
        from .research_steps import bounded_evidence
        return {**update, **bounded_evidence({**state, **update}, context)}
    return update


def build_bible_graph(context=None, *, checkpointer=None, interrupt_after=None):
    assert_local_tracing()
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import RetryPolicy
    from .profiles import profile_for
    profile = profile_for(context.env) if context is not None else None
    graph = StateGraph(GraphState, context_schema=BibleRuntime)
    nodes = {"route": route_request, "plan": plan_request, "assess_evidence": assess_evidence,
             "retrieve": retrieve, "budget": selection.budget_candidates,
             "screen": selection.screen_candidates, "rerank": selection.rerank_candidates, "recover": recover,
             "rerank_recovered": selection.rerank_recovered, "select_evidence": evidence, "enrich": enrich,
             "prepare": answer.prepare_answer, "generate": answer.generate_answer, "check_citations": answer.check_citations,
             "repair_citations": answer.repair_citations, "assess": answer.assess_answer, "check_facts": answer.check_facts,
             "repair_facts": answer.repair_facts, "validate": answer.validate_answer}
    if profile is not None:
        # Historical phrase-specific fact correction is not part of either
        # named research condition. Literal citation invariants remain below.
        nodes.pop('check_facts'); nodes.pop('repair_facts')
        async def routed_selection(state, ctx):
            from .profiles import execution_profile
            from .research_steps import rank_only_selection
            return (await selection.rerank_candidates(state, ctx)
                    if execution_profile(state, ctx.env).adaptive else rank_only_selection(state, ctx))
        nodes['rerank'] = routed_selection
    def bind(name, operation):
        async def run(state, runtime: Runtime[BibleRuntime]):
            ctx = runtime.context or context
            if ctx is None:
                raise ValueError("GRAPH_CONTEXT_REQUIRED")
            validate_state(state)
            # decode also copies every JSON container: a failing node cannot
            # mutate the previous checkpoint through nested aliases.
            working = decode(state, ctx)
            # Keep committed decision inputs deterministic during task replay.
            # I/O enforces the actual spending floor before any new HTTP call;
            # receipt application and commit then reconcile it into state.
            # Applying it here can change a budget-dependent schema and make
            # an already-paid response impossible to replay after node failure.
            ctx.check_deadline()
            started = time.monotonic()
            ctx.observe({"stage": "graph", "node": name, "event": "start"})
            try:
                update = operation(working, ctx)
                if inspect.isawaitable(update):
                    update = await update
                ctx.check_deadline()
                ctx.observe({"stage": "graph", "node": name, "event": "end", "milliseconds": (time.monotonic()-started)*1000})
                return commit_node(state, working, update, ctx, name)
            except BaseException as error:
                ctx.observe({"stage": "graph", "node": name, "event": "error", "code": getattr(error, "code", type(error).__name__)})
                raise
        return run
    for name, operation in nodes.items():
        graph.add_node(name, bind(name, operation), retry_policy=RetryPolicy(max_attempts=1))
    graph.add_edge(START, "route")
    def next_route(s):
        from .profiles import execution_profile
        if s.get('result'):
            return END
        if s['lane'] in {'conversation', 'clarification'}:
            return 'prepare'
        if (profile is not None and s.get('intentPlan')
                and not execution_profile(s, context.env).adaptive):
            return 'retrieve'
        return END if s.get("result") else {"conversation": "prepare", "direct": "enrich", "plan": "plan"}.get(s["lane"], "retrieve")
    graph.add_conditional_edges("route", next_route, [END, "prepare", "enrich", "retrieve", "plan"])
    graph.add_conditional_edges("plan", next_route, [END, "prepare", "enrich", "retrieve"])
    graph.add_edge("retrieve", "budget")
    graph.add_conditional_edges("budget", lambda s: "enrich" if s.get('contextFirst') else "screen" if s["needsScreening"] else "rerank", ["enrich", "screen", "rerank"])
    graph.add_edge("screen", "rerank")
    graph.add_conditional_edges("rerank", lambda s: "recover" if s["needsRecovery"] else "select_evidence", ["recover", "select_evidence"])
    graph.add_edge("recover", "rerank_recovered")
    graph.add_edge("rerank_recovered", "select_evidence")
    graph.add_conditional_edges("select_evidence", lambda s: END if s.get("result") else "enrich", [END, "enrich"])
    def after_enrich(s):
        from .profiles import execution_profile
        policy = execution_profile(s, context.env) if profile is not None else None
        return ('assess_evidence' if s.get('intentPlan') and
                (policy is None or profile.name == 'auto' or policy.adaptive) else 'prepare')
    graph.add_conditional_edges('enrich', after_enrich, ['assess_evidence', 'prepare'])
    graph.add_conditional_edges("assess_evidence", lambda s: END if s.get("result") else "recover" if s.get("needsRecovery") else "prepare", [END, "recover", "prepare"])
    graph.add_edge("prepare", "generate")
    graph.add_edge("generate", "check_citations")
    graph.add_conditional_edges("check_citations", lambda s: END if s.get("result") else "repair_citations" if s["needsCitationRepair"] or s.get('needsFinalReview') else "assess", [END, "repair_citations", "assess"])
    graph.add_edge("repair_citations", "assess")
    if profile is not None:
        graph.add_conditional_edges('assess', lambda s: END if s.get('result') else 'validate', [END, 'validate'])
    else:
        graph.add_conditional_edges("assess", lambda s: END if s.get("result") else "check_facts", [END, "check_facts"])
        graph.add_conditional_edges("check_facts", lambda s: "repair_facts" if s["conflicts"] else "validate", ["repair_facts", "validate"])
        graph.add_edge("repair_facts", "validate")
    graph.add_edge("validate", END)
    return graph.compile(checkpointer=checkpointer, interrupt_after=interrupt_after)


async def run_bible_graph(**options):
    assert_local_tracing()
    async with execution_resources(options) as options:
        state, context = create_execution(**options)
        graph = build_bible_graph(context)
        state = await invoke_graph(graph, state,
            {"recursion_limit": GRAPH_RECURSION_LIMIT, "callbacks": []}, context)
        return completed_result(state, context)
