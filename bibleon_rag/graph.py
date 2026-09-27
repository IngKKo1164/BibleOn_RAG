"""The actual 17-node LangGraph; no manual/JavaScript execution fallback."""
from __future__ import annotations

import asyncio
import inspect
import os
import time
from typing import Any, TypedDict

from . import answer_steps as answer
from . import selection_steps as selection
from .common import text_length, unique_verses
from .passages import parse_bible_references
from .runtime import BibleRuntime
from .study import expand_one_hop, retrieve_study_sources
from .sufficiency import missing_evidence_reply
from .transport import ProviderTransport
from .validation import fallback, unresolved_referent

GRAPH_RECURSION_LIMIT = 32
STATE_KEYS = ["lane", "retrieval", "screenedPool", "recoveryPool", "plan", "recoveryCount", "needsScreening", "needsRecovery",
              "evidence", "primaryEvidence", "sectionContext", "selectionCoverage", "studyTerms", "studySources", "crossReferences",
              "studyLoaded", "answerSchema", "partEvidence", "system", "languageSupport", "answerInput", "draft", "visibleTokens",
              "referenceEvidenceByPart", "needsCitationRepair", "confirmedContext", "conflicts", "usage", "result"]
GraphState = TypedDict("GraphState", {key: Any for key in STATE_KEYS}, total=False)
GraphOutput = TypedDict("GraphOutput", {"result": Any})


def assert_local_tracing():
    flags = ("LANGSMITH_TRACING", "LANGSMITH_TRACING_V2", "LANGCHAIN_TRACING", "LANGCHAIN_TRACING_V2", "LANGCHAIN_VERBOSE")
    if (any(os.getenv(k) and os.environ[k].lower() not in ("false", "0") for k in flags)
            or os.getenv("LANGSMITH_API_KEY") or os.getenv("LANGCHAIN_API_KEY")):
        raise RuntimeError("LANGGRAPH_EXTERNAL_TRACING_DISABLED")
    from langsmith.run_helpers import get_current_run_tree
    if get_current_run_tree() is not None:
        raise RuntimeError("LANGGRAPH_EXTERNAL_TRACING_DISABLED")


def route_request(state, context):
    if unresolved_referent(context.question, context.previous_evidence, context.passage):
        return {"result": fallback("uncertain", context.usage)}
    scope = context.scope
    direct = {**scope, "exact": True, "refs": scope["primary"]} if scope["lookup"] else parse_bible_references(context.corpus, context.question)
    if direct["exact"] and direct["invalid"]:
        return {"result": fallback("uncertain", context.usage)}
    if direct["exact"] and direct["refs"] and (scope["lookup"] or (len(direct["refs"]) <= 8 and sum(text_length(v["text"]) for v in direct["refs"]) <= 1800)):
        return {"result": {"kind": "answer", "text": "\n\n".join(f"**{v['reference']}**\n\n{v['text']}" for v in direct["refs"]),
                           "citations": direct["refs"], "contextSummary": "", "outOfScope": False, "needsClarification": False, "usage": context.usage}}
    context.get_study_index()
    return {"lane": "direct", "primaryEvidence": scope["primary"], "evidence": scope["primary"]+scope["context"]} if scope["direct"] else {"lane": "retrieve"}


async def retrieve(state, context):
    context.observe({"stage": "retrieve", "event": "start"})
    update = await selection.retrieve_candidates(state, context)
    context.usage.update(embeddingInputTokens=update["retrieval"]["inputTokens"], embeddingCalls=update["retrieval"]["embeddingCalls"])
    return update


async def recover(state, context):
    update = await selection.recover_candidates(state, context)
    context.usage.update(embeddingInputTokens=update["retrieval"]["inputTokens"], embeddingCalls=update["retrieval"]["embeddingCalls"])
    return update


def evidence(state, context):
    selected = selection.select_evidence(state, context)["selected"]
    if selected["plan"]["route"] != "bible":
        return {"result": fallback("uncertain" if selected["plan"]["route"] == "ambiguous" else selected["plan"]["route"], context.usage)}
    if not selected.get("evidence"):
        return {"result": missing_evidence_reply(context.usage)}
    return {"evidence": selected["evidence"], "primaryEvidence": selected["evidence"], "sectionContext": selected["sections"],
            "selectionCoverage": selected["plan"]["coverage"], "studyTerms": selected["plan"]["studyTerms"]}


def enrich(state, context):
    if not context.study_enabled:
        return {}
    context.observe({"stage": "study-retrieve", "event": "start"})
    try:
        index = context.get_study_index()
        context.check_deadline()
        if index is None:
            raise ValueError("STUDY_UNAVAILABLE")
        expanded = expand_one_hop(context.corpus, state["primaryEvidence"], context.question, index) if context.scope["crossrefs"] else {"roots": [], "linked": [], "relations": []}
        sources = retrieve_study_sources(state["primaryEvidence"], state.get("studyTerms", []), context.question, index, independent=True) if context.scope["sources"] else []
        evidence = unique_verses(state["evidence"]+expanded["linked"])
        context.observe({"stage": "study-retrieve", "result": {"primaryVerses": len(state["primaryEvidence"]), "roots": expanded["roots"],
            "linkedVerses": [v["id"] for v in expanded["linked"]], "relations": expanded["relations"], "sources": sources}})
        return {"evidence": evidence, "crossReferences": expanded["relations"], "studySources": sources, "studyLoaded": True}
    except Exception:
        context.check_deadline()
        context.observe({"stage": "study-unavailable", "result": {"degraded": True}})
        context.observe({"stage": "study-retrieve"})
        return {}


def build_bible_graph(context=None):
    assert_local_tracing()
    from langgraph.graph import StateGraph, START, END
    from langgraph.types import RetryPolicy
    graph = StateGraph(GraphState, output_schema=GraphOutput)
    nodes = {"route": route_request, "retrieve": retrieve, "budget": selection.budget_candidates,
             "screen": selection.screen_candidates, "rerank": selection.rerank_candidates, "recover": recover,
             "rerank_recovered": selection.rerank_recovered, "select_evidence": evidence, "enrich": enrich,
             "prepare": answer.prepare_answer, "generate": answer.generate_answer, "check_citations": answer.check_citations,
             "repair_citations": answer.repair_citations, "assess": answer.assess_answer, "check_facts": answer.check_facts,
             "repair_facts": answer.repair_facts, "validate": answer.validate_answer}
    def bind(name, operation):
        async def run(state):
            context.check_deadline()
            started = time.monotonic()
            context.observe({"stage": "graph", "node": name, "event": "start"})
            try:
                update = operation(state, context)
                if inspect.isawaitable(update):
                    update = await update
                context.check_deadline()
                context.observe({"stage": "graph", "node": name, "event": "end", "milliseconds": (time.monotonic()-started)*1000})
                return {**update, "usage": dict(context.usage)}
            except BaseException as error:
                context.observe({"stage": "graph", "node": name, "event": "error", "code": getattr(error, "code", type(error).__name__)})
                raise
        return run
    for name, operation in nodes.items():
        graph.add_node(name, bind(name, operation), retry_policy=RetryPolicy(max_attempts=1))
    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", lambda s: END if s.get("result") else "enrich" if s["lane"] == "direct" else "retrieve", [END, "enrich", "retrieve"])
    graph.add_edge("retrieve", "budget")
    graph.add_conditional_edges("budget", lambda s: "screen" if s["needsScreening"] else "rerank", ["screen", "rerank"])
    graph.add_edge("screen", "rerank")
    graph.add_conditional_edges("rerank", lambda s: "recover" if s["needsRecovery"] else "select_evidence", ["recover", "select_evidence"])
    graph.add_edge("recover", "rerank_recovered")
    graph.add_edge("rerank_recovered", "select_evidence")
    graph.add_conditional_edges("select_evidence", lambda s: END if s.get("result") else "enrich", [END, "enrich"])
    graph.add_edge("enrich", "prepare")
    graph.add_edge("prepare", "generate")
    graph.add_edge("generate", "check_citations")
    graph.add_conditional_edges("check_citations", lambda s: END if s.get("result") else "repair_citations" if s["needsCitationRepair"] else "assess", [END, "repair_citations", "assess"])
    graph.add_edge("repair_citations", "assess")
    graph.add_conditional_edges("assess", lambda s: END if s.get("result") else "check_facts", [END, "check_facts"])
    graph.add_conditional_edges("check_facts", lambda s: "repair_facts" if s["conflicts"] else "validate", ["repair_facts", "validate"])
    graph.add_edge("repair_facts", "validate")
    graph.add_edge("validate", END)
    return graph.compile()


async def run_bible_graph(**options):
    assert_local_tracing()
    transport = None
    if options.get("fetcher") is None:
        transport = ProviderTransport(options["env"])
        options["fetcher"] = transport.request
    task = watcher = None
    try:
        context = BibleRuntime(**options)
        context.check_deadline()
        graph = build_bible_graph(context)
        async with asyncio.timeout(110):
            task = asyncio.create_task(graph.ainvoke({}, config={"recursion_limit": GRAPH_RECURSION_LIMIT, "callbacks": []}))
            if context.cancel is not None:
                watcher = asyncio.create_task(context.cancel.wait())
                done, _ = await asyncio.wait((task, watcher), return_when=asyncio.FIRST_COMPLETED)
                if watcher in done and context.cancel.is_set():
                    raise asyncio.CancelledError()
            return (await task)["result"]
    finally:
        for pending in (task, watcher):
            if pending is not None and not pending.done():
                pending.cancel()
        if task is not None or watcher is not None:
            await asyncio.gather(*(p for p in (task, watcher) if p is not None), return_exceptions=True)
        if transport is not None:
            await transport.aclose()
