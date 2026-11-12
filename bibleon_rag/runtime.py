"""Per-invocation capabilities stay outside serializable LangGraph state."""
from __future__ import annotations

import asyncio
import json
import re
import time
from copy import deepcopy
from contextlib import asynccontextmanager

from langgraph.func import task
from langgraph.runtime import get_runtime
from langgraph.types import RetryPolicy

from .common import BibleProviderError, is_integer, json_text, sha256
from .config import MAX_REASONING_TOKENS, OPENAI_ENDPOINT, STAGES, discovery_seed_budget, reasoning_mode, study_discovery_mode, focus_schema_mode, target_policy_options
from .conversation import CONTEXT_PROMPT, context_wire
from .prompt_text import COMPACT_EVIDENCE_PROMPT
from .selection import compact_answer_input


def openai_body(stage, system, value, output_schema, *, env=None):
    config = STAGES.get(stage)
    if config is None:
        raise BibleProviderError("AI_NOT_CONFIGURED", 503)
    mode = reasoning_mode(env or {})
    none = (mode=='all-none' or mode=='selection-none' and stage in {'screen','rerank'}
            or mode=='plan-low' and stage!='intent'
            or mode=='plan-answer-low' and stage not in {'intent','answer'}
            or mode=='plan-review-answer-low' and stage not in {'intent','evidence_review','answer'}
            or mode=='judgement-low' and stage not in {'intent','evidence_review'})
    # Independent final checking is not one of the old selection-effort ablations.
    effort = config['effort'] if stage == 'final_review' else 'none' if none else config['effort']
    processing = (env or {}).get('BIBLE_PROCESSING_TIER', 'baseline')
    dispatch = {}
    if processing == 'answer-fast' and stage == 'answer' and value.get('evidence'):
        dispatch = {'service_tier': 'fast'}
    elif processing == 'decision-fast':
        # Move the existing premium to the measured planning/review bottleneck.
        # Explicit Standard elsewhere avoids a project default silently turning
        # the remaining stages into premium calls. No model/input/budget change.
        dispatch = {'service_tier': 'fast' if stage in {'intent', 'evidence_review'} else 'default'}
    elif processing == 'final-fast':
        dispatch = {'service_tier': 'fast' if stage in {'intent', 'final_review'} else 'default'}
    from .profiles import profile_for
    if profile_for(env or {}) is not None:
        dispatch = {'service_tier': 'default'}
    return {"model": config["model"], "reasoning_effort": effort,
            **dispatch,
            "max_completion_tokens": config["maxTokens"], "stream": False, "store": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "bible_"+stage, "strict": True, "schema": output_schema}},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json_text(value)}]}


def is_expected_model(stage, model):
    expected = STAGES.get(stage, {}).get("model")
    return bool(expected) and isinstance(model, str) and (model == expected or
            (model.startswith(expected+"-") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", model[len(expected)+1:]) is not None))


class BibleRuntime:
    def __init__(self, *, env, corpus, fetcher=None, observe=None, retriever=None,
                 study_index=None, cancel=None, data_directory=None):
        if env.get("BIBLE_AI_PROVIDER") != "openai" or env.get("BIBLE_RETRIEVAL_VERSION") != "v4":
            raise BibleProviderError("AI_NOT_CONFIGURED", 503)
        discovery_seed_budget(env)
        reasoning_mode(env)
        study_discovery_mode(env)
        focus_schema_mode(env)
        target_policy_options(env)
        from .profiles import profile_for
        profile_for(env)
        self.env, self.corpus = dict(env), corpus
        self.fetcher, self.observe = fetcher, observe or (lambda event: None)
        self.retriever, self.study_index = retriever, study_index
        self.cancel, self.expires_at = cancel, time.monotonic()+110
        self.data_directory = data_directory
        self._study_loaded = study_index is not None
        self.ledger = UsageLedger()
        self.active_ms_before = 0
        self.invocation_started_at = time.monotonic()

    def active_milliseconds(self):
        return self.active_ms_before + (time.monotonic()-self.invocation_started_at)*1000

    def check_deadline(self):
        if self.cancel is not None and self.cancel.is_set():
            raise asyncio.CancelledError()
        if time.monotonic() >= self.expires_at:
            raise TimeoutError("AI_TIMEOUT")

    def get_study_index(self, enabled=True):
        # Resource loading is cached; a disabled route must not cache absence.
        if self.study_index is None and not enabled:
            return None
        if not self._study_loaded:
            self._study_loaded = True
            from .data import DEFAULT_DATA, load_study_data
            from .study import StudyIndex
            try:
                self.study_index = StudyIndex(self.corpus, load_study_data(self.data_directory or DEFAULT_DATA))
            except (OSError, ValueError, KeyError):
                self.study_index = None
        return self.study_index

    def get_retriever(self):
        if self.retriever is None:
            from .retrieval import embedding_dimensions, load_bible_retriever
            self.retriever = load_bible_retriever(self.corpus, dimensions=embedding_dimensions(self.env))
        return self.retriever

    async def call(self, state, stage, system, value, output_schema):
        self.check_deadline()
        if state["conversation"] is not None:
            value = {**value, "conversation": context_wire(state["conversation"], state["contextPlan"])}
            system += "\n" + CONTEXT_PROMPT
        if stage == "answer":
            wire = compact_answer_input(value)
            if wire["compact"]:
                value, system = wire["input"], system+"\n"+COMPACT_EVIDENCE_PROMPT
        body = openai_body(stage, system, value, output_schema, env=self.env)
        budget = {k: deepcopy(state[k]) for k in ('usage', 'callAttempts', 'intentEnabled')}
        # Native tasks preserve completed I/O when the enclosing node fails.
        # Direct node/unit callers still use the same implementation without a graph.
        try:
            active_context = get_runtime().context
        except RuntimeError:
            active_context = None  # Explicit direct call outside a graph.
        if active_context is self:
            receipt = await completed_model_call(stage, body, output_schema['required'], budget)
        else:
            receipt = await self._call_once(stage, body, output_schema['required'], budget)
        if receipt['requestHash'] != sha256(json_text(body)) or receipt['stage'] != stage:
            raise ValueError('MODEL_RECEIPT_REQUEST_CHANGED')
        self.ledger.record(receipt)
        self.ledger.reconcile(state)
        self.check_deadline()
        # Domain decoding is allowed to mutate its answer, never the saved receipt.
        return deepcopy(receipt['response'])

    async def _call_once(self, stage, body, required, state):
        self.ledger.reconcile(state)
        self.check_deadline()
        if state['callAttempts'] >= (8 if state['intentEnabled'] else 6):
            raise BibleProviderError('AI_CALL_LIMIT')
        state['callAttempts'] += 1
        await self.ledger.flush(state)
        self.observe({'stage': stage, 'event': 'start'})
        async with asyncio.timeout(min(50, max(.001, self.expires_at-time.monotonic()))):
            response = await self.fetcher(OPENAI_ENDPOINT, body=body,
                                          headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.env.get('OPENAI_API_KEY', '')}"}, timeout=50)
        self.check_deadline()
        if not response.ok:
            raise BibleProviderError("AI_PROVIDER_ERROR", 429 if response.status == 429 else 502)
        data = response.data
        if not isinstance(data, dict) or not is_expected_model(stage, data.get("model")):
            raise BibleProviderError("AI_MODEL_MISMATCH")
        candidate = (data.get("choices") or [{}])[0]
        u = data.get("usage") or {}
        input_tokens, output_tokens = u.get("prompt_tokens"), u.get("completion_tokens")
        reasoning_tokens = (u.get("completion_tokens_details") or {}).get("reasoning_tokens")
        cached_tokens = (u.get('prompt_tokens_details') or {}).get('cached_tokens')
        cache_usage = {'cachedInputTokens': cached_tokens} if (is_integer(cached_tokens) and is_integer(input_tokens)
                      and 0 <= cached_tokens <= input_tokens) else {}
        write_tokens=(u.get('prompt_tokens_details') or {}).get('cache_write_tokens')
        write_reported=(is_integer(write_tokens) and is_integer(input_tokens)
                        and 0<=write_tokens<=input_tokens-cache_usage.get('cachedInputTokens',0))
        if write_reported:
            cache_usage['cacheWriteInputTokens']=write_tokens
        content = (candidate.get("message") or {}).get("content")
        try:
            result = json.loads(content)
        except (ValueError, TypeError):
            result = None
        self.observe({"stage": stage, "model": data.get("model"), "result": result if result is not None else {"incompleteAnswer": content},
                      'serviceTier':data.get('service_tier'), 'requestedTier':body.get('service_tier','auto'),
                      'cacheWriteReported':write_reported,
                      "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens, "reasoningTokens": reasoning_tokens, **cache_usage}, "finish": candidate.get("finish_reason")})
        if not all(is_integer(n) and n >= 0 for n in (input_tokens, output_tokens, reasoning_tokens)) or reasoning_tokens > output_tokens:
            raise BibleProviderError("AI_USAGE_UNVERIFIED")
        for key, increment in (("inputTokens", input_tokens), ("outputTokens", output_tokens), ("reasoningTokens", reasoning_tokens), ("totalTokens", input_tokens+output_tokens), ("calls", 1)):
            state["usage"][key] += increment
        await self.ledger.flush(state)
        if (output_tokens > STAGES[stage]["maxTokens"] or output_tokens-reasoning_tokens > STAGES[stage]["visibleTokens"]
                or state["usage"]["reasoningTokens"] > MAX_REASONING_TOKENS or candidate.get("finish_reason") != "stop"):
            raise BibleProviderError("AI_INCOMPLETE")
        if not isinstance(result, dict) or set(result) != set(required):
            raise BibleProviderError("AI_INVALID_ANSWER")
        return {'stage': stage, 'requestHash': sha256(json_text(body)),
                'callAttempts': state['callAttempts'], 'usage': dict(state['usage']),
                'response': {"result": result, "visibleTokens": output_tokens-reasoning_tokens}}


@task(retry_policy=RetryPolicy(max_attempts=1))
async def completed_model_call(stage, body, required, budget):
    """Execution-scoped paid result, not a cross-user or semantic answer cache.

    Headers/keys/clients are obtained from context, never task arguments. A
    response lost before the durable task write can still require another call.
    """
    return await get_runtime().context._call_once(stage, body, required, budget)


@asynccontextmanager
async def execution_resources(options):
    """One transport lifetime for ordinary and checkpointed execution."""
    from .transport import ProviderTransport
    transport = None
    values = dict(options)
    try:
        if values.get('fetcher') is None:
            transport = ProviderTransport(values['env'])
            values['fetcher'] = transport.request
        yield values
    finally:
        if transport is not None:
            await transport.aclose()


async def invoke_graph(graph, value, config, context, *, durability=None):
    """Shared deadline, cancellation and child-task cleanup, with no retry."""
    task_run = watcher = None
    context.check_deadline()
    try:
        async with asyncio.timeout(max(.001, context.expires_at-time.monotonic())):
            extra = {'durability': durability} if durability else {}
            task_run = asyncio.create_task(graph.ainvoke(value, config=config, context=context, **extra))
            if context.cancel is not None:
                watcher = asyncio.create_task(context.cancel.wait())
                await asyncio.wait((task_run, watcher), return_when=asyncio.FIRST_COMPLETED)
                if context.cancel.is_set():
                    raise asyncio.CancelledError()
            state = await task_run
            context.check_deadline()
            return state
    finally:
        pending = [p for p in (task_run, watcher) if p is not None]
        for child in pending:
            if not child.done():
                child.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)


class UsageLedger:
    """Write-ahead spending resource, not a second execution/decision state.

    A failed node is not checkpointed, but its attempted/paid calls still count.
    Only monotonic spending is reconciled into GraphState on entry/restart.
    """
    def __init__(self, attempts=0, usage=None, observer=None):
        self.attempts, self.usage, self.observer = attempts, dict(usage or {}), observer

    def reconcile(self, state):
        state["callAttempts"] = max(state["callAttempts"], self.attempts)
        for key, spent in self.usage.items():
            state["usage"][key] = max(state["usage"].get(key, 0), spent)

    def record(self, state):
        self.attempts = max(self.attempts, state["callAttempts"])
        for key, spent in state["usage"].items():
            self.usage[key] = max(self.usage.get(key, 0), spent)

    async def flush(self, state):
        self.record(state)
        if self.observer is not None:
            await self.observer(self.attempts, dict(self.usage))


def create_execution(*, question, history=None, previous_evidence=None, passage=None,
                     conversation=None, evidence_mode="adaptive", **resources):
    """Split public request inputs from runtime capabilities exactly once."""
    from .graph_state import initial_state
    context = BibleRuntime(**resources)
    state = initial_state(context, question=question, history=history, previous_evidence=previous_evidence,
                          passage=passage, conversation=conversation, evidence_mode=evidence_mode)
    return state, context
