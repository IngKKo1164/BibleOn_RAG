"""Per-invocation capabilities stay outside serializable LangGraph state."""
from __future__ import annotations

import asyncio
import json
import re
import time

from .common import BibleProviderError, compact_previous, is_integer, json_text
from .config import MAX_REASONING_TOKENS, OPENAI_ENDPOINT, STAGES
from .passages import resolve_bible_request, selection_parts
from .prompt_text import COMPACT_EVIDENCE_PROMPT
from .selection import compact_answer_input


def openai_body(stage, system, value, output_schema):
    config = STAGES.get(stage)
    if config is None:
        raise BibleProviderError("AI_NOT_CONFIGURED", 503)
    return {"model": config["model"], "reasoning_effort": config["effort"],
            "max_completion_tokens": config["maxTokens"], "stream": False, "store": False,
            "response_format": {"type": "json_schema", "json_schema": {"name": "bible_"+stage, "strict": True, "schema": output_schema}},
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": json_text(value)}]}


def is_expected_model(stage, model):
    expected = STAGES.get(stage, {}).get("model")
    return bool(expected) and isinstance(model, str) and (model == expected or
            (model.startswith(expected+"-") and re.fullmatch(r"\d{4}-\d{2}-\d{2}", model[len(expected)+1:]) is not None))


class BibleRuntime:
    def __init__(self, *, env, corpus, question, history=None, previous_evidence=None,
                 passage=None, fetcher=None, observe=None, retriever=None, study_index=None, cancel=None, data_directory=None):
        if env.get("BIBLE_AI_PROVIDER") != "openai" or env.get("BIBLE_RETRIEVAL_VERSION") != "v4":
            raise BibleProviderError("AI_NOT_CONFIGURED", 503)
        self.env, self.corpus, self.question = dict(env), corpus, question
        self.history, self.previous_evidence, self.passage = history or [], previous_evidence, passage or []
        self.fetcher, self.observe = fetcher, observe or (lambda event: None)
        self.retriever, self.study_index = retriever, study_index
        self.cancel, self.expires_at = cancel, time.monotonic()+110
        self.data_directory = data_directory
        self.scope = resolve_bible_request(corpus, question, self.passage)
        self.observe({"stage": "route", "result": {"route": self.scope["route"], "primaryVerses": len(self.scope["primary"]), "contextVerses": len(self.scope["context"])}})
        if self.scope["invalid"]:
            raise BibleProviderError("AI_REFERENCE_INVALID", 400)
        self.study_enabled = env.get("BIBLE_RAG_STUDY") == "one-hop" and (self.scope["sources"] or self.scope["crossrefs"])
        self.wire_previous = compact_previous(previous_evidence)
        self.question_parts = selection_parts(question)
        self.usage = {"inputTokens": 0, "outputTokens": 0, "reasoningTokens": 0, "answerTokens": 0, "totalTokens": 0, "calls": 0}
        self._study_loaded = study_index is not None

    def check_deadline(self):
        if self.cancel is not None and self.cancel.is_set():
            raise asyncio.CancelledError()
        if time.monotonic() >= self.expires_at:
            raise TimeoutError("AI_TIMEOUT")

    def get_study_index(self):
        if not self._study_loaded:
            self._study_loaded = True
            if self.study_enabled:
                from .data import DEFAULT_DATA, load_study_data
                from .study import StudyIndex
                try:
                    self.study_index = StudyIndex(self.corpus, load_study_data(self.data_directory or DEFAULT_DATA))
                except (OSError, ValueError, KeyError):
                    self.study_index = None
        return self.study_index

    async def call(self, stage, system, value, output_schema):
        self.check_deadline()
        if self.usage["calls"] >= 6:
            raise BibleProviderError("AI_CALL_LIMIT")
        self.observe({"stage": stage, "event": "start"})
        if stage == "answer":
            wire = compact_answer_input(value)
            if wire["compact"]:
                value, system = wire["input"], system+"\n"+COMPACT_EVIDENCE_PROMPT
        body = openai_body(stage, system, value, output_schema)
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
        content = (candidate.get("message") or {}).get("content")
        try:
            result = json.loads(content)
        except (ValueError, TypeError):
            result = None
        self.observe({"stage": stage, "model": data.get("model"), "result": result if result is not None else {"incompleteAnswer": content},
                      "usage": {"inputTokens": input_tokens, "outputTokens": output_tokens, "reasoningTokens": reasoning_tokens}, "finish": candidate.get("finish_reason")})
        if not all(is_integer(n) and n >= 0 for n in (input_tokens, output_tokens, reasoning_tokens)) or reasoning_tokens > output_tokens:
            raise BibleProviderError("AI_USAGE_UNVERIFIED")
        for key, increment in (("inputTokens", input_tokens), ("outputTokens", output_tokens), ("reasoningTokens", reasoning_tokens), ("totalTokens", input_tokens+output_tokens), ("calls", 1)):
            self.usage[key] += increment
        if (output_tokens > STAGES[stage]["maxTokens"] or output_tokens-reasoning_tokens > STAGES[stage]["visibleTokens"]
                or self.usage["reasoningTokens"] > MAX_REASONING_TOKENS or candidate.get("finish_reason") != "stop"):
            raise BibleProviderError("AI_INCOMPLETE")
        if not isinstance(result, dict) or set(result) != set(output_schema["required"]):
            raise BibleProviderError("AI_INVALID_ANSWER")
        return {"result": result, "visibleTokens": output_tokens-reasoning_tokens}
