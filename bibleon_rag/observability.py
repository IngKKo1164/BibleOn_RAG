"""Local, allowlisted observations. Never persist prompts, model text or credentials."""
from __future__ import annotations

import hashlib
import json
import math
import re
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import RESEARCH_DEFAULTS, STAGES
from .data import ROOT

OPS_ROOT = ROOT / "artifacts" / "mlops"
PHASES = {"retrieved_raw", "retrieved", "budgeted", "screen_preview", "screened", "rerank_input", "rerank_ranked", "study_selected", "final_review_input",
          "rerank_selected", "validated_selection", "repair_selection", "recovered", "final_selection",
          "excerpt", "generation_input", "direct", "context", "context_review", "citations", "profile_evidence"}
NODES = {"route", "retrieve", "budget", "screen", "rerank", "recover", "rerank_recovered", "select_evidence",
         "enrich", "prepare", "generate", "check_citations", "repair_citations", "assess", "check_facts",
         "repair_facts", "validate", "plan", "assess_evidence"}
USAGE_KEYS = {"inputTokens", "outputTokens", "reasoningTokens", "embeddingInputTokens", "cachedInputTokens", "cacheWriteInputTokens"}
VERSE_ID = re.compile(r"[a-z0-9]+:[1-9][0-9]*:[1-9][0-9]*")
ERROR_CODES = {"AI_NOT_CONFIGURED", "AI_REFERENCE_INVALID", "AI_CALL_LIMIT", "AI_PROVIDER_ERROR",
    "AI_MODEL_MISMATCH", "AI_USAGE_UNVERIFIED", "AI_INCOMPLETE", "AI_INVALID_ANSWER", "AI_CITATION_INVALID",
    "AI_EVIDENCE_ASSESSMENT_INVALID", "AI_FACTS_UNSUPPORTED", "AI_EVIDENCE_MISSING", "AI_EMPTY_ANSWER",
    "AI_LANGUAGE_INVALID", "AI_REPAIR_ASSESSMENT_INVALID", "AI_SELECTION_INVALID", "TimeoutError", "CancelledError",
    "INVALID_INTENT_PLAN", "EVIDENCE_REVIEW_INVALID", "SELECTION_STRUCTURE", "SELECTION_MEMBERSHIP",
    "SELECTION_FOCUS_COORDINATES", "SELECTION_FOCUS_MISSING", "ANSWER_ENDING_INVALID",
    "REVIEW_STRUCTURE", "REVIEW_QUOTE_INVALID", "REVIEW_SUPPORT_STATUS", "REVIEW_EVENT_PROOF_MISSING",
    "REVIEW_SCENE_SUBJECT_MISSING", "AI_SCENE_COVERAGE_INVALID", "AI_FINAL_REVIEW_INVALID"}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def safe_id(value):
    # Unknown candidate/source identifiers are hashed rather than leaking arbitrary strings.
    return value if isinstance(value, str) and re.fullmatch(r"[a-z0-9]+:[0-9]+:[0-9]+(?:-[0-9]+)?", value) else "id-" + digest(value)[:24]


def verse_ids(rows):
    values = [v.get("id") or f"{v.get('bookId')}:{v.get('chapter')}:{v.get('verse')}" for v in rows if isinstance(v, dict)]
    return sorted({v for v in values if VERSE_ID.fullmatch(v)})


def study_spans(text):
    """Deterministic verbatim spans, not semantic assertions; plaintext is never logged."""
    lines = text.splitlines()
    pieces = [text] + lines + [p for line in lines for p in re.split(r"(?<=[.!?])\s+", line)]
    return sorted({" ".join(p.split()) for p in pieces if len(" ".join(p.split())) >= 16})


def study_evidence_ids(sources):
    return sorted({"study:" + digest({"recordId": s["recordId"], "span": p})
        for s in sources if isinstance(s.get("recordId"), str)
        for p in study_spans(s.get("excerpt", ""))})


def snapshot(context, phase, *, pool=None, ids=None, verses=None, wire=None, attempt=0, sources=()):
    """Document membership and actual preview text coverage are separate facts."""
    documents, included = [], []
    chosen = set(ids) if ids is not None else None
    excerpts = {wire.canonical[c[0]]: c for c in wire.candidates} if wire is not None else {}
    hits = (pool or {}).get("hits", [])
    if ids is not None:
        by_id = {h["chunk"]["id"]: h for h in hits}
        hits = [by_id[i] for i in dict.fromkeys(ids) if i in by_id]
    for hit in hits:
        chunk = hit["chunk"]
        if chosen is not None and chunk["id"] not in chosen:
            continue
        all_ids = [f"{chunk['bookId']}:{chunk['chapter']}:{v['verse']}" for v in chunk["verses"]]
        if wire is None or excerpts.get(chunk["id"], [None, None, 0])[2] == 1:
            visible = all_ids
        else:
            # Only complete verse lines count as visible. A shortened preview is not full evidence.
            lines = set(excerpts.get(chunk["id"], [None, None, 0, ""])[3].splitlines())
            visible = [key for key, v in zip(all_ids, chunk["verses"]) if f"{v['verse']} {v['text']}" in lines]
        documents.append({"id": chunk["id"], "verseIds": all_ids, "visibleVerseIds": visible})
        included.extend(visible)
    if verses is not None:
        included = verse_ids(verses)
    context.observe({"stage": "evidence-flow", "phase": phase, "attempt": attempt,
                     "documents": documents, "verseIds": sorted(set(included)),
                     "studyEvidenceIds": study_evidence_ids(sources)})


def fingerprints(env):
    files = sorted((ROOT / "bibleon_rag").glob("*.py"))
    manifest = ROOT / "python-runtime-assets.json"
    options = {k: env.get(k, v) for k, v in RESEARCH_DEFAULTS.items()}
    from .profiles import PROFILE_KEY, PROFILE_VERSION_KEY
    for key in (PROFILE_KEY, PROFILE_VERSION_KEY):
        if key in env:
            options[key] = env[key]
    return {"code": digest({p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}),
            "assetsManifest": hashlib.sha256(manifest.read_bytes()).hexdigest() if manifest.exists() else "unavailable",
            "configuration": digest({"stages": STAGES, "options": options})}


def number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


class RunRecorder:
    """Fail-open observer: a broken disk must not discard a successfully generated answer."""
    def __init__(self, question, env, *, root=OPS_ROOT, mode="live", group=None):
        if mode not in {"live", "lookup", "replay"}:
            raise ValueError("INVALID_RUN_MODE")
        self.root = Path(root)
        self.run_id = uuid.uuid4().hex
        self.directory = self.root / "runs" / self.run_id
        self.started = time.monotonic()
        self.events, self.usage, self.warnings = [], {}, []
        self.summary = {"schemaVersion": 1, "runId": self.run_id, "mode": mode, "status": "running",
                        "questionHash": digest(question), "startedAt": datetime.now(timezone.utc).isoformat(),
                        "group": digest(group) if group is not None else None}
        try:
            self.summary["versions"] = fingerprints(env)
            self.summary["models"] = {stage: spec["model"] for stage, spec in STAGES.items()}
            self.directory.mkdir(parents=True, exist_ok=False)
            (self.directory / "events.jsonl").touch(exist_ok=False)
            self._write_summary()
        except Exception:
            self.warnings.append("LOCAL_RECORD_INITIALIZATION_FAILED")

    def _write_summary(self):
        temp = self.directory / "summary.tmp"
        temp.write_text(json.dumps(self.summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        temp.replace(self.directory / "summary.json")

    def observe(self, event):
        try:
            row = self._sanitize(event)
            if row is None:
                return
            row.update(sequence=len(self.events), elapsedMs=round((time.monotonic()-self.started)*1000, 3))
            self.events.append(row)
            with (self.directory / "events.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        except Exception:
            if "LOCAL_EVENT_WRITE_FAILED" not in self.warnings:
                self.warnings.append("LOCAL_EVENT_WRITE_FAILED")

    def _sanitize(self, event):
        stage = event.get("stage")
        if stage == 'processing-decision':
            from .routing import PROCESSING_VERSION
            if event.get('version') == PROCESSING_VERSION and event.get('profile') in {'simple', 'adaptive'}:
                row = {k: event[k] for k in ('profile', 'version', 'essentialNeedCount', 'threshold', 'basis')}
                self.summary['processing'] = row
                return {'type': stage, **row}
            return None
        if stage == 'route-decision':
            from .routing import ROUTES, VERSION
            if event.get('route') in ROUTES and event.get('policy') == VERSION:
                row = {'type': stage, 'route': event['route'], 'policy': VERSION,
                       'invalid': event.get('invalid') is True}
                self.summary['routing'] = row
                return row
            return None
        if stage in {'research-budget', 'research-selection', 'validation-contract', 'research-evidence-budget'}:
            data = event.get('result', {})
            allowed = {'research-budget': {'broad', 'retained', 'candidateLimit'},
                       'research-selection': {'candidateCount', 'selected', 'semanticSelectionEvaluated'},
                       'validation-contract': {'sourceIdentityChecked', 'semanticTruthCertified', 'automaticProseDeletion'},
                       'research-evidence-budget': {'verseLimit', 'characterLimit', 'retainedVerses', 'characters', 'sourceCharacters'}}[stage]
            values = {k: v for k, v in data.items() if k in allowed and
                      (type(v) is bool or type(v) is int and 0 <= v <= 100000)}
            if stage == 'research-evidence-budget':
                values['droppedVerses'] = len(data.get('droppedIds', []))
                values['droppedSources'] = len(data.get('droppedSourceIds', []))
            if data.get('profile') in {'simple', 'adaptive'}:
                values['profile'] = data['profile']
            return {'type': stage, **values}
        if (stage == 'plan-normalization' and event.get('reason') == 'unanchored_interpretation'
                and type(event.get('needsChanged')) is int and 0 <= event['needsChanged'] <= 6):
            return {'type': stage, 'reason': event['reason'], 'needsChanged': event['needsChanged']}
        if (stage == 'selection-budget' and type(event.get('omittedAssignments')) is int
                and 1 <= event['omittedAssignments'] <= 36):
            return {'type': stage, 'method': 'declared_ranked_top3_per_need',
                    'omittedAssignments': event['omittedAssignments']}
        if (stage == 'literal-scan' and type(event.get('checkedVerses')) is int
                and 1 <= event['checkedVerses'] <= 100000
                and type(event.get('spansChecked')) is int and 1 <= event['spansChecked'] <= 3
                and type(event.get('absentSpans')) is int and 0 <= event['absentSpans'] <= event['spansChecked']):
            return {'type': stage, 'method': 'normalized_joined_numbered_text',
                    **{key: event[key] for key in ('checkedVerses', 'spansChecked', 'absentSpans')}}
        if stage == 'selection-policy' and event.get('mode') in {'baseline', 'compact', 'fused', 'selection', 'strict', 'conditional', 'review-first'}:
            return {'type': stage, 'mode': event['mode'], 'merged': event.get('merged') is True}
        if stage == 'review-audit' and all(isinstance(event.get(k), int) and not isinstance(event[k], bool)
                                          and 0 <= event[k] <= 8 for k in ('retained', 'retracted')):
            return {'type': stage, 'retained': event['retained'], 'retracted': event['retracted']}
        if stage == "diagnostic" and event.get("component") in {"plan", "assess_evidence", "screen", "rerank", "validate"} and event.get("code") in ERROR_CODES:
            return {"type": "diagnostic", "component": event["component"], "code": event["code"]}
        if stage == "evidence-mode" and event.get("mode") in {"adaptive", "passage", "context"}:
            self.summary["evidenceMode"] = event["mode"]
            return {"type": "evidence-mode", "mode": event["mode"]}
        if stage in {"intent-decision", "evidence-review"}:
            kinds = {"context", "scripture", "history", "culture", "lexicon"}
            row = {"type": stage, "degraded": event.get("degraded") is True}
            if stage == "intent-decision":
                if event.get("intent") not in {"interpretation", "search", "social", "outOfScope", "safety", "clarify", "sourceUnavailable"}:
                    return None
                row.update(intent=event["intent"], needs=[k for k in event.get("needs", []) if k in kinds])
            else:
                row["checks"] = [{"kind": c["kind"], "status": c["status"]} for c in event.get("checks", [])
                    if c.get("kind") in kinds and c.get("status") in {"supported", "refuted", "missing", "conflict"}]
            return row
        if stage == "conversation":
            from .conversation import MODES
            if event.get("event") == "input" and event.get("mode") in MODES:
                state_hash = digest(event.get("state"))
                self.summary["conversationHash"] = state_hash
                return {"type": "conversation", "event": "input", "mode": event["mode"],
                        "stateHash": state_hash, "usesContext": event.get("usesContext") is True,
                        **{k: event[k] for k in ("messages", "characters") if number(event.get(k))}}
            if event.get("event") == "committed" and number(event.get("revision")):
                return {"type": "conversation", "event": "committed", "revision": event["revision"]}
            return None
        if stage == "evidence-flow" and event.get("phase") in PHASES:
            clean = lambda values: sorted({v for v in values if isinstance(v, str) and VERSE_ID.fullmatch(v)})
            study_ids = [v for v in event.get("studyEvidenceIds", []) if isinstance(v, str) and re.fullmatch(r"study:[a-f0-9]{64}", v)]
            return {"type": "evidence", "phase": event["phase"], "attempt": 1 if event.get("attempt") == 1 else 0,
                    **({"studyEvidenceIds": sorted(set(study_ids))} if study_ids else {}),
                    "verseIds": clean(event.get("verseIds", [])), "documents": [
                        {"id": safe_id(d["id"]), "verseIds": clean(d["verseIds"]), "visibleVerseIds": clean(d["visibleVerseIds"])}
                        for d in event.get("documents", [])]}
        if stage == "graph" and event.get("node") in NODES and event.get("event") in {"start", "end", "error"}:
            row = {"type": "node", "node": event["node"], "event": event["event"]}
            if event.get("code") in ERROR_CODES:
                row["code"] = event["code"]
            if number(event.get("milliseconds")):
                row["milliseconds"] = event["milliseconds"]
            return row
        if (stage == 'final_review' and event.get('event') == 'checked'
                and event.get('decision') == 'accepted'
                and type(event.get('targets')) is int and 0 <= event['targets'] <= 8
                and type(event.get('edits')) is int and 0 <= event['edits'] <= 8):
            return {'type': 'final_review', 'decision': 'accepted',
                    'targets': event['targets'], 'edits': event['edits']}
        if stage in STAGES and event.get("event") == "start":
            self.usage["modelCalls"] = self.usage.get("modelCalls", 0) + 1
            return {"type": "call", "stage": stage, "event": "start"}
        if stage in set(STAGES) | {"retrieve", "retrieval-recovery"} and isinstance(event.get("usage"), dict):
            usage = {k: v for k, v in event["usage"].items() if k in USAGE_KEYS and number(v)}
            if stage in STAGES:
                self.usage["reportedModelCalls"] = self.usage.get("reportedModelCalls", 0) + 1
            for k, v in usage.items():
                self.usage[k] = self.usage.get(k, 0) + v
            result = event.get("result") or {}
            return {"type": "usage", "stage": stage, "values": usage,
                    **({'serviceTier':event['serviceTier']} if event.get('serviceTier') in {'default','fast','priority','flex'} else {}),
                    **({'requestedTier':event['requestedTier']} if event.get('requestedTier') in {'auto','default','fast','priority','flex'} else {}),
                    **({'cacheWriteReported':event['cacheWriteReported']} if isinstance(event.get('cacheWriteReported'),bool) else {}),
                    "degraded": bool(result.get("degraded") or result.get("embeddingDegraded"))}
        if stage in {"retrieve", "retrieval-recovery", "study-unavailable"} and event.get("result", {}).get("degraded"):
            return {"type": "degraded", "stage": stage}
        return None

    def finish(self, *, result=None, error=None):
        if result is not None:
            self.observe({"stage": "evidence-flow", "phase": "citations", "verseIds": verse_ids(result.get("citations", []))})
        import asyncio
        status = "cancelled" if isinstance(error, (asyncio.CancelledError, KeyboardInterrupt)) else "failed" if error else "completed"
        self.summary.update(status=status, elapsedMs=round((time.monotonic()-self.started)*1000, 3),
                            usage=self.usage, warnings=list(self.warnings), eventCount=len(self.events))
        if result is not None and result.get("kind") in {"answer", "uncertain", "outOfScope", "greeting", "safety"}:
            self.summary["answerKind"] = result["kind"]
        if error is not None:
            # Exception messages and provider bodies may contain the question or a credential.
            self.summary["errorType"] = type(error).__name__ if type(error).__module__ == "builtins" else "ExecutionError"
            code = getattr(error, "code", type(error).__name__)
            if code in ERROR_CODES:
                self.summary["errorCode"] = code
        self.persist()

    def persist(self):
        try:
            self._write_summary()
        except Exception:
            if "LOCAL_SUMMARY_WRITE_FAILED" not in self.warnings:
                self.warnings.append("LOCAL_SUMMARY_WRITE_FAILED")
            self.summary["warnings"] = list(self.warnings)
