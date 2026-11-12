"""Private local LangGraph checkpoints, explicit resume and immutable-parent forks.

Trusted local callers only: owner_id is a namespace, never authentication.
Checkpoint contents are private working data, not MLOps logs or Git artifacts.
"""
from __future__ import annotations

from contextlib import asynccontextmanager
from copy import deepcopy
import json
from pathlib import Path
import time
import uuid
import zlib

import aiosqlite
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

from .common import json_text, sha256
from .config import RESEARCH_DEFAULTS
from .data import ROOT
from .graph import GRAPH_RECURSION_LIMIT, assert_local_tracing, build_bible_graph
from .graph_state import completed_result, encode, index_identity, validate_state
from .runtime import UsageLedger, create_execution, execution_resources, invoke_graph

DEFAULT_CHECKPOINT_DB = ROOT / "data/private/graph-checkpoints.sqlite3"
REQUEST_KEYS = ("question", "history", "previous_evidence", "passage", "conversation", "evidence_mode")
FORK_SETTINGS = {"BIBLE_EMBEDDING_DIMENSIONS", "BIBLE_LATENCY_MODE"}


class JsonCheckpointSerializer:
    """No pickle, arbitrary class loading, or fallback serialization."""
    def dumps_typed(self, value):
        return "json-zlib", zlib.compress(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8"))

    def loads_typed(self, value):
        kind, content = value
        if kind != "json-zlib":
            raise ValueError("CHECKPOINT_SERIALIZATION_FORMAT")
        return json.loads(zlib.decompress(content))


def settings(env):
    values = {key: env.get(key) for key in RESEARCH_DEFAULTS}
    from .profiles import PROFILE_KEY, PROFILE_VERSION_KEY
    for key in (PROFILE_KEY, PROFILE_VERSION_KEY):
        if key in env:
            values[key] = env[key]
    return values


def signature(context, state):
    # Bind checkpoints to actual source/data, not filenames or a mutable version label.
    source = "\n".join(p.name+":"+sha256(p.read_bytes()) for p in sorted((ROOT/"bibleon_rag").glob("*.py")))
    source += "\n"+sha256((ROOT/"requirements.lock").read_bytes())
    index = index_identity(context.retriever) if context.retriever is not None else None
    study = context.get_study_index(state['studyEnabled'])
    if study is not None and not hasattr(study, "checkpoint_identity"):
        study.checkpoint_identity = sha256(json_text(study.data))
    return {"schema": 1, "source": sha256(source), "corpus": sha256(json_text(context.corpus)),
            "index": index, "study": study.checkpoint_identity if study else None}


class CheckpointStore:
    def __init__(self, path=DEFAULT_CHECKPOINT_DB):
        self.path = Path(path).resolve()
        if str(self.path).startswith(("\\\\", "//")):
            raise ValueError("LOCAL_CHECKPOINT_STORAGE_REQUIRED")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = self.saver = None

    async def __aenter__(self):
        assert_local_tracing()
        self.conn = await aiosqlite.connect(self.path)
        await self.conn.execute("PRAGMA busy_timeout=5000")
        await self.conn.execute("""CREATE TABLE IF NOT EXISTS executions (
            thread_id TEXT PRIMARY KEY, execution_id TEXT NOT NULL, request TEXT NOT NULL,
            settings TEXT NOT NULL, signature TEXT NOT NULL, parent TEXT,
            status TEXT NOT NULL DEFAULT 'ready', attempts INTEGER NOT NULL DEFAULT 0,
            usage TEXT NOT NULL DEFAULT '{}', active_ms REAL NOT NULL DEFAULT 0,
            lease TEXT, lease_until REAL NOT NULL DEFAULT 0)""")
        await self.conn.commit()
        self.saver = AsyncSqliteSaver(self.conn, serde=JsonCheckpointSerializer())
        await self.saver.setup()
        return self

    async def __aexit__(self, *args):
        await self.conn.close()

    @staticmethod
    def key(owner_id, execution_id):
        if any(not isinstance(v, str) or not v.strip() or len(v) > 128 for v in (owner_id, execution_id)):
            raise ValueError("CHECKPOINT_ID_INVALID")
        return sha256(json_text([owner_id, execution_id]))

    def config(self, owner_id, execution_id, checkpoint_id=None):
        value = {"thread_id": self.key(owner_id, execution_id), "checkpoint_ns": ""}
        if checkpoint_id:
            value["checkpoint_id"] = checkpoint_id
        return {"configurable": value, "recursion_limit": GRAPH_RECURSION_LIMIT, "callbacks": []}

    async def _record(self, owner_id, execution_id):
        async with self.conn.execute("SELECT request,settings,signature,parent,status,attempts,usage,active_ms FROM executions WHERE thread_id=?",
                                     (self.key(owner_id, execution_id),)) as cursor:
            row = await cursor.fetchone()
        if row is None:
            raise ValueError("CHECKPOINT_EXECUTION_NOT_FOUND")
        return dict(zip(("request", "settings", "signature", "parent", "status", "attempts", "usage", "active_ms"),
                        [json.loads(v) if i in (0, 1, 2, 3, 6) and v is not None else v for i, v in enumerate(row)]))

    @asynccontextmanager
    async def _lease(self, owner_id, execution_id):
        key, lease = self.key(owner_id, execution_id), uuid.uuid4().hex
        cursor = await self.conn.execute("UPDATE executions SET lease=?,lease_until=? WHERE thread_id=? AND (lease IS NULL OR lease_until<?)",
                                         (lease, time.time()+125, key, time.time()))
        await self.conn.commit()
        if cursor.rowcount != 1:
            raise ValueError("CHECKPOINT_EXECUTION_BUSY")
        try:
            yield
        finally:
            await self.conn.execute("UPDATE executions SET lease=NULL,lease_until=0 WHERE thread_id=? AND lease=?", (key, lease))
            await self.conn.commit()

    async def create(self, *, owner_id, execution_id=None, **options):
        execution_id = execution_id or uuid.uuid4().hex
        state, context = create_execution(**options)
        request = {key: deepcopy(options[key]) for key in REQUEST_KEYS if key in options}
        # Encode/decode validates this payload independently of provider credentials.
        payload = json_text(encode(request))
        try:
            await self.conn.execute("INSERT INTO executions(thread_id,execution_id,request,settings,signature) VALUES(?,?,?,?,?)",
                (self.key(owner_id, execution_id), execution_id, payload, json_text(settings(options["env"])), json_text(signature(context, state))))
            await self.conn.commit()
        except aiosqlite.IntegrityError as error:
            raise ValueError("CHECKPOINT_EXECUTION_EXISTS") from error
        return execution_id

    async def history(self, *, owner_id, execution_id):
        await self._record(owner_id, execution_id)
        graph = build_bible_graph(checkpointer=self.saver)
        return [{"checkpointId": s.config["configurable"]["checkpoint_id"], "next": list(s.next),
                 "node": s.values.get("_lastNode"), "step": s.metadata.get("step"), "createdAt": s.created_at}
                async for s in graph.aget_state_history(self.config(owner_id, execution_id))]

    async def state(self, *, owner_id, execution_id, checkpoint_id=None):
        await self._record(owner_id, execution_id)
        graph = build_bible_graph(checkpointer=self.saver)
        snapshot = await graph.aget_state(self.config(owner_id, execution_id, checkpoint_id))
        if not snapshot.values:
            raise ValueError("CHECKPOINT_NOT_FOUND")
        return snapshot

    async def run(self, *, owner_id, execution_id, pause_after=None, **resources):
        """Start/resume the latest checkpoint; successful terminal results are reused.

        A deliberate older-checkpoint replay uses fork(), so the parent execution
        and any committed conversation remain unchanged. No implicit retries.
        """
        if set(resources) & set(REQUEST_KEYS):
            raise ValueError("CHECKPOINT_REQUEST_IS_FROZEN")
        async with self._lease(owner_id, execution_id):
            record = await self._record(owner_id, execution_id)
            if settings(resources["env"]) != record["settings"]:
                raise ValueError("CHECKPOINT_SETTINGS_CHANGED")
            async with execution_resources(resources) as resources:
                initial, context = create_execution(**resources, **record["request"])
                if signature(context, initial) != record["signature"]:
                    raise ValueError("CHECKPOINT_ASSETS_OR_CODE_CHANGED")
                context.expires_at = time.monotonic()+max(0, 110-record["active_ms"]/1000)
                context.active_ms_before = record["active_ms"]
                context.invocation_started_at = time.monotonic()
                key = self.key(owner_id, execution_id)
                async def budget(attempts, usage):
                    await self.conn.execute("UPDATE executions SET attempts=?,usage=? WHERE thread_id=?",
                                             (attempts, json_text(usage), key))
                    await self.conn.commit()
                context.ledger = UsageLedger(record['attempts'], record['usage'], budget)
                graph = build_bible_graph(checkpointer=self.saver, interrupt_after=pause_after)
                config = self.config(owner_id, execution_id)
                current = await graph.aget_state(config)
                # A canceled/paused execution never advances canonical conversation state.
                started = time.monotonic()
                failure = None
                try:
                    if not current.values or current.next:
                        context.check_deadline()
                        value = None if current.values else initial
                        await invoke_graph(graph, value, config, context, durability="sync")
                    current = await graph.aget_state(config)
                    state = deepcopy(current.values)
                    validate_state(state)
                    context.ledger.reconcile(state)
                    context.ledger.record(state)
                    status = "paused" if current.next else "completed"
                    if status == "completed" and not state.get('result'):
                        raise ValueError("CHECKPOINT_RESULT_MISSING")
                    result = completed_result(state, context) if status == 'completed' else None
                    if result is not None:
                        result['usage'] = dict(state['usage'])
                    return {"executionId": execution_id, "status": status, "next": list(current.next),
                            "checkpointId": current.config["configurable"]["checkpoint_id"], "result": result,
                            "usage": dict(state['usage']), "attempts": state['callAttempts']}
                except BaseException as error:
                    failure = error
                    raise
                finally:
                    await self.conn.execute("UPDATE executions SET status=?,active_ms=active_ms+?,attempts=?,usage=? WHERE thread_id=?",
                        ("failed" if failure else status, (time.monotonic()-started)*1000,
                         context.ledger.attempts, json_text(context.ledger.usage), key))
                    await self.conn.commit()

    async def fork(self, *, owner_id, execution_id, checkpoint_id, new_execution_id=None, **resources):
        """Native aupdate_state on a new thread, from an actual saved node output."""
        source = await self._record(owner_id, execution_id)
        saved = await self.state(owner_id=owner_id, execution_id=execution_id, checkpoint_id=checkpoint_id)
        if not saved.values.get("_lastNode") or not saved.next:
            raise ValueError("FORK_REQUIRES_EXECUTED_NONTERMINAL_CHECKPOINT")
        changed = {k for k, v in settings(resources["env"]).items() if source["settings"].get(k) != v}
        if changed - FORK_SETTINGS:
            raise ValueError("UNSUPPORTED_FORK_SETTINGS")
        if "BIBLE_EMBEDDING_DIMENSIONS" in changed and ("retrieval" in saved.values or not set(saved.next) <= {"retrieve", "enrich", "prepare"}):
            raise ValueError("DIMENSION_FORK_REQUIRES_PRE_RETRIEVAL_STATE")
        if "BIBLE_LATENCY_MODE" in changed and saved.next != ("budget",) and saved.next != ("retrieve",):
            raise ValueError("SELECTION_FORK_REQUIRES_PRE_SELECTION_STATE")
        initial, context = create_execution(**resources, **source["request"])
        current = signature(context, initial)
        allowed_signature = {"index"} if "BIBLE_EMBEDDING_DIMENSIONS" in changed else set()
        if any(current[k] != source["signature"][k] for k in current if k not in allowed_signature):
            raise ValueError("FORK_ASSETS_OR_CODE_CHANGED")
        if allowed_signature and current["index"]["chunks"] != source["signature"]["index"]["chunks"]:
            raise ValueError("FORK_CHUNKS_CHANGED")
        new_id = await self.create(owner_id=owner_id, execution_id=new_execution_id, **resources, **source["request"])
        values = deepcopy(saved.values)
        graph = build_bible_graph(checkpointer=self.saver)
        await graph.aupdate_state(self.config(owner_id, new_id), values, as_node=values["_lastNode"])
        copied = await graph.aget_state(self.config(owner_id, new_id))
        if copied.next != saved.next:
            raise ValueError("FORK_NEXT_NODE_MISMATCH")
        state = values
        validate_state(state)
        await self.conn.execute("UPDATE executions SET parent=?,attempts=?,usage=?,active_ms=? WHERE thread_id=?",
            (json_text({"executionId": execution_id, "checkpointId": checkpoint_id}), state["callAttempts"],
             json_text(state["usage"]), state.get("activeMilliseconds", 0), self.key(owner_id, new_id)))
        await self.conn.commit()
        return new_id

    async def delete(self, *, owner_id, execution_id):
        await self._record(owner_id, execution_id)
        async with self._lease(owner_id, execution_id):
            key = self.key(owner_id, execution_id)
            await self.saver.adelete_thread(key)
            await self.conn.execute("DELETE FROM executions WHERE thread_id=?", (key,))
            await self.conn.commit()
