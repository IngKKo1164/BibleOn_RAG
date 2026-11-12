"""Private local conversation storage; atomic, owner-scoped optimistic commits."""
from __future__ import annotations

import asyncio
import json
import sqlite3
from contextlib import closing
from pathlib import Path

from .conversation import empty_conversation
from .data import ROOT

DEFAULT_SESSION_DB = ROOT / "data/private/conversations.sqlite3"


class ConversationConflict(ValueError):
    def __init__(self):
        super().__init__("CONVERSATION_CHANGED_RETRY")


class ConversationStore:
    """Local trusted caller only. owner_id must come from auth in a future web adapter."""
    def __init__(self, path=DEFAULT_SESSION_DB):
        self.path = Path(path).resolve()
        if str(self.path).startswith(("\\\\", "//")):
            raise ValueError("LOCAL_CONVERSATION_STORAGE_REQUIRED")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(self._connect()) as db, db:
            db.execute("""CREATE TABLE IF NOT EXISTS conversations (
                owner TEXT NOT NULL, conversation TEXT NOT NULL, revision INTEGER NOT NULL,
                state TEXT NOT NULL, PRIMARY KEY(owner, conversation))""")

    def _connect(self):
        return sqlite3.connect(self.path, timeout=5)

    @staticmethod
    def _key(owner_id, conversation_id):
        if any(not isinstance(v, str) or not v.strip() or len(v) > 128 for v in (owner_id, conversation_id)):
            raise ValueError("CONVERSATION_ID_INVALID")
        return owner_id, conversation_id

    def load(self, owner_id, conversation_id):
        with closing(self._connect()) as db:
            row = db.execute("SELECT revision,state FROM conversations WHERE owner=? AND conversation=?",
                             self._key(owner_id, conversation_id)).fetchone()
        return (row[0], json.loads(row[1])) if row else (0, empty_conversation())

    def commit(self, owner_id, conversation_id, revision, state):
        key = self._key(owner_id, conversation_id)
        payload = json.dumps(state, ensure_ascii=False, allow_nan=False)
        if len(payload) > 20000:
            raise ValueError("CONVERSATION_STATE_TOO_LARGE")
        with closing(self._connect()) as db, db:
            if revision == 0:
                cursor = db.execute("INSERT OR IGNORE INTO conversations VALUES (?,?,1,?)", (*key, payload))
            else:
                cursor = db.execute("UPDATE conversations SET revision=revision+1,state=? WHERE owner=? AND conversation=? AND revision=?",
                                    (payload, *key, revision))
            if cursor.rowcount != 1:
                raise ConversationConflict()
        return revision + 1

    def reset(self, owner_id, conversation_id):
        # Retain a monotonically increasing tombstone to invalidate in-flight answers.
        key = self._key(owner_id, conversation_id)
        with closing(self._connect()) as db, db:
            db.execute("""INSERT INTO conversations VALUES (?,?,1,?)
                ON CONFLICT(owner,conversation) DO UPDATE SET revision=revision+1,state=excluded.state""",
                       (*key, json.dumps(empty_conversation())))


async def run_conversation_turn(*, store, owner_id, conversation_id, runner=None, **options):
    from .graph import run_bible_graph
    if any(key in options for key in ("conversation", "history", "previous_evidence")):
        raise ValueError("CONVERSATION_STATE_IS_STORE_MANAGED")
    revision, state = store.load(owner_id, conversation_id)
    result = await (runner or run_bible_graph)(conversation=state, **options)
    if options.get("cancel") is not None and options["cancel"].is_set():
        raise asyncio.CancelledError()
    # No await between cancellation check and transaction: stale/completed turns cannot interleave here.
    new_revision = store.commit(owner_id, conversation_id, revision, result["conversationState"])
    observer = options.get("observe")
    if observer is not None:
        observer({"stage": "conversation", "event": "committed", "revision": new_revision})
    return {key: value for key, value in result.items() if key != "conversationState"}
