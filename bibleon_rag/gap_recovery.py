"""One shared gap recovery, using stable needs and already verified evidence."""
import re
from .common import unique


def recovery_allowed(state, context):
    from .profiles import execution_profile
    policy = execution_profile(state, context.env)
    if policy is not None and not policy.adaptive:
        return False
    # Reserve rerank, second review, generation and one citation repair.
    return (state["intentEnabled"] and not state.get("recoveryCount", 0)
            and context.env.get("BIBLE_RETRIEVAL_REPAIR", "on") != "off"
            and state["callAttempts"] <= 4)


def gap_queries(state, context):
    parts = {p["partId"]: p["question"] for p in state["questionParts"]}
    expansion = context.env.get("BIBLE_REPAIR_QUERY_EXPANSION") != "off"
    rows, seen = [], set()
    gaps = [c for c in state.get("evidenceAssessment", [])
            if c["importance"] == "essential" and c["status"] in {"missing", "conflict"}]
    for gap in gaps:
        # Feedback is a retrieval hint, never a source or a new question/answer.
        hint = (gap.get("feedbackQuery") or gap.get("searchQuery") or gap["need"]) if expansion else ""
        query = parts[gap["partId"]] + ("\n확인할 근거: " + hint if hint else "")
        if query not in seen:
            rows.append({"partId": gap["partId"], "need": gap["need"], "query": query, "expanded": bool(hint)})
            seen.add(query)
        if len(rows) == 2:
            break
    return rows


def reviewed_background_queries(raw, checks, previous, *, ignore_invalid_hints=False):
    """Update only missing essential source relations, within the same 3-query cap."""
    rows = raw.get("backgroundRecoveryQueries")
    by_id = {c["needId"]: c for c in checks}
    if not isinstance(rows, list) or len(rows) > 2:
        raise ValueError("EVIDENCE_REVIEW_INVALID")
    additions = []
    for row in rows:
        if (not isinstance(row, dict) or set(row) != {"needId", "query"}
                or row.get("needId") not in by_id or not isinstance(row.get("query"), str)
                or not re.fullmatch(r"[ -~]{1,180}", row["query"]) or not row["query"].strip()):
            if ignore_invalid_hints:
                # Retrieval hints never approve or replace source checks. Keep
                # the original missing need and its ordinary recovery query.
                continue
            raise ValueError("EVIDENCE_REVIEW_INVALID")
        check = by_id[row["needId"]]
        if (check["importance"] == "essential" and check["status"] in {"missing", "conflict"}
                and check["kind"] in {"history", "culture", "lexicon"}):
            additions.append(row["query"].strip())
    return unique(additions + list(previous))[:3]


def merge_sources(state, incoming):
    """Keep supported excerpts and their IDs; never bind an old S-id to new text."""
    used = {a["evidenceId"] for c in state.get("evidenceAssessment", []) for a in c["supports"]}
    old = state.get("studySources", [])
    result = [s for s in old if s["id"] in used]
    records = {s["recordId"] for s in result}
    serial = max((int(s["id"][1:]) for s in old), default=0)
    chars = sum(len(s["excerpt"]) for s in result)
    for source in incoming:
        if source["recordId"] in records or len(result) >= 3 or chars+len(source["excerpt"]) > 4000:
            continue
        serial += 1
        result.append({**source, "id": f"S{serial}"})
        records.add(source["recordId"])
        chars += len(source["excerpt"])
    return result


def merge_plans(previous, incoming):
    """Retain per-part membership/focus as well as text, so later checks can cite it."""
    if incoming["route"] != "bible":
        return previous
    coverage = [{**p, "chunkIds": unique(p["chunkIds"] + next(
        (r["chunkIds"] for r in incoming["coverage"] if r["partId"] == p["partId"]), []))}
        for p in previous["coverage"]]
    return {**incoming, "coverage": coverage,
        "chunkIds": unique(previous["chunkIds"]+incoming["chunkIds"]),
        "rankedIds": unique(previous["rankedIds"]+incoming["rankedIds"]),
        "focus": {key: unique(previous["focus"].get(key, [])+incoming["focus"].get(key, []))
                  for key in unique(list(previous["focus"])+list(incoming["focus"]))}}
