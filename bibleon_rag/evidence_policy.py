"""Bounded evidence-preserving policies; no benchmark labels or answer mappings."""
from .common import unique
from .retrieval import candidate_excerpt


def reliability(state, context, stage):
    """Cumulative, independently replayable fixes; absent flag preserves old runs."""
    value = str(getattr(context, "env", {}).get("BIBLE_RELIABILITY_STAGE", "0"))
    if value not in {"0", "1", "2", "3", "4", "5", "6"}:
        raise ValueError("INVALID_RELIABILITY_STAGE")
    return state.get("intentEnabled", False) and int(value) >= stage


def policy_stage(env):
    value = str(env.get("BIBLE_EVIDENCE_POLICY_STAGE", "0"))
    if value not in {"0", "1", "2", "3", "4", "5"}:
        raise ValueError("INVALID_EVIDENCE_POLICY_STAGE")
    return int(value)


def enabled(state, context, stage):
    from .profiles import execution_profile
    profile = execution_profile(state, getattr(context, 'env', {}))
    if profile is not None and not profile.adaptive:
        return False
    return state["intentEnabled"] and policy_stage(getattr(context, "env", {})) >= stage


def need_queries(needs, original):
    """At most two additional semantic searches; original always remains first."""
    result, seen = [], {original.strip()}
    ordered = sorted(needs, key=lambda n: (n["importance"] != "essential", n["kind"] == "context"))
    for n in ordered:
        query = (n.get("searchQuery") or n["need"]).strip()
        if query and query not in seen:
            result.append({"id": n["needId"], "query": query})
            seen.add(query)
        if len(result) == 2:
            break
    return result


def protected_candidates(pool, per_branch=2):
    """Reserve independent need branches, never hard-coded scripture addresses."""
    branches = {}
    anchors = []
    for h in pool["hits"]:
        key = h["chunk"]["id"]
        for p in h.get("paths", []):
            if p["path"] == "explicit":
                anchors.append(key)
            if p["path"].startswith("need-") or p["path"] in {"early-crossref", "boundary-neighbour"}:
                branches.setdefault(p["path"], []).append((p["rank"], key))
    return unique(anchors + [key for rows in branches.values() for _, key in sorted(rows)[:per_branch]])[:24]


def balanced_pool(pool, limit):
    keep = set(unique(protected_candidates(pool, 4) + [h["chunk"]["id"] for h in pool["hits"]])[:limit])
    return {**pool, "hits": [h for h in pool["hits"] if h["chunk"]["id"] in keep],
            "candidates": [c for c in pool["candidates"] if c["id"] in keep]}


def preview_budgets(pool, characters=160):
    """Redistribute a fixed per-candidate allowance toward branch leaders."""
    protected = set(protected_candidates(pool))
    count = len(pool["hits"])
    boosted = min(len(protected), count)
    high = min(2*characters, characters*count//max(1, boosted))
    low = max(0, (characters*count-high*boosted)//max(1, count-boosted))
    return {h["chunk"]["id"]: high if h["chunk"]["id"] in protected else low for h in pool["hits"]}


def need_preview(chunk, question, needs, budget):
    """Round-robin complete verse windows for distinct needs, inside the old budget."""
    queries = unique([(n.get("searchQuery") or n["need"]) for n in needs] + [question])
    queues = [candidate_excerpt(chunk, q, max_characters=budget)["excerpt"].splitlines() for q in queries]
    selected, used = [], 0
    while any(queues):
        for q in queues:
            if not q:
                continue
            line = q.pop(0)
            if line not in selected and used + len(line) + bool(selected) <= budget:
                selected.append(line); used += len(line) + (len(selected) > 1)
    selected.sort(key=lambda line: int(line.split()[0]))
    return {"excerpt": "\n".join(selected), "complete": len(selected) == len(chunk["verses"])}


def need_parts(state, context):
    return [{"partId": n["needId"], "question": n["need"]} for n in state["intentPlan"]["needs"]]


def merge_need_selection(state, plan, context):
    by_need = {n["needId"]: n for n in state["intentPlan"]["needs"]}
    # Prioritize essential needs while keeping total selected chunks <= 6.
    rows = sorted(plan["coverage"], key=lambda p: by_need[p["partId"]]["importance"] != "essential")
    keys = unique(p["chunkIds"][i] for i in range(3) for p in rows if i < len(p["chunkIds"]))[:6]
    coverage = [{**p, "chunkIds": unique(key for n in rows if by_need[n["partId"]]["partId"] == p["partId"]
                 for key in n["chunkIds"] if key in keys)} for p in state["questionParts"]]
    return {**plan, "chunkIds": keys, "coverage": coverage, "focus": {k: plan["focus"][k] for k in keys},
            "needCoverage": rows, "missingParts": [p for p in coverage if not p["chunkIds"]]}


def bounded_need_selection(state, decoded, context):
    """A selected chunk without an actual focus is deferred, never given invented verses."""
    focus_ids = {f["chunkId"] for f in decoded["focus"]}
    needs = sorted(state["intentPlan"]["needs"], key=lambda n: n["importance"] != "essential")
    groups = [decoded["evidenceByPart"].get(n["needId"], []) for n in needs]
    selected = unique(row[i] for i in range(3) for row in groups
                      if i < len(row) and row[i] in focus_ids)[:6]
    return {**decoded, "evidenceByPart": {k: [i for i in ids if i in selected]
             for k, ids in decoded["evidenceByPart"].items()}}


def expand_discovery(pool, corpus, primary, question, study_index, *, seed_budget=12):
    """Add real neighbours/one-hop references before selection, with a fixed pool cap."""
    if type(seed_budget) is not int or seed_budget not in {12, 24, 48}:
        raise ValueError("INVALID_DISCOVERY_SEEDS")
    from .study import expand_one_hop
    index = pool["index"]
    roots = list(primary)
    seeds = unique(protected_candidates(pool, 1) + [h["chunk"]["id"] for h in pool["hits"][:seed_budget//2]])[:seed_budget]
    seeds_set = set(seeds)
    targets = []
    for h in pool["hits"]:
        c = h["chunk"]
        if c["id"] not in seeds_set:
            continue
        chapters = corpus[c["bookId"]]["chapters"]
        for chapter, verse in ((c["chapter"], c["start"]-1), (c["chapter"], c["end"]+1)):
            if verse < 1 and str(chapter-1) in chapters:
                chapter, verse = chapter-1, max(map(int, chapters[str(chapter-1)]))
            elif str(verse) not in chapters.get(str(chapter), {}) and str(chapter+1) in chapters:
                chapter, verse = chapter+1, 1
            targets.append(("boundary-neighbour", c["bookId"], chapter, verse))
    if study_index and roots:
        linked = expand_one_hop(corpus, roots, question, study_index)["linked"]
        targets += [("early-crossref", v["bookId"], v["chapter"], v["verse"]) for v in linked]
    by_id = {h["chunk"]["id"]: {**h, "paths": list(h.get("paths", []))} for h in pool["hits"]}
    added = []
    for rank, (path, book, chapter, verse) in enumerate(targets):
        for c in index.chunks:
            if c["bookId"] != book or c["chapter"] != chapter or not c["start"] <= verse <= c["end"]:
                continue
            if c["id"] not in by_id:
                if len(added) >= 24:
                    continue
                by_id[c["id"]] = {"chunk": c, "score": 0, "paths": []}
                added.append(c["id"])
            by_id[c["id"]]["paths"].append({"path": path, "rank": rank+1})
    ids = unique([h["chunk"]["id"] for h in pool["hits"][:420-len(added)]] + added)
    hits = [by_id[k] for k in ids]
    return {**pool, "hits": hits, "candidates": [{"id": h["chunk"]["id"], "reference": h["chunk"]["reference"],
        "complete": True, "excerpt": "\n".join(f"{v['verse']} {v['text']}" for v in h["chunk"]["verses"])} for h in hits]}
