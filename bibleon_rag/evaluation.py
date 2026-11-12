"""Reviewed verse-level evidence coverage; no gold means unknown, never correct."""
from __future__ import annotations

import json
import math
import re
from pathlib import Path

from .observability import VERSE_ID, digest
from .passages import verse_at

CATEGORIES = {"basic", "general", "advanced"}
STRATA = {"local_context", "cross_passage", "background", "boundary"}


def evidence_id(value):
    return isinstance(value, str) and bool(VERSE_ID.fullmatch(value) or
        re.fullmatch(r"study:[a-f0-9]{64}", value))


def review_status(case):
    return "human_reviewed" if case.get("reviewed") else "human_pending"


def validate_case(case, corpus=None):
    if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"].strip():
        raise ValueError("EVALUATION_CASE_ID_REQUIRED")
    if not isinstance(case.get("question"), str) or not case["question"].strip() or case.get("category") not in CATEGORIES:
        raise ValueError("EVALUATION_QUESTION_CATEGORY_REQUIRED")
    if not isinstance(case.get("reviewed"), bool):
        raise ValueError("EVALUATION_REVIEW_STATUS_REQUIRED")
    requirements = case.get("requirements", [])
    v2 = case.get("schemaVersion") == 2
    if not isinstance(requirements, list) or (case["reviewed"] and not requirements and not v2):
        raise ValueError("REVIEWED_EVIDENCE_REQUIRED")
    if v2:
        if case.get("stratum") not in STRATA or not isinstance(case.get("retrievalRequired"), bool):
            raise ValueError("EVALUATION_SCOPE_REQUIRED")
        if case.get("corpusAvailability") not in {"available", "partial", "unavailable", "not_applicable"}:
            raise ValueError("EVALUATION_AVAILABILITY_REQUIRED")
        if case.get("expectedBehavior") not in {"answer", "clarify", "correct_premise", "partial", "abstain"}:
            raise ValueError("EVALUATION_BEHAVIOR_REQUIRED")
        if not all(isinstance(case.get(k), str) and case[k].strip() for k in ("family", "selectionReason", "expectedAnswer")):
            raise ValueError("EVALUATION_PROVENANCE_REQUIRED")
        if not case.get("sources") or not isinstance(case.get("review"), dict):
            raise ValueError("EVALUATION_REVIEW_REQUIRED")
        review = case["review"]
        if review.get("status") != review_status(case):
            raise ValueError("EVALUATION_REVIEW_MISMATCH")
        if case["reviewed"] and (not review.get("reviewers") or not review.get("reviewedAt")):
            raise ValueError("HUMAN_REVIEW_PROVENANCE_REQUIRED")
        if case["expectedBehavior"] == "answer" and not requirements and case["corpusAvailability"] != "not_applicable":
            raise ValueError("REVIEWED_EVIDENCE_REQUIRED")
    seen = set()
    for req in requirements:
        if not isinstance(req, dict) or not isinstance(req.get("id"), str) or not req["id"] or req["id"] in seen:
            raise ValueError("DUPLICATE_OR_INVALID_REQUIREMENT")
        seen.add(req["id"])
        if req.get("importance", "essential") not in {"essential", "supplementary"}:
            raise ValueError("INVALID_EVIDENCE_IMPORTANCE")
        alternatives = req.get("alternatives")
        if not isinstance(alternatives, list) or not alternatives:
            raise ValueError("GOLD_ALTERNATIVES_REQUIRED")
        for bundle in alternatives:
            if not isinstance(bundle, list) or not bundle or any(not evidence_id(v) for v in bundle):
                raise ValueError("GOLD_CANONICAL_VERSE_IDS_REQUIRED")
            if len(bundle) != len(set(bundle)):
                raise ValueError("DUPLICATE_GOLD_VERSE")
            if corpus is not None:
                for value in bundle:
                    if value.startswith("study:"):
                        continue
                    book, chapter, verse = value.split(":")
                    try:
                        verse_at(corpus, {"bookId": book, "chapter": int(chapter), "verse": int(verse)})
                    except (ValueError, KeyError):
                        raise ValueError("GOLD_VERSE_NOT_IN_CORPUS") from None
    return case


def load_dataset(path, corpus=None):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schemaVersion") not in {1, 2} or not isinstance(data.get("cases"), list) or not data["cases"]:
        raise ValueError("INVALID_EVALUATION_DATASET")
    ids, questions = set(), set()
    for case in data["cases"]:
        if data["schemaVersion"] == 2:
            if case.get("schemaVersion") != 2:
                raise ValueError("CASE_SCHEMA_MISMATCH")
            sources = data.get("sources", {})
            if any(s not in sources for s in case.get("sources", [])):
                raise ValueError("EVALUATION_SOURCE_UNKNOWN")
            evidence = data.get("evidenceCatalog", {})
            for req in case.get("requirements", []):
                for bundle in req.get("alternatives", []):
                    if any(v.startswith("study:") and v not in evidence for v in bundle):
                        raise ValueError("STUDY_GOLD_UNKNOWN")
        validate_case(case, corpus)
        q = digest(" ".join(case["question"].split()))
        if case["id"] in ids or q in questions:
            raise ValueError("DUPLICATE_EVALUATION_CASE")
        ids.add(case["id"]); questions.add(q)
    return data


def coverage(verses, requirements):
    if not requirements:
        return {"anyEvidence": None, "allRequirements": None, "requirementRecall": None,
                "verseRecall": None, "requirements": []}
    available = set(verses)
    scores = [max(len(available & set(bundle))/len(bundle) for bundle in r["alternatives"]) for r in requirements]
    return {"anyEvidence": any(s > 0 for s in scores), "allRequirements": all(s == 1 for s in scores),
            "requirementRecall": sum(s == 1 for s in scores)/len(scores),
            "verseRecall": sum(scores)/len(scores), "requirements": scores}


def evaluate_events(events, case=None, *, provisional=False):
    if case is None or (not validate_case(case)["reviewed"] and not provisional):
        return {"status": "unscored", "reason": "NO_REVIEWED_GOLD", "answerCorrectness": "not_evaluated"}
    requirements = [r for r in case["requirements"] if r.get("importance", "essential") == "essential"]
    supplementary = [r for r in case["requirements"] if r.get("importance") == "supplementary"]
    scored, rows = {}, []
    for event in events:
        if event.get("type") != "evidence":
            continue
        key = f"{event['phase']}.{event['attempt']}"
        row = {"key": key, "phase": event["phase"], "attempt": event["attempt"],
               **coverage(event["verseIds"] + event.get("studyEvidenceIds", []), requirements),
               "supplementaryCoverage": coverage(event["verseIds"] + event.get("studyEvidenceIds", []), supplementary)}
        rows.append(row); scored[key] = row
    losses = []
    for row in rows:
        phase, attempt = row["phase"], row["attempt"]
        def existing(*phases):
            return next((scored[f"{p}.{attempt}"] for p in phases if f"{p}.{attempt}" in scored), None)
        parent = None
        if phase == "budgeted": parent = existing("retrieved")
        elif phase == "screen_preview": parent = existing("budgeted")
        elif phase == "screened": parent = existing("budgeted")
        elif phase == "rerank_input": parent = existing("recovered") if attempt else existing("screened", "budgeted")
        elif phase == "rerank_ranked": parent = existing("rerank_input")
        elif phase == "rerank_selected": parent = existing("rerank_ranked")
        elif phase == "validated_selection": parent = existing("rerank_selected")
        elif phase == "repair_selection": parent = existing("validated_selection")
        elif phase == "final_selection": parent = existing("repair_selection", "validated_selection")
        elif phase == "excerpt": parent = existing("final_selection")
        elif phase == "generation_input":
            parent = next((r for r in reversed(rows[:rows.index(row)]) if r["phase"] in {"excerpt", "direct", "context"}), None)
        elif phase == "citations":
            parent = next((r for r in reversed(rows[:rows.index(row)]) if r["phase"] in {"generation_input", "direct"}), None)
        if parent:
            for i, (before, after) in enumerate(zip(parent["requirements"], row["requirements"])):
                if before > after:
                    losses.append({"requirement": i, "from": parent["key"], "to": row["key"],
                                   "before": before, "after": after,
                                   "observation": "preview_omission" if phase == "screen_preview" else
                                       "citation_omission" if phase == "citations" else "evidence_loss"})
    final = next((r for r in reversed(rows) if r["phase"] in {"generation_input", "direct"}), None)
    for loss in losses:
        loss["presentInFinalInput"] = final["requirements"][loss["requirement"]] == 1 if final else None
    retrieved = scored.get("retrieved.0")
    v2 = case.get("schemaVersion") == 2
    retrieval_required = case.get("retrievalRequired", not any(r["phase"] == "direct" for r in rows))
    scripture_requirements = [r for r in requirements if all(VERSE_ID.fullmatch(v) for b in r["alternatives"] for v in b)]
    retrieved_event = next((e for e in events if e.get("type") == "evidence" and e.get("phase") == "retrieved" and e.get("attempt", 0) == 0), None)
    available = case.get("corpusAvailability", "available")
    rank_coverage = {}
    if retrieval_required and available == "available" and scripture_requirements:
        for k in (20, 50, 100):
            docs = retrieved_event.get("documents", []) if retrieved_event else []
            # This is requirement coverage at a fixed candidate prefix, not document qrel Recall@K.
            included = {v for d in docs[:k] for v in d.get("visibleVerseIds", [])}
            rank_coverage[str(k)] = coverage(included, scripture_requirements)
    return {"status": "scored" if case["reviewed"] else "provisional", "category": case["category"], "caseHash": digest(case["id"]),
            "schemaVersion": 2 if v2 else 1, "familyHash": digest(case.get("family", case["id"])),
            "reviewStatus": review_status(case), "stratum": case.get("stratum", "legacy"),
            "retrievalRequired": retrieval_required, "corpusAvailability": available,
            "expectedBehavior": case.get("expectedBehavior", "answer"), "requirementsCount": len(requirements),
            "supplementaryCount": len(supplementary),
            "rankedRequirementCoverage": rank_coverage,
            "initialScriptureCoverage": coverage(retrieved_event["verseIds"] if retrieved_event else [], scripture_requirements) if scripture_requirements else None,
            "initialRetrievalSkipped": retrieval_required and retrieved is None,
            "unnecessaryRetrieval": not retrieval_required and retrieved is not None,
            "goldHash": digest(case["requirements"]), "questionHash": digest(case["question"]),
            "stages": rows, "losses": losses,
            "initialRetrievalMiss": not retrieved["allRequirements"] if retrieved and requirements else None,
            "finalInputComplete": final["allRequirements"] if final else None,
            "answerCorrectness": "not_evaluated"}


def read_run(path):
    directory = Path(path)
    summary = json.loads((directory / "summary.json").read_text(encoding="utf-8"))
    events = [json.loads(line) for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines() if line]
    return summary, events


def aggregate(summaries, *, threshold=.95, min_samples=100):
    """Conservative denominator: missing stage records cannot silently improve the rate."""
    eligible = [s for s in summaries if s.get("evaluation", {}).get("status") == "scored" and s.get("mode") == "live"
                and s["evaluation"].get("retrievalRequired", False)
                and s["evaluation"].get("requirementsCount", 1) > 0
                and s["evaluation"].get("corpusAvailability", "available") == "available"]
    stages = {}
    unique = {}
    for summary in sorted(eligible, key=lambda s: (s.get("startedAt", ""), s.get("runId", ""))):
        unique.setdefault(summary["evaluation"]["caseHash"], summary)
    for phase in ("retrieved", "rerank_input", "final_selection", "excerpt", "generation_input"):
        scored = [next((r for r in reversed(s["evaluation"]["stages"]) if r["phase"] == phase), None) for s in eligible]
        observed = [r for r in scored if r is not None]
        n, success = len(eligible), sum(r["allRequirements"] is True for r in observed)
        rate = success/n if n else None
        # Repeated trials never increase the independent-question denominator.
        first = [next((r for r in s["evaluation"]["stages"] if r["phase"] == phase), None) for s in unique.values()]
        un = len(first)
        first_rate = sum(bool(r and r["allRequirements"]) for r in first)/un if un else None
        z = 1.96
        lower = ((first_rate+z*z/(2*un)-z*math.sqrt(first_rate*(1-first_rate)/un+z*z/(4*un*un)))/(1+z*z/un)) if un else None
        stages[phase] = {"eligible": n, "observed": len(observed), "complete": success, "completeRate": rate,
                         "anyEvidenceRate": sum(r["anyEvidence"] is True for r in observed)/n if n else None,
                         "uniqueFirstTrialCompleteRate": first_rate, "wilsonUniqueQuestions": un,
                         "wilsonLower95": lower}
    retrieval = stages["retrieved"]
    unique_cases = len({s["evaluation"]["caseHash"] for s in eligible})
    gate = "insufficient_data" if unique_cases < min_samples else "pass" if retrieval["uniqueFirstTrialCompleteRate"] >= threshold else "fail"
    categories = {}
    for category in sorted(CATEGORIES):
        subset = [s for s in eligible if s["evaluation"]["category"] == category]
        initial = [next((r for r in s["evaluation"]["stages"] if r["phase"] == "retrieved"), None) for s in subset]
        categories[category] = {"runs": len(subset),
            "retrievalCompleteRate": sum(bool(r and r["allRequirements"]) for r in initial)/len(subset) if subset else None,
            "finalInputCompleteRate": sum(s["evaluation"]["finalInputComplete"] is True for s in subset)/len(subset) if subset else None}
    return {"runs": len(summaries), "uniqueReviewedRetrievalCases": unique_cases, "liveReviewedRetrievalRuns": len(eligible), "stages": stages,
            "candidateCoverageGate": gate, "target": threshold, "minimumSamples": min_samples,
            "unscoredRuns": sum(s.get("evaluation", {}).get("status") != "scored" for s in summaries),
            "byCategory": categories,
            "answerCorrectness": "not_evaluated"}


def percentile(values, fraction):
    if not values:
        return None
    values = sorted(values)
    position = (len(values)-1)*fraction
    lower = int(position)
    return values[lower] + (values[min(lower+1, len(values)-1)]-values[lower])*(position-lower)


def pilot_report(summaries):
    """Exploratory scores remain visibly separate from the human-reviewed release gate."""
    scored = [s for s in summaries if s.get("evaluation", {}).get("status") in {"scored", "provisional"} and s.get("mode") == "live"]
    def group(rows):
        eligible = [s for s in rows if s["evaluation"]["retrievalRequired"] and s["evaluation"]["corpusAvailability"] == "available"
                    and s["evaluation"].get("initialScriptureCoverage") is not None]
        final = [s for s in rows if s["evaluation"]["requirementsCount"] and s["evaluation"]["corpusAvailability"] == "available"]
        mean = lambda values: sum(values)/len(values) if values else None
        times = [s["elapsedMs"] for s in rows]
        successful = [s["elapsedMs"] for s in rows if s.get("status") == "completed"]
        by_case = {}
        for s in rows:
            by_case.setdefault(s["evaluation"]["caseHash"], []).append(s)
        return {"runs": len(rows), "uniqueQuestions": len(by_case),
            "uniqueFamilies": len({s["evaluation"]["familyHash"] for s in rows}),
            "failures": sum(s.get("status") != "completed" for s in rows),
            "candidateEligibleRuns": len(eligible),
            "candidateEligibleQuestions": len({s["evaluation"]["caseHash"] for s in eligible}),
            "candidateAllRequiredScriptureRate": mean([bool(s["evaluation"]["initialScriptureCoverage"]["allRequirements"]) for s in eligible]),
            "candidateMeanRequirementRecall": mean([s["evaluation"]["initialScriptureCoverage"]["requirementRecall"] for s in eligible]),
            "candidatePrefixCoverage": {str(k): mean([s["evaluation"]["rankedRequirementCoverage"].get(str(k), {}).get("requirementRecall", 0) for s in eligible]) for k in (20, 50, 100)},
            "finalEligibleRuns": len(final),
            "finalAllRequiredEvidenceRate": mean([s["evaluation"]["finalInputComplete"] is True for s in final]),
            "requiredRetrievalSkipped": sum(s["evaluation"]["initialRetrievalSkipped"] for s in eligible),
            "unnecessaryRetrieval": sum(s["evaluation"]["unnecessaryRetrieval"] for s in rows),
            "latencyMsAll": {"p50": percentile(times, .5), "p95": percentile(times, .95), "mean": mean(times)},
            "latencyMsCompleted": {"p50": percentile(successful, .5), "p95": percentile(successful, .95)},
            "completedWithin15sRate": mean([s.get("status") == "completed" and s["elapsedMs"] <= 15000 for s in rows]),
            "uncertainResponses": sum(s.get("answerKind") == "uncertain" for s in rows),
            "usage": {k: sum(s.get("usage", {}).get(k, 0) for s in rows) for k in ("inputTokens", "outputTokens", "reasoningTokens", "modelCalls", "embeddingInputTokens")}}
    return {"labelStatus": "provisional" if any(s["evaluation"]["status"] == "provisional" for s in scored) else "human_reviewed",
            "answerCorrectness": "requires_human_review", "gate": aggregate(summaries),
            "overall": group(scored), "byStratum": {x: group([s for s in scored if s["evaluation"]["stratum"] == x]) for x in sorted(STRATA)},
            "byMode": {m: group([s for s in scored if s.get("evidenceMode", "adaptive") == m]) for m in sorted({s.get("evidenceMode", "adaptive") for s in scored})},
            "notes": ["Candidate metrics concern scripture retrieval only; background spans are evaluated in final input.",
                      "Prefix scores are essential-requirement coverage, not conventional document Recall@K.",
                      "Repeated trials are not independent questions. No population guarantee or answer-accuracy claim.",
                      "unnecessaryRetrieval means search occurred although draft gold did not require it; it is not automatically a routing error.",
                      "Latency starts before the recorded graph; resource preload and external answer review are excluded."]}


def compare_runs(left, right):
    def signature(s):
        return (s.get("questionHash"), s.get("mode"), s.get("evaluation", {}).get("goldHash"), s.get("conversationHash"))
    if signature(left) != signature(right):
        raise ValueError("COMPARE_REQUIRES_SAME_QUESTION_GOLD_AND_MODE")
    def stages(s):
        return {r["key"]: r for r in s.get("evaluation", {}).get("stages", [])}
    a, b = stages(left), stages(right)
    return {"left": left["runId"], "right": right["runId"], "status": [left["status"], right["status"]],
            "elapsedMsDelta": right["elapsedMs"]-left["elapsedMs"],
            "usageDelta": {k: right.get("usage", {}).get(k, 0)-left.get("usage", {}).get(k, 0)
                           for k in set(left.get("usage", {})) | set(right.get("usage", {}))},
            "coverage": [{"stage": key, "left": a.get(key, {}).get("requirementRecall"),
                           "right": b.get(key, {}).get("requirementRecall")} for key in sorted(set(a) | set(b))]}
