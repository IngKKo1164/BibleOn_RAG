"""Frozen, sequential evaluation batches; private answer review is explicitly opt-in."""
from __future__ import annotations

import argparse
import json
import random
import sys
import uuid
from datetime import datetime, timezone

from .data import ROOT
from .evaluation import load_dataset, pilot_report
from .observability import OPS_ROOT, digest, fingerprints


def trial_plan(cases, repeats, seed):
    if not 1 <= repeats <= 10:
        raise ValueError("EVALUATION_REPEATS_OUT_OF_RANGE")
    rng, plan = random.Random(seed), []
    for trial in range(1, repeats + 1):
        order = list(cases)
        rng.shuffle(order)
        plan.extend((trial, case) for case in order)
    return plan


async def run_evaluation_batch(args, ask):
    from .config import load_env
    from .data import load_corpus, load_study_data, verify_assets
    from .retrieval import load_bible_retriever, embedding_dimensions
    from .study import StudyIndex
    if not args.live:
        raise ValueError("PAID_CALL_REQUIRES_EVALUATE_LIVE")
    verify_assets()
    if hasattr(args, 'profile'):
        from .profiles import load_runtime_env
        env = load_runtime_env(ROOT, profile=args.profile, require_explicit_profile=True)
    else:
        env = load_env()  # Historical direct evaluation fixtures.
    corpus = load_corpus()
    dataset = load_dataset(args.dataset, corpus)
    plan = trial_plan(dataset["cases"], args.repeats, args.seed)
    if not env.get("OPENAI_API_KEY", "").strip():
        raise ValueError("RESEARCH_OPENAI_API_KEY_REQUIRED")
    resources = {"corpus": corpus, "env": env, "retriever": load_bible_retriever(corpus, dimensions=embedding_dimensions(env)),
                 "study": StudyIndex(corpus, load_study_data())}
    batch = uuid.uuid4().hex
    target = OPS_ROOT / "evaluations" / (batch + ".json")
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = {"batchId": batch, "startedAt": datetime.now(timezone.utc).isoformat(),
        "datasetHash": digest(dataset), "versions": fingerprints(env), "seed": args.seed,
        "expectedRuns": len(plan), "concurrency": 1, "automaticRetries": 0,
        "provisional": args.provisional, "evidenceMode": "adaptive",
        "plan": [{"trial": t, "caseHash": digest(c["id"])} for t, c in plan]}
    target.with_suffix(".plan.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    answer_file = ROOT / "artifacts" / "pilot-review" / batch / "answers.jsonl"
    if args.save_answers:
        answer_file.parent.mkdir(parents=True, exist_ok=False)
    print("EVALUATION_BATCH " + batch, file=sys.stderr, flush=True)
    for i, (trial, case) in enumerate(plan, 1):
        query = argparse.Namespace(command="ask", question=case["question"], live=True, no_record=False,
                                   mlflow=args.mlflow, provisional=args.provisional, evidence_mode="adaptive")
        result, error_type = None, None
        try:
            result = await ask(query, resources=resources, gold=case, group=batch)
        except Exception as error:
            error_type = type(error).__name__  # Never persist provider error bodies.
        if args.save_answers:
            with answer_file.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps({"caseId": case["id"], "trial": trial, "question": case["question"],
                    "expectedAnswer": case["expectedAnswer"], "result": result, "errorType": error_type,
                    "humanReview": {"status": "pending", "accuracy": None, "groundedness": None,
                                    "expectedBehaviorSatisfied": None}}, ensure_ascii=False) + "\n")
        print(f"EVALUATION_PROGRESS {i}/{len(plan)} trial={trial} case={digest(case['id'])[:12]} status={'failed' if error_type else 'completed'}", file=sys.stderr, flush=True)
    summaries = [json.loads(p.read_text(encoding="utf-8")) for p in (OPS_ROOT / "runs").glob("*/summary.json")]
    summaries = [s for s in summaries if s.get("group") == digest(batch)]
    report = {**manifest, "recordedRuns": len(summaries), **pilot_report(summaries)}
    if len(summaries) != len(plan) or any("evaluation" not in s or s.get("warnings") for s in summaries):
        report["gate"]["candidateCoverageGate"] = "incomplete_records"
    report["codeUnchangedDuringBatch"] = fingerprints(env) == manifest["versions"]
    if not report["codeUnchangedDuringBatch"]:
        report["gate"]["candidateCoverageGate"] = "configuration_changed"
    target.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    print("EVALUATION_REPORT " + str(target), file=sys.stderr)
    return report
