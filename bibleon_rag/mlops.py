"""Recorded graph execution and explicit local MLflow integration, without autologging."""
from __future__ import annotations

import os
from pathlib import Path

from .evaluation import evaluate_events, validate_case
from .observability import OPS_ROOT, RunRecorder, digest


def local_paths(root=OPS_ROOT):
    root = Path(root).resolve()
    if str(root).startswith(("\\\\", "//")):
        raise ValueError("LOCAL_TRACKING_DIRECTORY_REQUIRED")
    root.mkdir(parents=True, exist_ok=True)
    return root, "sqlite:///" + (root / "tracking.db").as_posix()


def local_client(root=OPS_ROOT):
    # Never honor inherited remote tracking/registry/workspace destinations.
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    from mlflow import MlflowClient
    root, uri = local_paths(root)
    return MlflowClient(tracking_uri=uri, registry_uri=uri, workspace_store_uri=uri)


def export_mlflow(recorder):
    client = local_client(recorder.root)
    artifacts = (recorder.root / "mlflow-artifacts").resolve().as_uri()
    name = "bibleon-rag-local"
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        try:
            client.create_experiment(name, artifact_location=artifacts)
        except Exception:
            if client.get_experiment_by_name(name) is None:
                raise
        experiment = client.get_experiment_by_name(name)
    if experiment.artifact_location != artifacts:
        raise ValueError("NONLOCAL_MLFLOW_ARTIFACT_LOCATION")
    summary = recorder.summary
    run = client.create_run(experiment.experiment_id, tags={"mlflow.runName": summary["runId"],
        "local_run_id": summary["runId"], "mode": summary["mode"], "status": summary["status"],
        "question_hash": summary["questionHash"], "gold_status": summary.get("evaluation", {}).get("status", "unscored")})
    run_id = run.info.run_id
    try:
        for key, value in summary.get("versions", {}).items():
            client.log_param(run_id, key+"_hash", value)
        for key, value in summary.get("models", {}).items():
            client.log_param(run_id, key+"_model", value)
        client.log_metric(run_id, "elapsed_ms", summary["elapsedMs"])
        for key, value in summary.get("usage", {}).items():
            client.log_metric(run_id, key, value)
        for event in recorder.events:
            if event["type"] == "node" and event["event"] == "end":
                client.log_metric(run_id, "node_ms."+event["node"], event["milliseconds"], step=event["sequence"])
            if event["type"] == "evidence":
                key = event["phase"]+"."+str(event["attempt"])
                client.log_metric(run_id, "documents."+key, len(event["documents"]))
                client.log_metric(run_id, "verses."+key, len(event["verseIds"]))
        for stage in summary.get("evaluation", {}).get("stages", []):
            for metric in ("anyEvidence", "allRequirements", "requirementRecall", "verseRecall"):
                if stage[metric] is not None:
                    client.log_metric(run_id, metric+"."+stage["key"], float(stage[metric]))
        # Both artifacts come from the allowlisted recorder, never provider inputs/outputs.
        client.log_dict(run_id, summary, "summary.json")
        client.log_dict(run_id, {"events": recorder.events}, "evidence-flow.json")
        client.set_terminated(run_id, status={"completed": "FINISHED", "failed": "FAILED", "cancelled": "KILLED"}[summary["status"]])
        return run_id
    except BaseException:
        client.set_terminated(run_id, status="FAILED")
        raise


async def run_recorded_graph(*, recorder=None, root=OPS_ROOT, mode="live", gold=None, mlflow=False, provisional=False,
                             store=None, owner_id=None, conversation_id=None, **options):
    from .graph import run_bible_graph
    if gold is not None:
        validate_case(gold, options["corpus"])
        if gold["question"] != options["question"]:
            raise ValueError("GOLD_QUESTION_MISMATCH")
    recorder = recorder or RunRecorder(options["question"], options["env"], root=root, mode=mode)
    recorder.summary.setdefault("versions", {})["corpus"] = digest(options["corpus"])
    original_observer = options.get("observe")
    def observe(event):
        recorder.observe(event)
        if original_observer is not None:
            original_observer(event)
    options["observe"] = observe
    result, failure = None, None
    try:
        if store is not None:
            from .sessions import run_conversation_turn
            result = await run_conversation_turn(store=store, owner_id=owner_id, conversation_id=conversation_id, **options)
        else:
            result = await run_bible_graph(**options)
        return result
    except BaseException as error:
        failure = error
        raise
    finally:
        recorder.finish(result=result, error=failure)
        try:
            recorder.summary["evaluation"] = evaluate_events(recorder.events, gold, provisional=provisional)
            if mlflow:
                recorder.summary["mlflowRunId"] = export_mlflow(recorder)
        except Exception:
            recorder.warnings.append("EVALUATION_OR_TRACKING_FAILED")
        recorder.summary["warnings"] = list(recorder.warnings)
        recorder.persist()


def serve_ui(*, root=OPS_ROOT, port=5000):
    if not 1024 <= port <= 65535:
        raise ValueError("INVALID_UI_PORT")
    root, uri = local_paths(root)
    # Launch without inherited remote tracing destinations or provider credentials.
    for key in list(os.environ):
        if key.startswith(("MLFLOW_", "LANGSMITH_", "LANGCHAIN_", "OTEL_")) or key in {"OPENAI_API_KEY", "QWEN_API_KEY"}:
            os.environ.pop(key, None)
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "true"
    from mlflow.cli import cli
    cli.main(args=["server", "--backend-store-uri", uri,
                   "--default-artifact-root", (root / "mlflow-artifacts").as_uri(),
                   "--no-serve-artifacts", "--host", "127.0.0.1", "--port", str(port), "--workers", "1"],
             prog_name="mlflow", standalone_mode=False)
    return 0
