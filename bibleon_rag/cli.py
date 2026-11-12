"""Python-only research entry points; paid operations require an explicit flag."""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

from .data import ROOT


def parser():
    result = argparse.ArgumentParser(prog="bibleon-rag", description="성경 근거 검색과 LangGraph 답변 생성")
    commands = result.add_subparsers(dest="command", required=True)
    for name in ("ask", "lookup"):
        command = commands.add_parser(name)
        command.add_argument("question")
        command.add_argument("--no-record", action="store_true", help="로컬 실행 기록 생략")
        command.add_argument("--mlflow", action="store_true", help="로컬 MLflow에도 기록")
        command.add_argument("--conversation", help="같은 ID로 이전 대화 계속 (비공개 로컬 저장)")
        command.add_argument("--owner", default="local", help="로컬 대화 소유자 네임스페이스")
        if name == "ask":
            command.add_argument("--live", action="store_true", help="유료 API 호출 허용")
            command.add_argument('--profile', choices=('auto', 'simple', 'adaptive'),
                                 help='질문별 자동 실행 / 고정 Simple·Adaptive 비교 조건')
            command.add_argument("--evidence-mode", choices=("adaptive", "passage", "context"), default="adaptive",
                                 help="질문별 RAG / 지정 본문만 / 고정 소제목 문맥 비교")
    command = commands.add_parser("check")
    command.add_argument("--with-data", action="store_true")
    command = commands.add_parser("check-git")
    command.add_argument("--staged", action="store_true")
    command = commands.add_parser('graph')
    command.add_argument('--profile', choices=('auto', 'simple', 'adaptive'))
    command = commands.add_parser('profile')
    command.add_argument('--profile', choices=('auto', 'simple', 'adaptive'))
    for name in ('answer-review-template', 'score-answer-use'):
        command = commands.add_parser(name)
        command.add_argument('bundle', type=Path, help='실제 답변·전달 근거·평가 문항의 비공개 JSON')
        if name == 'score-answer-use':
            command.add_argument('--review', type=Path, help='실제 답변·근거에 연결한 별도 의미 검수')
    for name in ("checkpoint-start", "checkpoint-resume", "checkpoint-fork", "checkpoint-history", "checkpoint-delete"):
        command = commands.add_parser(name)
        command.add_argument("question" if name == "checkpoint-start" else "execution")
        command.add_argument("--owner", default="local")
        if name in ("checkpoint-start", "checkpoint-resume", "checkpoint-fork"):
            command.add_argument("--index-dir", type=Path, help="별도 차원의 로컬 색인 경로")
        if name in ("checkpoint-start", "checkpoint-resume"):
            command.add_argument("--live", action="store_true", required=True)
            command.add_argument("--pause-after", choices=("plan", "retrieve", "prepare", "generate"))
            command.add_argument('--profile', choices=('auto', 'simple', 'adaptive'))
        if name == "checkpoint-fork":
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--dimensions", type=int, choices=(512, 1024))
    command = commands.add_parser("conversation-reset")
    command.add_argument("conversation")
    command.add_argument("--owner", default="local")
    for name in ("data-restore", "data-pack", "build-headings"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
    command = commands.add_parser("build-index")
    command.add_argument("--live", action="store_true")
    command = commands.add_parser("evaluate")
    command.add_argument("dataset", type=Path)
    command.add_argument("--live", action="store_true", help="평가 질문의 유료 호출 허용")
    command.add_argument('--profile', choices=('auto', 'simple', 'adaptive'))
    command.add_argument("--mlflow", action="store_true")
    command.add_argument("--provisional", action="store_true", help="사람 검수 전 초안의 잠정 점수 (공식 통과 판정 제외)")
    command.add_argument("--repeats", type=int, default=1)
    command.add_argument("--seed", type=int, default=20261019)
    command.add_argument("--save-answers", action="store_true", help="별도 비공개 검수 파일에 실제 답변 저장")
    command = commands.add_parser("runs")
    command.add_argument("--limit", type=int, default=10)
    command = commands.add_parser("inspect-run")
    command.add_argument("run_id")
    command = commands.add_parser("score-run")
    command.add_argument("run_id")
    command.add_argument("--dataset", type=Path, required=True)
    command.add_argument("--case", required=True)
    command = commands.add_parser("compare-runs")
    command.add_argument("left")
    command.add_argument("right")
    command = commands.add_parser("ops-ui")
    command.add_argument("--port", type=int, default=5000)
    return result


async def question(args, *, resources=None, gold=None, group=None):
    from .profiles import load_runtime_env
    from .data import load_corpus, load_study_data, verify_assets
    from .graph import run_bible_graph
    from .mlops import run_recorded_graph
    from .observability import RunRecorder
    from .passages import resolve_bible_request
    from .retrieval import load_bible_retriever, embedding_dimensions
    from .study import StudyIndex
    if args.command == "ask" and not args.live:
        raise ValueError("PAID_CALL_REQUIRES_ASK_LIVE")
    if resources is None:
        verify_assets()
        corpus, env = load_corpus(), load_runtime_env(ROOT, profile=getattr(args, 'profile', None),
                                                    require_explicit_profile=args.command == 'ask')
    else:
        corpus, env = resources["corpus"], resources["env"]
    options = dict(env=env, corpus=corpus, question=args.question, evidence_mode=getattr(args, "evidence_mode", "adaptive"))
    if args.command == "lookup":
        scope = resolve_bible_request(corpus, args.question, [])
        if not scope["lookup"] or scope["invalid"] or not scope["primary"]:
            raise ValueError("LOOKUP_REQUIRES_VALID_EXACT_REFERENCE")
        async def no_network(*args, **kwargs):
            raise ValueError("LOOKUP_NETWORK_FORBIDDEN")
        options["fetcher"] = no_network
    else:
        if not env.get("OPENAI_API_KEY", "").strip():
            raise ValueError("RESEARCH_OPENAI_API_KEY_REQUIRED")
        options.update(retriever=resources["retriever"] if resources else load_bible_retriever(corpus, dimensions=embedding_dimensions(env)),
                       study_index=resources["study"] if resources else StudyIndex(corpus, load_study_data()))
    session = {}
    if getattr(args, "conversation", None):
        from .sessions import ConversationStore
        session = {"store": ConversationStore(), "owner_id": args.owner, "conversation_id": args.conversation}
    if args.no_record:
        if args.mlflow:
            raise ValueError("MLFLOW_REQUIRES_RECORDING")
        if session:
            from .sessions import run_conversation_turn
            return await run_conversation_turn(**session, **options)
        return await run_bible_graph(**options)
    recorder = RunRecorder(args.question, env, mode="lookup" if args.command == "lookup" else "live", group=group)
    try:
        return await run_recorded_graph(recorder=recorder, gold=gold, mlflow=args.mlflow,
                                        provisional=getattr(args, "provisional", False), **session, **options)
    finally:
        print("로컬 실행 기록: " + str(recorder.directory), file=sys.stderr)
        if recorder.warnings:
            print("기록 경고: " + ", ".join(recorder.warnings), file=sys.stderr)


def run_path(run_id):
    import re
    from .observability import OPS_ROOT
    if not re.fullmatch(r"[a-f0-9]{32}", run_id):
        raise ValueError("INVALID_RUN_ID")
    return OPS_ROOT / "runs" / run_id


async def evaluate_dataset(args):
    from .pilot import run_evaluation_batch
    return await run_evaluation_batch(args, question)


async def checkpoint_command(args):
    from .checkpoints import CheckpointStore
    from .profiles import load_runtime_env
    from .data import DEFAULT_DATA, load_corpus, load_study_data
    from .retrieval import load_bible_retriever, embedding_dimensions
    from .study import StudyIndex
    async with CheckpointStore() as store:
        if args.command == "checkpoint-history":
            return await store.history(owner_id=args.owner, execution_id=args.execution)
        if args.command == "checkpoint-delete":
            await store.delete(owner_id=args.owner, execution_id=args.execution)
            return {"deleted": True}
        env, corpus = load_runtime_env(ROOT, profile=getattr(args, 'profile', None),
                                      require_explicit_profile=args.command == 'checkpoint-start'), load_corpus()
        if args.command != "checkpoint-start":
            saved = await store._record(args.owner, args.execution)
            for key, value in saved["settings"].items():
                if value is None:
                    env.pop(key, None)
                else:
                    env[key] = value
        if getattr(args, "dimensions", None):
            env["BIBLE_EMBEDDING_DIMENSIONS"] = str(args.dimensions)
        directory = args.index_dir or DEFAULT_DATA
        resources = {"env": env, "corpus": corpus,
            "retriever": load_bible_retriever(corpus, directory=directory, dimensions=embedding_dimensions(env)),
            "study_index": StudyIndex(corpus, load_study_data())}
        if args.command == "checkpoint-fork":
            execution = await store.fork(owner_id=args.owner, execution_id=args.execution,
                checkpoint_id=args.checkpoint, **resources)
            return {"executionId": execution, "executed": False, "networkCalls": 0}
        execution = (await store.create(owner_id=args.owner, question=args.question, **resources)
                     if args.command == "checkpoint-start" else args.execution)
        return await store.run(owner_id=args.owner, execution_id=execution,
            pause_after=[args.pause_after] if args.pause_after else None, **resources)



def execute(args):
    if args.command in ('answer-review-template', 'score-answer-use'):
        from .answer_use import review_template, score_answer_use
        bundle = json.loads(args.bundle.read_text(encoding='utf8'))
        values = [bundle[k] for k in ('case', 'result', 'materials')]
        if args.command == 'answer-review-template':
            return review_template(*values)
        review = json.loads(args.review.read_text(encoding='utf8')) if args.review else None
        return score_answer_use(*values, review)
    if args.command == 'profile':
        from .profiles import load_runtime_env, describe_profile
        return describe_profile(load_runtime_env(ROOT, profile=args.profile))
    if args.command.startswith("checkpoint-"):
        return asyncio.run(checkpoint_command(args))
    if args.command == "conversation-reset":
        from .sessions import ConversationStore
        ConversationStore().reset(args.owner, args.conversation)
        return {"reset": True}
    if args.command in ("ask", "lookup"):
        return asyncio.run(question(args))
    if args.command == "evaluate":
        return asyncio.run(evaluate_dataset(args))
    if args.command == "ops-ui":
        from .mlops import serve_ui
        return {"exitCode": serve_ui(port=args.port)}
    if args.command == "runs":
        from .observability import OPS_ROOT
        if not 1 <= args.limit <= 1000:
            raise ValueError("RUN_LIMIT_OUT_OF_RANGE")
        paths = sorted((OPS_ROOT / "runs").glob("*/summary.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        return [json.loads(p.read_text(encoding="utf-8")) for p in paths[:args.limit]]
    if args.command in ("inspect-run", "score-run", "compare-runs"):
        from .evaluation import compare_runs, evaluate_events, load_dataset, read_run
        from .observability import digest
        if args.command == "compare-runs":
            return compare_runs(read_run(run_path(args.left))[0], read_run(run_path(args.right))[0])
        summary, events = read_run(run_path(args.run_id))
        if args.command == "inspect-run":
            return {"summary": summary, "events": events}
        from .data import load_corpus
        dataset = load_dataset(args.dataset, load_corpus())
        case = next((c for c in dataset["cases"] if c["id"] == args.case), None)
        if case is None or digest(case["question"]) != summary["questionHash"]:
            raise ValueError("GOLD_QUESTION_MISMATCH")
        # Scoring an old run never overwrites its original experiment record.
        evaluation = evaluate_events(events, case)
        path = run_path(args.run_id) / ("evaluation-" + digest(case)[:24] + ".json")
        path.write_text(json.dumps(evaluation, ensure_ascii=False, indent=2), encoding="utf-8")
        return evaluation
    if args.command == "check":
        from .publication import check_source
        return check_source(with_data=args.with_data)
    if args.command == "check-git":
        from .publication import check_git
        return check_git(staged=args.staged)
    if args.command == "graph":
        from .graph import build_bible_graph
        from .profiles import load_runtime_env
        from .runtime import BibleRuntime
        return build_bible_graph(BibleRuntime(env=load_runtime_env(ROOT, profile=args.profile),
                                              corpus={})).get_graph().draw_mermaid()
    from .preparation import build_heading_boundaries, build_index, pack_corpus, restore_runtime_data
    if args.command == "data-restore":
        return restore_runtime_data(args.source)
    if args.command == "data-pack":
        return pack_corpus(args.source)
    if args.command == "build-headings":
        return build_heading_boundaries(args.source)
    if args.command == "build-index":
        return asyncio.run(build_index(live=args.live))
    raise ValueError("UNKNOWN_COMMAND")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parser().parse_args(argv)
    try:
        output = execute(args)
        print(output if isinstance(output, str) else json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except KeyboardInterrupt:
        print("작업을 취소했습니다.", file=sys.stderr)
        return 130
    except Exception as error:
        # Error text from network providers or JSON input can contain private data.
        code = getattr(error, "code", None)
        if code is None:
            import re
            candidate = str(error)
            code = candidate if isinstance(error, ValueError) and re.fullmatch(r"[A-Z_]+(?::[A-Za-z0-9_./, -]+)?", candidate) else type(error).__name__
        print(json.dumps({"error": code}, ensure_ascii=False), file=sys.stderr)
        return 1
