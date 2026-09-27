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
        if name == "ask":
            command.add_argument("--live", action="store_true", help="유료 API 호출 허용")
    command = commands.add_parser("check")
    command.add_argument("--with-data", action="store_true")
    command = commands.add_parser("check-git")
    command.add_argument("--staged", action="store_true")
    commands.add_parser("graph")
    for name in ("data-restore", "data-pack", "build-headings"):
        command = commands.add_parser(name)
        command.add_argument("source", type=Path)
    command = commands.add_parser("build-index")
    command.add_argument("--live", action="store_true")
    return result


async def question(args):
    from .config import load_env
    from .data import load_corpus, load_study_data, verify_assets
    from .graph import run_bible_graph
    from .passages import resolve_bible_request
    from .retrieval import load_bible_retriever
    from .study import StudyIndex
    if args.command == "ask" and not args.live:
        raise ValueError("PAID_CALL_REQUIRES_ASK_LIVE")
    verify_assets()
    corpus, env = load_corpus(), load_env()
    if args.command == "lookup":
        scope = resolve_bible_request(corpus, args.question, [])
        if not scope["lookup"] or scope["invalid"] or not scope["primary"]:
            raise ValueError("LOOKUP_REQUIRES_VALID_EXACT_REFERENCE")
        async def no_network(*args, **kwargs):
            raise ValueError("LOOKUP_NETWORK_FORBIDDEN")
        return await run_bible_graph(env=env, corpus=corpus, question=args.question, fetcher=no_network)
    if not env.get("OPENAI_API_KEY", "").strip():
        raise ValueError("RESEARCH_OPENAI_API_KEY_REQUIRED")
    return await run_bible_graph(env=env, corpus=corpus, question=args.question, history=[],
                                 retriever=load_bible_retriever(corpus), study_index=StudyIndex(corpus, load_study_data()))


def execute(args):
    if args.command in ("ask", "lookup"):
        return asyncio.run(question(args))
    if args.command == "check":
        from .publication import check_source
        return check_source(with_data=args.with_data)
    if args.command == "check-git":
        from .publication import check_git
        return check_git(staged=args.staged)
    if args.command == "graph":
        from .graph import build_bible_graph
        return build_bible_graph().get_graph().draw_mermaid()
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
