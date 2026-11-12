"""Source-only checks and a deny-by-default Git publication guard."""
from __future__ import annotations

import ast
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path

from .data import ROOT, verify_assets

DEPENDENCIES = {"httpx", "dotenv", "regex", "langgraph", "langsmith", "aiosqlite", "numpy"}
OPTIONAL_DEPENDENCIES = {"mlflow"}
SECRET_PATTERN = re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-(?:proj-)?|gsk_)[A-Za-z0-9_-]{30,}|\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}")
PRIVATE_SAMPLES = [".env.local", "data/runtime/corpus.json.gz", "data/private/sources/private.json", "artifacts/answer.json",
                   "tests/python/test_parity.py", "docs/RAG_ADAPTIVE_REPAIR.md", "AGENTS.md", "server/rag-runtime.mjs",
                   "package.json", "node_modules/module.js", ".venv/pyvenv.cfg", "bibleon_rag/unreviewed.py",
                   "artifacts/mlops/tracking.db", "artifacts/mlops/runs/sample/events.jsonl",
                   "data/private/conversations.sqlite3", "data/private/conversations.sqlite3-journal",
                   "data/private/graph-checkpoints.sqlite3", "data/private/graph-checkpoints.sqlite3-wal"]


def public_files(root=ROOT):
    lines = (root/".gitignore").read_text(encoding="utf-8").splitlines()
    if "*" not in lines:
        raise ValueError("PUBLICATION_MUST_DENY_BY_DEFAULT")
    files = {line[2:] for line in lines if line.startswith("!/") and not line.endswith("/")}
    for name in files:
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or any(c in name for c in "*?[]\\"):
            raise ValueError("INVALID_PUBLICATION_PATH")
    return sorted(files)


def check_source(*, with_data=False, root=ROOT):
    allowed = set(public_files(root))
    modules = 0
    for name in sorted(allowed):
        path = root/name
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"PUBLIC_SOURCE_MISSING:{name}")
        text = path.read_text(encoding="utf-8")
        if SECRET_PATTERN.search(text):
            raise ValueError(f"POTENTIAL_CREDENTIAL:{name}")
        if path.suffix != ".py":
            continue
        tree = ast.parse(text, filename=name)
        modules += 1
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.level:
                if node.level != 1:
                    raise ValueError(f"IMPORT_OUTSIDE_PACKAGE:{name}")
                if node.module and f"bibleon_rag/{node.module.replace('.', '/')}.py" not in allowed:
                    raise ValueError(f"PRIVATE_SOURCE_IMPORT:{name}:{node.module}")
                if not node.module:
                    for alias in node.names:
                        if f"bibleon_rag/{alias.name}.py" not in allowed:
                            raise ValueError(f"PRIVATE_SOURCE_IMPORT:{name}:{alias.name}")
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                imports = [node.module] if isinstance(node, ast.ImportFrom) else [a.name for a in node.names]
                for module in imports:
                    base = module.split(".")[0]
                    if base not in sys.stdlib_module_names | DEPENDENCIES | OPTIONAL_DEPENDENCIES | {"bibleon_rag"}:
                        raise ValueError(f"UNDECLARED_IMPORT:{name}:{base}")
    for name in DEPENDENCIES:
        if importlib.util.find_spec(name) is None:
            raise ValueError(f"DEPENDENCY_MISSING:{name}")
    from .graph import build_bible_graph
    node_count = len(build_bible_graph().get_graph().nodes)-2
    if node_count != 19:
        raise ValueError("GRAPH_TOPOLOGY_CHANGED")
    result = {"pythonModules": modules, "publicFiles": len(allowed), "graphNodes": node_count, "networkUsed": False}
    if with_data:
        from .data import load_corpus, load_study_data
        from .retrieval import load_bible_retriever
        from .study import StudyIndex
        directory = root/"data/runtime"
        result["verifiedAssets"] = len(verify_assets(directory, root/"python-runtime-assets.json"))
        corpus = load_corpus(directory)
        result["chunks"] = len(load_bible_retriever(corpus, directory=directory).chunks)
        StudyIndex(corpus, load_study_data(directory))
    return result


def git_output(args, *, root=ROOT, input=None, accepted=(0,)):
    process = subprocess.run(["git", *args], cwd=root, input=input, capture_output=True, encoding="utf-8", errors="strict",
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode not in accepted:
        raise ValueError("GIT_CHECK_FAILED:"+args[0])
    return process.stdout


def check_git(*, staged=False, root=ROOT):
    if Path(git_output(["rev-parse", "--show-toplevel"], root=root).strip()).resolve() != root.resolve():
        raise ValueError("INDEPENDENT_RESEARCH_GIT_REQUIRED")
    args = ["diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"] if staged else ["ls-files", "--cached", "--others", "--exclude-standard", "-z"]
    files = set(filter(None, git_output(args, root=root).split("\0")))
    allowed = set(public_files(root))
    forbidden = sorted(files-allowed)
    if forbidden:
        raise ValueError("PRIVATE_OR_UNAPPROVED_PATH:"+", ".join(forbidden))
    if staged:
        indexed = set(filter(None, git_output(["ls-files", "--cached", "-z"], root=root).split("\0")))
        if indexed != allowed:
            raise ValueError("STAGE_COMPLETE_REVIEWED_SOURCE_BEFORE_COMMIT")
    for name in files:
        if (root/name).is_symlink():
            raise ValueError("PUBLIC_SYMLINK_NOT_ALLOWED:"+name)
        content = git_output(["show", ":"+name], root=root) if staged else (root/name).read_text(encoding="utf-8")
        if SECRET_PATTERN.search(content):
            raise ValueError("POTENTIAL_CREDENTIAL:"+name)
    ignored = set(filter(None, git_output(["check-ignore", "--no-index", "--stdin", "-z"], root=root,
        input="\0".join(PRIVATE_SAMPLES)+"\0", accepted=(0, 1)).split("\0")))
    if ignored != set(PRIVATE_SAMPLES):
        raise ValueError("PRIVATE_IGNORE_RULE_MISSING")
    # Validate the staged rules too: changing a working copy must not hide unsafe staged bytes.
    if staged and ".gitignore" in files:
        expected = (root/".gitignore").read_text(encoding="utf-8").replace("\r\n", "\n")
        if git_output(["show", ":.gitignore"], root=root).replace("\r\n", "\n") != expected:
            raise ValueError("STAGED_IGNORE_DIFFERS_FROM_REVIEWED_RULES")
    return {"checkedFiles": len(files), "staged": staged, "privateDataExcluded": True}
