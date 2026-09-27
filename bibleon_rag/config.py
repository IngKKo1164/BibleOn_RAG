"""Research-only defaults and workspace-local credentials."""
from pathlib import Path

from dotenv import dotenv_values

from .data import ROOT

RESEARCH_DEFAULTS = {"BIBLE_AI_PROVIDER": "openai", "BIBLE_RAG_MODE": "hybrid", "BIBLE_RAG_SPEED": "integrated",
                     "BIBLE_RAG_STUDY": "one-hop", "BIBLE_RETRIEVAL_VERSION": "v4",
                     "BIBLE_RETRIEVAL_REPAIR": "on", "BIBLE_REPAIR_QUERY_EXPANSION": "on"}
MODEL = "gpt-6-luna"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
STAGES = {
    "screen": {"model": MODEL, "effort": "low", "maxTokens": 1800, "visibleTokens": 1000},
    "locate": {"model": MODEL, "effort": "low", "maxTokens": 1500, "visibleTokens": 450},
    "rerank": {"model": MODEL, "effort": "low", "maxTokens": 3000, "visibleTokens": 1800},
    "answer": {"model": MODEL, "effort": "low", "maxTokens": 4000, "visibleTokens": 3000},
    "repair": {"model": MODEL, "effort": "low", "maxTokens": 1500, "visibleTokens": 1000},
}
MAX_REASONING_TOKENS = 3000
KINDS = ["answer", "uncertain", "outOfScope", "greeting", "safety"]


def load_env(root: Path = ROOT):
    # Do not inherit application keys, search parent folders or interpolate env vars.
    values = dict(RESEARCH_DEFAULTS)
    keys = set(values) | {"OPENAI_API_KEY", "QWEN_API_KEY", "QWEN_WORKSPACE_ID"}
    for name in (".env", ".env.local"):
        path = root / name
        if path.is_file():
            values.update({k: v for k, v in dotenv_values(path, interpolate=False).items() if k in keys and v is not None})
    if values["BIBLE_AI_PROVIDER"] != "openai" or values["BIBLE_RETRIEVAL_VERSION"] != "v4":
        raise ValueError("LANGGRAPH_REQUIRES_RESEARCH_V4")
    return values
