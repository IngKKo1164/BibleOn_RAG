"""Bounded provider I/O with an injectable offline transport."""
from __future__ import annotations

import asyncio
import math
import re
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx

from .common import is_integer
from .retrieval import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL

QUERY_INSTRUCTION = "Given a Korean Bible question, retrieve Bible passages that directly answer the question."


@dataclass(frozen=True)
class HttpResult:
    status: int
    data: dict

    @property
    def ok(self):
        return 200 <= self.status < 300


class ProviderTransport:
    def __init__(self, env: dict):
        workspace = env.get("QWEN_WORKSPACE_ID", "").strip()
        if workspace and not re.fullmatch(r"[a-zA-Z0-9-]+", workspace):
            raise ValueError("INVALID_EMBEDDING_WORKSPACE")
        self.hosts = {"api.openai.com", "dashscope-intl.aliyuncs.com"}
        if workspace:
            self.hosts.add(f"{workspace}.ap-southeast-1.maas.aliyuncs.com")
        self.client = httpx.AsyncClient(follow_redirects=False, trust_env=False)

    async def request(self, url: str, *, body: dict, headers: dict, timeout: float) -> HttpResult:
        parsed = urlparse(url)
        if parsed.scheme != "https" or parsed.hostname not in self.hosts or parsed.username or parsed.port not in (None, 443):
            raise ValueError("PROVIDER_HOST_NOT_ALLOWED")
        from .common import json_text
        async with asyncio.timeout(timeout):
            response = await self.client.post(url, content=json_text(body).encode("utf-8"), headers=headers, timeout=timeout)
        # Never include error response bodies or authorization headers in errors.
        if not response.is_success:
            return HttpResult(response.status_code, {})
        try:
            data = response.json()
        except ValueError as error:
            raise ValueError("INVALID_PROVIDER_JSON") from error
        return HttpResult(response.status_code, data)

    async def aclose(self):
        await self.client.aclose()


async def embed_texts(texts: list[str], *, env: dict, fetcher, kind="document") -> dict:
    from .common import text_length
    if (not isinstance(texts, list) or not 1 <= len(texts) <= 10
            or any(not isinstance(s, str) or not s.strip() or text_length(s) > 16000 for s in texts)):
        raise ValueError("INVALID_EMBEDDING_INPUT")
    workspace = env.get("QWEN_WORKSPACE_ID", "").strip()
    if workspace and not re.fullmatch(r"[a-zA-Z0-9-]+", workspace):
        raise ValueError("INVALID_EMBEDDING_WORKSPACE")
    host = f"{workspace}.ap-southeast-1.maas.aliyuncs.com" if workspace else "dashscope-intl.aliyuncs.com"
    query = kind == "query"
    path = "/api/v1/services/embeddings/text-embedding/text-embedding" if query else "/compatible-mode/v1/embeddings"
    body = {"model": EMBEDDING_MODEL, "input": {"texts": texts},
            "parameters": {"dimension": 512, "text_type": "query", "instruct": QUERY_INSTRUCTION}} if query else {
                "model": EMBEDDING_MODEL, "input": texts, "dimensions": 512, "encoding_format": "float"}
    async with asyncio.timeout(12):
        response = await fetcher(f"https://{host}{path}", body=body,
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {env.get('QWEN_API_KEY', '')}"}, timeout=12)
    if not response.ok:
        raise ValueError("EMBEDDING_UNAVAILABLE")
    data = response.data
    rows = [{"index": r.get("text_index"), "embedding": r.get("embedding")} for r in data.get("output", {}).get("embeddings", [])] if query else data.get("data", [])
    tokens = data.get("usage", {}).get("total_tokens" if query else "prompt_tokens")
    if not is_integer(tokens) or tokens < 0 or len(rows) != len(texts):
        raise ValueError("INVALID_EMBEDDING_RESPONSE")
    vectors = [None] * len(texts)
    for row in rows:
        i, vector = row.get("index"), row.get("embedding")
        if (not is_integer(i) or not 0 <= i < len(texts) or vectors[int(i)] is not None
                or not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSIONS
                or any(type(n) not in (int, float) or not math.isfinite(n) for n in vector) or not any(vector)):
            raise ValueError("INVALID_EMBEDDING_VECTOR")
        vectors[int(i)] = vector
    return {"vectors": vectors, "inputTokens": int(tokens), "model": EMBEDDING_MODEL}
