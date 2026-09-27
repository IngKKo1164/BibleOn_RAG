"""Deterministic lexical, local-context and dense retrieval over canonical chunks."""
from __future__ import annotations

import math
import base64
import re
import unicodedata
from array import array
from collections import Counter
from pathlib import Path

from .common import json_text, sha256, text_length, unique
from .data import DEFAULT_DATA, load_headings, read_json
from .passages import CHUNK_VERSION, create_bible_chunks

RAG_LIMITS = {"candidates": 12, "selectedChunks": 3, "evidenceCharacters": 6000,
              "evidenceVerses": 96, "excerptCharacters": 1200, "candidateCharacters": 8400, "excerptMinimum": 420}
EMBEDDING_MODEL = "text-embedding-v4"
EMBEDDING_DIMENSIONS = 512
STOP = set("무엇 누구 누구지 누구야 뭐야 어떤 어떻게 왜 이유 의미 설명 설명해줘 설명해주세요 자세히 상세히 정확히 무슨 확정 확정할 근거 알려줘 알려주세요 알려줄래 말해줘 해주세요 있어 성경 본문 질문 관한 대한 대해서 대해 시대적 배경 들어 사람이 사람 그 이 저 것 수 중 더 한 할 한테 이것 그것 나는 나의 우리 위해 가장 중요한 태도 등 및 또 곧 즉 때 바 만 이다 하다 있다 있었다 있었는데".split())
EQUIVALENTS = [(r"오병이어", "오병이어 떡 다섯 물고기 두 마리 오천"),
               (r"칠병이어", "칠병이어 떡 일곱 생선 사천"), (r"설교|강해|말씀을?\s*듣", "강론 설교 말씀 듣"),
               (r"예수님", "예수"), (r"구원받[기가으을는]?", "구원 받"),
               (r"떨어(?:져|졌|진|짐)", "떨어지"), (r"씻기신", "씻기"), (r"추락(?:한|해|했|하여|하다)?", "떨어지")]
SUFFIX = re.compile(r"(?:에게서|에서는|으로는|이라고|이라는|하기를|하려고|에게|에서|으로|까지|부터|처럼|이랑|을|를|은|는|이|가|와|과|의|에|다)$")


def normalize_retrieval_query(text: str) -> str:
    value = str(text)
    for pattern, replacement in EQUIVALENTS:
        value = re.sub(pattern, replacement, value)
    return value


def rag_terms(text: str) -> list[str]:
    result = []
    for word in re.findall(r"[가-힣a-z]+", unicodedata.normalize("NFKC", str(text)).lower()):
        if word in STOP:
            continue
        if len(word) > 1:
            word = SUFFIX.sub("", word)
        if not word or word in STOP:
            continue
        result.append("w:" + word)
        result.extend("p:" + word[:n] for n in range(1, min(4, len(word)) + 1))
        result.extend("g:" + word[i:i+2] for i in range(len(word)-1))
    return result


def retrieval_scope(corpus: dict, text: str, *, coordinated_books=False) -> dict:
    original = unicodedata.normalize("NFKC", str(text))
    if re.search(r"말고|제외|아니라|성경\s*전체|전체\s*성경|다른\s*(?:책|성경)", original):
        return {"bookIds": [], "contentQuery": original}
    found, ranges = [], []
    if coordinated_books and corpus:
        names = sorted(((re.sub(r"\s", "", b["name"]), k) for k, b in corpus.items()), key=lambda p: -len(p[0]))
        alternatives, by_name = "|".join(re.escape(n) for n, _ in names), dict(names)
        chain = re.compile(r"(?<![가-힣])((?:" + alternatives + r")(?:\s*(?:와|과|및|,|·)\s*(?:" + alternatives + r"))+)(?:에서(?:는)?|에\s*(?:나오|기록|있는|등장))")
        for match in chain.finditer(original):
            found.extend(by_name[name] for name in re.split(r"\s*(?:와|과|및|,|·)\s*", match[1]))
            ranges.append(match.span())
    for book_id, book in corpus.items():
        name = re.escape(re.sub(r"\s", "", book["name"]))
        pattern = re.compile(r"(?<![가-힣])" + name + r"(?:에서(?:는)?|에\s*(?:나오|기록|있는|등장)|(?=\s*\d{1,3}(?:\s*[:장편])))")
        for match in pattern.finditer(original):
            found.append(book_id)
            if not any(a <= match.start() and b >= match.end() for a, b in ranges):
                ranges.append(match.span())
    content = original
    for start, end in sorted(ranges, reverse=True):
        content = content[:start] + " " + content[end:]
    return {"bookIds": unique(found), "contentQuery": content.strip()}


def retrieval_query(question: str, previous_evidence=None) -> str:
    followup = text_length(question) < 180 and re.search(r"그(?:\s*사람|\s*인물|\s*사건|\s*말씀|는|의|렇|래|럼)|이(?:\s*사람|\s*말씀|\s*사건)|나이|직업|더\s*(자세|구체)", question)
    return normalize_retrieval_query(previous_evidence["question"][:400] + "\n" + question if previous_evidence and followup else question)


def retrieval_variants(query: str) -> list[str]:
    variants = []
    def add(value):
        value = value.strip()
        if len(value) >= 4 and value != query and value not in variants:
            variants.append(value)
    for match in re.finditer(r'''[“‘"']([^“”‘’"'\n]{4,60})[”’"']''', query):
        add(match[1])
    if re.search(r"중.{0,10}(?:한|하나)|명단|뽑|선택|누가", query):
        for match in re.finditer(r"(?:^|\s)(열두|열한|열|일곱|다섯|네|세|두|십이|칠|\d+)\s+([가-힣]{2,10})", query):
            add(match[1] + " " + re.sub(r"(?:은|는|이|가|을|를|의)$", "", match[2]) + " 이름")
    for match in re.finditer(r"(?=([가-힣]{2,12})의\s+([가-힣]{2,12}))", query):
        last = re.sub(r"(?:은|는|이|가|를|을)$", "", match[2])
        if len(last) >= 2:
            add(match[1] + " " + last)
    clauses = [s.strip() for s in re.split(r"[?？]\s*|\s+(?:그리고|또한|아니면)\s+", query) if len(s.strip()) >= 8]
    if len(clauses) > 1:
        for clause in clauses:
            add(clause)
    return variants[:3]


class BibleRetriever:
    """Immutable corpus/index data; no question, account or graph state is cached."""
    def __init__(self, corpus, vectors=None, lexical_data=None, chunks=None, boundaries=None):
        self.corpus = corpus
        self.chunks = chunks if chunks is not None else create_bible_chunks(corpus, boundaries)
        self.vectors = vectors
        self.postings = dict(lexical_data["postings"]) if lexical_data else {}
        self.lengths = list(lexical_data["lengths"]) if lexical_data else []
        self.norms = []
        if not lexical_data:
            for i, chunk in enumerate(self.chunks):
                terms = rag_terms(chunk["text"])
                self.lengths.append(len(terms))
                for term, count in Counter(terms).items():
                    self.postings.setdefault(term, []).extend((i, count))
        if vectors is not None:
            if len(vectors) != len(self.chunks) * EMBEDDING_DIMENSIONS:
                raise ValueError("RAG_VECTOR_MISMATCH")
            for i in range(len(self.chunks)):
                row = vectors[i*512:(i+1)*512]
                self.norms.append(math.sqrt(sum(n*n for n in row)))
        self.avg_length = sum(self.lengths) / len(self.chunks) if self.chunks else 1
        self.avg_length = self.avg_length or 1
        self.window_view = None

    def lexical(self, query):
        scores = [0.0] * len(self.chunks)
        # Ordered uniqueness preserves floating-point accumulation and tie order.
        for term in unique(rag_terms(query)):
            posting = self.postings.get(term)
            if not posting:
                continue
            df = len(posting)/2
            idf = math.log(1 + (len(self.chunks)-df+.5)/(df+.5))
            weight = 1 if term.startswith("w:") else .6 if term.startswith("p:") else .15
            for at in range(0, len(posting), 2):
                i, tf = posting[at:at+2]
                scores[i] += weight*idf*tf*2.2/(tf+1.2*(.25+.75*self.lengths[i]/self.avg_length))
        return sorted(({"index": i, "score": s} for i, s in enumerate(scores) if s > 0), key=lambda r: (-r["score"], r["index"]))

    def focused(self, query):
        words = unique(t[2:6] for t in rag_terms(query) if t.startswith("w:"))
        clues = [(w, math.log(1+len(self.chunks)/(len(self.postings.get("p:"+w, []))/2+1))) for w in words]
        result = []
        for index, chunk in enumerate(self.chunks):
            best = 0
            for i in range(len(chunk["verses"])):
                text = " ".join(v["text"] for v in chunk["verses"][max(0, i-1):i+2])
                score = sum(weight for word, weight in clues if word in text)/(.8+.2*text_length(text)/180)
                best = max(best, score)
            if best > 0:
                result.append({"index": index, "score": best})
        return sorted(result, key=lambda r: (-r["score"], r["index"]))

    def rankings(self, query, *, vector=None, refs=None, book_scope=None):
        scope = retrieval_scope(self.corpus, query)
        books = unique(scope["bookIds"] if book_scope is None else book_scope)
        if books:
            books = unique(books + [r["bookId"] for r in refs or []])
        eligible = lambda i: not books or self.chunks[i]["bookId"] in books
        content = scope["contentQuery"] or query
        lex = [r for r in self.lexical(content) if eligible(r["index"])]
        focus = [r for r in self.focused(content) if eligible(r["index"])]
        dense = []
        if vector is not None and self.vectors is not None:
            norm = math.sqrt(sum(n*n for n in vector))
            for i in range(len(self.chunks)):
                if eligible(i):
                    dot = sum(a*b for a, b in zip(vector, self.vectors[i*512:(i+1)*512], strict=True))
                    dense.append({"index": i, "score": dot/(norm*self.norms[i] or 1)})
            dense.sort(key=lambda r: (-r["score"], r["index"]))
        return {"lex": lex, "focus": focus, "dense": dense, "scopedBooks": books}

    def serialize_lexical(self):
        return {"lengths": self.lengths, "postings": [[k, list(v)] for k, v in self.postings.items()]}


def load_bible_retriever(corpus, directory: Path = DEFAULT_DATA) -> BibleRetriever:
    metadata = read_json(directory / "index.json")
    headings = load_headings(directory)
    chunks = create_bible_chunks(corpus, headings["boundaries"])
    fingerprint = "\n".join(c["id"] + "\n" + c["text"] for c in chunks)
    if (metadata["version"] != CHUNK_VERSION or metadata["boundarySha256"] != headings["metadata"]["boundarySha256"]
            or metadata["model"] != EMBEDDING_MODEL or metadata["dimensions"] != 512
            or metadata["count"] != len(chunks) or metadata["chunkSha256"] != sha256(fingerprint)):
        raise ValueError("RAG_INDEX_MISMATCH")
    raw = (directory / "vectors.i8").read_bytes()
    if len(raw) != len(chunks)*512 or sha256(raw) != metadata["vectorSha256"]:
        raise ValueError("RAG_VECTOR_MISMATCH")
    lexical = read_json(directory / "lexical.json.gz")
    if len(lexical["lengths"]) != len(chunks) or lexical_fingerprint(lexical) != metadata["lexicalSha256"]:
        raise ValueError("RAG_LEXICAL_MISMATCH")
    return BibleRetriever(corpus, array("b", raw), lexical, chunks)


def lexical_fingerprint(lexical):
    """Canonical delta/varint wire digest, identical to the original index."""
    terms, counts, encoded = [], [], bytearray()
    def put(n):
        if not isinstance(n, int) or n < 0:
            raise ValueError("RAG_LEXICAL_MISMATCH")
        while n >= 128:
            encoded.append((n & 127) | 128)
            n >>= 7
        encoded.append(n)
    for term, rows in lexical["postings"]:
        if len(rows) % 2:
            raise ValueError("RAG_LEXICAL_MISMATCH")
        terms.append(term); counts.append(len(rows)//2)
        previous = 0
        for at in range(0, len(rows), 2):
            put(rows[at]-previous); put(rows[at+1]); previous = rows[at]
    return sha256(json_text({"lengths": lexical["lengths"], "terms": terms, "counts": counts,
                             "bytes": base64.b64encode(encoded).decode("ascii")}))


def candidate_excerpt(chunk, query, *, max_characters=1200, queries=()):
    budget = max(0, min(1200, math.floor(max_characters)))
    lines = [f"{v['verse']} {v['text']}" for v in chunk["verses"]]
    selected = chunk["verses"]
    if text_length("\n".join(lines)) > budget:
        terms = set(t for q in [query, *queries] for t in rag_terms(normalize_retrieval_query(q)))
        clues = [t[2:] for t in terms if t.startswith("w:")]
        matches = [(set(w for w in clues if w in v["text"]), set(rag_terms(v["text"])) & terms) for v in chunk["verses"]]
        best = (-1, 0, 0, 0)
        for start in range(len(lines)):
            length, words, grams = 0, set(), set()
            for end in range(start, len(lines)):
                length += text_length(lines[end]) + (1 if end > start else 0)
                if length > budget:
                    break
                words.update(matches[end][0]); grams.update(matches[end][1])
                score = sum(2+min(4, len(w)) for w in words)*100+len(grams)
                if (score, length) > best[:2]:
                    best = (score, length, start, end+1)
        selected = chunk["verses"][best[2]:best[3]]
    return {"id": chunk["id"], "reference": chunk["reference"],
            "excerpt": "\n".join(f"{v['verse']} {v['text']}" for v in selected),
            "excerptRange": f"{selected[0]['verse']}-{selected[-1]['verse']}" if selected else "",
            "complete": len(selected) == len(chunk["verses"])}


def quantize_vector(vector):
    scale = max(map(abs, vector), default=0) or 1
    # Python round uses ties-to-even; the source uses Math.round (toward +inf).
    return array("b", (math.floor(n/scale*127+.5) for n in vector))
