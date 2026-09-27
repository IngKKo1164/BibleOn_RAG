"""Bounded one-hop references and separately attributed historical sources."""
from __future__ import annotations

import re
import unicodedata

from .common import canonical_id_sort_key, text_length, unique
from .passages import verse_at
from .prompt_text import STUDY_SOURCES
from .retrieval import rag_terms
from .validation import has_apparent_death_claim, has_complete_answer_ending, korean_only

STUDY_LIMITS = {"roots": 6, "linkedVerses": 12, "linkedCharacters": 1500, "sources": 3,
                "sourceCharacters": 4000, "excerptCharacters": 2400, "notes": 2}
ALIASES = [(r"바리새", "Pharisees"), (r"사두개", "Sadducees"), (r"회당", "Synagogue"), (r"안식일", "Sabbath"),
           (r"유월절", "Passover"), (r"담무스", "Tammuz"), (r"니느웨", "Nineveh"), (r"사마리아", "Samaria"),
           (r"로마", "Rome"), (r"드로아", "Troas"), (r"유두고", "Eutychus"), (r"세례", "Baptism"),
           (r"성전", "Temple"), (r"성막", "Tabernacle"), (r"할례", "Circumcision"), (r"제사장", "Priest"),
           (r"대제사장", "High priest"), (r"구속|속량", "Redemption"), (r"구원", "Salvation"), (r"은혜", "Grace"),
           (r"믿음", "Faith"), (r"양선|선행", "Goodness"), (r"거룩|성화", "Sanctification"),
           (r"칭의", "Justification"), (r"화목", "Reconciliation"), (r"부활", "Resurrection"),
           (r"언약", "Covenant"), (r"바벨론", "Babylon"), (r"앗수르|아시리아", "Assyria"),
           (r"가나안", "Canaan"), (r"벧엘", "Bethel"), (r"욥", "Job"), (r"니고데모", "Nicodemus")]


def _indexed(data, key, default):
    if isinstance(data, list):
        return data[key] if isinstance(key, int) and 0 <= key < len(data) else default
    return data.get(str(key), data.get(key, default))


class StudyIndex:
    def __init__(self, corpus, data):
        canonical = [f"{b}:{c}:{v}" for b, book in corpus.items() for c in sorted(book["chapters"], key=int)
                     for v in sorted(book["chapters"][c], key=int)]
        if canonical != data["verseIds"]:
            raise ValueError("STUDY_CORPUS_MISMATCH")
        self.data, self.ids = data, {key: i for i, key in enumerate(canonical)}
        self.records, self.by_verse, self.titles = {}, {}, {}
        for row in data["records"]:
            self.records[row["id"]] = row
            for v in row["refs"]:
                self.by_verse.setdefault(v, []).append(row)
            if row["source"] in ("easton", "mhcc"):
                for word in unique(re.findall(r"[a-z]+", row["title"].lower())):
                    self.titles.setdefault(word, []).append(row)


def _term_score(text, terms):
    return sum(t in terms for t in rag_terms(text))


def expand_one_hop(corpus, primary, question, index, limits=None):
    limits = limits or STUDY_LIMITS
    existing, terms = {v["id"] for v in primary}, set(rag_terms(question))
    roots = sorted(enumerate(primary), key=lambda row: (-_term_score(row[1]["text"], terms), row[0]))[:limits["roots"]]
    candidates = {}
    for rank, (_, root) in enumerate(roots):
        for target in _indexed(index.data["graph"], index.ids.get(root["id"]), []):
            key = index.data["verseIds"][target]
            if not key or key in existing:
                continue
            book, chapter, verse = key.split(":")
            try:
                canonical = verse_at(corpus, {"bookId": book, "chapter": int(chapter), "verse": int(verse)})
            except ValueError:
                continue
            row = candidates.setdefault(key, {"verse": canonical, "roots": [], "score": 0})
            row["roots"].append(root["id"])
            row["score"] += 2/(rank+1) + _term_score(canonical["text"], terms)/8
    linked, relations, chars = [], [], 0
    for row in sorted(candidates.values(), key=lambda r: (-r["score"], index.ids[r["verse"]["id"]])):
        if len(linked) >= limits["linkedVerses"]:
            break
        if chars + text_length(row["verse"]["text"]) > limits["linkedCharacters"]:
            continue
        linked.append(row["verse"]); chars += text_length(row["verse"]["text"])
        relations.extend({"from": root, "to": row["verse"]["id"]} for root in row["roots"])
    return {"evidence": primary+linked, "linked": linked, "relations": relations, "roots": [v["id"] for _, v in roots]}


def normalize_study_terms(value):
    if not isinstance(value, list):
        return []
    return unique(s.lower().strip() for s in value if isinstance(s, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z '-]{0,47}", s))[:3]


def study_query_terms(question, model_hints=()):
    terms = [t for p, t in ALIASES if re.search(p, question)] + list(model_hints)
    terms += [m[0] for m in re.finditer(r"\b[A-Za-z][A-Za-z'-]{2,30}\b", question, re.A) if not re.fullmatch(r"[GH]\d+", m[0], re.I)]
    return unique(t.lower().strip() for t in terms if isinstance(t, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z '-]{0,47}", t))[:8]


def search_independent_study(index, terms, question):
    matches = {}
    for phrase in terms:
        tokens = re.findall(r"[a-z]+", phrase)
        for token in tokens:
            for row in index.titles.get(token, []):
                title = row["title"].lower()
                words = set(re.findall(r"[a-z]+", title))
                if not all(t in words for t in tokens):
                    continue
                score = 100 if title == phrase else 50+len(tokens)
                if row["id"] not in matches or matches[row["id"]]["score"] < score:
                    matches[row["id"]] = {"row": row, "score": score, "method": "independent-title"}
    for match in re.finditer(r"\bG(\d{1,5})\b", question, re.I | re.A):
        key = f"G{int(match[1]):04d}"
        record_id = index.data.get("aliases", {}).get("strong:"+key, "strong:"+key)
        row = index.records.get(record_id)
        if row and row["source"] == "strongGreek":
            matches[row["id"]] = {"row": row, "score": 120, "method": "explicit-lexicon-id"}
    return sorted(matches.values(), key=lambda r: (-r["score"], canonical_id_sort_key(r["row"]["id"])))[:32]


def _english_score(text, terms):
    return sum(term in text.lower() for term in terms)


def study_excerpt(text, terms, maximum=2400):
    paragraphs = [p.strip() for p in text.split("\n") if text_length(p.strip()) > 25]
    ranked = sorted(enumerate(paragraphs), key=lambda row: (-_english_score(row[1], terms), row[0]))
    selected, chars = [], 0
    for i, paragraph in ranked:
        length = text_length(paragraph)
        if length > maximum or chars + length + 1 > maximum:
            continue
        selected.append((i, paragraph)); chars += length+1
    return "\n".join(p for _, p in sorted(selected))


def retrieve_study_sources(primary, study_terms, question, index, *, independent=False):
    terms = study_query_terms(question, normalize_study_terms(study_terms)) if independent else normalize_study_terms(study_terms)
    candidates = {}
    requested = {f"G{int(s[1:]):04d}" for s in re.findall(r"\bG\d{1,5}\b", question, re.I | re.A)}
    korean_terms = set(rag_terms(question))
    roots = sorted(primary, key=lambda v: -_term_score(v["text"], korean_terms))[:12]
    for rank, v in enumerate(roots):
        n = index.ids.get(v["id"])
        for row in index.by_verse.get(n, []):
            candidate = candidates.setdefault(row["id"], {"row": row, "score": 0})
            candidate["score"] += (2/(rank+1))/(max(1, len(row["refs"]))**.5)
        for key in _indexed(index.data["greekTags"], n, []):
            if key not in requested:
                continue
            record_id = index.data["aliases"].get("strong:"+key, "strong:"+key)
            row = index.records.get(record_id)
            if row:
                candidates[record_id] = {"row": row, "score": 30}
    if independent:
        for match in search_independent_study(index, terms, question):
            existing = candidates.get(match["row"]["id"])
            if existing is not None:
                existing["score"] += match["score"]; existing["method"] = match["method"]
            else:
                candidates[match["row"]["id"]] = dict(match)
    for candidate in candidates.values():
        row = candidate["row"]
        candidate["score"] += _english_score(row["title"], terms)*25 + _english_score(row["text"], terms)*2
    chosen, seen, chars = [], set(), 0
    for candidate in sorted(candidates.values(), key=lambda r: (-r["score"], canonical_id_sort_key(r["row"]["id"]))):
        row = candidate["row"]
        if len(chosen) >= 3:
            break
        if candidate["score"] <= 0 or row["source"] in seen:
            continue
        excerpt = study_excerpt(row["text"], terms, min(2400, 4000-chars))
        if not excerpt:
            continue
        source = STUDY_SOURCES[row["source"]]
        item = {"id": f"S{len(chosen)+1}", "recordId": row["id"], "source": row["source"],
                "title": row["title"], "kind": source["kind"], "excerpt": excerpt}
        if independent:
            item.update(retrieval=candidate.get("method", "verse-linked"), license=source["license"], sourceUrl=source["url"])
        chosen.append(item); chars += text_length(excerpt); seen.add(row["source"])
    return chosen


def original_language_support(question, sources=()):
    requested = bool(re.search(r"원어|어원|헬라어|히브리어|아람어|\b[GH]\d{1,5}\b", question, re.I | re.A))
    available = not re.search(r"히브리어|아람어|\bH\d{1,5}\b", question, re.I | re.A) and any(s["kind"] == "lexicon_1890" for s in sources)
    return {"requested": requested, "available": bool(available), "limitation": "이 답변에서 확인한 자료만으로는 해당 한국어 표현에 대응하는 원어와 그 정확한 뜻을 확인하기 어렵습니다. 본문에서 드러나는 의미를 중심으로 이해해 주세요." if requested and not available else ""}


def preserve_original_language_limit(answer, support):
    if not support["limitation"]:
        return answer
    import regex
    covered = any(re.search(r"원어|어원|헬라어|히브리어|아람어", s) and re.search(r"(?:확인|단정|설명|특정).{0,25}(?:어렵|없|못)|자료.{0,20}(?:없|않)", s)
                  for s in regex.split(r"(?<=[.!?](?:\*\*)?)\s+", answer))
    return answer if covered else answer.strip()+"\n\n"+support["limitation"]


def validate_study_notes(notes, sources):
    if not isinstance(notes, list) or len(notes) > 2:
        return []
    by_id, accepted, seen = {s["id"]: s for s in sources}, [], set()
    normalize = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or ""))).strip()
    for note in notes:
        if not isinstance(note, dict):
            continue
        source, quote = by_id.get(note.get("sourceId")), normalize(note.get("quote"))
        if not source or source["id"] in seen or not 16 <= text_length(quote) <= 500 or quote not in normalize(source["excerpt"]):
            continue
        text = note.get("text", "")
        text = text.strip() if isinstance(text, str) else ""
        if not text or text_length(text) > 1200 or not korean_only(text) or not has_complete_answer_ending(text):
            continue
        if re.search(r"https?:|\]\(|<|>|\b(?:S\d|G\d{3,})\b", text, re.A):
            continue
        if re.search(r"taken up dead|was dead|restored him to life|brought to life again", quote, re.I) and has_apparent_death_claim(text):
            continue
        accepted.append({"source": source, "text": text, "quote": quote}); seen.add(source["id"])
    return accepted


def append_study_notes(answer, notes, used_crossrefs=False):
    blocks = [answer.strip()]
    if notes:
        blocks.append("### 참고 자료로 살펴본 배경")
        for note in notes:
            source = STUDY_SOURCES[note["source"]["source"]]
            blocks.append(f"**{source['title']}의 설명**\n\n{note['text']}\n\n[자료 출처]({source['url']})")
            if note["source"]["source"] == "strongGreek":
                step = STUDY_SOURCES["step"]
                blocks.append(f"원어 연결자료: [{step['title']}]({step['url']}) · [저작자표시 4.0]({step['licenseUrl']}). {step['changes']}")
        blocks.append("참고 자료의 설명과 해석은 성경 본문의 진술과 구분해 읽어 주세요.")
    if used_crossrefs:
        source = STUDY_SOURCES["crossrefs"]
        blocks.append(f"관주 자료: [{source['title']}]({source['url']}).")
    return "\n\n".join(blocks)
