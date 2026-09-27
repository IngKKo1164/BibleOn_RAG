"""Canonical references, request routing and heading-bounded chunk construction."""
from __future__ import annotations

import re
import unicodedata

from .common import is_integer, text_length, unique, unique_verses

BOOK_IDS = "genesis exodus leviticus numbers deuteronomy joshua judges ruth 1samuel 2samuel 1kings 2kings 1chronicles 2chronicles ezra nehemiah esther job psalms proverbs ecclesiastes songofsolomon isaiah jeremiah lamentations ezekiel daniel hosea joel amos obadiah jonah micah nahum habakkuk zephaniah haggai zechariah malachi matthew mark luke john acts romans 1corinthians 2corinthians galatians ephesians philippians colossians 1thessalonians 2thessalonians 1timothy 2timothy titus philemon hebrews james 1peter 2peter 1john 2john 3john jude revelation".split()
SHORT_NAMES = "창 출 레 민 신 수 삿 룻 삼상 삼하 왕상 왕하 대상 대하 스 느 에 욥 시 잠 전 아 사 렘 애 겔 단 호 욜 암 옵 욘 미 나 합 습 학 슥 말 마 막 눅 요 행 롬 고전 고후 갈 엡 빌 골 살전 살후 딤전 딤후 딛 몬 히 약 벧전 벧후 요일 요이 요삼 유 계".split()
CHUNK_VERSION = "krv-rnksv-heading-spans-v2"
CHUNK_POLICY = {"maxVerses": 24, "maxCharacters": 1800, "overlapVerses": 3, "boundary": "rnksv-heading-and-chapter"}
REQUEST_LIMITS = {"verses": 96, "characters": 12000, "contextVerses": 4}


def chapter_at(corpus: dict, book_id: str, chapter: int) -> dict:
    chapters = corpus.get(book_id, {}).get("chapters", {})
    return chapters.get(str(chapter), chapters.get(chapter, {}))


def text_at(corpus: dict, book_id: str, chapter: int, verse: int):
    row = chapter_at(corpus, book_id, chapter)
    return row.get(str(verse), row.get(verse))


def verse_at(corpus: dict, ref: dict) -> dict:
    if not isinstance(ref, dict) or not is_integer(ref.get("chapter")) or not is_integer(ref.get("verse")):
        raise ValueError("Invalid citation")
    book_id, chapter, verse = ref.get("bookId"), int(ref["chapter"]), int(ref["verse"])
    text = text_at(corpus, book_id, chapter, verse)
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Unknown citation")
    return {"id": f"{book_id}:{chapter}:{verse}", "bookId": book_id, "chapter": chapter, "verse": verse,
            "reference": f"{corpus[book_id]['name']} {chapter}:{verse}",
            "translation": "개역한글", "translationId": "KRV", "text": text}


def expand_references(corpus: dict, refs: list[dict]) -> list[dict]:
    if not isinstance(refs, list) or len(refs) > 8:
        raise ValueError("Invalid references")
    result = {}
    for ref in refs:
        verse_at(corpus, ref)
        for n in range(max(1, ref["verse"] - 4), ref["verse"] + 7):
            if text_at(corpus, ref["bookId"], ref["chapter"], n):
                row = verse_at(corpus, {**ref, "verse": n})
                result[row["id"]] = row
    if len(result) > 96 or sum(text_length(v["text"]) for v in result.values()) > 18000:
        raise ValueError("Evidence too long")
    return list(result.values())


def parse_bible_references(corpus: dict, text: str, *, max_matches=8, max_verses=20,
                           max_range_verses=20, chapter_mode="expand", known_omissions=frozenset()) -> dict:
    aliases = {}
    for book_id, book in corpus.items():
        aliases[re.sub(r"\s", "", book["name"])] = book_id
        if book_id in BOOK_IDS:
            aliases[SHORT_NAMES[BOOK_IDS.index(book_id)]] = book_id
    if not aliases:
        return {"refs": [], "exact": False, "invalid": False, "chapters": [], "mentions": 0}
    names = "|".join(re.escape(s) for s in sorted(aliases, key=lambda s: -len(s)))
    span_pattern = r"(?:\s*(?:[-~–—−]|부터)\s*\d{1,3}\s*절?(?:\s*까지)?)?"
    pattern = re.compile(r"(?<![가-힣])(" + names + r")\s*(\d{1,3})(?:\s*:\s*(\d{1,3})|\s*(?:장|편)(?:\s*(\d{1,3})(?:\s*절)?)?)(\s*절?" + span_pattern + r"(?:\s*[,，]\s*\d{1,3}(?:\s*:\s*\d{1,3})?\s*절?" + span_pattern + r")*)")
    normalized = unicodedata.normalize("NFKC", str(text))
    matches = list(pattern.finditer(normalized))
    refs, chapters = [], []
    invalid = len(matches) > max_matches
    for match in matches[:max_matches]:
        book_id, chapter = aliases[match[1]], int(match[2])
        row = chapter_at(corpus, book_id, chapter)
        if not row or re.match(r"\s*:", normalized[match.end():]):
            invalid = True
            continue
        first_text = match[3] or match[4]
        if not first_text and chapter_mode == "mention":
            chapters.append({"bookId": book_id, "chapter": chapter})
            continue
        if first_text:
            spans = re.split("[,，]", re.sub(r"절|까지|\s", "", first_text + match[5]).replace("부터", "-"))
        else:
            nums = list(map(int, row))
            spans = [f"{min(nums)}-{max(nums)}"]
        for span in spans:
            if ":" in span:
                chapter_text, span = span.split(":", 1)
                chapter = int(chapter_text)
            values = list(map(int, re.split("[-~–—−]", span)))
            first, last = values[0], values[-1]
            if last < first or last - first >= max_range_verses:
                invalid = True
                continue
            for verse in range(first, last + 1):
                value = text_at(corpus, book_id, chapter, verse)
                if (first < verse < last and f"{book_id}:{chapter}:{verse}" in known_omissions
                        and (value is None or re.fullmatch(r"\s*\(?없음\)?\s*", value))):
                    continue
                try:
                    refs.append(verse_at(corpus, {"bookId": book_id, "chapter": chapter, "verse": verse}))
                except ValueError:
                    invalid = True
    remaining = re.sub(r"[\s,.?!，。]", "", pattern.sub("", normalized))
    exact = bool(matches) and bool(re.fullmatch(r"(?:본문|말씀|보여줘|읽어줘|알려줘|보여주세요|읽어주세요)*", remaining))
    refs = unique_verses(refs)
    return {"refs": refs[:max_verses], "exact": exact, "invalid": invalid or len(refs) > max_verses,
            "chapters": chapters, "mentions": len(matches)}


def resolve_bible_request(corpus: dict, question: str, passage: list[dict] | None = None) -> dict:
    parsed = parse_bible_references(corpus, question, max_verses=96, max_range_verses=96)
    selected = unique_verses(verse_at(corpus, v) for v in passage or [])
    primary = parsed["refs"] if parsed["mentions"] else selected
    invalid = parsed["invalid"] or len(primary) > 96 or sum(text_length(v["text"]) for v in primary) > 12000
    direct = not invalid and bool(primary)
    lookup = direct and parsed["exact"]
    sources = bool(re.search(r"배경|역사|당시|문화|원어|어원|주석|사전|\b[GH]\d{1,5}\b", question, re.I | re.A))
    crossrefs = bool(re.search(r"관주|평행|교리|신학|교단|비교|차이|연결|성경\s*전체|다른\s*(?:성경|본문|구절)", question))
    context, ids = [], {v["id"] for v in primary}
    if direct and not lookup and re.search(r"문맥|앞뒤|전후|흐름", question):
        for anchor in (primary[0], primary[-1]):
            for offset in (-2, -1, 1, 2):
                if len(context) >= 4:
                    break
                try:
                    v = verse_at(corpus, {**anchor, "verse": anchor["verse"] + offset})
                    if v["id"] not in ids:
                        context.append(v)
                        ids.add(v["id"])
                except ValueError:
                    pass
    route = "lookup" if lookup else ("passage-study" if sources or crossrefs else "passage-explanation") if direct else "retrieval"
    return {"direct": direct, "lookup": lookup, "invalid": invalid, "primary": primary,
            "context": context, "sources": sources, "crossrefs": crossrefs, "route": route}


def selection_parts(question: str) -> list[dict]:
    clauses = [s.strip() for s in re.split(r"(?<=[?？])\s*|\s+(?:그리고|또한)\s+", question) if s.strip()]
    parts = clauses if len(clauses) <= 3 else clauses[:2] + [" ".join(clauses[2:])]
    return [{"partId": f"P{i+1}", "question": s} for i, s in enumerate(parts)]


def create_bible_chunks(corpus: dict, boundaries: dict | None = None) -> list[dict]:
    if boundaries is None:
        from .data import load_headings
        boundaries = load_headings()["boundaries"]
    chunks = []
    for book_id, book in corpus.items():
        for chapter_key in sorted(book["chapters"], key=int):
            chapter = book["chapters"][chapter_key]
            rows = [{"verse": int(v), "text": text} for v, text in sorted(chapter.items(), key=lambda kv: int(kv[0]))]
            if not rows:
                continue
            starts = set(boundaries.get(book_id, {}).get(str(chapter_key), []))
            offsets = sorted({0, *(i for i, row in enumerate(rows) if row["verse"] in starts)})
            for group, offset in enumerate(offsets):
                section = rows[offset:offsets[group+1] if group+1 < len(offsets) else len(rows)]
                section_start, section_end = section[0]["verse"], section[-1]["verse"]
                section_id = f"{book_id}:{chapter_key}:{section_start}-{section_end}"
                start = 0
                while start < len(section):
                    end, characters = start, 0
                    while end < len(section) and end - start < 24:
                        size = text_length(section[end]["text"])
                        if end > start and characters + size > 1800:
                            break
                        characters += size
                        end += 1
                    verses = section[start:end]
                    first, last = verses[0]["verse"], verses[-1]["verse"]
                    lines = "\n".join(f"{v['verse']} {v['text']}" for v in verses)
                    chunks.append({"id": f"{book_id}:{chapter_key}:{first}-{last}", "bookId": book_id,
                                   "chapter": int(chapter_key), "start": first, "end": last,
                                   "sectionId": section_id, "sectionStart": section_start, "sectionEnd": section_end,
                                   "sectionBoundary": "heading" if section_start in starts else "chapter",
                                   "reference": f"{book['name']} {chapter_key}:{first}-{last}", "verses": verses,
                                   "text": f"{book['name']} {chapter_key}장\n{lines}"})
                    if end == len(section):
                        break
                    start = max(start+1, end-3)
    return chunks


def compile_heading_boundaries(corpus: dict, documents: list[dict]) -> dict:
    by_book = {}
    for doc in documents:
        key = doc.get("book", {}).get("id")
        if key in by_book:
            raise ValueError("DUPLICATE_HEADING_BOOK")
        by_book[key] = doc
    if len(by_book) != len(corpus):
        raise ValueError("HEADING_BOOK_COUNT_MISMATCH")
    boundaries, differences = {}, []
    headings = distinct = 0
    for book_id, book in corpus.items():
        doc = by_book.get(book_id)
        if not doc or len(doc["chapters"]) != len(book["chapters"]):
            raise ValueError(f"HEADING_CHAPTER_MISMATCH:{book_id}")
        boundaries[book_id], seen = {}, set()
        for source in doc["chapters"]:
            chapter = chapter_at(corpus, book_id, source["chapter"])
            if not chapter or source["chapter"] in seen:
                raise ValueError(f"HEADING_CHAPTER_MISMATCH:{book_id}:{source['chapter']}")
            seen.add(source["chapter"])
            verse_ids, starts = set(), set()
            for verse in source["verses"]:
                n = verse["verse"]
                if not is_integer(n) or n < 1 or n in verse_ids:
                    raise ValueError("INVALID_HEADING_VERSE")
                verse_ids.add(n)
                labels = [s for s in verse.get("segments", []) if s["type"] == "heading"]
                if not labels:
                    continue
                headings += len(labels)
                if not isinstance(text_at(corpus, book_id, source["chapter"], n), str):
                    raise ValueError(f"UNALIGNED_HEADING:{book_id}:{source['chapter']}:{n}")
                starts.add(n)
            canonical = list(map(int, chapter))
            only_krv = [n for n in canonical if n not in verse_ids]
            only_new = [n for n in verse_ids if n not in canonical]
            if only_krv or only_new:
                differences.append({"bookId": book_id, "chapter": source["chapter"], "onlyInKrv": only_krv, "onlyInRnksv": only_new})
            distinct += len(starts)
            boundaries[book_id][str(source["chapter"])] = sorted(starts)
    return {"boundaries": boundaries, "audit": {"headings": headings, "distinctHeadingStarts": distinct,
            "differences": differences, "unalignedHeadings": 0,
            "alignment": "Heading verse coordinates checked against KRV. Not a full verse-by-verse translation alignment."}}
