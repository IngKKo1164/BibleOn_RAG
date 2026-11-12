"""Deterministic format, reference and narrow factual guards (not entailment proofs)."""
from __future__ import annotations

import unicodedata
import regex as re

from .common import text_length, unique
from .passages import parse_bible_references
from .prompt_text import OUT_OF_SCOPE_MESSAGE, UNCERTAIN_ANSWER_MESSAGE


def korean_only(text):
    if not isinstance(text, str):
        return False
    letters = re.findall(r"\p{L}", unicodedata.normalize("NFC", text))
    if not letters:
        return False
    foreign = 0
    for letter in letters:
        if re.fullmatch(r"\p{Script=Hangul}", letter):
            continue
        if not re.fullmatch(r"[\p{Script=Latin}\p{Script=Hebrew}\p{Script=Greek}]", letter):
            return False
        foreign += 1
    return foreign*100 <= len(letters)*3


def fallback(kind, usage):
    message = {"outOfScope": OUT_OF_SCOPE_MESSAGE, "greeting": "안녕하세요! 궁금한 말씀이나 신앙 이야기가 있으면 편하게 나눠 주세요.",
               "safety": "지금 많이 힘들고 무서우시겠어요. 혼자 견디지 말고 가까운 사람에게 지금 곁에 있어 달라고 연락해 주세요. 다칠 위험이 있다면 위험한 물건에서 멀어지고, 즉시 119나 112에 연락해 도움을 요청해 주세요. 자살예방상담전화 109에서도 도움을 받을 수 있어요."}.get(kind, UNCERTAIN_ANSWER_MESSAGE)
    return {"kind": kind, "text": message, "citations": [], "contextSummary": "", "outOfScope": kind == "outOfScope",
            "needsClarification": kind == "uncertain", "usage": usage}


def has_complete_answer_ending(text):
    if not isinstance(text, str) or not text.strip():
        return False
    body = re.sub(r"[\p{Extended_Pictographic}\p{Emoji_Modifier}\uFE0F\u200D\s]+$", "", text.strip()).strip()
    if not body:
        return False
    lines = re.split(r"\r?\n", body)
    fence, closed_at, has_code = None, -1, False
    for i, line in enumerate(lines):
        marker = re.match(r" {0,3}(`{3,}|~{3,})(.*)$", line)
        if fence is None and marker and not (marker[1][0] == "`" and "`" in marker[2]):
            fence, has_code = marker[1], False
        elif fence:
            if marker and marker[1][0] == fence[0] and len(marker[1]) >= len(fence) and not marker[2].strip():
                closed_at, fence = i if has_code else -1, None
            elif line.strip():
                has_code = True
    if fence:
        return False
    if closed_at == len(lines)-1:
        return True
    last = lines[-1]
    if re.search(r'''[.!?。！？][\s"'”’)*\]]*$''', last):
        return True
    if re.fullmatch(r"\s*(?:[-+*]|\d+[.)])\s+\S(?:.*\S)?", last) and re.search(r"[\p{L}\p{N}]", last):
        return not bool(re.search(r"(?:\s때|[가-힣](?:하며|하면서|하지만|때문에|위해서|통해서|하므로|한다면|있다면|있으며))[*_\s]*$", last))
    def cells(line):
        return [c.strip() for c in re.split(r"(?<!\\)\|", re.sub(r"^\||\|$", "", line.strip()))]
    for i in range(len(lines)-2, 0, -1):
        if not lines[i].strip():
            break
        delimiter = cells(lines[i])
        if (len(delimiter) >= 2 and all(re.fullmatch(r":?-{3,}:?", c) for c in delimiter)
                and len(cells(lines[i-1])) == len(delimiter)
                and all(len(cells(line)) == len(delimiter) and all(cells(line)) for line in lines[i+1:])):
            return True
    return False


def format_answer_part(text, *, continuation=False):
    value = re.sub(r"\r\n?", "\n", text.strip())
    if continuation:
        heading = re.search(r"^## [^\n]+$", value, re.M)
        if heading and heading.start() > 0:
            lead = value[:heading.start()].strip()
            if lead and not re.search(r"^#{1,6} ", lead, re.M):
                value = "\n\n".join(s for s in [heading[0], lead, value[heading.end():].strip()] if s)
    blocks = []
    for block in re.split(r"\n\s*\n", value):
        bold = re.fullmatch(r"\*\*([^*\n]+)\*\*", block)
        blocks.append(bold[1] if bold and text_length(bold[1]) > 80 else block)
    return "\n\n".join(blocks)


def inline_references_in_evidence(answer, corpus, evidence):
    parsed = parse_bible_references(corpus, answer, max_matches=128, max_verses=96, max_range_verses=96,
                                   chapter_mode="mention", known_omissions={"mark:9:44", "mark:9:46"})
    ids = {v["id"] for v in evidence}
    return (not parsed["invalid"] and all(v["id"] in ids for v in parsed["refs"])
            and all(any(v["id"].startswith(f"{c['bookId']}:{c['chapter']}:") for v in evidence) for c in parsed["chapters"]))


def complete_rag_answer(answer, corpus):
    if has_complete_answer_ending(answer):
        return True
    body = answer.strip()
    # Bold citation wrappers are presentation only. Coordinate validation remains strict.
    body = re.sub(r"\*\*(\([^()]+\))\*\*$", r"\1", body)
    tail = re.search(r"\(([^()]*)\)\s*$", body)
    if tail is None:
        return False
    parsed = parse_bible_references(corpus, tail[1], max_matches=128, max_verses=96, max_range_verses=96, chapter_mode="mention")
    if not parsed["exact"] or parsed["invalid"]:
        return False
    prefix = body[:tail.start()].strip()
    # Adjacent citation groups do not excuse unfinished prose.
    return bool(prefix) and (has_complete_answer_ending(prefix) or complete_rag_answer(prefix, corpus))


def unresolved_referent(question, previous_evidence=None, passage=()):
    if previous_evidence or passage:
        return False
    text = re.sub(r"[\s?!.,]", "", str(question))
    return bool(re.fullmatch(r"(?:성경(?:에|에서)(?:나오는|나온)?)?(?:그|이|저)(?:사람|인물|청년|여자|남자|사건|구절|말씀)(?:은|는|이|가|을|를)?(?:누구(?:였|이었)?(?:지|야|나요|어요)?|뭐(?:였|이었)?(?:지|야|예요)?|무엇(?:인가요|이었지)?|알려줘|설명해줘)", text))


# Compatibility for historical direct callers and frozen fixture tests only.
# Normal profile execution uses literal source contracts, not these heuristics.
def __getattr__(name):
    if name in {'has_apparent_death_claim', 'outcome_conflicts',
                'grounded_claim_failure', 'constrain_rag_draft'}:
        from . import legacy_validation
        return getattr(legacy_validation, name)
    raise AttributeError(name)
