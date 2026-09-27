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
    tail = re.search(r"\(([^()]*)\)\s*$", answer.strip())
    if tail is None:
        return False
    for segment in re.split(r"[;；]", tail[1]):
        parsed = parse_bible_references(corpus, segment, max_matches=128, max_verses=96, max_range_verses=96, chapter_mode="mention")
        if not parsed["exact"] or parsed["invalid"]:
            return False
    return has_complete_answer_ending(answer.strip()[:tail.start()])


def unresolved_referent(question, previous_evidence=None, passage=()):
    if previous_evidence or passage:
        return False
    text = re.sub(r"[\s?!.,]", "", str(question))
    return bool(re.fullmatch(r"(?:성경(?:에|에서)(?:나오는|나온)?)?(?:그|이|저)(?:사람|인물|청년|여자|남자|사건|구절|말씀)(?:은|는|이|가|을|를)?(?:누구(?:였|이었)?(?:지|야|나요|어요)?|뭐(?:였|이었)?(?:지|야|예요)?|무엇(?:인가요|이었지)?|알려줘|설명해줘)", text))


def has_apparent_death_claim(text):
    for match in re.finditer(r"(?:죽은\s*것으로|죽었다고)\s*(?:여[가-힣]*|보[가-힣]*|생각[가-힣]*|추정[가-힣]*|간주[가-힣]*)", str(text)):
        tail = re.split(r"[.!?]", str(text)[match.end():])[0]
        if not re.search(r"(?:다는|라는)\s*(?:뜻|말|표현|주장)?(?:은|이)?\s*(?:아니|아닙)", match[0]+tail):
            return True
    return False


def outcome_conflicts(answer, evidence):
    found = {}
    blocks = re.split(r"\n\s*\n", str(answer))
    for verse in evidence:
        if not re.search(r"일으켜\s*보니\s*죽었|죽었는지라", verse["text"]):
            continue
        match = re.search(r"([가-힣]{2,12})라\s*하는\s*(?:청년|사람|남자|여자)", verse["text"])
        if not match:
            continue
        subject = match[1]
        for i, original in enumerate(blocks):
            block = original.replace("**", "")
            if subject in block and has_apparent_death_claim(block):
                found[i] = {"rule": "declared-death", "verseId": verse["id"], "quote": verse["text"],
                            "subject": subject, "blockIndex": i, "original": original}
    return list(found.values())


def grounded_claim_failure(answer, evidence):
    text = str(answer).replace("**", "")
    canonical = "\n".join(v["text"] for v in evidence)
    patterns = [r"([가-힣]{2,15})(?:은|는|이|가)\s+[^.!?\n]{0,50}?(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:이었|였|입니|이다|로\s*일했)",
                r"(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:인|였던|이었던)\s+([가-힣]{2,15}?)(?=은|는|이|가|을|를|[\s,.!?]|$)"]
    for sentence in re.split(r"(?<=[.!?])\s+|\n", text):
        for order, pattern in enumerate(patterns):
            for match in re.finditer(pattern, sentence):
                subject, role = (match[1], match[2]) if order == 0 else (match[2], match[1])
                if re.match(r"(?:다는|라는)\s*(?:뜻|말|주장|설명)(?:은|이)?\s*(?:아닙|아니|틀렸|잘못)", sentence[match.end():]):
                    continue
                if not any(subject in v["text"] and role in v["text"] for v in evidence):
                    return "AI_FACTS_UNSUPPORTED"
    for match in re.finditer(r"[‘“]([^’”\n]{6,300})[’”](?:라고|고)\s*(?:말씀|말했|말합|기록|선포|외쳤)", text):
        quote, before = match[1], text[max(0, match.start()-50):match.start()]
        verbatim = re.search(r"원문|그대로\s*인용|직접\s*인용", before)
        promise = re.search(r"(?:헌금|기부|돈).{0,30}(?:언제나|항상|반드시|무조건|보장).{0,30}(?:부자|부유|구원|치유)", quote)
        if not verbatim and not promise:
            continue
        normalized = re.sub(r"[\s.,!?]", "", quote)
        stem = re.sub(r"(?:습니다|니다|느니라|으니|니라|으라|라|다)$", "", normalized)
        source = re.sub(r"[\s.,!?]", "", canonical)
        if normalized not in source and not (len(stem) >= 6 and stem in source):
            return "AI_QUOTE_INVALID"
    return None


PRECISION = re.compile(r"(?:제\s*)?(?:\d+|첫|두|세|네|일|이|삼|사)\s*(?:차|번째)\s*(?:전도|선교)\s*여행|(?:기원전|기원후|서기)\s*\d|\d+\s*세기|속주|항구\s*도시|헬라어|히브리어|아람어|육로|해로|통풍|공기.{0,12}탁|(?:주거용|주거\s*공간|상점|작업장|건축\s*(?:구조|양식)|환기\s*(?:부족|문제)|산소\s*부족)|당시.{0,60}(?:관습|풍습|기온|실내\s*온도|건물|창문\s*구조)")
LIMIT_ONLY = re.compile(r"(?:기록|명시|확인|특정|단정).{0,25}(?:않|없|어렵|불가|못)|알\s*수\s*없|추측할\s*수\s*없")


def constrain_rag_draft(answer, evidence, question="", *, lexicon_available=False):
    canonical = "\n".join(v["text"] for v in evidence)
    removed, rules, kept = [], [], []
    blocks = re.split(r"\n\s*\n", re.sub(r"^(#{1,6} [^\n]+)\n(?!\n)", r"\1\n\n", str(answer), flags=re.M))
    for block in blocks:
        sentences = [block] if re.match(r"^#{1,6} ", block) else re.split(r'''((?<=[.!?](?:\*\*|[”’"])?)\s+)''', block)
        safe = []
        for i, sentence in enumerate(sentences):
            if i % 2:
                safe.append(sentence)
                continue
            match = PRECISION.search(sentence) or re.search(r"관례|관습|풍습|피로.{0,12}(?:누적|쌓|때문)|아침\s*일찍|(?:공기|산소).{0,8}(?:부족|모자)|(?:실내|방\s*안|다락).{0,16}(?:더웠|더워|열기|무더)|(?:인원|사람).{0,8}많.{0,12}(?:시사|추정)", sentence)
            language = re.search(r"헬라어|히브리어|아람어", sentence)
            unsupported_language = not lexicon_available and language is not None and language[0] not in canonical
            actions = [("기도", r"기도(?:로|와(?=.{0,35}(?:살아|살렸|치유|회복되))|를\s*통해|를\s*통한|하여|하고|하자|하였|했|하셨|를.{0,8}(?:드|하))"),
                       ("안수", r"안수(?:로|하여|하고|하자|하였|했|하셨)"), ("금식", r"금식(?:으로|하여|하고|하자|하였|했|하셨)")]
            unsupported_action = any(re.search(p, sentence) and term not in canonical and not (term == "기도" and re.search(r"간구|하나님께.{0,16}구하|주께.{0,10}구하|여호와께.{0,20}부르짖", canonical)) for term, p in actions)
            language_match = match is not None and re.fullmatch(r"(?:헬라어|히브리어|아람어)", match[0])
            unsupported_precision = match is not None and match[0] not in canonical and not (language_match and lexicon_available)
            if (unsupported_language or unsupported_action or unsupported_precision) and not LIMIT_ONLY.search(sentence):
                removed.append(sentence)
                rules.append("language-source" if unsupported_language else "action-source" if unsupported_action else "precision-source")
            else:
                safe.append(sentence)
        cleaned = "".join(safe).strip()
        if cleaned:
            kept.append(cleaned)
    cleaned, pending = [], []
    for block in kept:
        heading = re.fullmatch(r"(#{1,6}) [^\n]+", block)
        if heading:
            pending = [row for row in pending if row[0] < len(heading[1])]
            pending.append((len(heading[1]), block))
        else:
            cleaned.extend([row[1] for row in pending] + [block]); pending = []
    if removed and re.search(r"배경|시대|역사|문화|연대", question):
        cleaned.append("구체적인 연대나 당시 생활환경처럼 이 본문에 기록되지 않은 배경은 확인하기 어렵습니다.")
    return {"answer": "\n\n".join(cleaned), "removed": removed, "rules": unique(rules)}
