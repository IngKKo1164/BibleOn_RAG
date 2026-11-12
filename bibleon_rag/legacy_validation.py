"""Historical symptom-based checks, retained only for old differential fixtures.

Named research profiles never call these semantic heuristics or delete prose.
"""
import regex as re
from .common import unique

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


def grounded_claim_failure(answer, evidence, *, precise_roles=False):
    text = str(answer).replace("**", "")
    canonical = "\n".join(v["text"] for v in evidence)
    patterns = [r"([가-힣]{2,15})(?:은|는|이|가)\s+[^.!?\n]{0,50}?(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:이었|였|입니|이다|로\s*일했)",
                r"(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:인|였던|이었던)\s+([가-힣]{2,15}?)(?=은|는|이|가|을|를|[\s,.!?]|$)"]
    if precise_roles:
        # Do not assign an embedded person's role to the reporting text/author:
        # e.g. "사도행전은 ... 갈리오가 ... 총독이었다고 기록합니다".
        patterns[0] = (r"([가-힣]{2,15}?)(?:은|는|이|가)\s+"
                       r"(?:(?![가-힣]{2,15}?(?:은|는|이|가)\s)[^.!?\n]){0,50}?"
                       r"(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:이었|였|입니|이다|로\s*일했)")
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
