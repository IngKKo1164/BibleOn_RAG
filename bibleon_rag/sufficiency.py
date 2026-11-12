"""Question-part evidence membership, literal support and bounded citation repair."""
from __future__ import annotations

import re

from .common import BibleProviderError, object_schema, text_length, unique_verses
from .validation import format_answer_part, inline_references_in_evidence

STATUSES = ["supported", "partial", "insufficient"]
GAPS = ["none", "detail_unstated", "background_unavailable", "passage_missing", "comparison_incomplete"]
MESSAGES = {
    "detail_unstated": "질문하신 세부 사항은 현재 확인한 본문에 명시되어 있지 않아 단정할 수 없습니다.",
    "background_unavailable": "추가적인 역사·배경 설명을 뒷받침할 자료가 충분하지 않아 그 부분은 단정하지 않겠습니다.",
    "passage_missing": "이 부분에 직접 답할 성경 근거를 충분히 찾지 못했습니다. 관련 인물이나 사건, 구절을 조금 더 구체적으로 알려 주세요.",
    "comparison_incomplete": "비교에 필요한 양쪽 근거를 모두 확인하지 못해 같은 사건이나 인물이라고 단정할 수 없습니다.",
}
CITATION_REPAIR_PROMPT = '이번 요청은 장절 배정 교정 1회다. answers와 citations만 반환한다. 해당 P의 allowedReferences 밖 장절이 기존 text에 섞였다. 이 목록의 연속 범위 안에서만 장절을 쓰고 빠진 절 번호를 범위로 잇지 않는다. 허용된 실제 본문으로 설명·깊이·##/### 구조를 보존하고 근거 없는 주장은 삭제하거나 한계를 밝힌다. answerOutline은 중복 방지용 구성이지 사실 근거가 아니다. 전체 결론은 P1만 쓰고 이후 P는 ## 제목으로 시작한다. 다른 P에서 이미 다룬 결론을 반복하거나 그 P에 배정된 본문이 자료 전체에 없다고 말하지 않는다. 요청하지 않은 역사 배경의 부족 안내는 만들지 않는다. 다른 P의 본문을 빌리지 않고 supports 중 하나 이상을 citations에도 넣는다.'


def sufficient_answer_schema(parts, evidence_by_part, *, strict_contract=False, reference_supports=False):
    props = {}
    for part in parts:
        ids = [v["id"] for v in evidence_by_part.get(part["partId"], [])]
        props[part["partId"]] = object_schema({
            "status": {"type": "string", "enum": STATUSES if ids else ["insufficient"]},
            "text": {"type": "string", **({} if ids else {"maxLength": 0})},
            "supports": {"type": "array", "maxItems": 2 if ids else 0, "items": object_schema({
                "verseId": {"type": "string", **({"enum": ids} if ids else {})},
                "quote": {"type": "string", "minLength": 4, "maxLength": 160}})},
            "gap": {"type": "string", "enum": GAPS if ids else [g for g in GAPS if g != "none"]}})
        support_key = "supportIds" if reference_supports else "supports"
        if reference_supports:
            props[part["partId"]]["properties"].pop("supports")
            props[part["partId"]]["properties"][support_key] = {
                "type": "array", "maxItems": 2 if ids else 0,
                "items": {"type": "string", **({"enum": ids} if ids else {})}}
            props[part["partId"]]["required"] = list(props[part["partId"]]["properties"])
        if strict_contract:
            from copy import deepcopy
            choices = []
            for status in (STATUSES if ids else ["insufficient"]):
                option = deepcopy(props[part["partId"]])
                p = option["properties"]
                p["status"]["enum"] = [status]
                p["gap"]["enum"] = ["none"] if status == "supported" else [g for g in GAPS if g != "none"]
                if status == "insufficient":
                    p["text"]["maxLength"] = 0
                    p[support_key]["maxItems"] = 0
                else:
                    p["text"]["minLength"] = 1
                    p[support_key]["minItems"] = 1
                choices.append(option)
            props[part["partId"]] = {"anyOf": choices}
    return object_schema(props)


def normalize_answer_linebreaks(draft):
    """Render accidentally double-escaped paragraph breaks; never alter evidence.

    Leave single literal escapes alone. Only answer prose is presentation data;
    canonical support quotes and source material must remain byte-for-byte.
    """
    answers=draft.get('answers') if isinstance(draft,dict) else None
    if isinstance(answers,dict):
        for row in answers.values():
            if isinstance(row,dict) and isinstance(row.get('text'),str):
                row['text']=row['text'].replace('\\n\\n','\n\n')


def decode_support_references(draft, evidence_by_part):
    """Bind model-selected IDs to canonical text; this is not semantic approval."""
    answers = draft.get("answers")
    if not isinstance(answers, dict) or set(answers) != set(evidence_by_part):
        raise BibleProviderError("AI_EVIDENCE_ASSESSMENT_INVALID")
    for key, row in answers.items():
        allowed = {v["id"]: v["text"] for v in evidence_by_part[key]}
        ids = row.get("supportIds") if isinstance(row, dict) else None
        if (not isinstance(row, dict) or set(row) != {"status", "gap", "text", "supportIds"}
                or not isinstance(ids, list) or len(ids) > 2
                or any(not isinstance(v, str) or v not in allowed for v in ids)
                or len(ids) != len(set(ids))):
            raise BibleProviderError("AI_EVIDENCE_ASSESSMENT_INVALID")
        row["supports"] = [{"verseId": v, "quote": allowed[v]} for v in ids]
        del row["supportIds"]
    return draft


def normalize_support_spacing(draft, evidence_by_part):
    """Resolve whitespace-only transcription differences to the exact source.

    Never repair paraphrases, different punctuation or wrongly assigned verses.
    This is a literal-match repair, not a factual/semantic approval.
    """
    for part, row in draft.get("answers", {}).items():
        allowed = {v["id"]: v["text"] for v in evidence_by_part.get(part, [])}
        for support in row.get("supports", []):
            quote = support.get("quote")
            text = allowed.get(support.get("verseId"), "")
            if isinstance(quote, str) and quote.strip() and quote not in text:
                match = re.search(r"\s+".join(re.escape(t) for t in quote.split()), text)
                if match and 4 <= text_length(match[0]) <= 160:
                    support["quote"] = match[0]
    return draft


def missing_evidence_reply(usage, *, ambiguous=False):
    return {"kind": "uncertain", "text": "어떤 인물이나 사건을 말씀하시는지 조금 더 구체적으로 알려 주세요." if ambiguous else MESSAGES["passage_missing"],
            "citations": [], "contextSummary": "", "outOfScope": False, "needsClarification": True, "usage": usage}


def validate_sufficient_answers(raw, parts, evidence_by_part, citations=(), *, corpus=None, reference_evidence_by_part=None,
                                needs_clarification=True):
    reference_evidence_by_part = evidence_by_part if reference_evidence_by_part is None else reference_evidence_by_part
    def invalid():
        raise BibleProviderError("AI_EVIDENCE_ASSESSMENT_INVALID")
    if not isinstance(raw, dict) or len(raw) != len(parts):
        invalid()
    citation_ids = {f"{c['bookId']}:{c['chapter']}:{c['verse']}" for c in citations}
    confirmed, blocks, checks = [], [], []
    for part in parts:
        row = raw.get(part["partId"])
        allowed = {v["id"]: v for v in evidence_by_part.get(part["partId"], [])}
        if (not isinstance(row, dict) or set(row) != {"gap", "status", "supports", "text"}
                or row["status"] not in STATUSES or row["gap"] not in GAPS or not isinstance(row["text"], str)
                or not isinstance(row["supports"], list) or len(row["supports"]) > 2):
            invalid()
        missing = row["status"] == "insufficient"
        if (row["status"] == "supported") != (row["gap"] == "none"):
            invalid()
        if missing:
            if row["text"].strip() or row["supports"]:
                invalid()
        else:
            if not row["text"].strip() or not row["supports"]:
                invalid()
            for support in row["supports"]:
                if (not isinstance(support, dict) or set(support) != {"quote", "verseId"}
                        or not isinstance(support["quote"], str) or text_length(support["quote"]) < 4
                        or support["verseId"] not in allowed or support["quote"] not in allowed[support["verseId"]]["text"]
                        or (text_length(support["quote"]) > 160 and support["quote"] != allowed[support["verseId"]]["text"])
                        or re.fullmatch(r"[\s(（]*없음[)）\s]*", allowed[support["verseId"]]["text"])):
                    invalid()
            if not any(s["verseId"] in citation_ids for s in row["supports"]):
                raise BibleProviderError("AI_CITATION_INVALID")
            if corpus is not None and not inline_references_in_evidence(row["text"], corpus, reference_evidence_by_part.get(part["partId"], [])):
                raise BibleProviderError("AI_CITATION_INVALID")
            confirmed.append(part["partId"])
        limitation = "" if row["gap"] == "none" else MESSAGES[row["gap"]]
        if row['gap'] == 'passage_missing' and not needs_clarification:
            # An unavailable source is not an ambiguous user request. Keep the
            # same gap/status and all membership checks; only the notice differs.
            limitation = '이 부분은 현재 확보한 성경 근거만으로 확인하지 못했습니다.'
        blocks.append("\n\n".join(s for s in [format_answer_part(row["text"], continuation=bool(blocks)), limitation] if s))
        checks.append({"partId": part["partId"], "status": row["status"], "gap": row["gap"], "supportVerseIds": [s["verseId"] for s in row["supports"]]})
    return {"text": "\n\n---\n\n".join(blocks), "confirmedParts": confirmed, "checks": checks,
            "allMissing": not confirmed, "partial": any(c["status"] != "supported" for c in checks)}


def build_part_evidence(parts, coverage, primary, section_context=()):
    result = {}
    for part in parts:
        selected = next((c for c in coverage if c["partId"] == part["partId"]), None)
        if selected is None:
            result[part["partId"]] = primary
            continue
        allowed = set()
        for verse in primary:
            for key in selected["chunkIds"]:
                m = re.fullmatch(r"([^:]+):(\d+):(\d+)-(\d+)", key)
                if m and verse["bookId"] == m[1] and verse["chapter"] == int(m[2]) and int(m[3]) <= verse["verse"] <= int(m[4]):
                    allowed.add(verse["id"])
        for section in section_context:
            if any(key in allowed for key in section["primaryVerseIds"]):
                allowed.update(section["contextVerseIds"])
        result[part["partId"]] = [v for v in primary if v["id"] in allowed]
    return result


def allowed_reference_ranges(verses, corpus):
    chapters = {}
    for v in verses:
        chapters.setdefault((v["bookId"], v["chapter"]), set()).add(v["verse"])
    output = []
    for (book_id, chapter), numbers in chapters.items():
        runs = []
        for n in sorted(numbers):
            if runs and runs[-1][1]+1 == n:
                runs[-1][1] = n
            else:
                runs.append([n, n])
        spans = ",".join(str(a) if a == b else f"{a}-{b}" for a, b in runs)
        output.append(f"{corpus[book_id]['name']} {chapter}:{spans}")
    return output


def repairable_support_quotes(row):
    """Only well-shaped prose with copied-support failures may use the one repair pass."""
    return (isinstance(row, dict) and set(row) == {"status", "gap", "supports", "text"}
            and row["status"] in {"supported", "partial"} and row["gap"] in GAPS
            and (row["status"] == "supported") == (row["gap"] == "none")
            and isinstance(row["text"], str) and bool(row["text"].strip())
            and isinstance(row["supports"], list) and 1 <= len(row["supports"]) <= 2
            and all(isinstance(s, dict) and set(s) == {"verseId", "quote"}
                    and isinstance(s["verseId"], str) and isinstance(s["quote"], str) for s in row["supports"]))


def invalid_citation_parts(*, draft, parts, part_evidence, reference_evidence_by_part, corpus, repair_supports=False):
    invalid = []
    for p in parts:
        try:
            validate_sufficient_answers({p["partId"]: draft.get("answers", {}).get(p["partId"])}, [p], part_evidence,
                                        draft["citations"], corpus=corpus, reference_evidence_by_part=reference_evidence_by_part)
        except BibleProviderError as error:
            support_copy_failure = (repair_supports and error.code == "AI_EVIDENCE_ASSESSMENT_INVALID"
                                    and repairable_support_quotes(draft.get("answers", {}).get(p["partId"])))
            if error.code != "AI_CITATION_INVALID" and not support_copy_failure:
                raise
            invalid.append(p)
    return invalid


def missing_anchor_parts(draft, parts, part_evidence, required_anchor=()):
    """Locate an already-supplied explicit anchor for the existing one repair.

    The final contract needs at least one requested verse, not every verse in
    a range. Do not create a citation or assign an anchor to an unrelated part.
    """
    anchors = {v['id'] for v in required_anchor}
    cited = {f"{c.get('bookId')}:{c.get('chapter')}:{c.get('verse')}"
             for c in draft.get('citations', []) if isinstance(c, dict)}
    if not anchors or anchors & cited:
        return []
    for part in parts:
        if anchors & {v['id'] for v in part_evidence.get(part['partId'], [])}:
            return [part]
    raise BibleProviderError('AI_CITATION_INVALID')


def citation_choices(verses, *, deduplicate=True):
    if not verses:
        # An empty anyOf is an invalid JSON Schema, even inside a zero-length
        # array. No Bible address can be proposed from an empty evidence set.
        return object_schema({})
    groups = {}
    for v in verses:
        groups.setdefault((v["bookId"], v["chapter"]), []).append(v["verse"])
    choices = [object_schema({"bookId": {"type": "string", "enum": [book]}, "chapter": {"type": "integer", "enum": [chapter]},
                              "verse": {"type": "integer", "enum": list(dict.fromkeys(numbers)) if deduplicate else numbers}})
               for (book, chapter), numbers in groups.items()]
    return choices[0] if len(choices) == 1 else {"anyOf": choices}


async def repair_part_citations(*, draft, parts, part_evidence, reference_evidence_by_part, corpus, system, call, observe, visible_tokens,
                               repair_supports=False, strict_contract=False, reference_supports=False, required_scenes=(), bind_visible_scenes=False,
                               required_anchor=()):
    invalid = invalid_citation_parts(draft=draft, parts=parts, part_evidence=part_evidence,
                                     reference_evidence_by_part=reference_evidence_by_part, corpus=corpus, repair_supports=repair_supports)
    from .review_memory import invalid_scene_parts, SCENE_ANSWER_PROMPT
    scene_invalid = invalid_scene_parts(draft, required_scenes, corpus)
    anchor_parts = missing_anchor_parts(draft, parts, part_evidence, required_anchor)
    invalid_ids = {p['partId'] for p in invalid} | scene_invalid | {p['partId'] for p in anchor_parts}
    invalid = [p for p in parts if p['partId'] in invalid_ids]
    if not invalid:
        return {"draft": draft, "visibleTokens": visible_tokens}
    observe({"stage": "citation-repair", "result": {"parts": [p["partId"] for p in invalid], "attempt": 1}})
    refs = unique_verses(v for p in invalid for v in reference_evidence_by_part.get(p["partId"], []))
    schema = object_schema({"answers": sufficient_answer_schema(invalid, part_evidence, strict_contract=strict_contract, reference_supports=reference_supports),
                            "citations": {"type": "array", "maxItems": 8, "items": citation_choices(refs, deduplicate=False)}})
    payload = {"questionParts": [{**p, "previousText": draft["answers"][p["partId"]]["text"]} for p in invalid],
               "answerOutline": [{"partId": p["partId"], "question": p["question"],
                                   "headings": re.findall(r"^#{2,3} (.+)$", draft["answers"].get(p["partId"], {}).get("text", ""), re.M)[:8]} for p in parts],
               "evidence": [{k: v[k] for k in ("id", "reference", "text")} for v in refs],
               "partEvidenceIds": {p["partId"]: [v["id"] for v in part_evidence[p["partId"]]] for p in invalid},
               "allowedReferences": {p["partId"]: allowed_reference_ranges(reference_evidence_by_part[p["partId"]], corpus) for p in invalid}}
    correction = "\n내부 supports 복사도 검사한다. 해당 절에서 연속된 4~160자 원문을 그대로 복사하고 띄어쓰기·철자·부정어를 바꾸지 않는다. 생략 부호로 두 구간을 잇지 않는다." if repair_supports else ""
    if reference_supports:
        correction = "\n내부 근거는 supports 복사 대신 supportIds에 해당 P의 실제 절 ID만 선택한다. 인용 원문은 프로그램이 정본에서 결합한다. 선택한 ID 중 하나 이상을 citations에도 넣는다."
    scenes = [r for r in required_scenes if r['partId'] in invalid_ids]
    if scenes:
        payload['requiredScenes'] = scenes
        correction += SCENE_ANSWER_PROMPT + '\n필수 장면 설명이 빠졌다면 제공된 원문으로 해당 문단 안에서 고친다.'
    repair_prompt = CITATION_REPAIR_PROMPT
    if anchor_parts:
        anchors = {v['id'] for v in required_anchor}
        payload['primaryReferences'] = {p['partId']: [v['id'] for v in part_evidence[p['partId']] if v['id'] in anchors]
                                        for p in anchor_parts}
        repair_prompt = repair_prompt.replace('해당 P의 allowedReferences 밖 장절이 기존 text에 섞였다.',
                                               '질문이 지정한 primaryReferences의 본문 인용이 빠졌다.')
        correction += ('\nprimaryReferences 중 하나 이상의 본문이 질문에 어떻게 연결되는지 실제 text에서 설명하고 '
                       '그 장절을 text와 citations에 함께 넣는다. 제공된 본문을 벗어난 설명이나 인용만 덧붙이지 않는다.')
    fixed = await call("answer", system+"\n"+repair_prompt+correction, payload, schema)
    normalize_answer_linebreaks(fixed['result'])
    if visible_tokens + fixed["visibleTokens"] > 3000:
        raise BibleProviderError("AI_INCOMPLETE")
    if reference_supports:
        decode_support_references(fixed["result"], {p["partId"]: part_evidence[p["partId"]] for p in invalid})
    if bind_visible_scenes:
        from .review_memory import bind_visible_scene_citations
        bind_visible_scene_citations(fixed['result'],scenes,corpus)
    validate_sufficient_answers(fixed["result"]["answers"], invalid, part_evidence, fixed["result"]["citations"],
                                corpus=corpus, reference_evidence_by_part=reference_evidence_by_part)
    if invalid_scene_parts(fixed['result'], scenes, corpus):
        raise BibleProviderError('AI_SCENE_COVERAGE_INVALID')
    answers = {**draft["answers"], **fixed["result"]["answers"]}
    support_ids = {s["verseId"] for a in answers.values() for s in a["supports"]}
    refs = {f"{c['bookId']}:{c['chapter']}:{c['verse']}": c for c in draft["citations"]+fixed["result"]["citations"]}
    anchors = {v['id'] for v in required_anchor}
    citations = [c for _, c in sorted(refs.items(), key=lambda pair: (-(pair[0] in anchors), -(pair[0] in support_ids)))][:8]
    validate_sufficient_answers(answers, parts, part_evidence, citations, corpus=corpus, reference_evidence_by_part=reference_evidence_by_part)
    merged = {**draft, 'answers': answers, 'citations': citations}
    if missing_anchor_parts(merged, parts, part_evidence, required_anchor):
        raise BibleProviderError('AI_CITATION_INVALID')
    if bind_visible_scenes:
        bind_visible_scene_citations(merged,required_scenes,corpus)
        validate_sufficient_answers(answers,parts,part_evidence,merged['citations'],corpus=corpus,
                                    reference_evidence_by_part=reference_evidence_by_part)
    if required_scenes:
        if invalid_scene_parts(merged, required_scenes, corpus):
            raise BibleProviderError('AI_SCENE_COVERAGE_INVALID')
    return {"draft": merged, "visibleTokens": visible_tokens + fixed["visibleTokens"]}
