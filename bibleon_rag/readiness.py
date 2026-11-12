"""Explicit partial-answer contracts after the final evidence review.

Quotes establish source identity. The review is a fallible semantic judgment;
these guards preserve its limits rather than certify the truth of prose.
"""
from copy import deepcopy

from .common import BibleProviderError

REVIEW_PROMPT = """
각 검토 항목의 reason에는 원문을 대조한 판정 이유를 짧게 쓴다. missing/conflict이면 무엇이 없거나 어느 대상·시기·주장이 충돌하는지 구체적으로 쓴다.
blocksConclusion은 이 항목을 확인하지 못하면 질문의 핵심 결론을 낼 수 없는지다. 부수 설명이 없는 것과 핵심 결론의 전제가 없는 것을 구별한다.
nextInformation은 missing/conflict일 때 필요한 본문·출처 또는 대상 확인 질문이다. supported/refuted이면 빈 문자열이고 blocksConclusion=false다.
주석의 다른 견해, 다른 발화자·상황·시기는 구분해 읽는다. 이를 곧 성경 자체의 모순으로 단정하지 않는다. 본문과 주석의 주장도 구분한다.
기존 needs가 원 질문을 빠뜨렸는지 독립적으로 읽는다. 질문의 조건·주체·비교 양쪽을 축소하지 않는다. 요청하지 않은 세부는 필수 요구로 늘리지 않는다.
"""
ANSWER_PROMPT = """
answerReadiness의 confirmed 항목만 근거가 확인된 설명으로 작성한다. unresolved 항목을 추측으로 채우지 않는다.
withholdConclusion=true이면 그 미확인 항목에 의존하는 핵심 결론은 내리지 말고, 독립적으로 확인된 부분만 설명한다.
coveredNeedIds에는 해당 문단에서 실제 설명한 confirmed의 needId를 적는다. ID만 적고 설명을 생략하지 않는다.
부분 답변의 미확인 사항·이유·추가 정보 제안은 별도로 표시되므로 같은 한계를 문단마다 반복하지 않는다.
"""


def fields():
    return {'reason': {'type': 'string', 'minLength': 1, 'maxLength': 160},
            'nextInformation': {'type': 'string', 'maxLength': 160},
            'blocksConclusion': {'type': 'boolean'}}


def details(raw, status):
    if (not isinstance(raw, dict) or type(raw.get('blocksConclusion')) is not bool
            or any(not isinstance(raw.get(k), str) or len(raw[k]) > 160
                   for k in ('reason', 'nextInformation')) or not raw['reason'].strip()):
        raise ValueError('INVALID_REVIEW_DETAILS')
    unresolved = status in {'missing', 'conflict'}
    if (unresolved != bool(raw['nextInformation'].strip())
            or not unresolved and raw['blocksConclusion']):
        raise ValueError('INVALID_REVIEW_DETAILS')
    return {k: raw[k].strip() if isinstance(raw[k], str) else raw[k] for k in fields()}


def attach_details(checks, rows):
    by_id = {r['needId']: r for r in rows}
    result = []
    for check in checks:
        raw = by_id[check['needId']]
        if check.get('invalidReason'):
            extra = {'reason': '선택한 원문 인용과 출처의 일치를 확인하지 못했습니다.',
                     'nextInformation': '이 내용을 확인할 성경 본문이나 출처를 알려 주시겠어요?',
                     'blocksConclusion': True}
        else:
            extra = details(raw, check['status'])
        result.append({**check, **extra})
    return result


def disposition(checks):
    essential = [c for c in checks if c['importance'] == 'essential']
    confirmed = [c for c in essential if c['status'] in {'supported', 'refuted'}]
    gaps = [c for c in essential if c['status'] in {'missing', 'conflict'}]
    return {'status': 'insufficient' if not confirmed else 'partial' if gaps else 'complete',
            'confirmed': [{k: c[k] for k in ('needId', 'partId', 'need', 'status')} for c in confirmed],
            'unresolved': [{k: c[k] for k in ('needId', 'partId', 'need', 'status',
                                              'reason', 'nextInformation', 'blocksConclusion')} for c in gaps],
            'withholdConclusion': any(c['blocksConclusion'] for c in gaps)}


def bind_answer_schema(schema, readiness):
    """Bind output coverage to reviewed needs, not to benchmark gold."""
    schema = deepcopy(schema)
    for part_id, part in schema['properties']['answers']['properties'].items():
        ids = [n['needId'] for n in readiness['confirmed'] if n['partId'] == part_id]
        missing = any(n['partId'] == part_id for n in readiness['unresolved'])
        status = 'insufficient' if not ids else 'partial' if missing else 'supported'
        options = part.get('anyOf', [part])
        keep = [p for p in options if status in p['properties']['status']['enum']]
        if not keep:
            # No actual scripture evidence is available for this part. Do not
            # fabricate supports just to satisfy a semantic review label.
            keep = [p for p in options if 'insufficient' in p['properties']['status']['enum']]
            ids = []
        for option in keep:
            option['properties']['coveredNeedIds'] = {'type': 'array', 'minItems': len(ids), 'maxItems': len(ids),
                'items': {'type': 'string', **({'enum': ids} if ids else {})}}
            option['required'].append('coveredNeedIds')
        schema['properties']['answers']['properties'][part_id] = {'anyOf': keep}
    return schema


def decode_coverage(draft, readiness):
    coverage = {}
    for part_id, row in draft.get('answers', {}).items():
        ids = row.pop('coveredNeedIds', None)
        expected = {n['needId'] for n in readiness['confirmed'] if n['partId'] == part_id}
        if row.get('status') == 'insufficient':
            expected = set()
        if (not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
                or len(ids) != len(set(ids)) or set(ids) != expected):
            raise BibleProviderError('AI_EVIDENCE_ASSESSMENT_INVALID')
        coverage[part_id] = ids
    return coverage


def unresolved_text(readiness):
    rows = readiness['unresolved']
    if not rows:
        return ''
    lines = ['**확인하지 못한 부분**']
    for row in rows:
        label = '자료의 설명을 함께 확인해야 합니다' if row['status'] == 'conflict' else '추가 근거가 필요합니다'
        lines.append(f"- **{row['need']}**: {row['reason']} {label}. {row['nextInformation']}")
    if readiness['withholdConclusion']:
        lines.append('이 부분이 확인되기 전에는 질문의 핵심 결론을 확정하기 어렵습니다.')
    return '\n'.join(lines)
