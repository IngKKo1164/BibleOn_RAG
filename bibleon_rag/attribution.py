"""Opt-in source-first generation wire; structural provenance is not entailment.

The existing answer call chooses actual per-part verse IDs before each prose
block. The adapter only joins text and binds supplied source hashes. It never
generates claims, fetches data, judges semantics or changes legacy answer caps.
"""
from copy import deepcopy

from .claim_review import digest
from .common import BibleProviderError, object_schema
from .sufficiency import GAPS


def source_first_schema(schema, evidence_by_part):
    result = deepcopy(schema)
    definitions = result.setdefault('$defs', {})
    signatures = {}
    for key, row in result['properties']['answers']['properties'].items():
        ids = [v['id'] for v in evidence_by_part[key]]
        signature = tuple(ids)
        if ids and signature not in signatures:
            name = 'generationVerse' + str(len(signatures) + 1)
            if name in definitions:
                raise ValueError('ATTRIBUTION_DEFINITION_COLLISION')
            signatures[signature] = name
            definitions[name] = {'type': 'string', 'enum': ids}
        item = {'$ref': '#/$defs/' + signatures[signature]} if ids else {'type': 'string'}
        for option in row.get('anyOf', [row]):
            fields = option['properties']
            if 'supportIds' not in fields or len(fields['status'].get('enum', [])) != 1:
                raise ValueError('SOURCE_FIRST_REQUIRES_ID_SUPPORTS')
            text = fields.pop('text')
            fields['supportIds']['items'] = deepcopy(item)
            block = object_schema({
                'evidenceIds': {'type': 'array', 'minItems': 1 if ids else 0, 'maxItems': min(3, len(ids)),
                                'items': deepcopy(item)},
                'text': {**text, 'minLength': 1},
            })
            insufficient = fields['status'].get('enum') == ['insufficient']
            fields['blocks'] = {'type': 'array', 'items': block,
                                **({'maxItems': 0} if insufficient else {'minItems': 1})}
            # Evidence precedes prose in the requested serial output. Status,
            # gaps and legacy support/citation rules still retain their scope.
            option['properties'] = {name: fields[name] for name in ['status', 'gap', 'supportIds', 'blocks']}
            option['required'] = list(option['properties'])
    return result


def source_first_prompt(system):
    old = 'questionParts의 각 P번호에 status, text, supportIds, gap을 작성한다.'
    if system.count(old) != 1:
        raise ValueError('SOURCE_FIRST_PROMPT_CONTRACT_CHANGED')
    return system.replace(old, 'questionParts의 각 P번호에 status, gap, supportIds, blocks를 작성한다.') + '''
각 blocks 항목은 evidenceIds를 먼저 고른 뒤 그 원문으로 뒷받침하는 text를 쓴다. 한 항목은 밀접한 설명 문장1~3개와 필요한 Markdown 제목·목록으로 구성한다.
evidenceIds는 해당 P의 실제 partEvidenceIds 중 최대3개다. 실제 주체·행동의 대상·조건·수량·앞뒤 관계를 그 원문과 맞추고, 출처가 존재한다는 이유만으로 새 사실을 덧붙이지 않는다.
기존 지침의 각 text는 blocks의 text를 순서대로 빈 줄로 이어 붙인 전체 답변을 뜻한다. 깊이·필수 설명·기존 분량 한도는 그대로이며 제목만 따로 만들거나 같은 결론을 반복하지 않는다.
insufficient는 blocks=[]·supportIds=[]다. 나머지 supportIds는 실제 blocks에서 고른 근거 중 핵심1~2개다. 인용 원문을 다시 출력하지 않으며 외부 배경은 기존 studyNotes 규칙을 따른다.
conversationUpdate가 있으면 blocks의 실제 답변 문장에서만 발췌한다. 별도의 사고과정·사실 추출 보고서·자기평가는 출력하지 않는다.'''


def decode_source_first(draft, evidence_by_part):
    """Atomically validate source scope, then render the same downstream shape."""
    def require(condition):
        if not condition:
            raise BibleProviderError('AI_ATTRIBUTION_INVALID')

    require(isinstance(draft, dict) and isinstance(draft.get('answers'), dict)
            and set(draft['answers']) == set(evidence_by_part))
    result = deepcopy(draft)
    record = {'version': 1, 'scope': 'generated_prose_before_corrections',
              'semanticApproval': False, 'parts': {}, 'sourceSha256': {}, 'normalizations': []}
    for key, row in result['answers'].items():
        allowed = {v['id']: v['text'] for v in evidence_by_part[key]}
        require(isinstance(row, dict) and set(row) == {'status', 'gap', 'supportIds', 'blocks'})
        blocks, supports = row['blocks'], row['supportIds']
        require(isinstance(blocks, list) and isinstance(supports, list) and len(supports) <= 2
                and all(isinstance(v, str) and v in allowed for v in supports)
                and len(set(supports)) == len(supports))
        require(row['status'] in {'supported', 'partial', 'insufficient'})
        require(row['gap'] in GAPS and ((row['status'] == 'supported') == (row['gap'] == 'none')))
        if row['status'] == 'insufficient':
            require(not blocks and not supports and row['gap'] != 'none')
        else:
            require(bool(blocks) and bool(supports))
        parts, spans, chosen = [], [], set()
        offset = 0
        for index, block in enumerate(blocks):
            require(isinstance(block, dict) and set(block) == {'evidenceIds', 'text'})
            raw_ids, text = block['evidenceIds'], block['text']
            require(isinstance(raw_ids, list) and 1 <= len(raw_ids) <= 3
                    and all(isinstance(v, str) and v in allowed for v in raw_ids)
                    and isinstance(text, str) and bool(text.strip()))
            ids = list(dict.fromkeys(raw_ids))
            if ids != raw_ids:
                record['normalizations'].append({'partId': key, 'block': index,
                                                'inputCount': len(raw_ids), 'sourceIds': ids})
            # Match the existing presentation-only paragraph normalization.
            text = text.replace('\\n\\n', '\n\n')
            if parts:
                offset += 2
            spans.append({'start': offset, 'end': offset + len(text),
                          'textSha256': digest(text), 'sourceIds': ids})
            parts.append(text)
            offset += len(text)
            chosen.update(ids)
            for v in ids:
                value = digest(allowed[v])
                require(v not in record['sourceSha256'] or record['sourceSha256'][v] == value)
                record['sourceSha256'][v] = value
        require(set(supports) <= chosen)
        row.pop('blocks')
        row['text'] = '\n\n'.join(parts)
        record['parts'][key] = {'textSha256': digest(row['text']), 'blocks': spans}
    return result, record


def attribution_status(record, draft, evidence_by_part):
    """Never rebind old provenance to prose changed by a later correction."""
    sources = {v['id']: v['text'] for values in evidence_by_part.values() for v in values}
    source_match = all(v in sources and digest(sources[v]) == value
                       for v, value in record['sourceSha256'].items())
    return {key: ('generated_text_unchanged' if source_match
                  and digest(draft.get('answers', {}).get(key, {}).get('text', '')) == row['textSha256']
                  else 'source_or_prose_changed') for key, row in record['parts'].items()}
