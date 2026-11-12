"""Bounded latency experiments; no answer truncation or skipped evidence review."""
from .common import BibleProviderError, json_text, object_schema


def latency_mode(state, context):
    mode = context.env.get('BIBLE_LATENCY_MODE', 'baseline')
    if mode not in {'baseline', 'compact', 'fused', 'selection', 'strict', 'conditional'}:
        raise ValueError('INVALID_LATENCY_MODE')
    return mode if state["intentEnabled"] else 'baseline'


def conditional_fusion(state, context):
    """Only simple textual explanation; temporal/background/ambiguous work stays split."""
    from .evidence_policy import reliability
    plan = state["intentPlan"] or {}
    needs = plan.get('needs', [])
    return (reliability(state, context, 5) and len(state["questionParts"]) == 1
            and plan.get('intent') in {'interpretation', 'search'} and 1 <= len(needs) <= 2
            and all(n['kind'] in {'context', 'scripture'} and n.get('proofKind') in {'general', 'text_scope'}
                    for n in needs))


def selection_schema(schema, visible_verses):
    """Encode the same allowed chunk/verse pairs once instead of N object schemas."""
    properties = {k: v for k, v in schema['properties'].items() if k != 'focus'}
    properties['focusRefs'] = {'type': 'array', 'maxItems': 24, 'items': {'type': 'string',
        'enum': [f'{key}:{verse}' for key, verses in visible_verses.items() for verse in verses]}}
    return {**schema, 'properties': properties,
            'required': [k if k != 'focus' else 'focusRefs' for k in schema['required']]}


def compact_coordinate_schema(schema,visible_verses):
    """Avoid repeating 128 object schemas; domain validation still binds each pair.

    The provider can propose a globally visible verse on the wrong candidate.
    Existing validate_reranking must reject it; never repair/invent that pair.
    Array/chunk/coordinate limits and the existing output shape stay unchanged.
    """
    from copy import deepcopy
    result=deepcopy(schema)
    result['properties']['focus']['items']=object_schema({
        'chunkId':{'$ref':'#/$defs/candidateId'},
        'verses':{'type':'array','minItems':1,'maxItems':4,
                  'items':{'type':'integer','enum':sorted({v for vs in visible_verses.values() for v in vs})}}})
    return result


def decode_selection_focus(raw, visible_verses):
    """Restore existing focus structure; reject unseen coordinates or excess focus."""
    def invalid():
        error = BibleProviderError('AI_SELECTION_INVALID')
        error.diagnostic = 'SELECTION_FOCUS_COORDINATES'
        raise error
    refs = raw.get('focusRefs')
    if not isinstance(refs, list) or len(refs) > 24 or 'focus' in raw:
        invalid()
    allowed = {f'{key}:{verse}': (key, verse) for key, verses in visible_verses.items() for verse in verses}
    grouped = {}
    for ref in refs:
        if not isinstance(ref, str) or ref not in allowed:
            invalid()
        key, verse = allowed[ref]
        group = grouped.setdefault(key, [])
        if verse not in group:
            group.append(verse)
        if len(group) > 4 or len(grouped) > 6:
            invalid()
    return {**{k: v for k, v in raw.items() if k != 'focusRefs'},
            'focus': [{'chunkId': key, 'verses': verses} for key, verses in grouped.items()]}


def bound_selection_schema(schema, visible_verses, parts):
    """One selected object binds candidate, visible coordinates and coverage.

    Avoid independent ranked/coverage/focus lists that can contradict each
    other. Retain the same six selected candidates and four verses per item.
    Domain validation still enforces at most three candidates per question.
    """
    from copy import deepcopy
    result = deepcopy(schema)
    for key in ('rankedIds', 'evidenceByPart', 'focus'):
        result['properties'].pop(key)
        result['required'].remove(key)
    choices = []
    for key, verses in visible_verses.items():
        numbers = sorted(set(verses))
        if not numbers or any(type(v) is not int or v < 1 for v in numbers):
            raise ValueError('FOCUS_SCHEMA_VERSES_INVALID')
        number = {'type': 'integer', **({'minimum': numbers[0], 'maximum': numbers[-1]}
            if numbers[-1] - numbers[0] + 1 == len(numbers) else {'enum': numbers})}
        choices.append(object_schema({'chunkId': {'type': 'string', 'enum': [key]},
            'verses': {'type': 'array', 'minItems': 1, 'maxItems': 4, 'items': number},
            'partIds': {'type': 'array', 'minItems': 1, 'maxItems': len(parts),
                        'items': {'$ref': '#/$defs/partId'}}}))
    if not choices:
        raise ValueError('EMPTY_SELECTION_PREVIEW')
    result['$defs']['partId'] = {'type': 'string', 'enum': [p['partId'] for p in parts]}
    result['properties']['selections'] = {'type': 'array', 'maxItems': min(6, 3 * len(parts)),
                                         'items': {'anyOf': choices}}
    result['required'].append('selections')
    # Factor only identical coordinate domains. Grouping by minimum/maximum
    # alone would silently admit holes or cross a verse into another passage.
    # Output fields and the canonical decoder are unchanged.
    compact = deepcopy(result)
    if 'selectedPartIds' in compact['$defs']:
        return result  # never overwrite an unrelated caller definition
    groups = {}
    for (key, verses), choice in zip(visible_verses.items(), choices):
        group = groups.setdefault(tuple(sorted(set(verses))), [])
        group.append((key, choice))
    compact['$defs']['selectedPartIds'] = deepcopy(choices[0]['properties']['partIds'])
    grouped_choices = []
    for group in groups.values():
        choice = deepcopy(group[0][1])
        choice['properties']['chunkId']['enum'] = [key for key, _ in group]
        choice['properties']['partIds'] = {'$ref': '#/$defs/selectedPartIds'}
        grouped_choices.append(choice)
    compact['properties']['selections']['items']['anyOf'] = grouped_choices

    def references(value, target):
        if isinstance(value, dict):
            return value.get('$ref') == target or any(references(v, target) for v in value.values())
        return isinstance(value, list) and any(references(v, target) for v in value)

    # The old ranked/focus fields were removed above. Repair schemas can still
    # use candidateId, so do not prune a definition that remains reachable.
    if not references(compact, '#/$defs/candidateId'):
        compact['$defs'].pop('candidateId', None)
    return compact if len(json_text(compact).encode('utf-8')) < len(json_text(result).encode('utf-8')) else result


def decode_bound_selection(raw, visible_verses, parts, *, recover_duplicates=False):
    """Decode declared bindings; optional hybrid recovery keeps first rank/focus.

    Validate even deferred rows. Merge duplicate visible coordinates within the
    existing four-verse cap, recording overflow without approving its meaning.
    Other callers remain strict; unique inputs retain their original result.
    """
    def invalid():
        error = BibleProviderError('AI_SELECTION_INVALID')
        error.diagnostic = 'SELECTION_FOCUS_COORDINATES'
        raise error
    if not isinstance(raw, dict) or any(k in raw for k in ('rankedIds','evidenceByPart','focus','focusRefs')):
        invalid()
    rows = raw.get('selections')
    if not isinstance(rows, list) or len(rows) > 6:
        invalid()
    coverage = {p['partId']: [] for p in parts}
    ranked, focus, omissions = [], [], []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {'chunkId','verses','partIds'}:
            invalid()
        key, verses, labels = row['chunkId'], row['verses'], row['partIds']
        if (not isinstance(key, str) or key not in visible_verses or (key in ranked and not recover_duplicates)
                or not isinstance(verses, list) or not 1 <= len(verses) <= 4
                or any(type(v) is not int or v not in visible_verses[key] for v in verses)
                or len(set(verses)) != len(verses) or not isinstance(labels, list)
                or not labels or any(not isinstance(p, str) or p not in coverage for p in labels)
                or len(set(labels)) != len(labels)):
            invalid()
        if key not in ranked:
            ranked.append(key); focus.append({'chunkId':key, 'verses':verses[:]})
            retained=verses
        else:
            # Match hybrid's existing duplicate recovery: preserve separately
            # valid candidates instead of discarding the entire model result.
            # A union still has at most4 declared visible verses per candidate.
            existing=next(f['verses'] for f in focus if f['chunkId']==key)
            additional=[v for v in verses if v not in existing]
            room=4-len(existing)
            existing.extend(additional[:room])
            retained=[v for v in verses if v in existing]
            deferred=additional[room:]
            if deferred:
                omissions.extend({'partId':label,'chunkId':key,'reason':'focus_limit',
                                  'deferredVerses':deferred[:]} for label in labels)
        for label in labels:
            # Never assign a new need to unrelated first-row verses when all
            # of that duplicate row's own coordinates were deferred.
            if not retained or key in coverage[label]:
                continue
            if len(coverage[label]) < 3:
                coverage[label].append(key)
            else:
                # Array-wide occurrence caps are not expressible in the
                # provider's schema subset. Apply the original ranked top-3
                # budget to valid declared bindings, not factual approval.
                omissions.append({'partId': label, 'chunkId': key})
    if raw.get('route') != 'bible' and rows:
        invalid()
    return {**{k:v for k,v in raw.items() if k != 'selections'},
            'rankedIds': ranked, 'evidenceByPart': coverage, 'focus': focus,
            **({'budgetOmissions': omissions} if omissions else {})}


def review_support_schema(evidence_ids, scripture_ids):
    """Scripture is already a short canonical verse; external sources need a span."""
    scripture = [v for v in evidence_ids if v in scripture_ids]
    sources = [v for v in evidence_ids if v not in scripture_ids]
    choices = []
    if scripture:
        choices.append(object_schema({'evidenceId': {'type': 'string', 'enum': scripture}}))
    if sources:
        choices.append(object_schema({'evidenceId': {'type': 'string', 'enum': sources},
                                      'quote': {'type': 'string', 'minLength': 4, 'maxLength': 500}}))
    if not choices:
        raise ValueError('EMPTY_REVIEW_MATERIALS')
    return choices[0] if len(choices) == 1 else {'anyOf': choices}


def bind_review_supports(raw, evidence):
    """Bind only supplied verse IDs, without granting a semantic supported status."""
    verses = {v['id']: v['text'] for v in evidence}
    result = {**raw, 'checks': []}
    for row in raw['checks']:
        if not isinstance(row, dict) or not isinstance(row.get('supports'), list):
            raise ValueError('EVIDENCE_REVIEW_INVALID')
        supports = []
        for support in row['supports']:
            if not isinstance(support, dict):
                raise ValueError('EVIDENCE_REVIEW_INVALID')
            key = support.get('evidenceId')
            if key in verses:
                if set(support) != {'evidenceId'}:
                    raise ValueError('EVIDENCE_REVIEW_INVALID')
                supports.append({'evidenceId': key, 'quote': verses[key]})
            else:
                supports.append(dict(support))  # Existing validator checks source/span.
        result['checks'].append({**row, 'supports': supports})
    return result


def bind_review_result(raw,evidence,*,keyed=False):
    """Bind new model choices before merging already validated retained checks."""
    if not keyed:return bind_review_supports(raw,evidence)
    checks=raw.get('checks')
    if not isinstance(checks,dict):raise ValueError('REVIEW_STRUCTURE')
    if any(not isinstance(v,dict) for v in checks.values()):raise ValueError('REVIEW_STRUCTURE')
    values=[{**v,'needId':k} for k,v in checks.items()]
    bound=bind_review_supports({**raw,'checks':values},evidence)
    return {**raw,'checks':{c['needId']:{k:v for k,v in c.items() if k!='needId'} for c in bound['checks']}}
