"""Explicit re-audit of previous evidence; retention is never automatic truth."""
from copy import deepcopy
import re

from .common import object_schema

PROOF_KINDS = ['general', 'text_scope', 'promise', 'event', 'sequence', 'comparison']
BASES = ['text_scope', 'observed_event', 'ordered_events', 'comparison',
         'promise_prediction', 'general', 'unavailable']
PROOF_PROMPT = """proofKind는 근거가 충족해야 할 형태다: text_scope=본문이 직접 말하는 범위와 추론의 한계, promise=이전에 어떤 약속·예고를 했는지 그 존재와 내용, event=실제로 이루어진 특정 행동(사명을 맡기거나 대답한 실제 발화도 포함), sequence=둘 이상의 사건과 선후, comparison=비교 양쪽, general=그 외. 약속이 있었는지를 묻는 need는 promise로, 약속이 실제 이루어졌는지는 event로 구분한다. 질문이 사건 이후의 회복·변화·결과를 묻는다면 sequence로 지정한다. 예고나 사전 약속은 이후 실제 행동의 증거가 아니다. 본문으로 추측의 한계를 설명하는 질문은 text_scope이며, 추측을 입증할 자료가 없다는 이유로 설명 자체를 불가능하게 만들지 않는다."""
AUDIT_PROMPT = """priorAudits는 이전에 원문 대조를 통과한 항목의 재검사다. 기존 판정은 정답이 아니며 보완된 문맥과 함께 다시 읽는다. 같은 need·동일 원문이 여전히 유효하면 retain/unchanged와 witnesses=[]로 유지한다. 다른 미확인 배경이 남았다는 이유만으로 철회하지 않는다. 기존 오독, 새 반증, 범위 불일치가 드러나면 retract와 해당 reason을 선택하고 그 판단의 실제 원문 witnesses를 적는다. 철회한 항목을 자동으로 다른 사실로 바꾸지 않는다. checks에서는 아직 해결되지 않은 need만 판정한다."""


def retained_checks(previous, needs, materials):
    """Only unchanged requirements with still present literal spans can be audited."""
    current = {n['needId']: n for n in needs}
    return {c['needId']: c for c in previous
            if c.get('status') in {'supported', 'refuted'} and c.get('supports')
            and c['needId'] in current
            and all(c.get(k) == current[c['needId']].get(k)
                    for k in ('partId', 'kind', 'need', 'importance', 'proofKind'))
            and all(s.get('quote') and s['quote'] in materials.get(s['evidenceId'], '')
                    for s in c['supports'])}


def delta_schema(schema, prior, materials):
    """Separate existing support audits from outstanding evidence obligations."""
    result = deepcopy(schema)
    checks = result['properties']['checks']['properties']
    for key in prior:
        checks.pop(key)
    result['properties']['checks']['required'] = list(checks)
    result['properties']['priorAudits'] = object_schema({key: object_schema({
        'decision': {'type': 'string', 'enum': ['retain', 'retract']},
        'reason': {'type': 'string', 'enum': ['unchanged', 'prior_misread', 'new_counterevidence', 'scope_mismatch']},
        'witnesses': {'type': 'array', 'maxItems': 3, 'items': object_schema({
            'evidenceId': {'type': 'string', 'enum': list(materials)},
            'quote': {'type': 'string', 'minLength': 4, 'maxLength': 500}})}
    }) for key in prior})
    result['required'].append('priorAudits')
    return result


def decode_delta(raw, prior, materials):
    """Reject silent resets; permit explicit, source-bound retraction of mistakes."""
    audits = raw.get('priorAudits')
    if not isinstance(audits, dict) or set(audits) != set(prior):
        raise ValueError('REVIEW_AUDIT_INVALID')
    checks = dict(raw.get('checks', {}))
    if set(checks) & set(prior):
        raise ValueError('REVIEW_AUDIT_DUPLICATE')
    for key, old in prior.items():
        audit = audits[key]
        witnesses = audit.get('witnesses')
        if not isinstance(witnesses, list) or len(witnesses) > 3:
            raise ValueError('REVIEW_AUDIT_INVALID')
        if audit.get('decision') == 'retain' and audit.get('reason') == 'unchanged' and not witnesses:
            checks[key] = {k: old[k] for k in ('status', 'supports')}
            for field in ('basis', 'actorMatch', 'decisiveEvidenceId', 'finding'):
                if field in old:
                    checks[key][field] = old[field]
        elif (audit.get('decision') == 'retract' and audit.get('reason') in
              {'prior_misread', 'new_counterevidence', 'scope_mismatch'} and witnesses):
            for s in witnesses:
                if (not isinstance(s, dict) or not isinstance(s.get('quote'), str)
                    or not 4 <= len(s['quote']) <= 500
                    or s['quote'] not in materials.get(s.get('evidenceId'), '')):
                    raise ValueError('REVIEW_AUDIT_WITNESS_INVALID')
            checks[key] = {'status': 'conflict', 'supports': [], 'basis': 'unavailable'}
        else:
            raise ValueError('REVIEW_AUDIT_INVALID')
    return {**raw, 'checks': checks}


def proof_schema(schema, *, scene_needs=(), evidence_ids=()):
    result = deepcopy(schema)
    if 'questionGaps' in result['properties']:
        gap = result['properties']['questionGaps']['items']
        gap['properties']['proofKind'] = {'type': 'string', 'enum': PROOF_KINDS}
        gap['required'].append('proofKind')
    critical = {n['needId'] for n in scene_needs if n.get('proofKind') in {'event', 'sequence'}}
    for key, item in result['properties']['checks']['properties'].items():
        item['properties']['basis'] = {'type': 'string', 'enum': BASES}
        item['required'].append('basis')
        if key in critical:
            item['properties']['actorMatch'] = {'type': 'string', 'enum': ['same_subject', 'group_only', 'unknown']}
            item['properties']['decisiveEvidenceId'] = {'type': 'string', 'enum': ['', *evidence_ids]}
            item['required'] += ['actorMatch', 'decisiveEvidenceId']
    return result


def finding_schema(schema, *, keyed):
    result = deepcopy(schema)
    rows = (result['properties']['checks']['properties'].values() if keyed else
            [result['properties']['checks']['items']])
    for item in rows:
        item['properties']['finding'] = {'type': 'string', 'maxLength': 120}
        item['required'].append('finding')
    return result


def bind_findings(checks, raw):
    """Carry public source-bound conclusions, not hidden reasoning or approval."""
    rows = {c['needId']: c for c in raw['checks']}
    result = []
    for check in checks:
        text = rows[check['needId']].get('finding', '')
        if not isinstance(text, str) or len(text) > 120:
            raise ValueError('EVIDENCE_REVIEW_INVALID')
        result.append({**check, 'finding': text.strip()
            if check['status'] in {'supported', 'refuted'} and check['supports'] else ''})
    return result


def validate_proofs(checks, raw, *, scenes=False, scope_witnesses=False, bound_events=False, scripture_ids=()):
    """A promise cannot fulfill an actual-event obligation, even with valid quotes."""
    rows = {c['needId']: c for c in raw['checks']}
    allowed = {'event': {'observed_event', 'ordered_events'}, 'sequence': {'ordered_events'},
               'promise': {'promise_prediction'}, 'comparison': {'comparison'}, 'text_scope': {'text_scope'}}
    if scope_witnesses:
        # A direct narrated event or promise is also evidence of what a passage
        # explicitly says. This never relaxes event/sequence/comparison demands.
        allowed['text_scope']={'text_scope','observed_event','ordered_events','comparison','promise_prediction','general'}
    result = []
    for c in checks:
        basis = rows[c['needId']].get('basis')
        if basis not in BASES:
            raise ValueError('REVIEW_PROOF_INVALID')
        c = {**c, 'basis': basis}
        row=rows[c['needId']]
        bound_event=(bound_events and c.get('proofKind')=='event' and basis in {'general','text_scope'}
                     and row.get('actorMatch')=='same_subject'
                     and row.get('decisiveEvidenceId') in scripture_ids
                     and row.get('decisiveEvidenceId') in {s['evidenceId'] for s in c['supports']})
        if c['status'] in {'supported', 'refuted'} and (
                basis == 'unavailable' or c.get('proofKind') in allowed and basis not in allowed[c['proofKind']] and not bound_event):
            c.update(status='missing', supports=[], invalidReason='REVIEW_EVENT_PROOF_MISSING')
        if scenes and c.get('proofKind') in {'event', 'sequence'}:
            row = rows[c['needId']]
            c.update(actorMatch=row.get('actorMatch', 'unknown'), decisiveEvidenceId=row.get('decisiveEvidenceId', ''))
            if c['status'] in {'supported', 'refuted'} and (c['actorMatch'] != 'same_subject'
                    or c['decisiveEvidenceId'] not in {s['evidenceId'] for s in c['supports']}):
                c.update(status='missing', supports=[], invalidReason='REVIEW_SCENE_SUBJECT_MISSING')
        result.append(c)
    return result


def limit_literal_supports(checks, observations):
    """A whole-corpus literal scan cannot certify events or semantic absence."""
    by_id = {row['id']: row for row in observations}
    result = []
    for check in checks:
        supports = []
        for support in check['supports']:
            observed = by_id.get(support['evidenceId'])
            request = check['need'].replace(observed['span'], '') if observed else check['need']
            source_request = re.search(r'기록|수록|출처|적혀|직접.{0,8}인용|(?:성경|본문).{0,16}(?:여부|나오|나와|있)', request)
            semantic_request = re.search(r'같은\s*뜻|의미|가르[치쳐침]|의역', request)
            if observed is None or (check.get('proofKind') == 'text_scope'
                    and check.get('basis') == 'text_scope'
                    and (observed['span'] in check['need'] or observed['span'] in check.get('searchQuery', ''))
                    and source_request and not semantic_request):
                if observed is not None:
                    from .common import json_text
                    material = json_text(observed)
                    # Bind the selected real observation, including its scope
                    # and zero count, instead of retaining only the echoed user
                    # phrase. This is an actual contiguous supplied JSON span.
                    quote = material[material.index('"span":'):material.index(',"checkedVerses":')]
                    supports.append({**support, 'quote': quote})
                else:
                    supports.append(support)
        row = {**check, 'supports': supports}
        if check['status'] in {'supported', 'refuted'} and not supports:
            row.update(status='missing', invalidReason='REVIEW_LITERAL_SCOPE_MISSING')
        result.append(row)
    return result


def join_decisive_supports(state,checks,raw,evidence):
    """Join two explicit model selections through supplied canonical records.

    The review's declared anchor is not a hidden/retrieved-but-unread verse.
    Validation of status, quotes, actor and temporal proof remains separate.
    """
    verses={v['id']:v['text'] for v in evidence}
    rows={r['needId']:r for r in raw['checks']};result=[];audits=[]
    for c in checks:
        row=rows[c['needId']];key=row.get('decisiveEvidenceId')
        if (c.get('proofKind') in {'event','sequence'} and c['status'] in {'supported','refuted'}
                and row.get('actorMatch')=='same_subject' and key in verses
                and key not in {s['evidenceId'] for s in c['supports']}):
            original=[s['evidenceId'] for s in c['supports']]
            # Keep the explicit event plus up to two already validated context
            # spans. All original material remains in the actual review input.
            c={**c,'supports':[{'evidenceId':key,'quote':verses[key]},*c['supports'][:2]]}
            audits.append({'needId':c['needId'],'action':'bind_declared_anchor',
                           'anchorId':key,'previousSupportIds':original,
                           'recoveryCount':state.get('recoveryCount',0)})
        result.append(c)
    state['reviewBindingAudits']=state.get('reviewBindingAudits',[])+audits
    return result


def proof_kind_audit_schema(needs):
    """Temporal/comparison obligations cannot be weakened by type correction."""
    flexible={'general','text_scope','promise'}
    return object_schema({n['needId']:{'type':'string','enum':sorted(flexible) if n['proofKind'] in flexible else [n['proofKind']]}
                          for n in needs})


def audit_proof_kinds(state,checks,raw):
    proposed=raw.get('reviewProofKinds')
    if not isinstance(proposed,dict) or set(proposed)!={c['needId'] for c in checks}:
        raise ValueError('INVALID_PROOF_KIND_AUDIT')
    result=[];audits=[];flexible={'general','text_scope','promise'}
    for c in checks:
        old,new=c['proofKind'],proposed[c['needId']]
        if new not in PROOF_KINDS or new!=old and (old not in flexible or new not in flexible):
            raise ValueError('INVALID_PROOF_KIND_AUDIT')
        accepted=(new==old or c['status'] in {'supported','refuted'} and bool(c['supports']))
        selected=new if accepted else old
        row={**c,'proofKind':selected}
        if new!=old:
            row['originalProofKind']=c.get('originalProofKind',old)
            audits.append({'needId':c['needId'],'from':old,'proposed':new,'accepted':accepted,
                           'supportIds':[s['evidenceId'] for s in c['supports']],
                           'recoveryCount':state.get('recoveryCount',0)})
        result.append(row)
    state['proofKindAudits']=state.get('proofKindAudits',[])+audits
    by_id={c['needId']:c for c in result}
    state['intentPlan']={**state['intentPlan'],'needs':[
        {**n,'proofKind':by_id[n['needId']]['proofKind'],
         **({'originalProofKind':by_id[n['needId']]['originalProofKind']} if 'originalProofKind' in by_id[n['needId']] else {})}
        for n in state['intentPlan']['needs']]}
    return result


def scene_requirements(checks, evidence_by_part):
    """Bind final explanation to the actual event anchor identified during review."""
    return [{k: c[k] for k in ('needId', 'partId', 'need', 'decisiveEvidenceId')}
            for c in checks if c.get('proofKind') in {'event', 'sequence'}
            and c.get('importance') == 'essential' and c.get('status') in {'supported', 'refuted'}
            and c.get('decisiveEvidenceId') in {v['id'] for v in evidence_by_part.get(c['partId'], [])}]


SCENE_ANSWER_PROMPT = ('\nrequiredScenes의 각 실제 장면을 text 안에서 구체적인 행동으로 설명하고 지정 decisiveEvidenceId의 장절을 같은 문단에 표기한다. '
    '단순히 다시 사명을 받았다/회복했다는 결론만 쓰지 말고 무엇을 말하거나 행했는지 쓴다. '
    '각 지정 장절은 citations에도 포함한다. 사건 순서는 실제 원문의 흐름을 보존하고, 부가적인 그 뒤/그 전에 같은 표현은 원문으로 확인될 때만 쓴다. '
    '별도 설명을 뒤에 덧붙여 늘리지 말고 기존 답변 안에 자연스럽게 포함하며 같은 설명을 반복하지 않는다.')


def _visible_clause(text, corpus):
    """Normalize presentation only; preserve words, subjects and all negations."""
    from .passages import parse_bible_references
    def citation(match):
        parsed = parse_bible_references(corpus, match[0], max_verses=96, max_range_verses=96)
        return '' if parsed['refs'] and not parsed['invalid'] else match[0]
    text = re.sub(r'\([^()\n]*\)|（[^（）\n]*）', citation, text)
    return re.sub(r'[\s*`_.,，。!?！？:;；]+', '', text)


def invalid_scene_parts(draft, requirements, corpus):
    """Check visible anchor coverage, not a model's duplicate self-attestation.

    Actor/event semantics are reviewed upstream and remain a human quality task.
    An inline reference alone cannot prove that its accompanying explanation is true.
    """
    from .passages import parse_bible_references
    citations = {f"{c.get('bookId')}:{c.get('chapter')}:{c.get('verse')}" for c in draft.get('citations', []) if isinstance(c, dict)}
    invalid = set()
    for r in requirements:
        text = draft.get('answers', {}).get(r['partId'], {}).get('text', '')
        # Inspect the actual rendered answer, not a second generated copy of it.
        # Standalone references/headings and hidden metadata are insufficient.
        prose_blocks = ['\n'.join(line for line in paragraph.splitlines()
                                 if not re.match(r'^\s*#{1,6}(?:\s|$)',line))
                        for paragraph in re.split(r'\n\s*\n', text)]
        bound = any(len(_visible_clause(paragraph, corpus)) >= 16
                    and r['decisiveEvidenceId'] in {v['id'] for v in parse_bible_references(
                        corpus, paragraph, max_verses=96, max_range_verses=96)['refs']}
                    for paragraph in prose_blocks)
        if not bound or r['decisiveEvidenceId'] not in citations:
            invalid.add(r['partId'])
    return invalid


def bind_visible_scene_citations(draft, requirements, corpus):
    """Bind metadata to existing visible, reviewed anchors; never invent prose.

    Preserve one supporting citation per part and all already-listed required
    scenes. If these cannot fit the existing eight-citation cap, leave the draft
    unchanged for the normal bounded correction/failure path.
    """
    from .passages import verse_at
    citations = draft.get('citations')
    if not isinstance(citations,list) or len(citations)>8:return
    try:
        rows = {verse_at(corpus,c)['id']:c for c in citations}
    except (ValueError,KeyError,TypeError):return
    mandatory=[]
    for r in requirements:
        try:
            book,chapter,verse=r['decisiveEvidenceId'].split(':')
            ref={'bookId':book,'chapter':int(chapter),'verse':int(verse)}
            verse_at(corpus,ref)
        except (ValueError,KeyError,TypeError):continue
        probe={**draft,'citations':[*citations,ref]}
        if not invalid_scene_parts(probe,[r],corpus):
            rows.setdefault(r['decisiveEvidenceId'],ref)
            mandatory.append(r['decisiveEvidenceId'])
    required=list(dict.fromkeys(mandatory))
    for answer in draft.get('answers',{}).values():
        ids=[s['verseId'] for s in answer.get('supports',[]) if s.get('verseId') in rows]
        if ids and not set(ids).intersection(required):required.append(ids[0])
    if len(required)>8:return
    keys=list(dict.fromkeys(required+list(rows)))[:8]
    draft['citations']=[rows[k] for k in keys]
