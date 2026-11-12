"""Common budgets and explicit baseline operations over the real graph state."""
from .common import text_length, unique
from .observability import snapshot
from .profiles import execution_profile


def cap_candidate_pool(pool, limit):
    """Preserve one real leader per branch, then ranked candidates, within K."""
    from .evidence_policy import protected_candidates
    leaders = protected_candidates(pool, 1)
    ids = unique(leaders + [h['chunk']['id'] for h in pool['hits']])[:limit]
    keep = set(ids)
    return {**pool, 'hits': [h for h in pool['hits'] if h['chunk']['id'] in keep],
            'candidates': [c for c in pool['candidates'] if c['id'] in keep]}


def budget_profile_candidates(state, context):
    policy = execution_profile(state, context.env)
    pool = cap_candidate_pool(state['retrieval'], policy.candidates)
    snapshot(context, 'budgeted', pool=pool)
    context.observe({'stage': 'research-budget', 'result': {
        'profile': policy.name, 'broad': len(state['retrieval']['hits']),
        'retained': len(pool['hits']), 'candidateLimit': policy.candidates}})
    # Both conditions see at most K candidates. Adaptive uses the existing
    # full reranker once; it no longer adds a separate screening LLM call.
    return {'screenedPool': pool, 'needsScreening': False, 'fusedSelection': False,
            'contextFirst': False}


def rank_only_selection(state, context):
    policy = execution_profile(state, context.env)
    pool = state['screenedPool']
    ids = [h['chunk']['id'] for h in pool['hits']][:policy.selected_chunks]
    plan = {'route': 'bible' if ids else 'ambiguous', 'chunkIds': ids,
            'rankedIds': [h['chunk']['id'] for h in pool['hits']],
            'focus': {h['chunk']['id']: [v['verse'] for v in h['chunk']['verses']]
                      for h in pool['hits'] if h['chunk']['id'] in ids},
            'coverage': [{'partId': p['partId'], 'chunkIds': list(ids)}
                         for p in state['questionParts']],
            'missingParts': [] if ids else list(state['questionParts']),
            'studyTerms': list(state.get('studyTerms', []))}
    snapshot(context, 'rerank_input', pool=pool)
    snapshot(context, 'rerank_selected', pool=pool, ids=ids)
    context.observe({'stage': 'research-selection', 'result': {
        'method': 'rank-only', 'candidateCount': len(pool['hits']),
        'selected': len(ids), 'semanticSelectionEvaluated': False}})
    return {'plan': plan, 'needsRecovery': False}


def bounded_evidence(state, context):
    """Retain whole, unchanged verses; protect actual anchors and prior supports.

    Budget exclusions are visible, never reclassified as supported. A quoted
    span being present establishes identity, not its semantic entailment.
    """
    policy = execution_profile(state, context.env)
    rows = state.get('evidence', [])
    by_id = {}
    for row in rows:
        if row['id'] in by_id and by_id[row['id']]['text'] != row['text']:
            raise ValueError('EVIDENCE_ID_REBOUND')
        by_id[row['id']] = row
    protected = {v['id'] for v in state['scope']['primary']}
    source_by_id = {}
    for source in state.get('studySources', []):
        if source['id'] in source_by_id and source_by_id[source['id']] != source:
            raise ValueError('SOURCE_ID_REBOUND')
        source_by_id[source['id']] = source
    protected_sources = set()
    for check in state.get('evidenceAssessment', []):
        if check.get('status') not in {'supported', 'refuted'}:
            continue
        for support in check.get('supports', []):
            key = support['evidenceId']
            material = by_id[key]['text'] if key in by_id else source_by_id.get(key, {}).get('excerpt', '')
            if not support.get('quote') or support['quote'] not in material:
                raise ValueError('PRIOR_SUPPORT_IDENTITY_CHANGED')
            (protected if key in by_id else protected_sources).add(key)
    if not protected <= by_id.keys():
        raise ValueError('REQUEST_ANCHOR_MISSING')
    ordered = ([row for row in by_id.values() if row['id'] in protected] +
               [row for row in by_id.values() if row['id'] not in protected])
    selected = set(); used = 0
    for row in ordered:
        size = text_length(row['text'])
        if len(selected) < policy.evidence_verses and used + size <= policy.evidence_characters:
            selected.add(row['id']); used += size
        elif row['id'] in protected:
            raise ValueError('PROTECTED_EVIDENCE_EXCEEDS_BUDGET')
    kept = [row for row in by_id.values() if row['id'] in selected]
    dropped = [row['id'] for row in by_id.values() if row['id'] not in selected]
    sources = []; source_used = 0; source_dropped = []
    source_order = ([s for s in source_by_id.values() if s['id'] in protected_sources] +
                    [s for s in source_by_id.values() if s['id'] not in protected_sources])
    for source in source_order:
        size = text_length(source.get('excerpt', ''))
        if len(sources) < policy.source_records and source_used + size <= policy.source_characters:
            sources.append(source); source_used += size
        else:
            if source['id'] in protected_sources:
                raise ValueError('PROTECTED_SOURCE_EXCEEDS_BUDGET')
            source_dropped.append(source['id'])
    sections = state.get('sectionContext', [])
    if dropped:
        # Do not describe a partially retained context as complete. Keep all
        # existing coordinates so missing context remains identifiable.
        sections = [{**s, 'complete': False} for s in sections]
    audit = {'verseLimit': policy.evidence_verses, 'characterLimit': policy.evidence_characters,
             'retainedVerses': len(kept), 'characters': used,
             'protectedIds': sorted(protected), 'droppedIds': dropped,
             'sourceCharacters': source_used, 'droppedSourceIds': source_dropped}
    context.observe({'stage': 'research-evidence-budget', 'result': audit})
    snapshot(context, 'profile_evidence', verses=kept, sources=sources)
    primary_ids = {v['id'] for v in state.get('primaryEvidence', rows)}
    return {'evidence': kept, 'primaryEvidence': [v for v in kept if v['id'] in primary_ids],
            'studySources': sources, 'sectionContext': sections,
            'evidenceBudget': audit}
