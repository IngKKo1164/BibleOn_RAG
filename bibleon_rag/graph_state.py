"""Versioned, JSON-only execution state. Capabilities never enter checkpoints."""
from copy import deepcopy
from typing import Literal, NotRequired, TypedDict

from .common import BibleProviderError, compact_previous, json_text, sha256
from .conversation import context_plan, normalize_conversation
from .passages import resolve_bible_request, selection_parts
from .retrieval import BibleRetriever

STATE_VERSION = 3
Json = str | int | float | bool | None | list['Json'] | dict[str, 'Json']
Record = dict[str, Json]


class QuestionPart(TypedDict):
    partId: str
    question: str


class EvidenceNeed(TypedDict):
    needId: str
    partId: str
    kind: str
    need: str
    importance: Literal['essential', 'supplementary']
    searchQuery: str
    evidenceScope: NotRequired[str]
    relation: NotRequired[str]


class IntentPlan(TypedDict, total=False):
    intent: str
    needs: list[EvidenceNeed]
    studyTerms: list[str]
    backgroundQueries: list[str]
    sourceLimitation: str
    sharedContextParts: list[str]


class GraphState(TypedDict, total=False):
    """Single execution state. Default replacement is intentional for sequential nodes.

    Evidence merges/retractions are domain operations, never blind list reducers.
    Objects in retrieval pools are encoded as verified resource identities.
    """
    stateVersion: int
    question: str
    history: list[Record]
    previousEvidence: Record | None
    passage: list[Record]
    evidenceMode: Literal['adaptive', 'passage', 'context']
    intentEnabled: bool
    intentPlan: IntentPlan | None
    routingDecision: Record
    processingProfile: Literal['simple', 'adaptive']
    processingDecision: Record
    clarificationRequest: Record
    answerReadiness: Record
    answerNeedCoverage: Record
    conversation: Record | None
    conversationUpdate: Record | None
    contextPlan: Record | None
    searchQuestion: str
    scope: Record
    studyEnabled: bool
    wirePrevious: Record | None
    questionParts: list[QuestionPart]
    callAttempts: int
    activeMilliseconds: float
    usage: dict[str, int]
    lane: str
    retrieval: Record
    screenedPool: Record
    recoveryPool: Record
    plan: Record
    recoveryCount: int
    needsScreening: bool
    needsRecovery: bool
    evidence: list[Record]
    primaryEvidence: list[Record]
    sectionContext: list[Record]
    selectionCoverage: list[Record]
    studyTerms: list[str]
    studyTermDiagnostics: list[Record]
    studySources: list[Record]
    crossReferences: list[Record]
    studyLoaded: bool
    answerSchema: Record
    partEvidence: dict[str, list[Record]]
    system: str
    languageSupport: Record
    answerInput: Record
    draft: Record
    visibleTokens: int
    referenceEvidenceByPart: dict[str, list[Record]]
    needsCitationRepair: bool
    needsFinalReview: bool
    finalReview: Record
    generationAttribution: Record
    confirmedContext: list[Record]
    conflicts: list[Record]
    result: Record
    evidenceAssessment: list[Record]
    evidenceBudget: Record
    sourceDiagnostics: list[Record]
    fusedSelection: bool
    contextFirst: bool
    requiredScenes: list[Record]
    evidenceAudits: list[Record]
    needScopeAudits: list[Record]
    proofKindAudits: list[Record]
    reviewBindingAudits: list[Record]
    reviewGapAudits: list[Record]
    sourceCoverage: Record
    literalChecks: list[Record]
    selectionBudgetAudits: list[Record]
    planNormalizationAudits: list[Record]
    _lastNode: str


def initial_state(context, *, question, history=None, previous_evidence=None, passage=None,
                  conversation=None, evidence_mode='adaptive') -> GraphState:
    env, corpus = context.env, context.corpus
    if evidence_mode not in {'adaptive', 'passage', 'context'}:
        raise ValueError('INVALID_EVIDENCE_MODE')
    from .profiles import automatic_routing
    if automatic_routing(env) and evidence_mode != 'adaptive':
        raise ValueError('AUTO_ROUTING_REQUIRES_FULL_PIPELINE')
    if env.get('BIBLE_INTENT_ROUTING', 'legacy') not in {'hybrid', 'legacy'}:
        raise ValueError('INVALID_INTENT_ROUTING')
    state = GraphState(stateVersion=STATE_VERSION, question=question, history=deepcopy(history or []),
        previousEvidence=deepcopy(previous_evidence), passage=deepcopy(passage or []), evidenceMode=evidence_mode,
        intentEnabled=evidence_mode == 'adaptive' and env.get('BIBLE_INTENT_ROUTING') == 'hybrid',
        intentPlan=None, conversationUpdate=None, callAttempts=0, activeMilliseconds=0,
        usage={'inputTokens': 0, 'outputTokens': 0, 'reasoningTokens': 0, 'answerTokens': 0, 'totalTokens': 0, 'calls': 0})
    if env.get('BIBLE_SOURCE_SCOPE') == 'qualified':
        from .config import source_coverage_contract
        state['sourceCoverage'] = source_coverage_contract(env)
    from .evidence_policy import policy_stage, reliability
    from .latency import latency_mode
    policy_stage(env)
    reliability(state, context, 1)
    latency_mode(state, context)
    state['conversation'] = normalize_conversation(conversation, corpus) if conversation is not None else None
    state['contextPlan'] = context_plan(question, state['conversation']) if conversation is not None else None
    state['searchQuestion'] = state['contextPlan']['query'] if state['contextPlan'] else question
    if conversation is not None:
        if history or previous_evidence:
            raise ValueError('CONVERSATION_HISTORY_CONFLICT')
        context.observe({'stage': 'conversation', 'event': 'input', 'mode': state['contextPlan']['mode'],
            'usesContext': state['contextPlan']['usesContext'], 'state': state['conversation'],
            'messages': len(state['conversation']['recentTurns']),
            'characters': sum(len(r['content']) for r in state['conversation']['recentTurns'])})
    scope = resolve_bible_request(corpus, question, state['passage'])
    if evidence_mode != 'adaptive' and not scope['direct']:
        raise ValueError('COMPARISON_REQUIRES_EXPLICIT_PASSAGE')
    if scope['direct'] and (not scope['lookup'] or automatic_routing(env)):
        if state['intentEnabled'] or evidence_mode == 'context':
            from .interpretation import passage_context
            scope.update(passage_context(corpus, scope['primary']))
        elif evidence_mode == 'passage':
            scope['context'] = []
    if evidence_mode != 'adaptive':
        scope.update(sources=False, crossrefs=False)
    context.observe({'stage': 'route', 'result': {'route': scope['route'], 'primaryVerses': len(scope['primary']),
                     'contextVerses': len(scope['context'])}})
    if scope['invalid'] and not state['intentEnabled']:
        raise BibleProviderError('AI_REFERENCE_INVALID', 400)
    state.update(scope=scope, studyEnabled=env.get('BIBLE_RAG_STUDY') == 'one-hop' and (scope['sources'] or scope['crossrefs']),
                 wirePrevious=compact_previous(previous_evidence), questionParts=selection_parts(question))
    context.observe({'stage': 'evidence-mode', 'mode': evidence_mode})
    return state


def index_identity(index):
    if not hasattr(index, "checkpoint_identity"):
        index.checkpoint_identity = {
            "dimensions": index.dimensions,
            "chunks": sha256(json_text(index.chunks)),
            "vectors": sha256(index.vectors.tobytes()) if index.vectors is not None else None,
        }
    return dict(index.checkpoint_identity)


def encode(value):
    if isinstance(value, BibleRetriever):
        return {"__bibleon_index__": index_identity(value)}
    if isinstance(value, dict):
        if any(not isinstance(k, str) for k in value):
            raise ValueError("NON_STRING_STATE_KEY")
        return {k: encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    if value is None or type(value) in (str, int, float, bool):
        return value
    raise ValueError("NON_SERIALIZABLE_GRAPH_STATE")


def decode(value, context):
    if isinstance(value, dict):
        if set(value) == {"__bibleon_index__"}:
            index = context.get_retriever()
            if index is None or index_identity(index) != value["__bibleon_index__"]:
                raise ValueError("CHECKPOINT_INDEX_CHANGED")
            return index
        result = {k: decode(v, context) for k, v in value.items()}
        if 'stateVersion' in result:
            for field in ('screenedPool', 'recoveryPool'):
                if field in result:
                    result[field] = expand_pool(result[field], result.get('retrieval'))
        return result
    if isinstance(value, list):
        return [decode(v, context) for v in value]
    return value


def expand_pool(pool, retrieval):
    """Resolve an ordered view of canonical candidates without duplicating text."""
    if not isinstance(pool, dict) or set(pool) != {'__bibleon_pool__'}:
        return pool
    view = pool['__bibleon_pool__']
    if not retrieval or set(view) != {'hitIds', 'candidateIds'}:
        raise ValueError('INVALID_GRAPH_POOL_VIEW')
    hits = {h['chunk']['id']: h for h in retrieval['hits']}
    candidates = {c['id']: c for c in retrieval['candidates']}
    if (any(not isinstance(ids, list) or any(not isinstance(i, str) for i in ids)
            or len(set(ids)) != len(ids) for ids in view.values())
            or set(view['hitIds']) != set(view['candidateIds'])
            or not set(view['hitIds']) <= hits.keys()):
        raise ValueError('INVALID_GRAPH_POOL_VIEW')
    return {**retrieval, 'hits': [hits[k] for k in view['hitIds']],
            'candidates': [candidates[k] for k in view['candidateIds']]}


def encode_state(working):
    # Encode the canonical retrieval only once. Pools are views when every
    # selected record and all metadata are unchanged; otherwise retain records.
    # The exact order of hits AND candidates is preserved independently.
    values = dict(working)
    retrieval = working.get('retrieval')
    if retrieval:
        for field in ('screenedPool', 'recoveryPool'):
            pool = working.get(field)
            if not pool or '__bibleon_pool__' in pool:
                continue
            view = {'__bibleon_pool__': {'hitIds': [h['chunk']['id'] for h in pool['hits']],
                    'candidateIds': [c['id'] for c in pool['candidates']]}}
            try:
                if expand_pool(view, retrieval) == pool:
                    values[field] = view
            except (ValueError, KeyError):
                pass  # An older screened pool can contain pre-recovery records.
    return encode(values)


def validate_state(state):
    if state.get('stateVersion') != STATE_VERSION:
        raise ValueError("CHECKPOINT_STATE_VERSION")
    if set(state) - GraphState.__annotations__.keys():
        raise ValueError('UNKNOWN_GRAPH_STATE_FIELD')
    required = {'question', 'scope', 'intentEnabled', 'intentPlan', 'studyEnabled', 'questionParts',
                'conversation', 'contextPlan', 'conversationUpdate', 'usage', 'callAttempts'}
    if required - state.keys() or not isinstance(state['question'], str):
        raise ValueError('INCOMPLETE_GRAPH_STATE')
    if type(state['callAttempts']) is not int or state['callAttempts'] < 0:
        raise ValueError('INVALID_GRAPH_BUDGET')
    if not isinstance(state['usage'], dict) or any(type(v) is not int or v < 0 for v in state['usage'].values()):
        raise ValueError('INVALID_GRAPH_USAGE')
    if 'processingProfile' in state and state['processingProfile'] not in {'simple', 'adaptive'}:
        raise ValueError('INVALID_EXECUTION_PROFILE')
    if 'processingDecision' in state:
        from .routing import PROCESSING_VERSION, ADAPTIVE_MINIMUM_NEEDS
        decision = state['processingDecision']
        if (not isinstance(decision, dict) or decision.get('version') != PROCESSING_VERSION
                or type(decision.get('essentialNeedCount')) is not int
                or not 1 <= decision['essentialNeedCount'] <= 6
                or decision.get('threshold') != ADAPTIVE_MINIMUM_NEEDS
                or decision.get('profile') != state.get('processingProfile')
                or decision['profile'] != ('adaptive' if decision['essentialNeedCount'] >= ADAPTIVE_MINIMUM_NEEDS else 'simple')):
            raise ValueError('INVALID_PROCESSING_DECISION')
    if 'routingDecision' in state:
        from .routing import ROUTES, VERSION
        decision = state['routingDecision']
        if (not isinstance(decision, dict) or decision.get('version') != VERSION
                or decision.get('route') not in ROUTES):
            raise ValueError('INVALID_ROUTING_DECISION')
    for field in ('intentEnabled', 'studyEnabled', 'needsRecovery', 'needsScreening', 'needsCitationRepair', 'contextFirst'):
        if field in state and type(state[field]) is not bool:
            raise ValueError('INVALID_GRAPH_DECISION')
    # Fail on duplicates instead of silently changing ranking or evidence limits.
    for field in ('evidence', 'primaryEvidence', 'studySources'):
        rows = state.get(field, [])
        if len({r['id'] for r in rows}) != len(rows):
            raise ValueError('DUPLICATE_GRAPH_EVIDENCE')
    for field in ('retrieval', 'screenedPool', 'recoveryPool'):
        pool = expand_pool(state.get(field), state.get('retrieval'))
        if pool:
            hits = [h['chunk']['id'] for h in pool['hits']]
            candidates = [c['id'] for c in pool['candidates']]
            if len(set(hits)) != len(hits) or len(set(candidates)) != len(candidates) or set(hits) != set(candidates):
                raise ValueError('INVALID_GRAPH_CANDIDATES')


def commit_node(before, working, update, context, name):
    """Commit one isolated node transaction through LangGraph channels only."""
    working.update(update)
    context.ledger.reconcile(working)
    context.ledger.record(working)
    working.update(_lastNode=name, activeMilliseconds=context.active_milliseconds())
    encoded = encode_state(working)
    validate_state(encoded)
    if 'routingDecision' in before and encoded.get('routingDecision') != before['routingDecision']:
        raise ValueError('ROUTING_DECISION_CHANGED')
    if 'processingDecision' in before and (encoded.get('processingDecision') != before['processingDecision']
            or encoded.get('processingProfile') != before.get('processingProfile')):
        raise ValueError('PROCESSING_DECISION_CHANGED')
    for field, identity in (('evidence', 'text'), ('primaryEvidence', 'text'), ('studySources', 'excerpt')):
        old = {r['id']: r for r in before.get(field, [])}
        for row in encoded.get(field, []):
            prior = old.get(row['id'])
            if prior is not None and (prior.get(identity) != row.get(identity)
                    or field == 'studySources' and prior.get('recordId') != row.get('recordId')):
                raise ValueError('GRAPH_EVIDENCE_ID_REBOUND')
    return {key: value for key, value in encoded.items() if key not in before or before[key] != value}


def completed_result(state, context):
    """Conversation metadata is advanced only after a terminal validated result."""
    from .conversation import advance_conversation
    result = deepcopy(state['result'])
    if state.get('conversation') is not None:
        result['conversationState'] = advance_conversation(state['conversation'], state['question'], result,
            state['contextPlan'], context.corpus, state.get('conversationUpdate'))
    return result
