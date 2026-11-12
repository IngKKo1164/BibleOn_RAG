"""Historical option compatibility and shared provider contracts.

Normal CLI settings are resolved by profiles.load_runtime_env. The old option
dictionary remains for frozen direct-call fixtures, not new parameter sweeps.
"""
from pathlib import Path

from dotenv import dotenv_values

from .data import ROOT

RESEARCH_DEFAULTS = {"BIBLE_AI_PROVIDER": "openai", "BIBLE_RAG_MODE": "hybrid", "BIBLE_RAG_SPEED": "integrated",
                     "BIBLE_RAG_STUDY": "one-hop", "BIBLE_RETRIEVAL_VERSION": "v4",
                     "BIBLE_RETRIEVAL_REPAIR": "on", "BIBLE_REPAIR_QUERY_EXPANSION": "on",
                     "BIBLE_INTENT_ROUTING": "hybrid", "BIBLE_EVIDENCE_POLICY_STAGE": "5",
                     "BIBLE_SET_SELECTION": "off", "BIBLE_BACKGROUND_SEARCH": "bilingual",
                     "BIBLE_RELIABILITY_STAGE": "3", "BIBLE_LATENCY_MODE": "baseline",
                     "BIBLE_EMBEDDING_DIMENSIONS": "1024",
                     "BIBLE_CONTEXT_RETRIEVAL": "baseline",
                     "BIBLE_DISCOVERY_SEEDS": "12", "BIBLE_REASONING_MODE": "baseline",
                     "BIBLE_STUDY_DISCOVERY": "baseline", "BIBLE_FOCUS_SCHEMA":"baseline",
                     "BIBLE_SOURCE_ROUTING":"baseline", "BIBLE_EXPLANATION_POLICY":"baseline",
                     "BIBLE_REVIEW_WIRE":"baseline", "BIBLE_NEED_AUDIT":"baseline",
                     "BIBLE_PROSE_POLICY":"baseline", "BIBLE_PROOF_BINDING":"baseline",
                     "BIBLE_ANSWER_COVERAGE":"baseline", "BIBLE_PROCESSING_TIER":"baseline",
                     "BIBLE_CONTEXT_LEADERS":"baseline", "BIBLE_SOURCE_SCOPE":"baseline",
                     "BIBLE_AMBIGUITY_ROUTING":"baseline", "BIBLE_SUPERSCRIPTIONS":"baseline",
                     "BIBLE_REFERENCE_ROOTS":"baseline", "BIBLE_LITERAL_CHECK":"baseline",
                     "BIBLE_REVIEW_FINDINGS":"baseline", "BIBLE_FINAL_REVIEW":"baseline"}
MODEL = "gpt-6-luna"
OPENAI_ENDPOINT = "https://api.openai.com/v1/chat/completions"
STAGES = {
    "intent": {"model": MODEL, "effort": "low", "maxTokens": 1600, "visibleTokens": 1000},
    "evidence_review": {"model": MODEL, "effort": "low", "maxTokens": 2200, "visibleTokens": 1600},
    "screen": {"model": MODEL, "effort": "low", "maxTokens": 1800, "visibleTokens": 1000},
    "locate": {"model": MODEL, "effort": "low", "maxTokens": 1500, "visibleTokens": 450},
    "rerank": {"model": MODEL, "effort": "low", "maxTokens": 3000, "visibleTokens": 1800},
    "answer": {"model": MODEL, "effort": "low", "maxTokens": 4000, "visibleTokens": 3000},
    "repair": {"model": MODEL, "effort": "low", "maxTokens": 1500, "visibleTokens": 1000},
    "final_review": {"model": MODEL, "effort": "low", "maxTokens": 1500, "visibleTokens": 1000},
}
MAX_REASONING_TOKENS = 3000
KINDS = ["answer", "uncertain", "outOfScope", "greeting", "safety"]


def discovery_seed_budget(env):
    value = str(env.get("BIBLE_DISCOVERY_SEEDS", "12"))
    if value not in {"12", "24", "48"}:
        raise ValueError("INVALID_DISCOVERY_SEEDS")
    return int(value)


def reasoning_mode(env):
    value = env.get("BIBLE_REASONING_MODE", "baseline")
    if value not in {"baseline", "selection-none", "all-none", "plan-low", "judgement-low", "plan-answer-low", "plan-review-answer-low"}:
        raise ValueError("INVALID_REASONING_MODE")
    return value


def study_discovery_mode(env):
    value = env.get('BIBLE_STUDY_DISCOVERY', 'baseline')
    if value not in {'baseline', 'planned'}:
        raise ValueError('INVALID_STUDY_DISCOVERY')
    return value


def focus_schema_mode(env):
    value=env.get('BIBLE_FOCUS_SCHEMA','baseline')
    if value not in {'baseline','compact','refs','bound'} or value in {'compact','bound'} and env.get('BIBLE_LATENCY_MODE')=='selection':
        raise ValueError('INVALID_FOCUS_SCHEMA')
    return value


def bilingual_background(env):
    return env.get('BIBLE_BACKGROUND_SEARCH') in {'bilingual', 'anchored'}


def checked_answer_coverage(env):
    return env.get('BIBLE_ANSWER_COVERAGE') in {'checked', 'source-first'}


def target_policy_options(env):
    if env.get('BIBLE_BACKGROUND_SEARCH', 'legacy') not in {'legacy', 'bilingual', 'anchored'}:
        raise ValueError('INVALID_BACKGROUND_SEARCH')
    if env.get('BIBLE_CONTEXT_RETRIEVAL','baseline') not in {'baseline','conditional','review-first'}:
        raise ValueError('INVALID_CONTEXT_RETRIEVAL')
    if env.get('BIBLE_SOURCE_ROUTING','baseline') not in {'baseline','inventory'}:
        raise ValueError('INVALID_SOURCE_ROUTING')
    if env.get('BIBLE_EXPLANATION_POLICY','baseline') not in {'baseline','explicit'}:
        raise ValueError('INVALID_EXPLANATION_POLICY')
    if env.get('BIBLE_REVIEW_WIRE','baseline') not in {'baseline','canonical'}:
        raise ValueError('INVALID_REVIEW_WIRE')
    if env.get('BIBLE_NEED_AUDIT','baseline') not in {'baseline','bounded'}:
        raise ValueError('INVALID_NEED_AUDIT')
    if env.get('BIBLE_PROSE_POLICY','baseline') not in {'baseline','direct'}:
        raise ValueError('INVALID_PROSE_POLICY')
    if env.get('BIBLE_PROOF_BINDING','baseline') not in {'baseline','witness'}:
        raise ValueError('INVALID_PROOF_BINDING')
    if env.get('BIBLE_ANSWER_COVERAGE','baseline') not in {'baseline','checked','source-first'}:
        raise ValueError('INVALID_ANSWER_COVERAGE')
    if env.get('BIBLE_ANSWER_COVERAGE') == 'source-first' and (
            str(env.get('BIBLE_RELIABILITY_STAGE', '0')) not in {'1','2','3','4','5','6'}
            or env.get('BIBLE_INTENT_ROUTING') != 'hybrid'):
        raise ValueError('SOURCE_FIRST_REQUIRES_HYBRID_ID_SUPPORTS')
    if env.get('BIBLE_PROCESSING_TIER','baseline') not in {'baseline','answer-fast','decision-fast','final-fast'}:
        raise ValueError('INVALID_PROCESSING_TIER')
    if env.get('BIBLE_FINAL_REVIEW','baseline') not in {'baseline','patch','canonical'}:
        raise ValueError('INVALID_FINAL_REVIEW')
    if env.get('BIBLE_CONTEXT_LEADERS','baseline') not in {'baseline','same-book','anchor-aware'}:
        raise ValueError('INVALID_CONTEXT_LEADERS')
    if env.get('BIBLE_SOURCE_SCOPE','baseline') not in {'baseline','qualified'}:
        raise ValueError('INVALID_SOURCE_SCOPE')
    if env.get('BIBLE_AMBIGUITY_ROUTING','baseline') not in {'baseline','resolve'}:
        raise ValueError('INVALID_AMBIGUITY_ROUTING')
    if env.get('BIBLE_SUPERSCRIPTIONS','baseline') not in {'baseline','web'}:
        raise ValueError('INVALID_SUPERSCRIPTIONS')
    if env.get('BIBLE_REFERENCE_ROOTS','baseline') not in {'baseline','provided'}:
        raise ValueError('INVALID_REFERENCE_ROOTS')
    if env.get('BIBLE_LITERAL_CHECK','baseline') not in {'baseline','bounded'}:
        raise ValueError('INVALID_LITERAL_CHECK')
    if env.get('BIBLE_REVIEW_FINDINGS','baseline') not in {'baseline','concise'}:
        raise ValueError('INVALID_REVIEW_FINDINGS')


def source_coverage_contract(env=None):
    """Describe the supplied data, not facts allegedly absent from Scripture.

    Runtime corpus.json.gz has numbered verse strings only. Editorial section
    coordinates and bounded study excerpts do not restore canonical preambles.
    Return a fresh JSON value so executions cannot mutate a shared inventory.
    """
    result = {'scripture': {'translation': 'KRV', 'unit': 'numbered_verse_text',
                          'notSupplied': ['canonical_superscriptions', 'book_introductions', 'edition_footnotes']},
            'sections': 'RNKSV_editorial_section_coordinates',
            'study': 'bounded_verbatim_excerpts',
            'absenceRule': 'not_supplied_is_not_proof_of_absence'}
    if (env or {}).get('BIBLE_SUPERSCRIPTIONS') == 'web':
        result['additionalScripture'] = 'WEB translated Psalm superscriptions; use only actually supplied records'
    return result


def load_env(root: Path = ROOT):
    # Do not inherit application keys, search parent folders or interpolate env vars.
    values = dict(RESEARCH_DEFAULTS)
    keys = set(values) | {"OPENAI_API_KEY", "QWEN_API_KEY", "QWEN_WORKSPACE_ID"}
    for name in (".env", ".env.local"):
        path = root / name
        if path.is_file():
            values.update({k: v for k, v in dotenv_values(path, interpolate=False).items() if k in keys and v is not None})
    if values["BIBLE_AI_PROVIDER"] != "openai" or values["BIBLE_RETRIEVAL_VERSION"] != "v4":
        raise ValueError("LANGGRAPH_REQUIRES_RESEARCH_V4")
    target_policy_options(values)
    discovery_seed_budget(values)
    reasoning_mode(values)
    study_discovery_mode(values)
    focus_schema_mode(values)
    return values
