"""A routed runtime and two frozen comparison conditions.

Historical low-level APIs retain their original settings for local replay.
Normal CLI requests use one of these versioned conditions and cannot enable
retired per-case filters, experimental verifier wires or premium dispatch.
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import warnings

from dotenv import dotenv_values

VERSION = 'tradeoff-v1'
PROFILE_KEY = 'BIBLE_RESEARCH_PROFILE'
PROFILE_VERSION_KEY = 'BIBLE_PROFILE_VERSION'


@dataclass(frozen=True)
class ResearchProfile:
    name: str
    adaptive: bool | None
    # Common comparison constraints, not exposed tuning switches.
    candidates: int = 24
    selected_chunks: int = 6
    evidence_verses: int = 96
    evidence_characters: int = 12000
    source_records: int = 3
    source_characters: int = 4000


PROFILES = {'auto': ResearchProfile('auto', None),
            'simple': ResearchProfile('simple', False),
            'adaptive': ResearchProfile('adaptive', True)}


def automatic_routing(env):
    policy = profile_for(env)
    return policy is not None and policy.name == 'auto'


def execution_profile(state, env):
    """Resolve a turn's stored decision without mutating shared runtime settings."""
    policy = profile_for(env)
    if policy is None or policy.name != 'auto':
        return policy
    selected = state.get('processingProfile')
    if selected not in {'simple', 'adaptive'}:
        raise ValueError('EXECUTION_PROFILE_NOT_SELECTED')
    return PROFILES[selected]


def profile_for(env):
    name = env.get(PROFILE_KEY)
    if name is None:
        return None  # Explicitly preserved historical Python callers.
    if name not in PROFILES or env.get(PROFILE_VERSION_KEY) != VERSION:
        raise ValueError('INVALID_RESEARCH_PROFILE')
    for key, expected in resolved_settings(name).items():
        if env.get(key) != expected:
            raise ValueError('PROFILE_SETTINGS_CONFLICT:' + key)
    return PROFILES[name]


def resolved_settings(name):
    from .config import RESEARCH_DEFAULTS
    if name not in PROFILES:
        raise ValueError('INVALID_RESEARCH_PROFILE')
    return {**RESEARCH_DEFAULTS, PROFILE_KEY: name, PROFILE_VERSION_KEY: VERSION,
            'BIBLE_EVIDENCE_POLICY_STAGE': '3', 'BIBLE_RELIABILITY_STAGE': '3',
            'BIBLE_CONTEXT_RETRIEVAL': 'baseline', 'BIBLE_LATENCY_MODE': 'baseline',
            'BIBLE_PROCESSING_TIER': 'baseline', 'BIBLE_ANSWER_COVERAGE': 'baseline',
            'BIBLE_FINAL_REVIEW': 'baseline',
            'BIBLE_RETRIEVAL_REPAIR': 'off' if name == 'simple' else 'on',
            'BIBLE_REPAIR_QUERY_EXPANSION': 'off' if name == 'simple' else 'on'}


def load_runtime_env(root: Path, *, profile=None, require_explicit_profile=False):
    """Read credentials locally; never inherit the application's environment.

    Old flags are inventoried, not applied to normal runs. The policy's exact
    resolved settings remain checkpoint-bound and observable.
    """
    stored = {}
    for name in ('.env', '.env.local'):
        path = root / name
        if path.is_file():
            stored.update({k: v for k, v in dotenv_values(path, interpolate=False).items()
                           if v is not None})
    if require_explicit_profile and profile is None and not stored.get(PROFILE_KEY):
        raise ValueError('RESEARCH_PROFILE_SELECTION_REQUIRED')
    chosen = profile or stored.get(PROFILE_KEY, 'simple')
    if chosen not in PROFILES:
        raise ValueError('INVALID_RESEARCH_PROFILE')
    dimensions = stored.get('BIBLE_EMBEDDING_DIMENSIONS', '1024')
    if dimensions != '1024':
        raise ValueError('RESEARCH_PROFILE_REQUIRES_1024_INDEX')
    env = resolved_settings(chosen)
    env.update({key: stored.get(key, '') for key in
                ['OPENAI_API_KEY', 'QWEN_API_KEY', 'QWEN_WORKSPACE_ID']})
    # Fix the established operations. Unproven late-stage experiments are not
    # promoted merely because they exist in the historical implementation.
    legacy = sorted(k for k in stored if k.startswith('BIBLE_') and
                    k not in {PROFILE_KEY, 'BIBLE_EMBEDDING_DIMENSIONS'})
    if legacy:
        warnings.warn('Legacy research flags are not applied to profile runs: ' +
                      ', '.join(legacy), UserWarning, stacklevel=2)
    return env


def describe_profile(env):
    policy = profile_for(env)
    if policy is None:
        raise ValueError('NAMED_RESEARCH_PROFILE_REQUIRED')
    return {'version': VERSION, **asdict(policy),
            'modelWeightsChanged': False, 'qualityEffect': 'requires_actual_evaluation',
            'commonRouter': 'structured intent and conversation routing',
            'router': 'LLM intent and needs, then essential-need-count-v1' if policy.name == 'auto' else 'fixed comparison condition',
            'selection': ('essential needs: 1 rank-only; 2+ model selection' if policy.name == 'auto' else
                          'model needs selection' if policy.adaptive else 'rank-only selection'),
            'review': 'both routes before generation' if policy.name == 'auto' else 'adaptive only',
            'recovery': ('adaptive route only; at most once; no simple escalation' if policy.name == 'auto' else
                         'at most one missing-need search' if policy.adaptive else 'none'),
            'semanticRuleFilters': False, 'automaticProseDeletion': False,
            'generationModel': 'gpt-6-luna', 'reasoning': 'low',
            'processingTier': 'standard',
            'constraintsAreNotSemanticProof': True}
