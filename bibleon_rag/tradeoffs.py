"""Complete-cohort, source-bound quality/cost/latency comparisons. No API I/O."""
from collections import Counter
import math
import statistics

CONTEXT_KEYS = ('datasetSha256', 'goldSha256', 'assetsSha256', 'model',
                'answerTokenLimit', 'timingKind', 'qualityRubric', 'comparisonBudget')


def finite_nonnegative(value):
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def summarize_condition(manifest, rows, grades):
    """Never turn missing/failed/ungraded rows into a smaller winning cohort."""
    planned = manifest['plannedIds']
    if not planned or len(planned) != len(set(planned)):
        raise ValueError('INVALID_PLANNED_COHORT')
    if any(key not in manifest for key in CONTEXT_KEYS):
        raise ValueError('COMPARISON_CONTEXT_REQUIRED')
    eligible = set(manifest['evidenceApplicableIds'])
    boundary = set(manifest['boundaryIds'])
    if eligible & boundary or eligible | boundary != set(planned):
        raise ValueError('ALL_CASE_METRIC_ACCOUNTING_REQUIRED')
    ids = [row['caseId'] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= set(planned):
        raise ValueError('RESULT_ID_MISMATCH')
    grade_ids = [g['caseId'] for g in grades]
    if len(grade_ids) != len(set(grade_ids)) or not set(grade_ids) <= set(ids):
        raise ValueError('GRADE_ID_MISMATCH')
    by_id = {row['caseId']: row for row in rows}
    by_grade = {g['caseId']: g for g in grades}
    use_protocol = manifest.get('answerUseProtocol', 'legacy_quality_only')
    if use_protocol not in {'legacy_quality_only', 'evidence-use-v1'}:
        raise ValueError('INVALID_ANSWER_USE_PROTOCOL')
    use_scores = []
    for cid, grade in by_grade.items():
        if not grade.get('resultSha256') or grade['resultSha256'] != by_id[cid].get('resultSha256'):
            raise ValueError('GRADE_RESULT_BINDING_REQUIRED')
        if not finite_nonnegative(grade.get('quality')) or grade['quality'] > 8:
            raise ValueError('INVALID_QUALITY_SCORE')
        if type(grade.get('materialError')) is not bool:
            raise ValueError('MATERIAL_ERROR_JUDGMENT_REQUIRED')
        if use_protocol == 'evidence-use-v1' and grade.get('answerUse') is not None:
            score = grade['answerUse']
            if (score.get('protocol') != use_protocol or score.get('caseId') != cid
                    or score.get('bindings', {}).get('resultSha256') != by_id[cid]['resultSha256']):
                raise ValueError('ANSWER_USE_RESULT_BINDING_REQUIRED')
            if any(not by_id[cid].get(key) or score.get('bindings', {}).get(key) != by_id[cid][key]
                   for key in ('caseSha256', 'inputSha256')):
                raise ValueError('ANSWER_USE_CASE_INPUT_BINDING_REQUIRED')
            if cid in eligible and score['availability']['allRequirements'] is not by_id[cid].get('finalComplete'):
                raise ValueError('ANSWER_USE_INPUT_COVERAGE_MISMATCH')
            use_scores.append(score)
    times = [row['seconds'] for row in rows if finite_nonnegative(row.get('seconds'))]
    costs = [row['estimatedUsd'] for row in rows if finite_nonnegative(row.get('estimatedUsd'))]
    errors = [row['caseId'] for row in rows if row.get('errorType')]
    missing = sorted(set(planned) - set(ids))
    scores = [g['quality'] for g in grades]
    complete = sum(cid in by_id and by_id[cid].get('finalComplete') is True for cid in eligible)
    kinds = Counter(row.get('kind', 'unknown') for row in rows)
    within = sum(t <= 15 for t in times)
    context = {k: manifest[k] for k in CONTEXT_KEYS}
    context['answerUseProtocol'] = use_protocol
    context.update(plannedIds=sorted(planned), evidenceApplicableIds=sorted(eligible), boundaryIds=sorted(boundary))
    result = {'condition': manifest['condition'], 'context': context,
              'planned': len(planned), 'recorded': len(rows), 'missingIds': missing,
              'executionErrorIds': errors, 'graded': len(grades),
              'ungradedIds': sorted(set(planned) - set(by_grade)),
              'observedTimeCount': len(times), 'observedCostCount': len(costs),
              'observedMeanSeconds': statistics.mean(times) if times else None,
              'observedMedianSeconds': statistics.median(times) if times else None,
              'observedP95Seconds': sorted(times)[max(0, math.ceil(len(times)*.95)-1)] if times else None,
              'observedMeanUsd': statistics.mean(costs) if costs else None,
              'observedQualityOf8': statistics.mean(scores) if scores else None,
              'materialErrors': sum(g['materialError'] for g in grades),
              'within15Seconds': within, 'outcomes': dict(kinds),
              'evidenceApplicable': len(eligible), 'finalComplete': complete,
              'boundaryCount': len(boundary)}
    use_complete = (len(use_scores) == len(planned) and all(s['reviewStatus'] == 'reviewed' for s in use_scores))
    result.update(answerUseReviewed=len([s for s in use_scores if s['reviewStatus'] == 'reviewed']),
        useCompleteCases=sum(s.get('useComplete') is True for s in use_scores if s['caseId'] in eligible),
        useErrorCases=sum(bool(s.get('additionalMaterialErrors')) or any(
            r['use'] in {'distorted', 'unsupported'} for r in s.get('requirements', [])) for s in use_scores))
    result['fullyMeasured'] = (not missing and len(times) == len(costs) == len(grades) == len(planned)
                               and (use_protocol == 'legacy_quality_only' or use_complete))
    result['qualityIsNotEvidenceRecall'] = True
    return result


def compare_frontier(summaries):
    """No arbitrary combined score or automatic promotion from a single target.

    Mean/p95 latency, cost, quality, material errors, execution errors and final
    complete evidence all remain separate. Replay timing cannot compete with
    live timing, and historical annotation changes cannot win a new condition.
    """
    if not summaries:
        raise ValueError('COMPARISON_REQUIRED')
    context = summaries[0]['context']
    if any(s['context'] != context for s in summaries):
        raise ValueError('INCOMPARABLE_EXPERIMENT_CONTEXTS')
    if any(not s['fullyMeasured'] for s in summaries):
        return {'status': 'incomplete', 'frontier': [], 'automaticPromotion': False,
                'reason': 'Missing time, cost, execution rows or source-bound grades'}
    if context['timingKind'] != 'live_full_request':
        return {'status': 'diagnostic_only', 'frontier': [], 'automaticPromotion': False,
                'reason': 'Replayed/reconstructed timing does not establish live serving performance'}
    if len(summaries) < 2 or len({s['condition'] for s in summaries}) != len(summaries):
        return {'status': 'comparison_required', 'frontier': [], 'automaticPromotion': False,
                'reason': 'At least two distinct measured conditions are required'}
    def axes(s):
        values = (s['observedMeanSeconds'], s['observedP95Seconds'], s['observedMeanUsd'],
                -s['observedQualityOf8'], s['materialErrors'], len(s['executionErrorIds']),
                -s['finalComplete'])
        return values + (-s['useCompleteCases'], s['useErrorCases']) if context['answerUseProtocol'] == 'evidence-use-v1' else values
    def dominates(a, b):
        x, y = axes(a), axes(b)
        return all(i <= j for i, j in zip(x, y)) and any(i < j for i, j in zip(x, y))
    frontier = [s['condition'] for s in summaries
                if not any(dominates(other, s) for other in summaries)]
    return {'status': 'measured_tradeoffs', 'frontier': frontier,
            'automaticPromotion': False, 'axes': ['mean_latency', 'p95_latency', 'cost',
            'quality', 'material_errors', 'execution_errors', 'complete_evidence'] + (
                ['correct_evidence_use', 'evidence_use_errors'] if context['answerUseProtocol'] == 'evidence-use-v1' else []),
            'samplingUncertaintyNotResolvedByParetoMembership': True}
