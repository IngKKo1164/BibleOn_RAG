"""Offline evidence availability and source-bound semantic use reviews.

No provider calls, answer rewriting or automatic semantic approval. Literal
span/quote checks bind a review to real material; the reviewer judges meaning.
"""
from copy import deepcopy
from collections import Counter

from .evaluation import coverage, evidence_id
from .observability import digest

VERSION = 'evidence-use-v1'
USE_STATUSES = {'supported', 'partial', 'omitted', 'distorted', 'unsupported', 'unclear'}


def branches(case):
    return case.get('answerVariants') or [
        {'id': 'main', 'expectedAnswer': case['expectedAnswer'], 'requirements': case['requirements']}]


def essential(branch, *, scripture_only=False):
    result = []
    for requirement in branch['requirements']:
        if requirement.get('importance', 'essential') != 'essential':
            continue
        row = deepcopy(requirement)
        if scripture_only:
            from .observability import VERSE_ID
            row['alternatives'] = [bundle for bundle in row['alternatives']
                                   if all(VERSE_ID.fullmatch(v) for v in bundle)]
        if row['alternatives']:
            result.append(row)
    return result


def case_coverage(available, case, *, scripture_only=False):
    results = []
    for branch in branches(case):
        requirements = essential(branch, scripture_only=scripture_only)
        result = coverage(available, requirements)
        results.append({'id': branch['id'], **result, 'requirementIds': [r['id'] for r in requirements]})
    eligible = [r for r in results if r['requirementRecall'] is not None]
    if not eligible:
        return {**coverage([], []), 'branchId': None, 'completeBranchIds': [], 'branches': results}
    best = max(eligible, key=lambda r: (r['requirementRecall'], r['verseRecall']))
    return {**best, 'branchId': best['id'],
            'completeBranchIds': [r['id'] for r in eligible if r['allRequirements']], 'branches': results}


def bindings(case, result, materials):
    if not isinstance(result, dict) or not isinstance(result.get('text'), str):
        raise ValueError('ANSWER_RESULT_REQUIRED')
    if not isinstance(materials, dict) or any(not evidence_id(k) or not isinstance(v, str) or not v
                                               for k, v in materials.items()):
        raise ValueError('ACTUAL_INPUT_MATERIALS_REQUIRED')
    return {'caseId': case['id'], 'caseSha256': digest(case),
            'resultSha256': digest(result), 'inputSha256': digest(materials)}


def review_template(case, result, materials):
    return {'version': VERSION, 'bindings': bindings(case, result, materials),
            'reviewer': {'kind': '', 'name': '', 'independent': False, 'blinded': False},
            'variantId': None, 'requirements': [], 'additionalMaterialClaimsReviewed': False,
            'additionalErrors': [], 'behavior': None,
            'notice': 'Empty/unreviewed is unknown, not correct. Review the actual answer and source meaning.'}


def _spans(values, text, *, required):
    if not isinstance(values, list) or any(not isinstance(s, str) for s in values) or len(values) != len(set(values)):
        raise ValueError('INVALID_ANSWER_SPANS')
    if required and not values:
        raise ValueError('VISIBLE_ANSWER_SPAN_REQUIRED')
    if any(not isinstance(s, str) or not s.strip() or s not in text for s in values):
        raise ValueError('ANSWER_SPAN_NOT_IN_ACTUAL_OUTPUT')
    return values


def _supports(values, materials, *, required):
    if not isinstance(values, list) or required and not values:
        raise ValueError('ACTUAL_SUPPORT_REQUIRED')
    ids = []
    for value in values:
        if not isinstance(value, dict):
            raise ValueError('INVALID_SUPPORT')
        key, quote = value.get('evidenceId'), value.get('quote')
        if key not in materials or not isinstance(quote, str) or not quote.strip() or quote not in materials[key]:
            raise ValueError('SUPPORT_NOT_IN_ACTUAL_INPUT')
        ids.append(key)
    if len(ids) != len(set(ids)):
        raise ValueError('DUPLICATE_SUPPORT')
    return ids


def _reason(row):
    value = row.get('reason')
    if not isinstance(value, str) or not value.strip():
        raise ValueError('SEMANTIC_REVIEW_REASON_REQUIRED')


def score_answer_use(case, result, materials, review=None):
    """Keep strict qrel coverage unchanged, then independently review usage.

    A supported answer may use valid alternate input not listed in frozen gold.
    That disagreement is reported without expanding or weakening the gold.
    """
    bound = bindings(case, result, materials)
    availability = case_coverage(materials, case)
    output = {'protocol': VERSION, 'caseId': case['id'], 'bindings': bound, 'availability': availability,
              'reviewStatus': 'pending', 'semanticTruthAutomaticallyCertified': False,
              'requirements': [], 'reviewedRequirements': 0, 'useComplete': None,
              'additionalMaterialErrors': None, 'behavior': None}
    if review is None:
        return output
    if review.get('version') != VERSION or review.get('bindings') != bound:
        raise ValueError('REVIEW_BINDING_MISMATCH')
    reviewer = review.get('reviewer', {})
    if reviewer.get('kind') not in {'human', 'assistant_direct', 'llm_judge'} or not reviewer.get('name'):
        raise ValueError('REVIEWER_REQUIRED')
    if any(type(reviewer.get(k)) is not bool for k in ('independent', 'blinded')):
        raise ValueError('REVIEW_SCOPE_REQUIRED')
    variant = next((b for b in branches(case) if b['id'] == review.get('variantId')), None)
    if variant is None:
        raise ValueError('REVIEW_VARIANT_REQUIRED')
    expected = {r['id']: r for r in essential(variant)}
    rows = review.get('requirements')
    if not isinstance(rows, list) or any(not isinstance(r, dict) for r in rows):
        raise ValueError('REQUIREMENT_REVIEWS_REQUIRED')
    ids = [r.get('requirementId') for r in rows]
    if any(not isinstance(i, str) for i in ids) or len(ids) != len(set(ids)) or not set(ids) <= expected.keys():
        raise ValueError('REVIEW_REQUIREMENT_MISMATCH')
    text = result['text']
    checked = []
    for row in rows:
        status = row.get('status')
        if status not in USE_STATUSES:
            raise ValueError('INVALID_USE_STATUS')
        _reason(row)
        spans = _spans(row.get('answerSpans'), text, required=status not in {'omitted', 'unclear'})
        if status == 'omitted' and spans:
            raise ValueError('OMISSION_CANNOT_HAVE_ANSWER_SPAN')
        support_ids = _supports(row.get('supports'), materials, required=status in {'supported', 'partial', 'distorted'})
        requirement = expected[row['requirementId']]
        acquired = coverage(materials, [requirement])['allRequirements']
        checked.append({'requirementId': row['requirementId'], 'claim': requirement['claim'],
            'acquired': acquired, 'use': status, 'answerSpans': spans, 'supports': row['supports'],
            'reason': row['reason'], 'goldAlternativeReviewNeeded': status == 'supported' and not acquired,
            'citationIdsAreNotSemanticCredit': True})
    additional = review.get('additionalErrors')
    if type(review.get('additionalMaterialClaimsReviewed')) is not bool or not isinstance(additional, list):
        raise ValueError('ADDITIONAL_CLAIM_REVIEW_REQUIRED')
    for row in additional:
        if not isinstance(row, dict) or row.get('status') not in {'distorted', 'unsupported'}:
            raise ValueError('INVALID_ADDITIONAL_ERROR')
        _reason(row)
        _spans(row.get('answerSpans'), text, required=True)
        _supports(row.get('supports'), materials, required=row['status'] == 'distorted')
    behavior = review.get('behavior')
    if behavior is not None:
        if not isinstance(behavior, dict) or behavior.get('status') not in {'appropriate', 'inappropriate', 'unclear'}:
            raise ValueError('INVALID_BEHAVIOR_REVIEW')
        _reason(behavior)
        _spans(behavior.get('answerSpans'), text, required=True)
    complete_review = (set(ids) == expected.keys() and review['additionalMaterialClaimsReviewed']
                       and all(r['use'] != 'unclear' for r in checked)
                       and (bool(expected) or behavior is not None and behavior['status'] != 'unclear'))
    output.update(reviewStatus='reviewed' if complete_review else 'partial', reviewer=reviewer,
        variantId=variant['id'], requirements=checked, reviewedRequirements=len(checked),
        expectedRequirements=len(expected), additionalMaterialErrors=len(additional),
        additionalErrors=additional, behavior=behavior,
        selectedVariantAvailability=coverage(materials, list(expected.values())))
    if complete_review and expected:
        output['useComplete'] = all(r['use'] == 'supported' for r in checked) and not additional
    return output


def summarize_answer_use(cases, scores):
    planned_ids = [c['id'] for c in cases]
    if not planned_ids or len(set(planned_ids)) != len(planned_ids):
        raise ValueError('INVALID_PLANNED_COHORT')
    by_id = {s['caseId']: s for s in scores}
    if len(by_id) != len(scores) or not by_id.keys() <= set(planned_ids):
        raise ValueError('SCORED_COHORT_MISMATCH')
    eligible_ids = {c['id'] for c in cases if c.get('corpusAvailability', 'available') == 'available'
                    and any(essential(b) for b in branches(c))}
    applicable = [s for s in scores if s['caseId'] in eligible_ids]
    reviewed = [s for s in scores if s['reviewStatus'] == 'reviewed']
    rows = [r for s in applicable for r in s['requirements']]
    acquired = [r for r in rows if r['acquired']]
    used = sum(r['use'] == 'supported' for r in acquired)
    pending_ids = sorted(set(planned_ids) - {s['caseId'] for s in reviewed})
    quadrants = Counter()
    for score in applicable:
        found = score['availability']['allRequirements'] is True
        used_all = score['useComplete']
        quadrants[('acquired' if found else 'not_fully_acquired') + '/' + (
            'correctly_used' if used_all is True else 'use_incomplete' if used_all is False else 'unreviewed')] += 1
    return {'planned': len(planned_ids), 'recorded': len(scores),
        'missingIds': sorted(set(planned_ids)-by_id.keys()), 'pendingReviewIds': pending_ids,
        'applicableCases': len(eligible_ids), 'availabilityCompleteCases': sum(s['availability']['allRequirements'] is True for s in applicable),
        'useCompleteCases': sum(s['useComplete'] is True for s in applicable),
        'allUseScoresReviewed': not pending_ids,
        'reviewedRequirementStatuses': dict(Counter(r['use'] for r in rows)),
        'acquiredReviewedRequirements': len(acquired), 'correctlyUsedAcquiredRequirements': used,
        'contextUseAmongAcquired': used/len(acquired) if acquired and not pending_ids else None,
        'availabilityByUse': dict(quadrants),
        'incompleteUseCaseIds': sorted(s['caseId'] for s in applicable if s['useComplete'] is False),
        'meaningErrorCaseIds': sorted(s['caseId'] for s in applicable if any(
            r['use'] in {'distorted', 'unsupported'} for r in s['requirements']) or s['additionalMaterialErrors']),
        'goldAlternativeReviewCases': sorted({s['caseId'] for s in scores if any(r['goldAlternativeReviewNeeded'] for r in s['requirements'])}),
        'additionalMaterialErrorCases': [s['caseId'] for s in scores if s['additionalMaterialErrors']],
        'behaviorReviews': dict(Counter(s['behavior']['status'] for s in reviewed if s['behavior'])),
        'allAnswersCorrectnessNotImplied': True}
