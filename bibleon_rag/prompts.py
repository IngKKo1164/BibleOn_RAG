"""Compose the preserved prompts without repeating passage text or metadata."""
import re

from .common import unique
from .prompt_text import (INTEGRATED_BASE, PASSAGE_BASE, INTEGRATED_GROUNDING,
                          STUDY_GROUNDING, COVERAGE_GROUNDING, SUFFICIENCY_PROMPT)


def build_integrated_system_prompt(corpus, evidence, study=False):
    books = {v["bookId"] for v in evidence}
    ids = ", ".join(f"{key}={book['name']}" for key, book in corpus.items() if key in books)
    return INTEGRATED_BASE + "\n사용 가능한 bookId: " + ids + "\n" + (STUDY_GROUNDING if study else INTEGRATED_GROUNDING) + "\n" + COVERAGE_GROUNDING


def build_passage_system_prompt(corpus, evidence, study=False):
    ids = ", ".join(f"{key}={corpus[key]['name']}" for key in unique(v["bookId"] for v in evidence))
    return PASSAGE_BASE + "\n사용 가능한 bookId: " + ids + ("\n" + STUDY_GROUNDING if study else "") + "\n" + COVERAGE_GROUNDING


def with_sufficiency_prompt(prompt):
    value = re.sub(r'^\{"answer":.*\}$', "구체적인 출력 필드는 제공된 JSON 스키마와 아래 질문별 근거 지침을 따른다.", prompt, flags=re.M)
    for a, b in [
        ("전부 무관하면 kind=outOfScope, citations=[]이고 answer는 아래 범위 밖 안내만 쓴다.", "전부 무관하면 kind=outOfScope, citations=[]로 표시한다. 범위 밖 안내는 서버가 제공한다."),
        ("핵심 답을 확신할 수 없으면 kind=uncertain이다.", "모든 질문 부분의 핵심 근거가 부족한 경우에만 kind=uncertain이다."),
        ("kind=uncertain, citations=[]로 하고 answer에는 아래 안내만 그대로 쓴다.", "답변할 수 없는 질문 부분은 아래 질문별 근거 지침에 따라 표시한다.")]:
        value = value.replace(a, b, 1)
    value = value.replace("JSON은 answer, kind, citations", "JSON은 answers, kind, citations")
    value = value.replace("answer는 한국어로 완결한 답변,", "answers의 각 text는 한국어로 완결한 답변,", 1)
    value = value.replace("answer 안에서", "첫 답변 text에서")
    return re.sub(r"\banswer(?=는|의|와|에는)", "각 답변 text", value, flags=re.A) + "\n" + SUFFICIENCY_PROMPT
