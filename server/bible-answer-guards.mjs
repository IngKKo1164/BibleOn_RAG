import { OUT_OF_SCOPE_MESSAGE, UNCERTAIN_ANSWER_MESSAGE } from './rag-answer-guidelines.mjs';
// Preserve the import name while applying the owner's small gloss allowance.
export { isKoreanExplanation as koreanOnly } from './bible-language.mjs';
export function quoteMatches(quote, text) {
  if (typeof quote !== 'string' || typeof text !== 'string') return false;
  const parts = quote.split(/\.{3}|…+/).map(s => s.trim()).filter(Boolean);
  if (!parts.length || parts.some(s => s.length < 3)) return false;
  let position = 0;
  return parts.every(part => { const at = text.indexOf(part, position); if (at < 0) return false; position = at + part.length; return true; });
}
export const fallback = (kind, usage) => ({ kind, text: kind === 'outOfScope' ? OUT_OF_SCOPE_MESSAGE : kind === 'greeting' ? '안녕하세요! 궁금한 말씀이나 신앙 이야기가 있으면 편하게 나눠 주세요.' : kind === 'safety' ? '지금 많이 힘들고 무서우시겠어요. 혼자 견디지 말고 가까운 사람에게 지금 곁에 있어 달라고 연락해 주세요. 다칠 위험이 있다면 위험한 물건에서 멀어지고, 즉시 119나 112에 연락해 도움을 요청해 주세요. 자살예방상담전화 109에서도 도움을 받을 수 있어요.' : UNCERTAIN_ANSWER_MESSAGE, citations: [], contextSummary: '', outOfScope: kind === 'outOfScope', needsClarification: kind === 'uncertain', usage });
