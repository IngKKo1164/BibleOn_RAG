export const BIBLE_LANGUAGE_PROMPT = 'answer는 한국어 중심의 자연스러운 설명으로 쓴다. 이해를 돕는 영어 표현과 히브리어·그리스어·라틴어 원어는 한국어 뜻과 함께 짧게 병기할 수 있다. 외국어 글자는 전체 언어 글자의 3% 이내로 제한한다. 공백·숫자·마크다운 기호로 비율을 늘리지 않는다. 외국어 문단이나 불필요한 언어 전환은 하지 않는다. 원어는 제공된 근거로 확인할 수 있을 때만 사용하며 의미·어원을 지어내지 않는다. 인명·지명과 성경 장절은 개역한글 표기를 중심으로 전달한다.';

// A size/script check, not a semantic judgment of whether a gloss is useful.
// Count letters so padding with whitespace, numbers or Markdown cannot hide a
// foreign passage. Hebrew pointing/accents do not inflate the letter count.
export function isKoreanExplanation(text) {
  if (typeof text !== 'string') return false;
  const letters = [...text.normalize('NFC').matchAll(/\p{L}/gu)].map(m => m[0]);
  if (!letters.length) return false;
  let foreign = 0;
  for (const letter of letters) {
    if (/\p{Script=Hangul}/u.test(letter)) continue;
    if (!/[\p{Script=Latin}\p{Script=Hebrew}\p{Script=Greek}]/u.test(letter)) return false;
    foreign++;
  }
  return foreign * 100 <= letters.length * 3;
}
