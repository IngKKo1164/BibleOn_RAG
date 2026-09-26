import { hasCompleteAnswerEnding } from './answer-completion.mjs';
import { parseBibleReferences } from './bible-rag.mjs';
// These checks supplement semantic review; they do not prove every assertion.
// With a Scripture-only corpus, reconstructed historical precision is outside
// the available evidence. Do not let the model's own approval override that.
export function unresolvedReferent(question, previousEvidence, passage = []) {
  if (previousEvidence || passage.length) return false;
  const text=String(question).replace(/[\s?!.,]/g,'');
  return /^(?:성경(?:에|에서)(?:나오는|나온)?)?(?:그|이|저)(?:사람|인물|청년|여자|남자|사건|구절|말씀)(?:은|는|이|가|을|를)?(?:누구(?:였|이었)?(?:지|야|나요|어요)?|뭐(?:였|이었)?(?:지|야|예요)?|무엇(?:인가요|이었지)?|알려줘|설명해줘)$/.test(text);
}
const precision = /(?:제\s*)?(?:\d+|첫|두|세|네|일|이|삼|사)\s*(?:차|번째)\s*(?:전도|선교)\s*여행|(?:기원전|기원후|서기)\s*\d|\d+\s*세기|속주|항구\s*도시|헬라어|히브리어|아람어|육로|해로|통풍|공기.{0,12}탁|(?:주거용|주거\s*공간|상점|작업장|건축\s*(?:구조|양식)|환기\s*(?:부족|문제)|산소\s*부족)|당시.{0,60}(?:관습|풍습|기온|실내\s*온도|건물|창문\s*구조)/;
const limitOnly = /(?:기록|명시|확인|특정|단정).{0,25}(?:않|없|어렵|불가|못)|알\s*수\s*없|추측할\s*수\s*없/;
export function constrainRagDraft(answer, evidence, question='', { lexiconAvailable = false } = {}) {
  const canonical=evidence.map(v=>v.text).join('\n'), removed=[], rules=[];
  // Preserve explanatory glosses; language size/script validation is separate.
  const korean=String(answer);
  const blocks=korean.replace(/^(#{1,6} [^\n]+)\n(?!\n)/gm,'$1\n\n').split(/\n\s*\n/), kept=[];
  for(const block of blocks){
    // Closing emphasis/quotes belong to the preceding sentence. Preserve the
    // separators so a neighboring valid sentence is never removed with it.
    const sentences=/^#{1,6} /.test(block)?[block]:block.split(/((?<=[.!?](?:\*\*|[”’"])?)\s+)/u);
    const safe=sentences.filter((sentence,i)=>{
      if(i%2)return true;
      const match=sentence.match(precision) || sentence.match(/관례|관습|풍습|피로.{0,12}(?:누적|쌓|때문)|아침\s*일찍|(?:공기|산소).{0,8}(?:부족|모자)|(?:실내|방\s*안|다락).{0,16}(?:더웠|더워|열기|무더)|(?:인원|사람).{0,8}많.{0,12}(?:시사|추정)/);
      const unsupportedLanguage=!lexiconAvailable && /헬라어|히브리어|아람어/.test(sentence) && !canonical.includes(sentence.match(/헬라어|히브리어|아람어/)[0]);
      const unsupportedAction=[['기도',/기도(?:로|와(?=.{0,35}(?:살아|살렸|치유|회복되))|를\s*통해|를\s*통한|하여|하고|하자|하였|했|하셨|를.{0,8}(?:드|하))/],['안수',/안수(?:로|하여|하고|하자|하였|했|하셨)/],['금식',/금식(?:으로|하여|하고|하자|하였|했|하셨)/]]
        .some(([term,pattern])=>pattern.test(sentence)&&!canonical.includes(term)&&!(term==='기도'&&/간구|하나님께.{0,16}구하|주께.{0,10}구하|여호와께.{0,20}부르짖/.test(canonical)));
      const languageMatch = match && /^(?:헬라어|히브리어|아람어)$/.test(match[0]);
      const unsupportedPrecision=match && !canonical.includes(match[0]) && !(languageMatch && lexiconAvailable);
      const unsafe=(unsupportedLanguage || unsupportedPrecision || unsupportedAction) && !limitOnly.test(sentence);
      if(unsafe){removed.push(sentence);rules.push(unsupportedLanguage?'language-source':unsupportedAction?'action-source':'precision-source');}return !unsafe;
    });
    const cleaned=safe.join('').trim();if(cleaned)kept.push(cleaned);
  }
  // A removed paragraph must not leave an empty heading behind.
  const cleaned=[];let pending=[];
  for(const block of kept){const heading=block.match(/^(#{1,6}) [^\n]+$/);if(heading){pending=pending.filter(row=>row.level<heading[1].length);pending.push({level:heading[1].length,block});}else{cleaned.push(...pending.map(row=>row.block),block);pending=[];}}
  if(removed.length&&/배경|시대|역사|문화|연대/.test(question))cleaned.push('구체적인 연대나 당시 생활환경처럼 이 본문에 기록되지 않은 배경은 확인하기 어렵습니다.');
  return {answer:cleaned.join('\n\n'),removed,rules:[...new Set(rules)]};
}
export function completeRagAnswer(answer,corpus){
  if(hasCompleteAnswerEnding(answer))return true;
  const tail=answer.trim().match(/\(([^()]*)\)\s*$/);if(!tail)return false;
  // Multiple book references commonly use semicolons. Every segment must be
  // a complete valid reference, never arbitrary prose hidden after a citation.
  const segments=tail[1].split(/[;；]/);
  return segments.every(segment=>{
    const parsed=parseBibleReferences(corpus,segment,{maxMatches:128,maxVerses:96,maxRangeVerses:96,chapterMode:'mention'});
    return parsed.exact&&!parsed.invalid;
  })&&hasCompleteAnswerEnding(answer.trim().slice(0,tail.index));
}
