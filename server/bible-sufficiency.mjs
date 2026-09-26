import { BibleProviderError } from './bible-errors.mjs';
import {inlineReferencesInEvidence} from './bible-integrated.mjs';
import {formatAnswerPart} from './answer-markdown.mjs';

const object=properties=>({type:'object',additionalProperties:false,properties,required:Object.keys(properties)});
const statuses=['supported','partial','insufficient'];
const gaps=['none','detail_unstated','background_unavailable','passage_missing','comparison_incomplete'];
const messages={
  detail_unstated:'질문하신 세부 사항은 현재 확인한 본문에 명시되어 있지 않아 단정할 수 없습니다.',
  background_unavailable:'추가적인 역사·배경 설명을 뒷받침할 자료가 충분하지 않아 그 부분은 단정하지 않겠습니다.',
  passage_missing:'이 부분에 직접 답할 성경 근거를 충분히 찾지 못했습니다. 관련 인물이나 사건, 구절을 조금 더 구체적으로 알려 주세요.',
  comparison_incomplete:'비교에 필요한 양쪽 근거를 모두 확인하지 못해 같은 사건이나 인물이라고 단정할 수 없습니다.',
};

export function sufficientAnswerSchema(parts, evidenceByPart) {
  return object(Object.fromEntries(parts.map(part=>{
    const ids=(evidenceByPart.get(part.partId)||[]).map(v=>v.id);
    return [part.partId,object({
      status:{type:'string',enum:ids.length?statuses:['insufficient']},
      text:{type:'string',...(ids.length?{}:{maxLength:0})},
      supports:{type:'array',maxItems:ids.length?2:0,items:object({
        verseId:{type:'string',...(ids.length?{enum:ids}:{})},
        quote:{type:'string',minLength:4,maxLength:160},
      })},
      gap:{type:'string',enum:ids.length?gaps:gaps.filter(g=>g!=='none')},
    })];
  })));
}

export const SUFFICIENCY_PROMPT = `이번 응답은 전체 answer 문자열 대신 answers 객체로 작성한다. questionParts의 각 P번호에 status, text, supports, gap을 작성한다. kind와 citations, 요청된 경우 studyNotes는 기존 스키마를 따른다.
각 text는 한 답변의 연속된 부분이다. 전체 결론은 첫 P에서만 제시하고 다음 P는 내용을 나타내는 ## 제목으로 바로 시작한다. 공통 결론·설명은 한 번만 하며, 각 부분의 서로 다른 근거와 논점은 충분히 풀어 쓴다. 상세 답은 ## 주제 아래 ### 하위 논점을 묶는다. P번호나 ‘질문 1’은 text에 쓰지 않는다. 부분 사이 구분선은 서버가 넣는다.
각 부분을 실제 본문과 대조한다. supported는 질문의 핵심을 근거로 답할 수 있음, partial은 핵심은 확인되지만 요청한 세부·배경·비교 일부가 부족함, insufficient는 핵심 답을 뒷받침할 근거가 없음이다. 검색 점수나 후보 선택 자체는 충분성의 증명이 아니다. 틀린 전제를 반증하는 본문이 있으면 supported이며, 나이·직업·동기 등이 없다는 이유만으로 확인된 인물·사건 전체를 insufficient로 만들지 않는다.
supported/partial의 text는 기존 깊이와 형식을 유지하여 확인된 내용만 충분히 설명한다. 각 text의 장절도 해당 P의 partEvidenceIds(및 여기서 직접 연결된 관주) 안에서만 인용한다. 다른 P에만 배정된 본문을 가져오지 않는다. supports에는 해당 부분의 partEvidenceIds에서 실제 설명을 지지하는 절 1~2개와 그 절의 연속된 원문 4~160자를 그대로 복사한다. 내부 검증용 quote는 필요한 핵심 문구만 짧게 복사하고 띄어쓰기·철자도 고치지 않는다. 각 부분의 supports 중 적어도 한 절을 citations에도 포함한다. 구절에 같은 단어가 있다는 이유만으로 다른 인물·사건·부정 표현을 근거로 삼지 않는다.
insufficient의 text는 빈 문자열, supports는 []다. 모르는 핵심을 기억으로 채우거나 성경 전체에 없다고 선언하지 않는다. gap은 supported일 때 none, 나머지는 detail_unstated(명시되지 않은 세부), background_unavailable(요청한 배경 자료 부족), passage_missing(핵심 본문 미확인), comparison_incomplete(비교 한쪽 부족) 중 선택한다. 다른 P에만 배정된 구절을 전체 자료에 없는 것으로 설명하지 않는다. 요청하지 않은 배경을 억지로 부족 항목으로 만들지 않는다. 부족 안내는 서버가 덧붙이므로 text에 정형 안내를 반복하지 않는다. 부족한 부분과 확인된 부분을 분리한다. 성경 질문의 모든 부분이 insufficient이면 kind=uncertain, 그 외에는 kind=answer다. 별도의 outOfScope/greeting/safety 경로이면 그 kind를 유지하고 모든 부분의 text="", supports=[], status=insufficient, gap=passage_missing, citations=[]로 둔다.`;

export function withSufficiencyPrompt(prompt) {
  return prompt
    .replace(/^\{"answer":.*\}$/gm,'구체적인 출력 필드는 제공된 JSON 스키마와 아래 질문별 근거 지침을 따른다.')
    .replace('전부 무관하면 kind=outOfScope, citations=[]이고 answer는 아래 범위 밖 안내만 쓴다.','전부 무관하면 kind=outOfScope, citations=[]로 표시한다. 범위 밖 안내는 서버가 제공한다.')
    .replace('핵심 답을 확신할 수 없으면 kind=uncertain이다.','모든 질문 부분의 핵심 근거가 부족한 경우에만 kind=uncertain이다.')
    .replace('kind=uncertain, citations=[]로 하고 answer에는 아래 안내만 그대로 쓴다.','답변할 수 없는 질문 부분은 아래 질문별 근거 지침에 따라 표시한다.')
    .replace(/JSON은 answer, kind, citations/g,'JSON은 answers, kind, citations')
    .replace('answer는 한국어로 완결한 답변,','answers의 각 text는 한국어로 완결한 답변,')
    .replace(/answer 안에서/g,'첫 답변 text에서')
    .replace(/\banswer(?=는|의|와|에는)/g,'각 답변 text')+'\n'+SUFFICIENCY_PROMPT;
}

export function missingEvidenceReply(usage, {ambiguous=false}={}) {
  return {kind:'uncertain',text:ambiguous?'어떤 인물이나 사건을 말씀하시는지 조금 더 구체적으로 알려 주세요.':messages.passage_missing,
    citations:[],contextSummary:'',outOfScope:false,needsClarification:true,usage};
}

// Verifies structure, part coverage, literal support and reference membership.
// The model's semantic status is still fallible; this is NOT an entailment proof.
export function validateSufficientAnswers(raw, parts, evidenceByPart, citations=[], {corpus,referenceEvidenceByPart=evidenceByPart}={}) {
  const invalid=()=>{throw new BibleProviderError('AI_EVIDENCE_ASSESSMENT_INVALID');};
  if(!raw||typeof raw!=='object'||Array.isArray(raw)||Object.keys(raw).length!==parts.length)invalid();
  const citationIds=new Set(citations.map(c=>`${c.bookId}:${c.chapter}:${c.verse}`));
  const confirmed=[],blocks=[],checks=[];
  for(const part of parts){
    const row=raw[part.partId],allowed=new Map((evidenceByPart.get(part.partId)||[]).map(v=>[v.id,v]));
    if(!row||Object.keys(row).sort().join(',')!=='gap,status,supports,text'||!statuses.includes(row.status)||!gaps.includes(row.gap)||typeof row.text!=='string'||!Array.isArray(row.supports)||row.supports.length>2)invalid();
    const missing=row.status==='insufficient';
    if((row.status==='supported')!==(row.gap==='none'))invalid();
    if(missing){if(row.text.trim()||row.supports.length)invalid();}
    else {
      if(!row.text.trim()||!row.supports.length)invalid();
      for(const support of row.supports){
        if(!support||Object.keys(support).sort().join(',')!=='quote,verseId'||typeof support.quote!=='string'||support.quote.length<4||support.quote.length>160
          ||!allowed.get(support.verseId)?.text.includes(support.quote)||/^[\s(（]*없음[)）\s]*$/.test(allowed.get(support.verseId)?.text||''))invalid();
      }
      if(!row.supports.some(s=>citationIds.has(s.verseId)))throw new BibleProviderError('AI_CITATION_INVALID');
      if(corpus&&!inlineReferencesInEvidence(row.text,corpus,referenceEvidenceByPart.get(part.partId)||[]))throw new BibleProviderError('AI_CITATION_INVALID');
      confirmed.push(part.partId);
    }
    const limitation=row.gap==='none'?'':messages[row.gap];
    // The model's content headings carry meaning; internal P numbers do not.
    // Keep missing parts visible even when their generated text is empty.
    blocks.push([formatAnswerPart(row.text,{continuation:blocks.length>0}),limitation].filter(Boolean).join('\n\n'));
    checks.push({partId:part.partId,status:row.status,gap:row.gap,supportVerseIds:row.supports.map(s=>s.verseId)});
  }
  return {text:blocks.join('\n\n---\n\n'),confirmedParts:confirmed,checks,
    allMissing:confirmed.length===0,partial:checks.some(c=>c.status!=='supported')};
}

export function buildPartEvidence(parts, coverage, primary, sectionContext=[]) {
  return new Map(parts.map(part=>{
    const selected=coverage.find(c=>c.partId===part.partId);
    if(!selected)return [part.partId,primary]; // Explicit passage/provider without part selection.
    const ids=new Set(selected.chunkIds);
    const allowed=new Set();
    for(const verse of primary){
      for(const id of ids){
        const m=id.match(/^([^:]+):(\d+):(\d+)-(\d+)$/);
        if(m&&verse.bookId===m[1]&&verse.chapter===Number(m[2])&&verse.verse>=Number(m[3])&&verse.verse<=Number(m[4]))allowed.add(verse.id);
      }
    }
    for(const section of sectionContext){
      if(section.primaryVerseIds.some(id=>allowed.has(id)))section.contextVerseIds.forEach(id=>allowed.add(id));
    }
    return [part.partId,primary.filter(v=>allowed.has(v.id))];
  }));
}
