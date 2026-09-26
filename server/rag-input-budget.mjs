// A smaller first pass, not a smaller corpus. The complete retrieved union is
// retained locally for one recovery. No labels or known answer IDs are used.
export const INPUT_BUDGETS=Object.freeze({standard:128,complex:160});
export function firstPassLimit(question,parts=[]){
  return parts.length>1||/비교|차이|관계|모순|각각|원어|어원|배경|역사/.test(question)
    ?INPUT_BUDGETS.complex:INPUT_BUDGETS.standard;
}
export function budgetCandidatePool(pool,question,parts=[]){
  const limit=firstPassLimit(question,parts);
  if(pool.hits.length<=limit)return pool;
  const keep=new Set(pool.hits.slice(0,limit).map(h=>h.chunk.id));
  // User-selected/explicit references must never disappear behind a budget.
  for(const h of pool.hits)if(h.paths?.some(p=>p.path==='explicit'))keep.add(h.chunk.id);
  return {...pool,hits:pool.hits.filter(h=>keep.has(h.chunk.id)),candidates:pool.candidates.filter(c=>keep.has(c.id))};
}

export const COMPACT_EVIDENCE_PROMPT='전송 형식: evidence의 각 행은 [verseId,정본 원문]이다. verseId는 bookId:장:절이며 bookId의 한국어 이름은 위 목록을 따른다. 원문과 각 질문별 허용 근거는 생략하지 않았다. sectionContext는 제공한 본문의 원래 소제목 범위와 누락 상태다.';

export function compactAnswerInput(input){
  if(!Array.isArray(input.evidence)||JSON.stringify(input).length<3200)return {input,compact:false};
  if(!input.evidence.every(v=>typeof v.id==='string'&&typeof v.text==='string'))return {input,compact:false};
  const result={...input,evidence:input.evidence.map(v=>[v.id,v.text])};
  // Same questions and evidence membership are already in the two canonical
  // fields. These IDs are internal selection bookkeeping, not source text.
  if(input.questionParts&&input.partEvidenceIds)delete result.questionCoverage;
  if(input.sectionContext)result.sectionContext=input.sectionContext.map(({bookId,chapter,start,end,complete,omittedVerses})=>({bookId,chapter,start,end,complete,omittedVerses}));
  for(const key of ['history','sectionContext','crossReferences','studySources','contextVerseIds'])if(Array.isArray(result[key])&&!result[key].length)delete result[key];
  if(result.previousEvidence===null)delete result.previousEvidence;
  return {input:result,compact:true};
}
