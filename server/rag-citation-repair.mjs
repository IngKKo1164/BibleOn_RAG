import {validateSufficientAnswers,sufficientAnswerSchema} from './bible-sufficiency.mjs';
import {BibleProviderError} from './bible-errors.mjs';
const object=properties=>({type:'object',additionalProperties:false,properties,required:Object.keys(properties)});
export const CITATION_REPAIR_PROMPT='이번 요청은 장절 배정 교정 1회다. answers와 citations만 반환한다. 해당 P의 allowedReferences 밖 장절이 기존 text에 섞였다. 이 목록의 연속 범위 안에서만 장절을 쓰고 빠진 절 번호를 범위로 잇지 않는다. 허용된 실제 본문으로 설명·깊이·##/### 구조를 보존하고 근거 없는 주장은 삭제하거나 한계를 밝힌다. answerOutline은 중복 방지용 구성이지 사실 근거가 아니다. 전체 결론은 P1만 쓰고 이후 P는 ## 제목으로 시작한다. 다른 P에서 이미 다룬 결론을 반복하거나 그 P에 배정된 본문이 자료 전체에 없다고 말하지 않는다. 요청하지 않은 역사 배경의 부족 안내는 만들지 않는다. 다른 P의 본문을 빌리지 않고 supports 중 하나 이상을 citations에도 넣는다.';

export function allowedReferenceRanges(verses,corpus){
  const chapters=new Map();
  for(const v of verses){const key=`${v.bookId}:${v.chapter}`;if(!chapters.has(key))chapters.set(key,{bookId:v.bookId,chapter:v.chapter,verses:new Set()});chapters.get(key).verses.add(v.verse);}
  return [...chapters.values()].map(c=>{
    const numbers=[...c.verses].sort((a,b)=>a-b),runs=[];
    for(const n of numbers){const last=runs.at(-1);if(last&&last[1]+1===n)last[1]=n;else runs.push([n,n]);}
    return `${corpus[c.bookId].name} ${c.chapter}:${runs.map(([a,b])=>a===b?a:`${a}-${b}`).join(',')}`;
  });
}

// Only deterministic per-part citation failures trigger this one bounded
// correction. No new search, no self-approval, no relaxation of evidence rules.
export function invalidCitationParts({draft,parts,partEvidence,referenceEvidenceByPart,corpus}){
  const invalid=[];
  for(const p of parts){
    try{validateSufficientAnswers({[p.partId]:draft.answers?.[p.partId]},[p],partEvidence,draft.citations,{corpus,referenceEvidenceByPart});}
    catch(error){if(error.code!=='AI_CITATION_INVALID')throw error;invalid.push(p);}
  }
  return invalid;
}

export async function repairPartCitations({draft,parts,partEvidence,referenceEvidenceByPart,corpus,system,call,observe,visibleTokens}){
  const invalid=invalidCitationParts({draft,parts,partEvidence,referenceEvidenceByPart,corpus});
  if(!invalid.length)return {draft,visibleTokens};
  observe({stage:'citation-repair',result:{parts:invalid.map(p=>p.partId),attempt:1}});
  const refs=[...new Map(invalid.flatMap(p=>referenceEvidenceByPart.get(p.partId)||[]).map(v=>[v.id,v])).values()];
  const groups=new Map();
  for(const v of refs){const k=`${v.bookId}:${v.chapter}`;if(!groups.has(k))groups.set(k,{bookId:v.bookId,chapter:v.chapter,verses:[]});groups.get(k).verses.push(v.verse);}
  const choices=[...groups.values()].map(g=>object({bookId:{type:'string',enum:[g.bookId]},chapter:{type:'integer',enum:[g.chapter]},verse:{type:'integer',enum:g.verses}}));
  const schema=object({answers:sufficientAnswerSchema(invalid,partEvidence),citations:{type:'array',maxItems:8,items:choices.length===1?choices[0]:{anyOf:choices}}});
  const input={questionParts:invalid.map(p=>({...p,previousText:draft.answers[p.partId].text})),
    answerOutline:parts.map(p=>({partId:p.partId,question:p.question,
      headings:[...(draft.answers[p.partId]?.text||'').matchAll(/^#{2,3} (.+)$/gm)].map(m=>m[1]).slice(0,8)})),
    evidence:refs.map(v=>({id:v.id,reference:v.reference,text:v.text})),
    partEvidenceIds:Object.fromEntries(invalid.map(p=>[p.partId,partEvidence.get(p.partId).map(v=>v.id)])),
    allowedReferences:Object.fromEntries(invalid.map(p=>[p.partId,allowedReferenceRanges(referenceEvidenceByPart.get(p.partId),corpus)]))};
  const fixed=await call('answer',system+'\n'+CITATION_REPAIR_PROMPT,input,schema);
  // Counting both drafts is conservative, and prevents repair from enlarging
  // the owner's 3,000-token visible-output ceiling.
  if(visibleTokens+fixed.visibleTokens>3000)throw new BibleProviderError('AI_INCOMPLETE');
  validateSufficientAnswers(fixed.result.answers,invalid,partEvidence,fixed.result.citations,{corpus,referenceEvidenceByPart});
  const answers={...draft.answers,...fixed.result.answers};
  const supportIds=new Set(Object.values(answers).flatMap(a=>a.supports.map(s=>s.verseId)));
  const references=new Map([...draft.citations,...fixed.result.citations].map(c=>[`${c.bookId}:${c.chapter}:${c.verse}`,c]));
  const citations=[...references].sort(([a],[b])=>Number(supportIds.has(b))-Number(supportIds.has(a))).slice(0,8).map(([,c])=>c);
  const merged={...draft,answers,citations};
  validateSufficientAnswers(answers,parts,partEvidence,citations,{corpus,referenceEvidenceByPart});
  return {draft:merged,visibleTokens:visibleTokens+fixed.visibleTokens};
}
