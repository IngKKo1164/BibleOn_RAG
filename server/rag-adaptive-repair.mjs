import {BibleProviderError} from './bible-errors.mjs';

// REPAIR-inspired bounded adaptation, NOT the paper's trained PSR, dense
// Bradley–Terry reward or sliding-window SAR reproduction. No hidden CoT.
export const REPAIR_LIMITS=Object.freeze({requirements:6,queries:2,newCandidates:48,rerankCandidates:48});
export const repairEnabled=env=>env?.BIBLE_RETRIEVAL_REPAIR!=='off';
const object=properties=>({type:'object',additionalProperties:false,properties,required:Object.keys(properties)});
export const REPAIR_PROMPT=`추가 점검: 본문 주소나 인물명을 모르는 질문도 정상적인 검색 질문이다. 이름이 없거나 현재 후보가 약하다는 이유로 outOfScope로 분류하지 않는다. 구체적인 사건 단서가 있으면 bible로 두고 부족한 근거를 표시한다.
requirements는 답에 필요한 검증 가능한 조건 목록이다(총 6개 이하, bible/ambiguous는 각 P 최소 1개). 숨겨진 사고과정이나 답변을 쓰지 않는다. need에는 질문의 행동·상황·결과·부정 중 구별에 중요한 조건을 간결히 적는다. 인물/사건 식별과 비교 양쪽은 essential, 부수적인 배경은 supplementary다. 사건을 찾아야 한다면 이름만 일치하는 것으로 만족하지 말고 식별 조건을 먼저 확인한다.
status: supported=조건이 선택 본문으로 확인됨, refuted=선택 본문이 질문의 잘못된 전제를 직접 반박함, missing=확인 근거 없음, conflict=후보 사건을 아직 구별 못함. 틀린 전제를 바로잡을 직접 근거가 있으면 refuted이지 conflict가 아니다. supported/refuted에는 선택한 핵심 절의 짧고 연속된 원문 인용을 anchors에 넣는다. 다른 사건의 비슷한 낱말만으로 충족 처리하지 않는다. 근거 없는 나이·동기·역사 배경은 missing이다. missing/conflict의 anchors는 []다.
미충족 조건의 query에는 원 질문을 보완할 짧은 검색 단서를 적는다. 일상어를 본문의 표현/행동 관계로 바꾸되, 부정·인과·질문의 의도를 유지한다. 추측한 인명·지명은 검색 가설일 뿐이다. 본문에 확인 안 된 장절 좌표나 가상의 본문 문장은 만들지 않는다. supported/refuted의 query는 빈 문자열이다. 앞선 requirements나 검색 가설은 사실이 아니며 새 본문으로 다시 판단한다. 이전 선택을 무조건 보존하지 않는다.`;

export function repairRequirementsSchema(parts,id){
  return {type:'array',maxItems:REPAIR_LIMITS.requirements,items:object({
    partId:{type:'string',enum:parts.map(p=>p.partId)},
    need:{type:'string',minLength:1,maxLength:120},
    importance:{type:'string',enum:['essential','supplementary']},
    status:{type:'string',enum:['supported','refuted','missing','conflict']},
    anchors:{type:'array',maxItems:2,items:object({chunkId:id,verse:{type:'integer',minimum:1},quote:{type:'string',minLength:2,maxLength:120}})},
    query:{type:'string',maxLength:160},
  })};
}

export function assessRepair(plan,requirements,parts,retrieval){
  const invalid=()=>{throw new BibleProviderError('AI_REPAIR_ASSESSMENT_INVALID');};
  if(!Array.isArray(requirements)||requirements.length>REPAIR_LIMITS.requirements)invalid();
  const chunks=new Map(retrieval.hits.map(h=>[h.chunk.id,h.chunk]));
  const seen=new Set();
  for(const r of requirements){
    if(!r||!parts.some(p=>p.partId===r.partId)||typeof r.need!=='string'||!r.need.trim()||r.need.length>120||
      !['essential','supplementary'].includes(r.importance)||!['supported','refuted','missing','conflict'].includes(r.status)||
      typeof r.query!=='string'||r.query.length>160||!Array.isArray(r.anchors)||r.anchors.length>2)invalid();
    const key=`${r.partId}:${r.need.trim()}`;if(seen.has(key))invalid();seen.add(key);
    const supported=['supported','refuted'].includes(r.status);
    if(supported?(!r.anchors.length||r.query!==''):(r.anchors.length!==0))invalid();
    for(const a of r.anchors){
      const text=chunks.get(a?.chunkId)?.verses.find(v=>v.verse===a.verse)?.text;
      if(!plan.coverage.find(p=>p.partId===r.partId)?.chunkIds.includes(a?.chunkId)||!plan.focus.get(a.chunkId)?.includes(a.verse)||
        typeof a.quote!=='string'||a.quote.trim().length<2||a.quote.length>120||!text?.includes(a.quote))invalid();
    }
  }
  if(['bible','ambiguous'].includes(plan.route)&&parts.some(p=>!requirements.some(r=>r.partId===p.partId)))invalid();
  if(!['bible','ambiguous'].includes(plan.route)&&requirements.length)invalid();
  const unresolved=requirements.filter(r=>['missing','conflict'].includes(r.status));
  // A ranked chunk alone no longer proves coverage. Do not propagate a wrong
  // event when its identifying conditions are unresolved. Partial background
  // may stay unresolved while a genuinely identified event remains usable.
  const coverage=plan.coverage.map(p=>{
    const rs=requirements.filter(r=>r.partId===p.partId);
    const blocked=rs.some(r=>r.importance==='essential'&&['missing','conflict'].includes(r.status));
    const accepted=new Set(rs.flatMap(r=>r.anchors.map(a=>a.chunkId)));
    return {...p,chunkIds:blocked?[]:p.chunkIds.filter(id=>accepted.has(id))};
  });
  const chunkIds=[...new Set(coverage.flatMap(p=>p.chunkIds))];
  return {...plan,coverage,chunkIds,focus:new Map([...plan.focus].filter(([id])=>chunkIds.includes(id))),
    requirements,unresolved,missingParts:coverage.filter(p=>!p.chunkIds.length||unresolved.some(r=>r.partId===p.partId))};
}

export function repairQueries(plan,parts,{expand=true}={}){
  const weight=r=>(r.importance==='essential'?4:0)+(r.status==='conflict'?2:1);
  const targets=[...(plan.unresolved||[])].sort((a,b)=>weight(b)-weight(a));
  const seen=new Set(),queries=[];
  for(const r of targets){
    const part=parts.find(p=>p.partId===r.partId);
    if(!part)continue;
    // Original language always remains in the query. A model hypothesis is
    // used only as a search hint, never as a citation or a forced book scope.
    const hint=expand?r.query.trim():'';
    const query=hint?`${part.question}\n검색 단서: ${hint}`:part.question;
    if(seen.has(query))continue;seen.add(query);
    queries.push({partId:r.partId,need:r.need,query,expanded:!!hint});
    if(queries.length===REPAIR_LIMITS.queries)break;
  }
  if(!queries.length)for(const part of plan.missingParts.slice(0,REPAIR_LIMITS.queries))queries.push({partId:part.partId,query:part.question,expanded:false});
  return queries;
}

export function recoveryRerankPool(retrieval,plan,branches){
  const byId=new Map(retrieval.hits.map(h=>[h.chunk.id,h])),keep=new Set();
  const add=id=>{if(byId.has(id)&&keep.size<REPAIR_LIMITS.rerankCandidates)keep.add(id);};
  plan.chunkIds.forEach(add);
  // Reserve space for both original candidates and exploratory query branches.
  plan.rankedIds.slice(0,12).forEach(add);
  for(let i=0;i<12;i++)for(const rows of branches)add(rows[i]?.chunk.id);
  retrieval.hits.forEach(h=>add(h.chunk.id));
  return {...retrieval,hits:[...keep].map(id=>byId.get(id)),candidates:retrieval.candidates.filter(c=>keep.has(c.id))};
}
