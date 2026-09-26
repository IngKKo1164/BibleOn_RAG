import {retrieveBroadBible, discoverCandidates, DISCOVERY_LIMITS} from './rag-discovery.mjs';
import {RERANK_PROMPT, rerankingSchema, validateReranking} from './rag-reranking.mjs';
import {focusedEvidence} from './rag-focused-evidence.mjs';
import {expandOneHop} from './bible-study.mjs';
import {candidateWire, screeningSchema, decodeScreening, decodeReranking, SCREEN_PROMPT, SCREEN_LIMITS} from './rag-candidate-wire.mjs';
import {budgetCandidatePool} from './rag-input-budget.mjs';
import {embedTexts} from './rag-embedding.mjs';
import {repairEnabled,REPAIR_PROMPT,REPAIR_LIMITS,assessRepair,repairQueries,recoveryRerankPool} from './rag-adaptive-repair.mjs';


export async function retrieveCandidates(state, context) {
  const {corpus,question,previousEvidence,passage,env,fetcher,signal,retriever,observe,studyIndex}=context;
  let {retrieval}=state;
  retrieval=await retrieveBroadBible({corpus,question,previousEvidence,passage,env,fetcher,signal,retriever,studyIndex});
  const reportPool=()=>observe({stage:'retrieve',result:{candidateCount:retrieval.hits.length,
    candidateIds:retrieval.hits.map(h=>h.chunk.id),characters:retrieval.candidates.reduce((n,c)=>n+c.excerpt.length,0),
    degraded:retrieval.degraded,queryVariants:1},usage:{embeddingInputTokens:retrieval.inputTokens}});
  reportPool();
  return {retrieval};
}

export async function budgetCandidates(state, context) {
  const {question,questionParts,observe}=context;
  let {retrieval,screenedPool}=state;
  screenedPool=budgetCandidatePool(retrieval,question,questionParts);
  observe({stage:'candidate-budget',result:{broad:retrieval.hits.length,firstPass:screenedPool.hits.length,deferred:retrieval.hits.length-screenedPool.hits.length}});
  return {screenedPool,needsScreening:screenedPool.candidates.reduce((n,c)=>n+c.excerpt.length,0)>SCREEN_LIMITS.thresholdCharacters};
}

export async function screenCandidates(state, context) {
  const {question,questionParts,previousEvidence,call,observe}=context;
  let {retrieval,screenedPool}=state;
    const wire=candidateWire(screenedPool,{preview:true,question:retrieval.query||question});
    const {result}=await call('screen',SCREEN_PROMPT,{question,questionParts,candidates:wire.candidates,
      ...(previousEvidence?{previousQuestion:previousEvidence.question}:{})},screeningSchema(questionParts,wire.ids));
    const ids=decodeScreening(result,questionParts,wire),keep=new Set(ids);
    // Empty screening is not proof of unanswerability. Inspect the broad pool.
    screenedPool=ids.length?{...retrieval,hits:retrieval.hits.filter(h=>keep.has(h.chunk.id)),candidates:retrieval.candidates.filter(c=>keep.has(c.id))}:retrieval;
    observe({stage:'screening',result:{candidateIds:screenedPool.hits.map(h=>h.chunk.id),candidateCount:screenedPool.hits.length,
      previewCharacters:wire.candidates.reduce((n,c)=>n+c[3].length,0),fallback:!ids.length}});
  return {screenedPool};
}

async function rerankPool(pool,recovery,context){
  const {question,questionParts,previousEvidence,call,env}=context;
    const repair=repairEnabled(env);
    const wire=candidateWire(pool);
    const {result}=await call('rerank',RERANK_PROMPT+(repair?'\n'+REPAIR_PROMPT:''),{question,questionParts,candidates:wire.candidates,
      ...(recovery?{missingParts:recovery.missingParts.map(p=>p.partId),previouslySelected:recovery.coverage.map(p=>({partId:p.partId,chunkIds:p.chunkIds.map(id=>wire.aliases.get(id)).filter(Boolean)})),
        ...(repair?{unresolvedRequirements:recovery.unresolved?.map(({partId,need,status})=>({partId,need,status}))}:{})}:{}),
      ...(previousEvidence?{previousQuestion:previousEvidence.question}: {})},rerankingSchema(questionParts,wire.ids,{repair}));
    const decoded=decodeReranking(result,wire),plan=validateReranking(decoded,questionParts,pool);
    return repair?assessRepair(plan,decoded.requirements,questionParts,pool):plan;
}


export async function rerankCandidates(state, context) {
  let {retrieval,screenedPool,plan,recoveryCount=0}=state;
  plan=await rerankPool(screenedPool,null,context);
  return {plan,recoveryCount:0,needsRecovery:((plan.route==='bible'||(repairEnabled(context.env)&&plan.route==='ambiguous'))&&plan.missingParts.length>0)||(screenedPool!==retrieval&&['ambiguous','outOfScope'].includes(plan.route))};
}

export async function recoverCandidates(state, context) {
  const {corpus,question,questionParts,observe,studyIndex,env,fetcher}=context;
  const signal=context.deadline||context.signal;
  let {retrieval,plan,recoveryCount=0}=state;
    const repair=repairEnabled(env);
    if(plan.route!=='bible')plan={...plan,missingParts:questionParts};
    recoveryCount=1;
    const additional=[],branches=[];
    const queries=repair?repairQueries(plan,questionParts,{expand:env?.BIBLE_REPAIR_QUERY_EXPANSION!=='off'}):plan.missingParts.map(p=>({query:p.question,partId:p.partId,expanded:false}));
    let vectors=[],embeddingInputTokens=0,embeddingCalls=0,embeddingDegraded=false;
    const expanded=queries.filter(q=>q.expanded);
    if(expanded.length){
      signal?.throwIfAborted();embeddingCalls=1;
      try{
        const response=await embedTexts(expanded.map(q=>q.query),{env,fetcher,signal,type:'query'});
        vectors=response.vectors;embeddingInputTokens=response.inputTokens;
      }catch(error){if(signal?.aborted)throw error;embeddingDegraded=true;}
    }
    for(const q of queries){
      signal?.throwIfAborted();
      const vector=q.expanded?vectors[expanded.indexOf(q)]:undefined;
      const more=discoverCandidates(retrieval.index,corpus,q.query,{retry:true,vector,scopeQuestion:retrieval.query||question});
      const rows=more.hits.slice(0,24);branches.push(rows);additional.push(...rows);
    }
    // One-hop from already selected, model-identified anchors only. Never
    // recursively expand linked verses or use the whole lexical top hit as fact.
    if(studyIndex&&plan.chunkIds.length){
      const primary=focusedEvidence(corpus,retrieval,plan,[],{evidenceCharacters:6000,evidenceVerses:96}).evidence;
      const linked=expandOneHop(corpus,primary,question,studyIndex).linked;
      const neighbors=[];
      for(const v of linked){const chunk=retrieval.index.chunks.find(c=>c.bookId===v.bookId&&c.chapter===v.chapter&&c.start<=v.verse&&c.end>=v.verse);if(chunk)neighbors.push({chunk,score:0,paths:[{path:'one-hop-recovery',rank:1}]});}
      branches.push(neighbors);additional.push(...neighbors);
    }
    const known=new Set(retrieval.hits.map(h=>h.chunk.id)),added=[];
    const cap=DISCOVERY_LIMITS.candidates+(repair?REPAIR_LIMITS.newCandidates:0);
    for(const h of additional){if(known.has(h.chunk.id))continue;if(added.length>=REPAIR_LIMITS.newCandidates||known.size>=cap)break;known.add(h.chunk.id);added.push(h);}
    retrieval={...retrieval,hits:[...retrieval.hits,...added],candidates:[...retrieval.candidates,...added.map(h=>({id:h.chunk.id,reference:h.chunk.reference,complete:true,excerpt:h.chunk.verses.map(v=>`${v.verse} ${v.text}`).join('\n')}))],
      inputTokens:retrieval.inputTokens+embeddingInputTokens,embeddingCalls:(retrieval.embeddingCalls||1)+embeddingCalls,degraded:retrieval.degraded||embeddingDegraded};
    if(retrieval.candidates.reduce((n,c)=>n+c.excerpt.length,0)>DISCOVERY_LIMITS.characters)throw Error('DISCOVERY_CHARACTER_LIMIT');
    const recoveryPool=repair?recoveryRerankPool(retrieval,plan,branches):retrieval;
    observe({stage:'retrieval-recovery',result:{attempt:1,method:repair?'repair-inspired-v1':'legacy',missingParts:plan.missingParts.map(p=>p.partId),
      targets:queries.map(({partId,need,expanded})=>({partId,need,expanded})),added:added.length,candidateIds:retrieval.hits.map(h=>h.chunk.id),
      rerankIds:recoveryPool.hits.map(h=>h.chunk.id),embeddingCalls,embeddingDegraded},usage:{embeddingInputTokens}});
  return {retrieval,plan,recoveryCount,recoveryPool};
}

export async function rerankRecovered(state, context) {
  const {questionParts}=context;
  let {retrieval,plan}=state;
  const recovered=await rerankPool(state.recoveryPool||retrieval,plan,context);
    if(repairEnabled(context.env))return {plan:recovered};
    if(recovered.route==='bible'){
      // Preserve coverage already found if the second selection overlooks it.
      const coverage=recovered.coverage.map(p=>p.chunkIds.length?p:plan.coverage.find(c=>c.partId===p.partId));
      const chosen=new Set();
      for(let i=0;i<3;i++)for(const part of coverage){const id=part.chunkIds[i];if(id&&chosen.size<6)chosen.add(id);}
      const ids=[...chosen],focus=ids.map(chunkId=>({chunkId,verses:recovered.focus.get(chunkId)||plan.focus.get(chunkId)}));
      plan=validateReranking({route:'bible',rankedIds:[...new Set([...ids,...recovered.rankedIds,...plan.rankedIds])].slice(0,12),
        evidenceByPart:Object.fromEntries(coverage.map(p=>[p.partId,p.chunkIds.filter(id=>chosen.has(id))])),focus,
        studyTerms:[...new Set([...recovered.studyTerms,...plan.studyTerms])].slice(0,3)},questionParts,retrieval);
    }else plan=recovered;
  return {plan};
}

export async function selectEvidence(state, context) {
  const {corpus,passage,observe}=context;
  let {retrieval,plan,recoveryCount=0}=state;
  observe({stage:'selection',result:{chunkIds:plan.chunkIds,rankedIds:plan.rankedIds,parts:plan.coverage.length,
    missingParts:plan.missingParts.map(p=>p.partId),recoveryCount,
    ...(plan.requirements?{requirements:plan.requirements.map(({partId,need,status,importance})=>({partId,need,status,importance}))}:{})}});
  if(plan.route!=='bible'||!plan.chunkIds.length)return {selected:{plan,retrieval,recoveryCount}};
  const selected=focusedEvidence(corpus,retrieval,plan,passage);
  observe({stage:'evidence',result:{verses:selected.evidence.length,anchorIds:selected.anchorIds,
    retainedVerseIds:selected.evidence.map(v=>v.id),incompleteSections:selected.sections.filter(s=>!s.complete).length}});
  return {selected:{plan,retrieval,...selected,recoveryCount}};
}
