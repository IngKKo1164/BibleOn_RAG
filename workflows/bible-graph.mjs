import {Annotation,StateGraph,START,END} from '@langchain/langgraph';
import {getCurrentRunTree} from 'langsmith/singletons/traceable';
import {createBibleRuntime} from '../server/rag-runtime.mjs';
import {parseBibleReferences} from '../server/bible-rag.mjs';
import {unresolvedReferent} from '../server/rag-answer-safety.mjs';
import {fallback} from '../server/bible-answer-guards.mjs';
import {missingEvidenceReply} from '../server/bible-sufficiency.mjs';
import {loadBibleStudy,expandOneHop,retrieveStudySources} from '../server/bible-study.mjs';
import {retrieveCandidates,budgetCandidates,screenCandidates,rerankCandidates,recoverCandidates,rerankRecovered,selectEvidence} from '../server/rag-selection-steps.mjs';
import {prepareAnswer,generateAnswer,checkCitations,repairCitations,assessAnswer,checkFacts,repairFacts,validateAnswer} from '../server/rag-answer-steps.mjs';

// Last-value channels, not an ever-growing message log. No checkpoint/store/
// cache or automatic node retries. Credentials and corpus stay in the closure.
const stateKeys=[
  'lane','retrieval','screenedPool','recoveryPool','plan','recoveryCount','needsScreening','needsRecovery',
  'evidence','primaryEvidence','sectionContext','selectionCoverage','studyTerms',
  'studySources','crossReferences','studyLoaded','answerSchema','partEvidence','system',
  'languageSupport','answerInput','draft','visibleTokens','referenceEvidenceByPart',
  'needsCitationRepair','confirmedContext','conflicts','usage','result',
];
const State=Annotation.Root(Object.fromEntries(stateKeys.map(key=>[key,Annotation()])));
const Output=Annotation.Root({result:Annotation()});
export const GRAPH_RECURSION_LIMIT=32;

function assertLocalTracing(){
  // Fail before graph construction. An async-local false flag alone does not
  // prevent every upstream client /info probe with inherited tracing settings.
  const flags=['LANGSMITH_TRACING','LANGSMITH_TRACING_V2','LANGCHAIN_TRACING','LANGCHAIN_TRACING_V2','LANGCHAIN_VERBOSE'];
  if(flags.some(key=>process.env[key]&&!/^(false|0)$/i.test(process.env[key]))||
    process.env.LANGSMITH_API_KEY||process.env.LANGCHAIN_API_KEY||getCurrentRunTree(true)){
    throw Error('LANGGRAPH_EXTERNAL_TRACING_DISABLED: remove inherited LangSmith/LangChain tracing, verbose flags and keys for this private research run.');
  }
}

function routeRequest(state,context){
  const {corpus,question,previousEvidence,passage,scope,usage}=context;
  if(unresolvedReferent(question,previousEvidence,passage))return {result:fallback('uncertain',usage)};
  const direct=scope.lookup?{...scope,exact:true,refs:scope.primary}:parseBibleReferences(corpus,question);
  if(direct.exact&&direct.invalid)return {result:fallback('uncertain',usage)};
  if(direct.exact&&direct.refs.length&&(scope.lookup||(direct.refs.length<=8&&direct.refs.reduce((n,v)=>n+v.text.length,0)<=1800))){
    return {result:{kind:'answer',text:direct.refs.map(v=>`**${v.reference}**\n\n${v.text}`).join('\n\n'),
      citations:direct.refs,contextSummary:'',outOfScope:false,needsClarification:false,usage}};
  }
  context.getStudyIndex();
  return scope.direct?{lane:'direct',primaryEvidence:scope.primary,evidence:[...scope.primary,...scope.context]}:{lane:'retrieve'};
}

async function retrieve(state,context){
  context.observe({stage:'retrieve',event:'start'});
  const update=await retrieveCandidates(state,{...context,signal:context.deadline,studyIndex:await context.getStudyIndex()});
  context.usage.embeddingInputTokens=update.retrieval.inputTokens;
  context.usage.embeddingCalls=update.retrieval.embeddingCalls;
  return update;
}

async function recover(state,context){
  const update=await recoverCandidates(state,{...context,studyIndex:await context.getStudyIndex()});
  context.usage.embeddingInputTokens=update.retrieval.inputTokens;
  context.usage.embeddingCalls=update.retrieval.embeddingCalls;
  return update;
}

async function evidence(state,context){
  const {selected}=await selectEvidence(state,context);
  if(selected.plan.route!=='bible')return {result:fallback(selected.plan.route==='ambiguous'?'uncertain':selected.plan.route,context.usage)};
  if(!selected.evidence?.length)return {result:missingEvidenceReply(context.usage)};
  return {evidence:selected.evidence,primaryEvidence:selected.evidence,sectionContext:selected.sections,
    selectionCoverage:selected.plan.coverage,studyTerms:selected.plan.studyTerms};
}

async function enrich(state,context){
  const {corpus,question,scope,studyEnabled,observe,deadline}=context;
  if(!studyEnabled)return {};
  observe({stage:'study-retrieve',event:'start'});
  try{
    const index=await context.getStudyIndex();
    deadline.throwIfAborted();
    if(!index)throw Error('STUDY_UNAVAILABLE');
    const expanded=scope.crossrefs?expandOneHop(corpus,state.primaryEvidence,question,index):{roots:[],linked:[],relations:[]};
    const studySources=scope.sources?retrieveStudySources(state.primaryEvidence,state.studyTerms||[],question,index,{independent:true}):[];
    const evidence=[...new Map([...state.evidence,...expanded.linked].map(v=>[v.id,v])).values()];
    observe({stage:'study-retrieve',result:{primaryVerses:state.primaryEvidence.length,roots:expanded.roots,
      linkedVerses:expanded.linked.map(v=>v.id),relations:expanded.relations,sources:studySources}});
    return {evidence,crossReferences:expanded.relations,studySources,studyLoaded:true};
  }catch{
    deadline.throwIfAborted();
    observe({stage:'study-unavailable',result:{degraded:true}});
    observe({stage:'study-retrieve'});
    return {};
  }
}

// Execution and local diagram export consume this single topology definition.
export function buildBibleGraph(context){
  assertLocalTracing();
  const graph=new StateGraph({stateSchema:State,output:Output});
  const nodes={route:routeRequest,retrieve,budget:budgetCandidates,screen:screenCandidates,
    rerank:rerankCandidates,recover,rerank_recovered:rerankRecovered,select_evidence:evidence,enrich,
    prepare:prepareAnswer,generate:generateAnswer,check_citations:checkCitations,
    repair_citations:repairCitations,assess:assessAnswer,check_facts:checkFacts,
    repair_facts:repairFacts,validate:validateAnswer};
  for(const [node,operation] of Object.entries(nodes)){
    graph.addNode(node,async state=>{
      context.deadline.throwIfAborted();
      const started=performance.now();
      context.observe({stage:'graph',node,event:'start'});
      try{
        const update=await operation(state,context);
        context.deadline.throwIfAborted();
        context.observe({stage:'graph',node,event:'end',milliseconds:performance.now()-started});
        return {...update,usage:{...context.usage}};
      }catch(error){
        context.observe({stage:'graph',node,event:'error',code:error.code||error.name});
        throw error;
      }
    },{retryPolicy:{maxAttempts:1}});
  }
  graph.addEdge(START,'route');
  graph.addConditionalEdges('route',s=>s.result?END:s.lane==='direct'?'enrich':'retrieve',[END,'enrich','retrieve']);
  graph.addEdge('retrieve','budget');
  graph.addConditionalEdges('budget',s=>s.needsScreening?'screen':'rerank',['screen','rerank']);
  graph.addEdge('screen','rerank');
  graph.addConditionalEdges('rerank',s=>s.needsRecovery?'recover':'select_evidence',['recover','select_evidence']);
  graph.addEdge('recover','rerank_recovered');
  graph.addEdge('rerank_recovered','select_evidence');
  graph.addConditionalEdges('select_evidence',s=>s.result?END:'enrich',[END,'enrich']);
  graph.addEdge('enrich','prepare');
  graph.addEdge('prepare','generate');
  graph.addEdge('generate','check_citations');
  graph.addConditionalEdges('check_citations',s=>s.result?END:s.needsCitationRepair?'repair_citations':'assess',[END,'repair_citations','assess']);
  graph.addEdge('repair_citations','assess');
  graph.addConditionalEdges('assess',s=>s.result?END:'check_facts',[END,'check_facts']);
  graph.addConditionalEdges('check_facts',s=>s.conflicts.length?'repair_facts':'validate',['repair_facts','validate']);
  graph.addEdge('repair_facts','validate');
  graph.addEdge('validate',END);
  return graph.compile();
}

export async function runBibleGraph(options){
  if(options.env.BIBLE_AI_PROVIDER!=='openai'||options.env.BIBLE_RETRIEVAL_VERSION!=='v4')throw Error('LANGGRAPH_REQUIRES_RESEARCH_V4');
  assertLocalTracing();
  const context=createBibleRuntime(options);
  let pendingStudy;
  context.getStudyIndex=()=>pendingStudy??=(context.studyIndex?Promise.resolve(context.studyIndex):context.studyEnabled?loadBibleStudy(context.corpus,context.fetcher).catch(()=>null):Promise.resolve(null));
  const graph=buildBibleGraph(context);
  return (await graph.invoke({}, {signal:context.deadline,recursionLimit:GRAPH_RECURSION_LIMIT,callbacks:[]})).result;
}
