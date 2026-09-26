import {MAX_REASONING_TOKENS,selectionParts} from './rag-contracts.mjs';
import {compactVerses, compactPrevious} from './bible-efficient.mjs';
import {compactAnswerInput, COMPACT_EVIDENCE_PROMPT} from './rag-input-budget.mjs';
import {createEvidenceReader} from './bible-evidence.mjs';
import {BibleProviderError} from './bible-errors.mjs';
import {resolveBibleRequest} from './bible-request.mjs';
import {OPENAI_BIBLE_STAGES, OPENAI_BIBLE_ENDPOINT, openAIBibleBody, isExpectedOpenAIModel} from './bible-openai.mjs';

// Per-invocation capabilities remain outside serializable LangGraph state.
export function createBibleRuntime({env,corpus,question,history,previousEvidence=null,passage=[],fetcher=fetch,observe=()=>{},retriever,studyIndex,signal}) {
  if (env.BIBLE_AI_PROVIDER !== 'openai' || env.BIBLE_RETRIEVAL_VERSION !== 'v4') throw new BibleProviderError('AI_NOT_CONFIGURED', 503);
  const model = OPENAI_BIBLE_STAGES.answer.model;
  const scope = resolveBibleRequest(corpus, question, passage);
  observe({stage:'route', result:{route:scope.route,primaryVerses:scope.primary.length,contextVerses:scope.context.length}});
  if (scope.invalid) throw new BibleProviderError('AI_REFERENCE_INVALID', 400);
  const studyEnabled = env.BIBLE_RAG_STUDY === 'one-hop' && (scope.sources || scope.crossrefs);
  const wireVerses = compactVerses;
  const wirePrevious = compactPrevious(previousEvidence);
  const {verseAt, expand} = createEvidenceReader(corpus);
  const usage = {inputTokens:0,outputTokens:0,reasoningTokens:0,answerTokens:0,totalTokens:0,calls:0};
  const deadline = AbortSignal.any([AbortSignal.timeout(110000), ...(signal ? [signal] : [])]);
  const questionParts = selectionParts(question);
  const call = async (stage, system, input, outputSchema) => {
    deadline.throwIfAborted();
    if (usage.calls >= 6) throw new BibleProviderError('AI_CALL_LIMIT');
    observe({stage,event:'start'});
    if (stage === 'answer') {
      const wire = compactAnswerInput(input);
      if (wire.compact) { input = wire.input; system += '\n' + COMPACT_EVIDENCE_PROMPT; }
    }
    const response = await fetcher(OPENAI_BIBLE_ENDPOINT, {
      method:'POST',headers:{'Content-Type':'application/json',Authorization:`Bearer ${env.OPENAI_API_KEY}`},
      signal:AbortSignal.any([deadline,AbortSignal.timeout(50000)]),
      body:JSON.stringify(openAIBibleBody(stage,system,input,outputSchema)),
    });
    deadline.throwIfAborted();
    if (!response.ok) throw new BibleProviderError('AI_PROVIDER_ERROR',response.status === 429 ? 429 : 502);
    const data = await response.json();
    if (!isExpectedOpenAIModel(stage,data.model)) throw new BibleProviderError('AI_MODEL_MISMATCH');
    const candidate = data.choices?.[0], u = data.usage;
    const inputTokens=u?.prompt_tokens,outputTokens=u?.completion_tokens,reasoningTokens=u?.completion_tokens_details?.reasoning_tokens;
    let observedResult;
    try { observedResult=JSON.parse(candidate?.message?.content); } catch { observedResult={incompleteAnswer:candidate?.message?.content}; }
    observe({stage,model:data.model,result:observedResult,usage:{inputTokens,outputTokens,reasoningTokens},finish:candidate?.finish_reason});
    // Completion usage already includes reasoning; do not bill it twice.
    if (![inputTokens,outputTokens,reasoningTokens].every(n=>Number.isInteger(n)&&n>=0)||reasoningTokens>outputTokens) throw new BibleProviderError('AI_USAGE_UNVERIFIED');
    usage.inputTokens+=inputTokens;usage.outputTokens+=outputTokens;usage.reasoningTokens+=reasoningTokens;usage.totalTokens+=inputTokens+outputTokens;usage.calls++;
    const stageExceeded=outputTokens>OPENAI_BIBLE_STAGES[stage].maxTokens||outputTokens-reasoningTokens>OPENAI_BIBLE_STAGES[stage].visibleTokens;
    if (stageExceeded||usage.reasoningTokens>MAX_REASONING_TOKENS||candidate?.finish_reason!=='stop') throw new BibleProviderError('AI_INCOMPLETE');
    let result;
    try { result=JSON.parse(candidate.message.content); } catch { throw new BibleProviderError('AI_INVALID_ANSWER'); }
    if (!result||typeof result!=='object'||Array.isArray(result)||Object.keys(result).some(k=>!outputSchema.required.includes(k))||outputSchema.required.some(k=>!(k in result))) throw new BibleProviderError('AI_INVALID_ANSWER');
    return {result,visibleTokens:outputTokens-reasoningTokens};
  };
  const bookIds=Object.entries(corpus).map(([id,b])=>`${id}=${b.name}`).join(', ');
  const inputs={question,history,previousEvidence:wirePrevious,passage:wireVerses(passage)};
  return {env,corpus,question,history,previousEvidence,passage,fetcher,observe,retriever,studyIndex,model,scope,
    studyEnabled,wireVerses,wirePrevious,verseAt,expand,usage,deadline,questionParts,call,bookIds,inputs,
    openai:true,research:true,integrated:true,efficient:true,gemini:false,ragEnabled:true,sufficiencyEnabled:true};
}
