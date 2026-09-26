import {retrievalQuery,retrievalScope,retrievalVariants,parseBibleReferences,loadBibleRetriever} from './bible-rag.mjs';
import {embedTexts} from './rag-embedding.mjs';
import {lexicalViewBranches} from './rag-lexical-views.mjs';
import {searchIndependentStudy,studyQueryTerms} from './study-query.mjs';

export const DISCOVERY_LIMITS=Object.freeze({branch:100,candidates:420,characters:760000,selected:6,final:12});

// Preserve whole branch unions until the semantic reranker. No gold IDs or
// question-specific expected passages belong in this module.
export function discoverCandidates(index,corpus,question,{vector,refs=[],extraBranches=[],preparedQuery,retry=false,views=true,scopeQuestion}={}) {
  if(typeof index.rankings!=='function')throw Error('BROAD_RETRIEVER_REQUIRED');
  const query=preparedQuery??retrievalQuery(question),scope=retrievalScope(corpus,scopeQuestion??query,{coordinatedBooks:true});
  const lists=index.rankings(query,{vector,refs,bookScope:scope.bookIds});
  const branches=[['lexical',lists.lex],['focused',lists.focus],['dense',lists.dense],
    ...(views?lexicalViewBranches(index,corpus,query,{retry,scopeQuestion}):[]),...extraBranches];
  const found=new Map();
  const add=(i,path,rank)=>{
    const chunk=index.chunks[i];if(!chunk||(scope.bookIds.length&&!scope.bookIds.includes(chunk.bookId)))return;
    if(!found.has(chunk.id))found.set(chunk.id,{chunk,score:0,paths:[]});
    const hit=found.get(chunk.id);hit.score+=1/(10+rank+1);hit.paths.push({path,rank:rank+1});
  };
  for(const ref of refs)for(let i=0;i<index.chunks.length;i++){
    const c=index.chunks[i];if(c.bookId===ref.bookId&&c.chapter===ref.chapter&&c.start<=ref.verse&&c.end>=ref.verse)add(i,'explicit',0);
  }
  for(const [path,rows]of branches)rows.slice(0,DISCOVERY_LIMITS.branch).forEach((r,n)=>add(r.index,path,n));
  for(const [n,variant]of retrievalVariants(query).entries()){
    const s=retrievalScope(corpus,variant,{coordinatedBooks:true}),books=new Set(scopeQuestion!==undefined?scope.bookIds:(s.bookIds.length?s.bookIds:scope.bookIds));
    index.lexical(s.contentQuery||variant).filter(r=>!books.size||books.has(index.chunks[r.index].bookId)).slice(0,8).forEach((r,i)=>add(r.index,`clause-${n}`,i));
  }
  const hits=[...found.values()].sort((a,b)=>b.score-a.score||a.chunk.id.localeCompare(b.chunk.id));
  // Fail explicitly instead of claiming recall after silently discarding a tail.
  if(hits.length>DISCOVERY_LIMITS.candidates)throw Error('DISCOVERY_POOL_LIMIT');
  const candidates=hits.map(h=>({id:h.chunk.id,reference:h.chunk.reference,
    excerpt:h.chunk.verses.map(v=>`${v.verse} ${v.text}`).join('\n'),complete:true}));
  if(candidates.reduce((n,c)=>n+c.excerpt.length,0)>DISCOVERY_LIMITS.characters)throw Error('DISCOVERY_CHARACTER_LIMIT');
  return {hits,candidates,query,bookScope:scope.bookIds};
}

export function independentStudyBranch(index,studyIndex,question) {
  if(!studyIndex||!/배경|역사|당시|문화|원어|어원|주석|사전|교리|신학|\b[GH]\d{1,5}\b/i.test(question))return [];
  const matches=searchIndependentStudy(studyIndex,studyQueryTerms(question),question).slice(0,4),seen=new Set(),rows=[];
  // Source references propose Bible candidates; neither the dictionary nor a
  // title match proves that a verse answers the user's question.
  for(const match of matches){
    const ids=new Set(match.row.refs.map(n=>studyIndex.data.verseIds[n]));let perSource=0;
    for(let i=0;i<index.chunks.length;i++){
      const c=index.chunks[i];if(seen.has(i)||!c.verses.some(v=>ids.has(`${c.bookId}:${c.chapter}:${v.verse}`)))continue;
      seen.add(i);rows.push({index:i,score:match.score});if(++perSource>=8)break;
    }
  }
  return rows.slice(0,24);
}

export async function retrieveBroadBible({corpus,question,previousEvidence,passage=[],env,fetcher=fetch,signal,retriever,extraBranches=[],studyIndex}) {
  const index=retriever||await loadBibleRetriever(corpus),query=retrievalQuery(question,previousEvidence);
  let vector,inputTokens=0,degraded=false;
  try {const r=await embedTexts([query],{env,fetcher,signal,type:'query'});vector=r.vectors[0];inputTokens=r.inputTokens;}
  catch(error){if(signal?.aborted)throw error;degraded=true;}
  const followup=previousEvidence&&query!==retrievalQuery(question)?previousEvidence.passages||[]:[];
  const refs=[...parseBibleReferences(corpus,question).refs,...passage,...followup];
  const independent=independentStudyBranch(index,studyIndex,query);
  return {...discoverCandidates(index,corpus,query,{vector,refs,extraBranches:[...extraBranches,['independent-study-reference',independent]],preparedQuery:query}),index,vector,inputTokens,embeddingCalls:1,degraded};
}
