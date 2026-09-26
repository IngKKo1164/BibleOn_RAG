import {createEvidenceReader} from './bible-evidence.mjs';
import {RAG_LIMITS} from './bible-rag.mjs';

// Every model-selected anchor survives. Additional context fills the remaining
// shared budget in rounds, so a long first chunk cannot starve another part.
export function focusedEvidence(corpus,retrieval,plan,passage=[],limits=RAG_LIMITS) {
  const {verseAt}=createEvidenceReader(corpus),byId=new Map(retrieval.hits.map(h=>[h.chunk.id,h.chunk]));
  const selected=new Map(),queues=[],sections=[];let chars=0;
  const add=(v,required=false)=>{
    const row=verseAt(v);if(selected.has(row.id))return true;
    if(selected.size>=limits.evidenceVerses||chars+row.text.length>limits.evidenceCharacters){if(required)throw Error('REQUIRED_EVIDENCE_LIMIT');return false;}
    selected.set(row.id,row);chars+=row.text.length;return true;
  };
  passage.forEach(v=>add(v,true));
  for(const id of plan.chunkIds){
    const c=byId.get(id),focus=plan.focus.get(id);if(!c||!focus?.length)throw Error('UNRETRIEVED_FOCUS');
    for(const verse of focus){if(!c.verses.some(v=>v.verse===verse))throw Error('INVALID_FOCUS_VERSE');add({bookId:c.bookId,chapter:c.chapter,verse},true);}
    const start=c.sectionStart??c.start,end=c.sectionEnd??c.end,queue=[];
    for(let verse=start;verse<=end;verse++)if(corpus[c.bookId]?.chapters[c.chapter]?.[verse])queue.push({bookId:c.bookId,chapter:c.chapter,verse});
    queue.sort((a,b)=>Math.min(...focus.map(n=>Math.abs(a.verse-n)))-Math.min(...focus.map(n=>Math.abs(b.verse-n)))||a.verse-b.verse);
    queues.push(queue);sections.push({id:c.sectionId||c.id,bookId:c.bookId,chapter:c.chapter,start,end,selectedChunkId:id});
  }
  while(queues.some(q=>q.length))for(const q of queues){const next=q.shift();if(next)add(next);}
  const bookOrder=new Map(Object.keys(corpus).map((b,i)=>[b,i]));
  const evidence=[...selected.values()].sort((a,b)=>bookOrder.get(a.bookId)-bookOrder.get(b.bookId)||a.chapter-b.chapter||a.verse-b.verse);
  const context=sections.map(s=>{
    const chunk=byId.get(s.selectedChunkId),present=evidence.filter(v=>v.bookId===s.bookId&&v.chapter===s.chapter&&v.verse>=s.start&&v.verse<=s.end);
    const primaryVerseIds=present.filter(v=>v.verse>=chunk.start&&v.verse<=chunk.end).map(v=>v.id);
    const contextVerseIds=present.filter(v=>v.verse<chunk.start||v.verse>chunk.end).map(v=>v.id);
    const count=Object.keys(corpus[s.bookId].chapters[s.chapter]).filter(v=>+v>=s.start&&+v<=s.end).length;
    return {...s,sectionId:s.id,primaryVerseIds,contextVerseIds,complete:present.length===count,omittedVerses:count-present.length};
  });
  return {evidence,sections:context,anchorIds:plan.chunkIds.flatMap(id=>{const c=byId.get(id);return plan.focus.get(id).map(n=>`${c.bookId}:${c.chapter}:${n}`);})};
}
