import './lib/workspace.mjs';
import { readFileSync, writeFileSync, mkdirSync, existsSync, renameSync } from 'node:fs';
import { parseEnv } from 'node:util';
import { createBibleChunks, CHUNK_VERSION } from '../server/rag-chunks.mjs';
import { metadata as boundaryMetadata } from '../server/data/rag-heading-boundaries.mjs';
import { embedTexts, EMBEDDING_MODEL, EMBEDDING_DIMENSIONS } from '../server/rag-embedding.mjs';
import { packRagIndex, hash } from './lib/pack-rag-index.mjs';
const live=process.argv.includes('--live'), root='artifacts/rag-heading-v2', cache=`${root}/embeddings`;
const corpusBytes=readFileSync('server/data/krv.json'),corpus=JSON.parse(corpusBytes),chunks=createBibleChunks(corpus);
if(hash(corpusBytes)!==boundaryMetadata.corpusSha256)throw Error('HEADING_CORPUS_MISMATCH');
mkdirSync(cache,{recursive:true});
const key=c=>hash(`${EMBEDDING_MODEL}:${EMBEDDING_DIMENSIONS}:${c.text}`);
const cached=c=>{const file=`${cache}/${key(c)}.json`;if(!existsSync(file))return null;const r=JSON.parse(readFileSync(file,'utf8'));if(r.textHash!==hash(c.text)||r.model!==EMBEDDING_MODEL||r.vector?.length!==EMBEDDING_DIMENSIONS||!r.vector.every(Number.isFinite)||!r.vector.some(n=>n!==0))throw Error('INVALID_CACHED_DOCUMENT_VECTOR');return r.vector;};
const pending=chunks.filter(c=>!cached(c));
const plan={version:CHUNK_VERSION,chunkCount:chunks.length,sectionCount:new Set(chunks.map(c=>c.sectionId)).size,cachedDocuments:chunks.length-pending.length,newDocuments:pending.length,newDocumentCharacters:pending.reduce((n,c)=>n+c.text.length,0),model:EMBEDDING_MODEL,dimensions:EMBEDDING_DIMENSIONS,providerPayload:'KRV text and book/chapter label only; no RNKSV heading text or full text',boundarySha256:boundaryMetadata.boundarySha256};
writeFileSync(`${root}/embedding-plan.json`,JSON.stringify(plan,null,2));console.log(JSON.stringify(plan));
if(pending.length&&!live){console.log('Prepared locally. New document embeddings need --live and this research workspace\'s .env.local QWEN_API_KEY. No network calls were made.');process.exit(0);}
if(pending.length){
  // Never inherit an application credential or read the parent environment.
  const env=existsSync('.env.local')?parseEnv(readFileSync('.env.local','utf8')):{};
  if(!env.QWEN_API_KEY?.trim())throw Error('RESEARCH_QWEN_API_KEY_REQUIRED');
  let completed=0,inputTokens=0;
  for(let i=0;i<pending.length;i+=10){
    const batch=pending.slice(i,i+10);let result;
    for(let attempt=0;attempt<4;attempt++){
      try{result=await embedTexts(batch.map(c=>c.text),{env,type:'document',fetcher:(url,init)=>fetch(url,{...init,redirect:'error'})});break;}
      catch(error){if(attempt===3||error.status===401||error.code==='AccessDenied')throw error;await new Promise(r=>setTimeout(r,error.status===429?15000:1000*(attempt+1)));}
    }
    batch.forEach((c,j)=>writeFileSync(`${cache}/${key(c)}.json`,JSON.stringify({model:EMBEDDING_MODEL,textHash:hash(c.text),vector:result.vectors[j],origin:'new-document-embedding'})));
    inputTokens+=result.inputTokens;completed+=batch.length;
    writeFileSync(`${root}/embedding-usage.json`,JSON.stringify({completedThisRun:completed,inputTokens,remaining:pending.length-completed}));
    if(completed%100===0||completed===pending.length)console.log(JSON.stringify({completed,total:pending.length,inputTokens}));
  }
}
const packed=packRagIndex(corpus,chunks,chunks.map(cached),corpusBytes,boundaryMetadata.boundarySha256);
const target='server/data/rag-index-packed.mjs';
writeFileSync(`${target}.tmp`,packed.module);renameSync(`${target}.tmp`,target);
writeFileSync(`${root}/index-manifest.json`,JSON.stringify(packed.metadata,null,2));
const assets=JSON.parse(readFileSync('runtime-assets.json','utf8'));
for(const file of ['rag-index-packed.mjs','rag-heading-boundaries.mjs']){
  const entry={file,sha256:hash(readFileSync(`server/data/${file}`))};
  const index=assets.findIndex(a=>a.file===file);
  if(index<0)assets.push(entry);else assets[index]=entry;
}
writeFileSync('runtime-assets.json',JSON.stringify(assets,null,2)+'\n');
console.log(JSON.stringify({complete:true,count:chunks.length,version:CHUNK_VERSION}));
