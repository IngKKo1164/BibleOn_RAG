import { createBibleChunks, CHUNK_VERSION } from './rag-chunks.mjs';
import { embedTexts, EMBEDDING_DIMENSIONS, EMBEDDING_MODEL } from './rag-embedding.mjs';
import { createEvidenceReader } from './bible-evidence.mjs';
import { metadata as headingMetadata } from './data/rag-heading-boundaries.mjs';

const loaded = new WeakMap();
export const RAG_LIMITS = Object.freeze({ candidates: 12, selectedChunks: 3, evidenceCharacters: 6000, evidenceVerses: 96,
  excerptCharacters: 1200, candidateCharacters: 8400, excerptMinimum: 420 });
const shortNames = '창 출 레 민 신 수 삿 룻 삼상 삼하 왕상 왕하 대상 대하 스 느 에 욥 시 잠 전 아 사 렘 애 겔 단 호 욜 암 옵 욘 미 나 합 습 학 슥 말 마 막 눅 요 행 롬 고전 고후 갈 엡 빌 골 살전 살후 딤전 딤후 딛 몬 히 약 벧전 벧후 요일 요이 요삼 유 계'.split(' ');
const bookIds = 'genesis exodus leviticus numbers deuteronomy joshua judges ruth 1samuel 2samuel 1kings 2kings 1chronicles 2chronicles ezra nehemiah esther job psalms proverbs ecclesiastes songofsolomon isaiah jeremiah lamentations ezekiel daniel hosea joel amos obadiah jonah micah nahum habakkuk zephaniah haggai zechariah malachi matthew mark luke john acts romans 1corinthians 2corinthians galatians ephesians philippians colossians 1thessalonians 2thessalonians 1timothy 2timothy titus philemon hebrews james 1peter 2peter 1john 2john 3john jude revelation'.split(' ');
const stop = new Set('무엇 누구 누구지 누구야 뭐야 어떤 어떻게 왜 이유 의미 설명 설명해줘 설명해주세요 자세히 상세히 정확히 무슨 확정 확정할 근거 알려줘 알려주세요 알려줄래 말해줘 해주세요 있어 성경 본문 질문 관한 대한 대해서 대해 시대적 배경 들어 사람이 사람 그 이 저 것 수 중 더 한 할 한테 이것 그것 나는 나의 우리 위해 가장 중요한 태도 등 및 또 곧 즉 때 바 만 이다 하다 있다 있었다 있었는데'.split(' '));
// Vocabulary equivalents are retrieval hints, never evidence or answers.
// They contain no verse IDs and are independent of the evaluation dataset.
const equivalents = [
  [/오병이어/g, '오병이어 떡 다섯 물고기 두 마리 오천'],
  [/칠병이어/g, '칠병이어 떡 일곱 생선 사천'],
  [/설교|강해|말씀을?\s*듣/g, '강론 설교 말씀 듣'],
  [/예수님/g, '예수'], [/구원받[기가으을는]?/g, '구원 받'],
  [/떨어(?:져|졌|진|짐)/g, '떨어지'], [/씻기신/g, '씻기'],
  [/추락(?:한|해|했|하여|하다)?/g, '떨어지'],
];
export function normalizeRetrievalQuery(text) {
  let query = String(text);
  for (const [pattern, replacement] of equivalents) query = query.replace(pattern, replacement);
  return query;
}
export function ragTerms(text) {
  const terms = [];
  for (let word of String(text).normalize('NFKC').toLowerCase().match(/[가-힣a-z]+/g) || []) {
    if (stop.has(word)) continue;
    if (word.length > 1) word = word.replace(/(?:에게서|에서는|으로는|이라고|이라는|하기를|하려고|에게|에서|으로|까지|부터|처럼|이랑|을|를|은|는|이|가|와|과|의|에|다)$/, '');
    if (!word.length || stop.has(word)) continue;
    terms.push(`w:${word}`);
    for (let size = 1; size <= Math.min(4, word.length); size++) terms.push(`p:${word.slice(0, size)}`);
    for (let i = 0; i < word.length - 1; i++) terms.push(`g:${word.slice(i, i + 2)}`);
  }
  return terms;
}

// A person's name alone is not a Bible-book restriction. Only explicit book
// locations/references qualify; negative and whole-Bible requests stay global.
export function retrievalScope(corpus, text, {coordinatedBooks=false}={}) {
  const original = String(text).normalize('NFKC');
  if (/말고|제외|아니라|성경\s*전체|전체\s*성경|다른\s*(?:책|성경)/.test(original)) return {bookIds:[], contentQuery:original};
  const found = [], ranges = [];
  if(coordinatedBooks){
    const names=Object.entries(corpus).map(([id,b])=>[b.name.replace(/\s/g,''),id]).sort((a,b)=>b[0].length-a[0].length);
    const alternatives=names.map(([name])=>name).join('|'), byName=new Map(names);
    // A coordinated chain must end in an explicit book-location suffix.
    // Names in an ordinary sentence still do not become forced book filters.
    const chain=new RegExp(`(?<![가-힣])((?:${alternatives})(?:\\s*(?:와|과|및|,|·)\\s*(?:${alternatives}))+)(?:에서(?:는)?|에\\s*(?:나오|기록|있는|등장))`,'g');
    for(const match of original.matchAll(chain)){
      for(const name of match[1].split(/\s*(?:와|과|및|,|·)\s*/))found.push(byName.get(name));
      ranges.push([match.index,match.index+match[0].length]);
    }
  }
  for (const [bookId, book] of Object.entries(corpus)) {
    const name = book.name.replace(/\s/g, '');
    const pattern = new RegExp(`(?<![가-힣])${name}(?:에서(?:는)?|에\\s*(?:나오|기록|있는|등장)|(?=\\s*\\d{1,3}(?:\\s*[:장편])))`, 'g');
    for (const match of original.matchAll(pattern)) {found.push(bookId); if(!ranges.some(([a,b])=>a<=match.index&&b>=match.index+match[0].length))ranges.push([match.index,match.index+match[0].length]);}
  }
  let contentQuery=original;
  for (const [start,end] of ranges.sort((a,b)=>b[0]-a[0])) contentQuery=contentQuery.slice(0,start)+' '+contentQuery.slice(end);
  return {bookIds:[...new Set(found)],contentQuery:contentQuery.trim()};
}

export function parseBibleReferences(corpus, text, { maxMatches = 8, maxVerses = 20, maxRangeVerses = 20, chapterMode = 'expand', knownOmissions = new Set() } = {}) {
  const aliases = new Map();
  for (const [id, book] of Object.entries(corpus)) {
    aliases.set(book.name.replace(/\s/g, ''), id);
    const at = bookIds.indexOf(id); if (at >= 0) aliases.set(shortNames[at], id);
  }
  const names = [...aliases.keys()].sort((a, b) => b.length - a.length).join('|');
  const range = '(?:\\s*(?:[-~–—−]|부터)\\s*\\d{1,3}\\s*절?(?:\\s*까지)?)?';
  const pattern = new RegExp(`(?<![가-힣])(${names})\\s*(\\d{1,3})(?:\\s*:\\s*(\\d{1,3})|\\s*(?:장|편)(?:\\s*(\\d{1,3})(?:\\s*절)?)?)(\\s*절?${range}(?:\\s*[,，]\\s*\\d{1,3}(?:\\s*:\\s*\\d{1,3})?\\s*절?${range})*)`, 'g');
  const refs = [], chapters = [], { verseAt } = createEvidenceReader(corpus); let invalid = false;
  const matches = [...String(text).normalize('NFKC').matchAll(pattern)];
  if (matches.length > maxMatches) invalid = true;
  for (const match of matches.slice(0, maxMatches)) {
    const bookId = aliases.get(match[1]); let chapter = Number(match[2]);
    const row = corpus[bookId]?.chapters[chapter];
    if (!row || /^\s*:/.test(String(text).normalize('NFKC').slice(match.index + match[0].length))) { invalid = true; continue; }
    const firstText = match[3] || match[4];
    if (!firstText && chapterMode === 'mention') { chapters.push({ bookId, chapter }); continue; }
    const spans = firstText ? (firstText + match[5]).replace(/절|까지|\s/g, '').replace(/부터/g, '-').split(/[,，]/) : [`${Math.min(...Object.keys(row).map(Number))}-${Math.max(...Object.keys(row).map(Number))}`];
    for (let span of spans) {
      // A comma may repeat/change the chapter: John 3:1, 7:50–52.
      // This is a list of ranges, not a range crossing chapter boundaries.
      if (span.includes(':')) { const parts = span.split(':'); chapter = Number(parts[0]); span = parts[1]; }
      const [first, end] = span.split(/[-~–—−]/).map(Number), last = end ?? first;
      if (!Number.isInteger(first) || last < first || last - first >= maxRangeVerses) { invalid = true; continue; }
      for (let verse = first; verse <= last; verse++) {
        // Only explicitly verified translation omissions inside a range may
        // be skipped. Missing ordinary verses, endpoints and lone references
        // still fail; source membership is checked by the caller afterwards.
        const text=corpus[bookId]?.chapters?.[chapter]?.[verse];
        if(verse>first&&verse<last&&knownOmissions.has(`${bookId}:${chapter}:${verse}`)&&
          (text===undefined||/^\s*\(?없음\)?\s*$/.test(text)))continue;
        try { refs.push(verseAt({ bookId, chapter, verse })); }
        catch { invalid = true; }
      }
    }
  }
  const remaining = String(text).normalize('NFKC').replace(pattern, '').replace(/[\s,.?!，。]/g, '');
  const exact = matches.length > 0 && /^(?:본문|말씀|보여줘|읽어줘|알려줘|보여주세요|읽어주세요)*$/.test(remaining);
  const unique = [...new Map(refs.map(v => [v.id, v])).values()];
  if (unique.length > maxVerses) invalid = true;
  return { refs: unique.slice(0, maxVerses), exact, invalid, chapters, mentions: matches.length };
}

export function createBibleRetriever(corpus, vectors = null, lexicalData = null, chunks = createBibleChunks(corpus)) {
  const postings = new Map(lexicalData?.postings || []), lengths = lexicalData?.lengths || [], norms = [];
  let totalLength = lexicalData ? lengths.reduce((a,b)=>a+b,0) : 0;
  for (let i = 0; i < chunks.length; i++) {
    if (!lexicalData) {
    const terms = ragTerms(chunks[i].text), counts = new Map();
    lengths.push(terms.length); totalLength += terms.length;
    for (const term of terms) counts.set(term, (counts.get(term) || 0) + 1);
    for (const [term, frequency] of counts) { let posting = postings.get(term); if (!posting) postings.set(term, posting = []); posting.push(i, frequency); }
    }
    if (vectors) { let square = 0; for (let j = 0; j < EMBEDDING_DIMENSIONS; j++) square += vectors[i * EMBEDDING_DIMENSIONS + j] ** 2; norms.push(Math.sqrt(square)); }
  }
  for (const [term, posting] of postings) if (!(posting instanceof Uint16Array) && !(posting instanceof Uint32Array)) postings.set(term, Uint32Array.from(posting));
  const avgLength = totalLength / chunks.length || 1;
  const lexical = query => {
    const scores = new Float64Array(chunks.length);
    for (const term of new Set(ragTerms(query))) {
      const posting = postings.get(term); if (!posting) continue;
      const idf = Math.log(1 + (chunks.length - posting.length / 2 + .5) / (posting.length / 2 + .5));
      const weight = term.startsWith('w:') ? 1 : term.startsWith('p:') ? .6 : .15;
      for (let j = 0; j < posting.length; j += 2) {
        const i = posting[j], tf = posting[j + 1];
        scores[i] += weight * idf * tf * 2.2 / (tf + 1.2 * (.25 + .75 * lengths[i] / avgLength));
      }
    }
    return [...scores].map((score, index) => ({ index, score })).filter(v => v.score > 0).sort((a, b) => b.score - a.score || a.index - b.index);
  };
  const focused = query => {
    const words = [...new Set(ragTerms(query).filter(t => t.startsWith('w:')).map(t => t.slice(2, 6)))];
    const clues = words.map(word => ({ word, weight: Math.log(1 + chunks.length / ((postings.get(`p:${word}`)?.length || 0) / 2 + 1)) }));
    return chunks.map((chunk, index) => {
      let best = 0;
      for (let i = 0; i < chunk.verses.length; i++) {
        const text = chunk.verses.slice(Math.max(0, i - 1), i + 2).map(v => v.text).join(' ');
        const score = clues.reduce((n, clue) => n + (text.includes(clue.word) ? clue.weight : 0), 0) / (.8 + .2 * text.length / 180);
        best = Math.max(best, score);
      }
      return { index, score: best };
    }).filter(r => r.score > 0).sort((a,b) => b.score-a.score || a.index-b.index);
  };
  const rankings = (query, { vector, refs = [], bookScope } = {}) => {
    const parsedScope=retrievalScope(corpus,query);
    const scopedBooks=new Set(bookScope ?? parsedScope.bookIds);
    // References supplied by callers remain eligible in a comparison/follow-up.
    refs.forEach(ref=>{if(scopedBooks.size)scopedBooks.add(ref.bookId);});
    const eligible=index=>!scopedBooks.size||scopedBooks.has(chunks[index].bookId);
    const contentQuery=parsedScope.contentQuery || query;
    const lex = lexical(contentQuery).filter(r=>eligible(r.index)), focus = focused(contentQuery).filter(r=>eligible(r.index)), dense = [];
    if (vector && vectors) {
      const norm = Math.sqrt(vector.reduce((sum, n) => sum + n * n, 0));
      for (let i = 0; i < chunks.length; i++) {
        if (!eligible(i)) continue;
        let dot = 0; for (let j = 0; j < EMBEDDING_DIMENSIONS; j++) dot += vector[j] * vectors[i * EMBEDDING_DIMENSIONS + j];
        dense.push({ index: i, score: dot / (norm * norms[i] || 1) });
      }
      dense.sort((a, b) => b.score - a.score || a.index - b.index);
    }
    return {lex,focus,dense,scopedBooks};
  };
  const search = (query, { vector, refs = [], limit = RAG_LIMITS.candidates, bookScope } = {}) => {
    const {lex,focus,dense,scopedBooks}=rankings(query,{vector,refs,bookScope}), fused=new Map();
    for (const list of [focus.slice(0, 80), lex.slice(0, 80), dense.slice(0, 80)]) list.forEach((row, rank) => fused.set(row.index, (fused.get(row.index) || 0) + 1 / (10 + rank + 1)));
    for (const ref of refs) {
      const matches = chunks.map((c, index) => ({ c, index })).filter(({ c }) => c.bookId === ref.bookId && c.chapter === ref.chapter && c.start <= ref.verse && c.end >= ref.verse);
      matches.sort((a, b) => Math.abs((a.c.start + a.c.end) / 2 - ref.verse) - Math.abs((b.c.start + b.c.end) / 2 - ref.verse));
      if (matches.length) fused.set(matches[0].index, 1 + (fused.get(matches[0].index) || 0));
    }
    const results = [];
    const ranked = [...fused].sort((a, b) => b[1] - a[1] || a[0] - b[0]);
    // Preserve each search method's strongest candidate when the other method
    // disagrees, while still prioritizing explicit user references.
    const reserved = [focus[0], dense[0], lex[0], focus[1], dense[1], lex[1], focus[2], dense[2], dense[3]].filter(Boolean);
    const bookLeaders=scopedBooks.size>1?[...scopedBooks].flatMap(book=>{
      const row=ranked.find(([index])=>chunks[index].bookId===book);return row?[row]:[];
    }):[];
    const priority = [...ranked.filter(([,score]) => score >= 1), ...bookLeaders, ...reserved.map(row => [row.index, fused.get(row.index)]), ...ranked];
    for (const [index, score] of priority) {
      const chunk = chunks[index];
      if (results.some(r => r.chunk.bookId === chunk.bookId && r.chunk.chapter === chunk.chapter
        && Math.max(0, Math.min(r.chunk.end, chunk.end) - Math.max(r.chunk.start, chunk.start) + 1) / Math.min(r.chunk.verses.length, chunk.verses.length) > .6)) continue;
      results.push({ chunk, score, denseScore: dense.find(r => r.index === index)?.score ?? null });
      if (results.length >= limit) break;
    }
    return results;
  };
  return { chunks, search, rankings, lexical, focused, serializeLexical:()=>({lengths,postings:[...postings].map(([term,rows])=>[term,[...rows]])}) };
}

export async function loadBibleRetriever(corpus) {
  if (!loaded.has(corpus)) loaded.set(corpus, (async () => {
    const { loadRagVectors, loadRagLexical, metadata } = await import('./data/rag-index-packed.mjs');
    const chunks = createBibleChunks(corpus), fingerprint = chunks.map(c => `${c.id}\n${c.text}`).join('\n');
    const digest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', new TextEncoder().encode(fingerprint))), b => b.toString(16).padStart(2, '0')).join('');
    if (metadata.version !== CHUNK_VERSION || metadata.boundarySha256 !== headingMetadata.boundarySha256 || metadata.model !== EMBEDDING_MODEL || metadata.dimensions !== EMBEDDING_DIMENSIONS || metadata.count !== chunks.length || digest !== metadata.chunkSha256) throw Error('RAG_INDEX_MISMATCH');
    const vectors = await loadRagVectors();
    if (vectors.length !== chunks.length * EMBEDDING_DIMENSIONS) throw Error('RAG_VECTOR_MISMATCH');
    const vectorDigest = Array.from(new Uint8Array(await crypto.subtle.digest('SHA-256', vectors)), b => b.toString(16).padStart(2, '0')).join('');
    if (vectorDigest !== metadata.vectorSha256) throw Error('RAG_VECTOR_MISMATCH');
    const lexicalData = loadRagLexical ? await loadRagLexical() : null;
    return createBibleRetriever(corpus, vectors, lexicalData);
  })().catch(error => { loaded.delete(corpus); throw error; }));
  return loaded.get(corpus);
}

export function retrievalQuery(question, previousEvidence) {
  const followup = question.length < 180 && /그(?:\s*사람|\s*인물|\s*사건|\s*말씀|는|의|렇|래|럼)|이(?:\s*사람|\s*말씀|\s*사건)|나이|직업|더\s*(자세|구체)/.test(question);
  return normalizeRetrievalQuery(previousEvidence && followup ? `${previousEvidence.question.slice(0, 400)}\n${question}` : question);
}
// Extra lexical queries preserve explicit phrases and separate question parts.
// No generated answers, verse IDs, or additional embedding requests are used.
export function retrievalVariants(query) {
  const variants=[];
  const add=value=>{const s=value.trim();if(s.length>=4&&s!==query&&!variants.includes(s))variants.push(s);};
  for(const match of query.matchAll(/[“‘"']([^“”‘’"'\n]{4,60})[”’"']/g)) add(match[1]);
  // Group membership needs a list, not merely another mention of the group.
  // This linguistic expansion contains no Biblical names or answer locations.
  if(/중.{0,10}(?:한|하나)|명단|뽑|선택|누가/.test(query))for(const match of query.matchAll(/(?:^|\s)(열두|열한|열|일곱|다섯|네|세|두|십이|칠|\d+)\s+([가-힣]{2,10})/g))add(`${match[1]} ${match[2].replace(/(?:은|는|이|가|을|를|의)$/, '')} 이름`);
  // Lookahead permits overlapping possessives: '바울의 육체의 가시'.
  for(const match of query.matchAll(/(?=([가-힣]{2,12})의\s+([가-힣]{2,12}))/g)) {
    const last=match[2].replace(/(?:은|는|이|가|를|을)$/, '');
    if(last.length>=2)add(`${match[1]} ${last}`);
  }
  const clauses=query.split(/[?？]\s*|\s+(?:그리고|또한|아니면)\s+/).map(s=>s.trim()).filter(s=>s.length>=8);
  if(clauses.length>1)for(const clause of clauses)add(clause);
  return variants.slice(0,3);
}
const verseLines = chunk => chunk.verses.map(v=>`${v.verse} ${v.text}`);
export function candidateExcerpt(chunk, query, { maxCharacters = RAG_LIMITS.excerptCharacters, queries = [] } = {}) {
  const budget = Math.max(0,Math.min(RAG_LIMITS.excerptCharacters,Math.floor(maxCharacters)));
  const lines = verseLines(chunk), full = lines.join('\n');
  let selected = chunk.verses;
  if (full.length > budget) {
    const terms = new Set([query,...queries].flatMap(q=>ragTerms(normalizeRetrievalQuery(q))));
    const clues = [...terms].filter(t=>t.startsWith('w:')).map(t=>t.slice(2));
    const matches = chunk.verses.map(v=>({words:clues.filter(w=>v.text.includes(w)),grams:new Set(ragTerms(v.text).filter(t=>terms.has(t)))}));
    let best = {score:-1, length:0, start:0, end:0};
    // Preserve a continuous window, including subjects, negations and outcomes.
    // Distinct clues prevent repeated wording from overwhelming another fact.
    for (let start=0; start<lines.length; start++) {
      let length=0;const words=new Set(),grams=new Set();
      for (let end=start;end<lines.length;end++) {
        length += lines[end].length+(end>start?1:0);
        if (length>budget) break;
        matches[end].words.forEach(w=>words.add(w));matches[end].grams.forEach(t=>grams.add(t));
        const score = [...words].reduce((n,w)=>n+2+Math.min(4,w.length),0)*100+grams.size;
        if (score>best.score || (score===best.score && length>best.length)) best={score,length,start,end:end+1};
      }
    }
    selected = chunk.verses.slice(best.start,best.end);
  }
  return {id:chunk.id, reference:chunk.reference,
    excerpt:selected.map(v=>`${v.verse} ${v.text}`).join('\n'),
    excerptRange:selected.length?`${selected[0].verse}-${selected.at(-1).verse}`:'',
    complete:selected.length===chunk.verses.length};
}

export function candidateExcerpts(hits, query, queries = []) {
  const chunks=hits.map(h=>h.chunk), targets=chunks.map(c=>Math.min(verseLines(c).join('\n').length,RAG_LIMITS.excerptCharacters));
  if(chunks.length>RAG_LIMITS.candidates)throw Error('TOO_MANY_RAG_CANDIDATES');
  const budgets=targets.map(n=>Math.min(n,RAG_LIMITS.excerptMinimum));
  let remaining=RAG_LIMITS.candidateCharacters-budgets.reduce((a,b)=>a+b,0);
  // First finish small complete passages. Then share remaining space across
  // long passages; candidate rank does not monopolize the evidence budget.
  const short=chunks.map((c,i)=>({i,size:verseLines(c).join('\n').length})).filter(r=>r.size<=RAG_LIMITS.excerptCharacters)
    .sort((a,b)=>(a.size-budgets[a.i])-(b.size-budgets[b.i])||a.i-b.i);
  for(const {i,size} of short){const extra=size-budgets[i];if(extra<=remaining){budgets[i]+=extra;remaining-=extra;}}
  while(remaining>0){const pending=budgets.map((n,i)=>i).filter(i=>budgets[i]<targets[i]);if(!pending.length)break;
    const share=Math.max(1,Math.floor(remaining/pending.length));
    for(const i of pending){const extra=Math.min(share,targets[i]-budgets[i],remaining);budgets[i]+=extra;remaining-=extra;}
  }
  return chunks.map((chunk,i)=>candidateExcerpt(chunk,query,{maxCharacters:budgets[i],queries}));
}
export async function retrieveBible({ corpus, question, previousEvidence, passage = [], env, fetcher = fetch, signal, retriever }) {
  const index = retriever || await loadBibleRetriever(corpus), parsed = parseBibleReferences(corpus, question);
  const query = retrievalQuery(question, previousEvidence);
  let vector, inputTokens = 0, degraded = false;
  try { const result = await embedTexts([query], { env, fetcher, signal, type: 'query' }); vector = result.vectors[0]; inputTokens = result.inputTokens; }
  catch { degraded = true; }
  const followupRefs=previousEvidence && query!==normalizeRetrievalQuery(question) ? previousEvidence.passages || [] : [];
  const refs=[...parsed.refs,...passage,...followupRefs], main=index.search(query,{vector,refs});
  const variants=retrievalVariants(query);
  const scope=retrievalScope(corpus,query);
  const alternatives=variants.map(q=>index.search(q,{limit:3,bookScope:retrievalScope(corpus,q).bookIds.length?undefined:scope.bookIds}));
  const hits=[];
  // Preserve the leading hybrid candidate and each phrase's strongest match.
  const ordered=[main[0],...alternatives.map(rows=>rows[0]),...main,...alternatives.flat()].filter(Boolean);
  for(const row of ordered){
    if(hits.some(h=>h.chunk.id===row.chunk.id || (h.chunk.bookId===row.chunk.bookId&&h.chunk.chapter===row.chunk.chapter&&Math.max(0,Math.min(h.chunk.end,row.chunk.end)-Math.max(h.chunk.start,row.chunk.start)+1)/Math.min(h.chunk.verses.length,row.chunk.verses.length)>.6)))continue;
    hits.push(row);if(hits.length>=RAG_LIMITS.candidates)break;
  }
  return { hits, candidates: candidateExcerpts(hits, scope.contentQuery || query, variants), inputTokens, degraded, queryVariants:1+variants.length, bookScope:scope.bookIds };
}
export function selectRagEvidence(corpus, retrieval, selectedIds, passage = []) {
  if (!Array.isArray(selectedIds) || selectedIds.length > RAG_LIMITS.selectedChunks || new Set(selectedIds).size !== selectedIds.length) throw Error('INVALID_RAG_SELECTION');
  const byId = new Map(retrieval.hits.map(h => [h.chunk.id, h.chunk]));
  const { verseAt } = createEvidenceReader(corpus), verses = new Map(passage.map(v => { const canonical = verseAt(v); return [canonical.id, canonical]; }));
  let chars = [...verses.values()].reduce((n, v) => n + v.text.length, 0);
  if (chars > RAG_LIMITS.evidenceCharacters || verses.size > RAG_LIMITS.evidenceVerses) throw Error('RAG_EVIDENCE_LIMIT');
  for (const id of selectedIds) {
    const chunk = byId.get(id); if (!chunk) throw Error('UNRETRIEVED_CHUNK');
    const rows = [...new Set(chunk.verses.map(v=>v.verse))].sort((a,b)=>a-b)
      .map(verse => verseAt({ bookId: chunk.bookId, chapter: chunk.chapter, verse })).filter(v => !verses.has(v.id));
    const added = rows.reduce((n, v) => n + v.text.length, 0);
    // All selected evidence must reach generation. Never report a selection
    // as fulfilled after silently discarding an entire part of the question.
    if (chars + added > RAG_LIMITS.evidenceCharacters || verses.size + rows.length > RAG_LIMITS.evidenceVerses) throw Error('RAG_EVIDENCE_LIMIT');
    rows.forEach(v => verses.set(v.id, v)); chars += added;
  }
  return [...verses.values()];
}
