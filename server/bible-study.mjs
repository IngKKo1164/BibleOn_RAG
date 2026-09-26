import { BIBLE_STUDY_ASSET } from './data/bible-study-asset.mjs';
import { gunzipSync } from 'node:zlib';
import { createHash } from 'node:crypto';
import { Buffer } from 'node:buffer';
import { createEvidenceReader } from './bible-evidence.mjs';
import { ragTerms } from './bible-rag.mjs';
import { koreanOnly } from './bible-answer-guards.mjs';
import { hasCompleteAnswerEnding } from './answer-completion.mjs';
import { hasApparentDeathClaim } from './bible-claim-guards.mjs';
import {studyQueryTerms,searchIndependentStudy} from './study-query.mjs';

export const STUDY_LIMITS = Object.freeze({ roots: 6, linkedVerses: 12, linkedCharacters: 1500, sources: 3, sourceCharacters: 4000, excerptCharacters: 2400, notes: 2 });
export const STUDY_SOURCES = Object.freeze({
  crossrefs: { title: '대한성서공회 관주성경 간이국한문 계열 전사본', url: 'http://www.sihong.pe.kr/', license: 'Underlying 1964 edition: economic copyright expired 2014-12-31; coordinates only; not a verified 1962 facsimile' },
  mhcc: { title: '매튜 헨리 간결 주석', url: 'https://www.crosswire.org/sword/modules/ModInfo.jsp?modName=MHCC', license: 'Public Domain', kind: 'commentary_opinion' },
  easton: { title: '이스턴 성경사전', url: 'https://www.crosswire.org/sword/modules/ModInfo.jsp?modName=Easton', license: 'Public Domain', kind: 'historical_dictionary_1897' },
  strongGreek: { title: '스트롱 헬라어 사전', url: 'https://www.crosswire.org/sword/modules/ModInfo.jsp?modName=StrongsGreek', license: 'Public Domain', kind: 'lexicon_1890' },
  step: { title: '스텝바이블 · 틴데일하우스', url: 'https://github.com/STEPBible/STEPBible-Data', license: 'CC BY 4.0', licenseUrl: 'https://creativecommons.org/licenses/by/4.0/', changes: '공통 본문의 구절·원어 번호를 발췌하고 재구성했습니다.' },
});

const cache = new WeakMap();
export function createStudyIndex(corpus, data) {
  const canonicalIds = Object.entries(corpus).flatMap(([b, book]) => Object.entries(book.chapters).flatMap(([c, chapter]) => Object.keys(chapter).map(v => `${b}:${c}:${v}`)));
  if (canonicalIds.length !== data.verseIds.length || canonicalIds.some((id, i) => id !== data.verseIds[i])) throw Error('STUDY_CORPUS_MISMATCH');
  const ids = new Map(data.verseIds.map((id, i) => [id, i])), records = new Map(), byVerse = new Map();
  for (const row of data.records) {
    records.set(row.id, row);
    for (const v of row.refs) { if (!byVerse.has(v)) byVerse.set(v, []); byVerse.get(v).push(row); }
  }
  return { data, ids, records, byVerse };
}
export async function loadBibleStudy(corpus, fetcher = fetch) {
  if (!cache.has(corpus)) {
    const pending = (async () => {
      const response = await fetcher(BIBLE_STUDY_ASSET.publicUrl, { signal: AbortSignal.timeout(8000) });
      if (!response.ok) throw Error('STUDY_ASSET_UNAVAILABLE');
      const bytes = Buffer.from(await response.arrayBuffer());
      if (bytes.length !== BIBLE_STUDY_ASSET.bytes || createHash('sha256').update(bytes).digest('hex') !== BIBLE_STUDY_ASSET.sha256) throw Error('STUDY_ASSET_HASH_MISMATCH');
      const raw = gunzipSync(bytes, { maxOutputLength: BIBLE_STUDY_ASSET.uncompressedBytes });
      return createStudyIndex(corpus, JSON.parse(raw.toString('utf8')));
    })();
    cache.set(corpus, pending);
    pending.catch(() => { if (cache.get(corpus) === pending) cache.delete(corpus); });
  }
  return cache.get(corpus);
}

const termScore = (text, terms) => ragTerms(text).reduce((n, term) => n + Number(terms.has(term)), 0);
export function expandOneHop(corpus, primary, question, index, limits = STUDY_LIMITS) {
  const { verseAt } = createEvidenceReader(corpus), existing = new Set(primary.map(v => v.id));
  const terms = new Set(ragTerms(question));
  const roots = primary.map((verse, order) => ({ verse, order, score: termScore(verse.text, terms) }))
    .sort((a, b) => b.score - a.score || a.order - b.order).slice(0, limits.roots);
  const candidates = new Map();
  // Freeze the roots before expansion. Targets are never enqueued as new roots.
  for (const [rank, root] of roots.entries()) {
    const n = index.ids.get(root.verse.id);
    for (const target of index.data.graph[n] || []) {
      const id = index.data.verseIds[target];
      if (!id || existing.has(id)) continue;
      const [bookId, chapter, verse] = id.split(':');
      let canonical; try { canonical = verseAt({ bookId, chapter: +chapter, verse: +verse }); } catch { continue; }
      if (!candidates.has(id)) candidates.set(id, { verse: canonical, roots: [], score: 0 });
      const row = candidates.get(id);
      row.roots.push(root.verse.id);
      row.score += 2 / (rank + 1) + termScore(canonical.text, terms) / 8;
    }
  }
  const linked = [], relations = []; let chars = 0;
  for (const row of [...candidates.values()].sort((a, b) => b.score - a.score || index.ids.get(a.verse.id) - index.ids.get(b.verse.id))) {
    if (linked.length >= limits.linkedVerses) break;
    if (chars + row.verse.text.length > limits.linkedCharacters) continue;
    linked.push(row.verse); chars += row.verse.text.length;
    for (const from of row.roots) relations.push({ from, to: row.verse.id });
  }
  return { evidence: [...primary, ...linked], linked, relations, roots: roots.map(r => r.verse.id) };
}

export function normalizeStudyTerms(value) {
  if (!Array.isArray(value)) return [];
  return [...new Set(value.filter(s => typeof s === 'string' && /^[a-zA-Z][a-zA-Z '-]{0,47}$/.test(s)).map(s => s.toLowerCase().trim()))].slice(0, 3);
}
const englishScore = (text, terms) => terms.reduce((score, term) => score + (text.toLowerCase().includes(term) ? 1 : 0), 0);

// Preserve complete paragraphs, including their qualifications. A paragraph
// that cannot fit is omitted, not silently truncated or reduced to a clause.
export function studyExcerpt(text, terms, maximum = STUDY_LIMITS.excerptCharacters) {
  const paragraphs = text.split('\n').map(p => p.trim()).filter(p => p.length > 25);
  const ranked = paragraphs.map((p, i) => ({ p, i, score: englishScore(p, terms) }))
    .sort((a, b) => b.score - a.score || a.i - b.i);
  const selected = []; let chars = 0;
  for (const row of ranked) {
    const value = row.p;
    if (value.length > maximum) continue;
    if (chars + value.length + 1 > maximum) continue;
    selected.push({ text: value, index: row.i }); chars += value.length + 1;
  }
  return selected.sort((a, b) => a.index - b.index).map(r => r.text).join('\n');
}

export function retrieveStudySources(primary, studyTerms, question, index, {independent=false}={}) {
  const terms = independent?studyQueryTerms(question,normalizeStudyTerms(studyTerms)):normalizeStudyTerms(studyTerms), candidates = new Map();
  // A verse can contain many Greek words (e.g. all nine fruits). A model's
  // English translation of a Korean term is not verified word alignment.
  // Use the lexicon only for an explicitly requested, verse-attested number.
  const requestedGreek = new Set((question.match(/\bG\d{1,5}\b/gi) || []).map(s => 'G' + String(Number(s.slice(1))).padStart(4, '0')));
  const koreanTerms = new Set(ragTerms(question));
  const roots = primary.map(v => ({ v, score: termScore(v.text, koreanTerms) })).sort((a, b) => b.score - a.score).slice(0, 12);
  for (const [rank, { v }] of roots.entries()) {
    const n = index.ids.get(v.id);
    for (const row of index.byVerse.get(n) || []) {
      if (!candidates.has(row.id)) candidates.set(row.id, { row, score: 0 });
      candidates.get(row.id).score += (2 / (rank + 1)) / Math.sqrt(Math.max(1, row.refs.length));
    }
    for (const key of index.data.greekTags[n] || []) {
      if (!requestedGreek.has(key)) continue;
      const id = index.data.aliases['strong:' + key] || 'strong:' + key, row = index.records.get(id);
      if (row) candidates.set(id, { row, score: 30 });
    }
  }
  if(independent)for(const match of searchIndependentStudy(index,terms,question)){
    const existing=candidates.get(match.row.id);
    if(existing){existing.score+=match.score;existing.method=match.method;}
    else candidates.set(match.row.id,{...match});
  }
  for (const candidate of candidates.values()) {
    const { row } = candidate;
    candidate.score += englishScore(row.title, terms) * 25 + englishScore(row.text, terms) * 2;
  }
  const chosen = [], seenTypes = new Set(); let characters = 0;
  const ranked = [...candidates.values()].sort((a, b) => b.score - a.score || a.row.id.localeCompare(b.row.id));
  // Reserve room for different kinds of evidence; don't fill every slot with
  // near-identical commentary on neighboring verses.
  for (const { row, score, method } of ranked) {
    if (chosen.length >= STUDY_LIMITS.sources) break;
    if (score <= 0) continue;
    if (seenTypes.has(row.source)) continue;
    const excerpt = studyExcerpt(row.text, terms, Math.min(STUDY_LIMITS.excerptCharacters, STUDY_LIMITS.sourceCharacters - characters));
    if (!excerpt) continue;
    chosen.push({ id: `S${chosen.length + 1}`, recordId: row.id, source: row.source, title: row.title, kind: STUDY_SOURCES[row.source].kind, excerpt,
      ...(independent?{retrieval:method||'verse-linked',license:STUDY_SOURCES[row.source].license,sourceUrl:STUDY_SOURCES[row.source].url}: {}) });
    characters += excerpt.length; seenTypes.add(row.source);
  }
  return chosen;
}

const normalizeQuote = s => String(s || '').normalize('NFKC').replace(/\s+/g, ' ').trim();
export function originalLanguageSupport(question,sources=[]) {
  const requested=/원어|어원|헬라어|히브리어|아람어|\b[GH]\d{1,5}\b/i.test(question);
  const available=!/히브리어|아람어|\bH\d{1,5}\b/i.test(question) && sources.some(s=>s.kind==='lexicon_1890');
  return {requested,available,limitation:requested&&!available?'이 답변에서 확인한 자료만으로는 해당 한국어 표현에 대응하는 원어와 그 정확한 뜻을 확인하기 어렵습니다. 본문에서 드러나는 의미를 중심으로 이해해 주세요.':''};
}
export function preserveOriginalLanguageLimit(answer,support) {
  if(!support.limitation)return answer;
  const covered=answer.split(/(?<=[.!?](?:\*\*)?)\s+/u).some(s=>/원어|어원|헬라어|히브리어|아람어/.test(s)&&/(?:확인|단정|설명|특정).{0,25}(?:어렵|없|못)|자료.{0,20}(?:없|않)/.test(s));
  return covered?answer:answer.trim()+'\n\n'+support.limitation;
}
export function validateStudyNotes(notes, sources) {
  if (!Array.isArray(notes) || notes.length > STUDY_LIMITS.notes) return [];
  const byId = new Map(sources.map(s => [s.id, s])), accepted = [], seen = new Set();
  for (const note of notes) {
    const source = byId.get(note?.sourceId), quote = normalizeQuote(note?.quote);
    if (!source || seen.has(source.id) || quote.length < 16 || quote.length > 500 || !normalizeQuote(source.excerpt).includes(quote)) continue;
    const text = typeof note.text === 'string' ? note.text.trim() : '';
    if (!text || text.length > 1200 || !koreanOnly(text) || !hasCompleteAnswerEnding(text)) continue;
    // External notes are not a route around Bible citation checks, links, or UI
    // source attribution. Every note is separately labeled by its actual source.
    if (/https?:|\]\(|<|>|\b(?:S\d|G\d{3,})\b/.test(text)) continue;
    // A verbatim quote is not enough if the paraphrase explicitly turns the
    // source's declared death into apparent death. This is a narrow conflict
    // check, not a claim that all accepted translations are verified.
    if(/taken up dead|was dead|restored him to life|brought to life again/i.test(quote)&&hasApparentDeathClaim(text))continue;
    accepted.push({ source, text, quote }); seen.add(source.id);
  }
  return accepted;
}
export function appendStudyNotes(answer, notes, usedCrossrefs = false) {
  const blocks = [answer.trim()];
  if (notes.length) {
    blocks.push('### 참고 자료로 살펴본 배경');
    for (const note of notes) {
      const source = STUDY_SOURCES[note.source.source];
      blocks.push(`**${source.title}의 설명**\n\n${note.text}\n\n[자료 출처](${source.url})`);
      if (note.source.source === 'strongGreek') blocks.push(`원어 연결자료: [${STUDY_SOURCES.step.title}](${STUDY_SOURCES.step.url}) · [저작자표시 4.0](${STUDY_SOURCES.step.licenseUrl}). ${STUDY_SOURCES.step.changes}`);
    }
    blocks.push('참고 자료의 설명과 해석은 성경 본문의 진술과 구분해 읽어 주세요.');
  }
  if (usedCrossrefs) blocks.push(`관주 자료: [${STUDY_SOURCES.crossrefs.title}](${STUDY_SOURCES.crossrefs.url}).`);
  return blocks.join('\n\n');
}
