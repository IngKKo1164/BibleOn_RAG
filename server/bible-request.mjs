import { parseBibleReferences } from './bible-rag.mjs';
import { createEvidenceReader } from './bible-evidence.mjs';

export const REQUEST_LIMITS = Object.freeze({ verses: 96, characters: 12000, contextVerses: 4 });
export function resolveBibleRequest(corpus, question, passage = []) {
  const parsed = parseBibleReferences(corpus, question, { maxVerses: REQUEST_LIMITS.verses, maxRangeVerses: REQUEST_LIMITS.verses });
  const { verseAt } = createEvidenceReader(corpus);
  const selected = [...new Map(passage.map(v => { const canonical = verseAt(v); return [canonical.id, canonical]; })).values()];
  const primary = parsed.mentions ? parsed.refs : selected;
  const tooLarge = primary.length > REQUEST_LIMITS.verses || primary.reduce((n, v) => n + v.text.length, 0) > REQUEST_LIMITS.characters;
  const invalid = parsed.invalid || tooLarge;
  const direct = !invalid && primary.length > 0;
  const lookup = direct && parsed.exact;
  const sources = /배경|역사|당시|문화|원어|어원|주석|사전|\b[GH]\d{1,5}\b/i.test(question);
  const crossrefs = /관주|평행|교리|신학|교단|비교|차이|연결|성경\s*전체|다른\s*(?:성경|본문|구절)/.test(question);
  const wantsContext = /문맥|앞뒤|전후|흐름/.test(question);
  const context = [], ids = new Set(primary.map(v => v.id));
  if (direct && !lookup && wantsContext) {
    for (const anchor of [primary[0], primary.at(-1)]) for (const offset of [-2, -1, 1, 2]) {
      if (context.length >= REQUEST_LIMITS.contextVerses) break;
      try {
        const v = verseAt({ ...anchor, verse: anchor.verse + offset });
        if (!ids.has(v.id)) { context.push(v); ids.add(v.id); }
      } catch { /* Chapter boundaries are not invented. */ }
    }
  }
  return { direct, lookup, invalid, primary, context, sources, crossrefs, route: lookup ? 'lookup' : direct ? (sources || crossrefs ? 'passage-study' : 'passage-explanation') : 'retrieval' };
}
