import { boundaries } from './data/rag-heading-boundaries.mjs';
export const CHUNK_VERSION = 'krv-rnksv-heading-spans-v2';
export const CHUNK_POLICY = Object.freeze({ maxVerses: 24, maxCharacters: 1800, overlapVerses: 3, boundary: 'rnksv-heading-and-chapter' });

// Each heading/chapter span is a parent. Oversized parents split into bounded
// children; overlap is allowed only inside that parent, never across a heading.
export function createBibleChunks(corpus, headingBoundaries = boundaries) {
  const chunks = [];
  for (const [bookId, book] of Object.entries(corpus)) for (const [chapterKey, chapter] of Object.entries(book.chapters)) {
    const rows = Object.entries(chapter).map(([verse, text]) => ({verse:Number(verse), text})).sort((a,b)=>a.verse-b.verse);
    if (!rows.length) continue;
    const starts = new Set(headingBoundaries[bookId]?.[chapterKey] || []);
    const offsets = [...new Set([0, ...rows.flatMap((row,i)=>starts.has(row.verse)?[i]:[])])].sort((a,b)=>a-b);
    for (let group = 0; group < offsets.length; group++) {
      const section = rows.slice(offsets[group], offsets[group+1] ?? rows.length);
      const sectionStart = section[0].verse, sectionEnd = section.at(-1).verse;
      const sectionId = `${bookId}:${chapterKey}:${sectionStart}-${sectionEnd}`;
      for (let start = 0; start < section.length;) {
        let end = start, characters = 0;
        while (end < section.length && end-start < CHUNK_POLICY.maxVerses) {
          if (end > start && characters+section[end].text.length > CHUNK_POLICY.maxCharacters) break;
          characters += section[end++].text.length;
        }
        const verses = section.slice(start,end), first = verses[0].verse, last = verses.at(-1).verse;
        chunks.push({id:`${bookId}:${chapterKey}:${first}-${last}`, bookId, chapter:Number(chapterKey), start:first, end:last,
          sectionId, sectionStart, sectionEnd, sectionBoundary:starts.has(sectionStart)?'heading':'chapter',
          reference:`${book.name} ${chapterKey}:${first}-${last}`, verses,
          text:`${book.name} ${chapterKey}장\n${verses.map(v=>`${v.verse} ${v.text}`).join('\n')}`});
        if (end === section.length) break;
        start = Math.max(start+1,end-CHUNK_POLICY.overlapVerses);
      }
    }
  }
  return chunks;
}
