// Source headings supply boundaries only. Translation text and heading labels
// never enter the KRV index or provider payloads through this module.
export function compileHeadingBoundaries(corpus, documents) {
  const byBook = new Map();
  for (const doc of documents) {
    if (byBook.has(doc.book?.id)) throw Error('DUPLICATE_HEADING_BOOK');
    byBook.set(doc.book?.id, doc);
  }
  const boundaries = {}, differences = [];
  let headings = 0, distinctHeadingStarts = 0;
  if (byBook.size !== Object.keys(corpus).length) throw Error('HEADING_BOOK_COUNT_MISMATCH');
  for (const [bookId, book] of Object.entries(corpus)) {
    const doc = byBook.get(bookId);
    if (!doc || doc.chapters.length !== Object.keys(book.chapters).length) throw Error(`HEADING_CHAPTER_MISMATCH:${bookId}`);
    boundaries[bookId] = {};
    const seen = new Set();
    for (const source of doc.chapters) {
      const chapter = book.chapters[source.chapter];
      if (!chapter || seen.has(source.chapter)) throw Error(`HEADING_CHAPTER_MISMATCH:${bookId}:${source.chapter}`);
      seen.add(source.chapter);
      const verseIds = new Set(), starts = new Set();
      for (const verse of source.verses) {
        if (!Number.isInteger(verse.verse) || verse.verse < 1 || verseIds.has(verse.verse)) throw Error('INVALID_HEADING_VERSE');
        verseIds.add(verse.verse);
        const labels = verse.segments?.filter(s => s.type === 'heading') || [];
        if (!labels.length) continue;
        headings += labels.length;
        // Never guess a merged-verse mapping or shift a boundary by row index.
        if (typeof chapter[verse.verse] !== 'string') throw Error(`UNALIGNED_HEADING:${bookId}:${source.chapter}:${verse.verse}`);
        starts.add(verse.verse);
      }
      const canonical = Object.keys(chapter).map(Number);
      const onlyInKrv = canonical.filter(v => !verseIds.has(v));
      const onlyInRnksv = [...verseIds].filter(v => typeof chapter[v] !== 'string');
      if (onlyInKrv.length || onlyInRnksv.length) differences.push({bookId, chapter: source.chapter, onlyInKrv, onlyInRnksv});
      distinctHeadingStarts += starts.size;
      boundaries[bookId][source.chapter] = [...starts].sort((a,b) => a-b);
    }
  }
  return {boundaries, audit: {headings, distinctHeadingStarts, differences, unalignedHeadings: 0,
    alignment: 'Heading verse coordinates checked against KRV. Not a full verse-by-verse translation alignment.'}};
}
