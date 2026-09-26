// Reference resolution is restricted to the bundled canonical text. It cannot
// fetch URLs or accept text invented by a model or client.
const readers = new WeakMap();
export function createEvidenceReader(corpus) {
  if (readers.has(corpus)) return readers.get(corpus);
  const verseAt = (ref) => {
    if (!ref || !Number.isInteger(ref.chapter) || !Number.isInteger(ref.verse)) throw Error('Invalid citation');
    const book = corpus[ref.bookId], text = book?.chapters?.[String(ref.chapter)]?.[String(ref.verse)];
    if (typeof text !== 'string' || !text.trim()) throw Error('Unknown citation');
    return { id: `${ref.bookId}:${ref.chapter}:${ref.verse}`, bookId: ref.bookId, chapter: ref.chapter, verse: ref.verse,
      reference: `${book.name} ${ref.chapter}:${ref.verse}`, translation: '개역한글', translationId: 'KRV', text };
  };
  const expand = refs => {
    if (!Array.isArray(refs) || refs.length > 8) throw Error('Invalid references');
    const verses = new Map();
    for (const ref of refs) {
      verseAt(ref);
      // Nearby context preserves speakers, locations, and outcomes. Bounded
      // at 96 verses / 18k characters, independent of model-provided ranges.
      for (let v = Math.max(1, ref.verse - 4); v <= ref.verse + 6; v++) {
        if (!corpus[ref.bookId].chapters[String(ref.chapter)][String(v)]) continue;
        const item = verseAt({ ...ref, verse: v }); verses.set(item.id, item);
      }
    }
    const all = [...verses.values()];
    if (all.length > 96 || all.reduce((n, v) => n + v.text.length, 0) > 18000) throw Error('Evidence too long');
    return all;
  };
  const reader = { verseAt, expand }; readers.set(corpus, reader); return reader;
}
