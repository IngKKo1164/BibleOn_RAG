// Keep canonical text intact; omit fields that repeat the same reference or
// describe the UI card. The server still keeps full verses for validation/UI.
export const compactVerses = verses => verses.map(({ id, reference, text }) => ({ id, reference, text }));
export const compactPrevious = previous => previous ? { question: previous.question, passages: compactVerses(previous.passages) } : null;

