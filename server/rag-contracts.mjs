// Structured output contract used by the current retrieval and answer graph.
export const MAX_REASONING_TOKENS = 3000;
export const kinds = ['answer', 'uncertain', 'outOfScope', 'greeting', 'safety'];
export const schema = properties => ({ type: 'object', additionalProperties: false, properties, required: Object.keys(properties) });
export function selectionParts(question) {
  const clauses=String(question).split(/(?<=[?？])\s*|\s+(?:그리고|또한)\s+/).map(s=>s.trim()).filter(Boolean);
  const parts=clauses.length<=3?clauses:[...clauses.slice(0,2),clauses.slice(2).join(' ')];
  return parts.map((question,i)=>({partId:`P${i+1}`,question}));
}
export const PROVIDER_ANSWER_SCHEMA = schema({
  answer: { type: 'string' },
  kind: { type: 'string', enum: kinds },
  citations: {
    type: 'array', maxItems: 8,
    items: schema({ bookId: { type: 'string' }, chapter: { type: 'integer' }, verse: { type: 'integer' } }),
  },
});
