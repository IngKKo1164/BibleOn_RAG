import { BibleProviderError } from './bible-errors.mjs';

// One owner-selected model for discovery, selection, generation and repair.
// Request input cannot change the provider or choose a cheaper fallback.
export const OPENAI_BIBLE_MODEL = 'gpt-6-luna';
export const OPENAI_BIBLE_STAGES = Object.freeze({
  screen: Object.freeze({ model: OPENAI_BIBLE_MODEL, effort: 'low', maxTokens: 1800, visibleTokens: 1000 }),
  locate: Object.freeze({ model: OPENAI_BIBLE_MODEL, effort: 'low', maxTokens: 1500, visibleTokens: 450 }),
  rerank: Object.freeze({ model: OPENAI_BIBLE_MODEL, effort: 'low', maxTokens: 3000, visibleTokens: 1800 }),
  answer: Object.freeze({ model: OPENAI_BIBLE_MODEL, effort: 'low', maxTokens: 4000, visibleTokens: 3000 }),
  repair: Object.freeze({ model: OPENAI_BIBLE_MODEL, effort: 'low', maxTokens: 1500, visibleTokens: 1000 }),
});
export const OPENAI_BIBLE_ENDPOINT = 'https://api.openai.com/v1/chat/completions';

export function openAIBibleBody(stage, system, input, outputSchema) {
  const config = OPENAI_BIBLE_STAGES[stage];
  if (!config) throw new BibleProviderError('AI_NOT_CONFIGURED', 503);
  return {
    model: config.model, reasoning_effort: config.effort,
    max_completion_tokens: config.maxTokens, stream: false, store: false,
    response_format: { type: 'json_schema', json_schema: { name: `bible_${stage}`, strict: true, schema: outputSchema } },
    messages: [{ role: 'system', content: system }, { role: 'user', content: JSON.stringify(input) }],
  };
}

export function isExpectedOpenAIModel(stage, model) {
  const expected = OPENAI_BIBLE_STAGES[stage]?.model;
  return !!expected && (model === expected || (typeof model === 'string' && model.startsWith(expected + '-') && /^\d{4}-\d{2}-\d{2}$/.test(model.slice(expected.length + 1))));
}
