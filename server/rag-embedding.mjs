export const EMBEDDING_MODEL = 'text-embedding-v4';
export const EMBEDDING_DIMENSIONS = 512;
export const QUERY_INSTRUCTION = 'Given a Korean Bible question, retrieve Bible passages that directly answer the question.';
export async function embedTexts(texts, { env, fetcher = fetch, signal, type = 'document' } = {}) {
  if (!Array.isArray(texts) || !texts.length || texts.length > 10 || texts.some(t => typeof t !== 'string' || !t.trim() || t.length > 16000)) throw Error('INVALID_EMBEDDING_INPUT');
  const workspace = env.QWEN_WORKSPACE_ID?.trim();
  if (workspace && !/^[a-zA-Z0-9-]+$/.test(workspace)) throw Error('INVALID_EMBEDDING_WORKSPACE');
  const host = workspace ? `${workspace}.ap-southeast-1.maas.aliyuncs.com` : 'dashscope-intl.aliyuncs.com';
  const query = type === 'query';
  const response = await fetcher(`https://${host}${query ? '/api/v1/services/embeddings/text-embedding/text-embedding' : '/compatible-mode/v1/embeddings'}`, { method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${env.QWEN_API_KEY}` },
    signal: signal ? AbortSignal.any([signal, AbortSignal.timeout(12000)]) : AbortSignal.timeout(12000),
    body: JSON.stringify(query ? { model: EMBEDDING_MODEL, input: { texts }, parameters: { dimension: EMBEDDING_DIMENSIONS, text_type: 'query', instruct: QUERY_INSTRUCTION } }
      : { model: EMBEDDING_MODEL, input: texts, dimensions: EMBEDDING_DIMENSIONS, encoding_format: 'float' }) });
  const data = await response.json();
  if (!response.ok) throw Object.assign(Error('EMBEDDING_UNAVAILABLE'), { status: response.status, code: data.error?.code || data.code });
  if (query) { data.data = data.output?.embeddings?.map(row => ({ index: row.text_index, embedding: row.embedding })); data.usage = { prompt_tokens: data.usage?.total_tokens }; }
  const inputTokens = data.usage?.prompt_tokens;
  if (!Number.isInteger(inputTokens) || inputTokens < 0 || data.data?.length !== texts.length) throw Error('INVALID_EMBEDDING_RESPONSE');
  const vectors = [];
  for (const row of data.data) {
    if (!Number.isInteger(row.index) || row.index < 0 || row.index >= texts.length || vectors[row.index]
      || row.embedding?.length !== EMBEDDING_DIMENSIONS || !row.embedding.every(Number.isFinite)
      || !row.embedding.some(n => n !== 0)) throw Error('INVALID_EMBEDDING_VECTOR');
    vectors[row.index] = row.embedding;
  }
  return { vectors, inputTokens, model: EMBEDDING_MODEL };
}
export function quantizeVector(vector) {
  const scale = Math.max(...vector.map(Math.abs)) || 1;
  return Int8Array.from(vector, n => Math.round(n / scale * 127));
}
