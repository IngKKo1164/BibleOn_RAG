import './lib/workspace.mjs';
import { loadEnv } from './lib/rag-env.mjs';
import { loadPackedBibleCorpus } from '../server/data/krv-packed.mjs';
import { runBibleGraph } from '../workflows/bible-graph.mjs';
import { createStudyIndex } from '../server/bible-study.mjs';
import { loadStudyData } from '../server/data/bible-study-packed.mjs';
const args = process.argv.slice(2);
if (args.shift() !== '--live' || !args.length) throw Error('Paid model call: npm run ask -- --live "질문". Put provider keys in this workspace only.');
const env = loadEnv();
const hosts = new Set(['api.openai.com', 'dashscope-intl.aliyuncs.com']);
const workspace = env.QWEN_WORKSPACE_ID?.trim();
if (workspace) {
  if (!/^[a-zA-Z0-9-]+$/.test(workspace)) throw Error('INVALID_EMBEDDING_WORKSPACE');
  hosts.add(`${workspace}.ap-southeast-1.maas.aliyuncs.com`);
}
const fetcher = (input, init) => {
  const url = new URL(typeof input === 'string' || input instanceof URL ? input : input.url);
  if (url.protocol !== 'https:' || !hosts.has(url.hostname)) throw Error(`Research call not permitted: ${url.hostname}`);
  return fetch(input, { ...init, redirect: 'error' });
};
const corpus = await loadPackedBibleCorpus();
const result = await runBibleGraph({ env, corpus, question: args.join(' '), history: [], studyIndex: createStudyIndex(corpus, loadStudyData()), fetcher });
console.log(JSON.stringify(result, null, 2));
