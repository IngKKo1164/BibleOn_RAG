import {existsSync,readFileSync} from 'node:fs';
import {resolve} from 'node:path';
import {parseEnv} from 'node:util';
import {workspaceRoot} from './workspace.mjs';
import {RESEARCH_DEFAULTS} from '../../server/research-config.mjs';

export function loadEnv() {
  const values={...RESEARCH_DEFAULTS};
  for (const name of ['.env','.env.local']) {
    const file=resolve(workspaceRoot,name);
    if (existsSync(file)) Object.assign(values,parseEnv(readFileSync(file,'utf8')));
  }
  Object.assign(values,process.env);
  if (values.BIBLE_AI_PROVIDER!=='openai'||values.BIBLE_RETRIEVAL_VERSION!=='v4') throw Error('This entry point requires the current OpenAI/LangGraph pipeline.');
  const keys=new Set(['OPENAI_API_KEY','QWEN_API_KEY','QWEN_WORKSPACE_ID',...Object.keys(RESEARCH_DEFAULTS)]);
  return Object.fromEntries(Object.entries(values).filter(([key])=>keys.has(key)));
}
