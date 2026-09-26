import './lib/workspace.mjs';
import { spawnSync } from 'node:child_process';
import { resolve } from 'node:path';
import { workspaceRoot } from './lib/workspace.mjs';
const git = args => {
  const result = spawnSync('git', args, { encoding: 'utf8', windowsHide: true });
  if (result.status !== 0) throw Error(`Git check failed: ${args[0]}`);
  return result.stdout;
};
if (resolve(git(['rev-parse', '--show-toplevel']).trim()) !== resolve(workspaceRoot)) throw Error('Run only inside the independent graduation Git repository');
const staged = process.argv.includes('--staged');
const files = [...new Set(git(staged ? ['diff', '--cached', '--name-only', '--diff-filter=ACMR', '-z'] : ['ls-files', '--cached', '--others', '--exclude-standard', '-z']).split('\0').filter(Boolean))];
const rootFiles = new Set(['.gitignore','.env.example','README.md','package.json','package-lock.json','runtime-assets.json']);
const allowed = p => rootFiles.has(p) || /^(?:server|scripts|workflows|\.githooks)\//.test(p);
const localDocument = p => /(?:^|\/)(?:docs?|reports?|history|changelog)(?:\/|$)|(?:^|\/)(?:CHANGELOG|HISTORY)[^/]*$|\.(?:docx?|hwpx?|pdf|odt|rtf)$/i.test(p);
// --no-index also catches files previously tracked or force-added despite .gitignore.
const ignored = new Set();
if (files.length) {
  const result = spawnSync('git', ['check-ignore', '--no-index', '--stdin', '-z'], { input: files.join('\0') + '\0', encoding: 'utf8', windowsHide: true });
  if (![0, 1].includes(result.status)) throw Error('Could not verify Git ignore rules');
  for (const path of result.stdout.split('\0').filter(Boolean)) ignored.add(path);
}
const forbidden = p => !allowed(p) || localDocument(p) || ignored.has(p) || /^server\/data\//.test(p) && p !== 'server/data/bible-study-packed.mjs' || /(?:^|\/)(?:\.env(?:\.|$)|node_modules\/)|\.(?:pem|key|p12|pfx|jks|keystore)$/i.test(p) && p !== '.env.example';
const failures = files.filter(forbidden).map(p => `Private or unapproved path: ${p}`);
// Reject disallowed paths before reading source blobs from the index.
if (failures.length) throw Error(failures.join('\n'));
// Inspect staged bytes, not just the working copy, when used as a commit hook.
const { readFileSync } = await import('node:fs');
for (const file of files.filter(p => !forbidden(p))) {
  const text = staged ? git(['show', `:${file}`]) : readFileSync(file, 'utf8');
  if (/-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|\b(?:sk-(?:proj-)?|gsk_)[A-Za-z0-9_-]{30,}|\beyJ[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}\.[A-Za-z0-9_-]{20,}/.test(text)) failures.push(`Potential credential: ${file}`);
}
for (const sample of ['data/private/sources/private.txt','data/private/derived/vectors.json','artifacts/run/answer.json','server/data/krv.json','.env.local','docs/RAG_ADAPTIVE_REPAIR.md','scripts/docs/change.md','tests/report.docx','workflows/report.hwp','workflows/report.hwpx','workflows/report.pdf','tests/ai/langgraph.test.mjs','scripts/compare-gemma-nova.mjs','scripts/evaluate-rag-v4.mjs','server/gemma-provider.mjs','server/bible-pipeline-manual.mjs','server/new-unreviewed-file.mjs','AGENTS.md']) {
  if (spawnSync('git', ['check-ignore', '--quiet', '--no-index', sample], { windowsHide: true }).status !== 0) failures.push(`Ignore rule missing: ${sample}`);
}
if (failures.length) throw Error(failures.join('\n'));
console.log(JSON.stringify({ gitRoot: 'graduation-research', checkedFiles: files.length, staged, privateDataExcluded: true, localReportsExcluded: true }));
