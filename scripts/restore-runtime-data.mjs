import './lib/workspace.mjs';
import { readFileSync, existsSync, mkdirSync, copyFileSync, lstatSync } from 'node:fs';
import { resolve, dirname, sep } from 'node:path';
import { createHash } from 'node:crypto';
const source = process.argv[2];
if (!source) throw Error('Use npm run data:restore -- <authorized-backup-server/data-directory>');
const sourceRoot = resolve(source), targetRoot = resolve('server/data');
const assets = JSON.parse(readFileSync('runtime-assets.json','utf8'));
const planned = assets.map(({ file, sha256 }) => {
  if (!/^[a-z0-9.-]+$/.test(file)) throw Error('Invalid runtime asset name');
  const from = resolve(sourceRoot, file), to = resolve(targetRoot, file);
  if (!from.startsWith(sourceRoot+sep) || !to.startsWith(targetRoot+sep) || lstatSync(from).isSymbolicLink()) throw Error('Invalid runtime asset path');
  const bytes = readFileSync(from);
  if (createHash('sha256').update(bytes).digest('hex') !== sha256) throw Error(`Asset does not match this research baseline: ${file}`);
  if (existsSync(to) && !readFileSync(to).equals(bytes)) throw Error(`Refusing to overwrite modified research data: ${file}`);
  return { from, to };
});
for (const {from,to} of planned) { mkdirSync(dirname(to),{recursive:true}); if (!existsSync(to)) copyFileSync(from,to); }
console.log(JSON.stringify({restored:planned.length, networkUsed:false, gitExcluded:true}));
