import './lib/workspace.mjs';
import {readFileSync,existsSync,lstatSync} from 'node:fs';
import {dirname,resolve,sep} from 'node:path';
import {spawnSync} from 'node:child_process';
import {workspaceRoot} from './lib/workspace.mjs';

// Validate only the exact publication allowlist, including in a fresh clone.
// Historical local experiments and tests are deliberately outside this scope.
const root=resolve(workspaceRoot);
const files=readFileSync('.gitignore','utf8').split(/\r?\n/)
  .map(line=>line.match(/^!\/([^*?\[\]]+[^/])$/)?.[1]).filter(Boolean);
const allowed=new Set(files);
const dependencies=JSON.parse(readFileSync('package.json','utf8')).dependencies;
const assets=JSON.parse(readFileSync('runtime-assets.json','utf8'));
const runtimeData=new Set(assets.map(asset=>'server/data/'+asset.file));
let modules=0;
for (const file of allowed) {
  const absolute=resolve(root,file);
  if (!absolute.startsWith(root+sep)||!existsSync(absolute)||!lstatSync(absolute).isFile()||lstatSync(absolute).isSymbolicLink()) throw Error(`Missing or unsafe public file: ${file}`);
  if (!file.endsWith('.mjs')) continue;
  const text=readFileSync(absolute,'utf8');
  const imports=[...text.matchAll(/^(?:import|export)\b[^\n]*?\bfrom\s*['"]([^'"]+)['"]|^import\s*['"]([^'"]+)['"]|\bimport\s*\(\s*['"]([^'"]+)['"]/gm)];
  for (const match of imports) {
    const specifier=match[1]||match[2]||match[3];
    if (specifier.startsWith('node:')) continue;
    if (specifier.startsWith('.')) {
      const target=resolve(dirname(absolute),specifier);
      if (!target.startsWith(root+sep)) throw Error(`Import escapes repository: ${file}`);
      const name=target.slice(root.length+1).split(sep).join('/');
      if (!allowed.has(name)&&!runtimeData.has(name)) throw Error(`Import not published or declared as private data: ${file} -> ${specifier}`);
    } else {
      const name=specifier.startsWith('@')?specifier.split('/').slice(0,2).join('/'):specifier.split('/')[0];
      if (!dependencies[name]) throw Error(`Undeclared package: ${file} -> ${name}`);
    }
  }
  const syntax=spawnSync(process.execPath,['--check',absolute],{encoding:'utf8',windowsHide:true});
  if (syntax.status!==0) throw Error(`Invalid module: ${file}\n${syntax.stderr}`);
  modules++;
}
for (const [name,command] of Object.entries(JSON.parse(readFileSync('package.json','utf8')).scripts)) {
  const entry=command.match(/^node (\S+\.mjs)(?:\s|$)/)?.[1];
  if (!entry||!allowed.has(entry)) throw Error(`Package command references an unpublished file: ${name}`);
}
const missingData=[...runtimeData].filter(file=>!existsSync(file));
let graphNodes;
if (process.argv.includes('--with-data')) {
  if (missingData.length) throw Error(`Restore authorized runtime data first: ${missingData.join(', ')}`);
  const {buildBibleGraph}=await import('../workflows/bible-graph.mjs');
  const graph=await buildBibleGraph(null).getGraphAsync();
  if (!graph.nodes.rerank||!graph.nodes.generate) throw Error('Required graph nodes are missing');
  graphNodes=Object.keys(graph.nodes).length-2;
}
console.log(JSON.stringify({publishedFiles:allowed.size,checkedModules:modules,publicImportClosure:true,missingPrivateAssets:missingData.length,graphNodes,networkCalls:0}));
