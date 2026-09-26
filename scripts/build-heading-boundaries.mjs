import './lib/workspace.mjs';
import { readFileSync, readdirSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { compileHeadingBoundaries } from '../server/rag-heading-sections.mjs';
const hash = value => createHash('sha256').update(value).digest('hex');
const source = 'data/private/sources/rnksv';
const files = readdirSync(source).filter(f => f.endsWith('.json')).sort();
const documents = files.map(f => JSON.parse(readFileSync(`${source}/${f}`, 'utf8')));
const corpus = JSON.parse(readFileSync('server/data/krv.json', 'utf8'));
const result = compileHeadingBoundaries(corpus, documents);
const metadata = {version:'rnksv-heading-coordinates-v1', corpusSha256:hash(readFileSync('server/data/krv.json')),
  sourceSha256:hash(files.map(f => `${f}:${hash(readFileSync(`${source}/${f}`))}`).join('\n')),
  boundarySha256:hash(JSON.stringify(result.boundaries)), chapterBoundary:true, textTranslation:'KRV', headingLabelsIncluded:false};
mkdirSync('data/private/derived/heading-chunks', {recursive:true});
writeFileSync('server/data/rag-heading-boundaries.mjs', `// Generated private coordinate-only asset. No translation text or heading labels.\nexport const metadata = ${JSON.stringify(metadata)};\nexport const boundaries = ${JSON.stringify(result.boundaries)};\n`);
writeFileSync('data/private/derived/heading-chunks/boundary-audit.json', JSON.stringify({metadata, ...result.audit}, null, 2));
console.log(JSON.stringify({metadata, headings:result.audit.headings, distinctHeadingStarts:result.audit.distinctHeadingStarts, differingChapters:result.audit.differences.length, unalignedHeadings:0}));
