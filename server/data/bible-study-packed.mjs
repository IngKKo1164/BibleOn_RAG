// Local import/evaluation helper only; NEVER bundled into the Edge Function.
import { readFileSync } from 'node:fs';
import { gunzipSync } from 'node:zlib';
export function loadStudyData() { return JSON.parse(gunzipSync(readFileSync(new URL('./bible-study.json.gz', import.meta.url))).toString('utf8')); }
