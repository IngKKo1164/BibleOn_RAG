import { fileURLToPath } from 'node:url';
export const workspaceRoot = fileURLToPath(new URL('../../', import.meta.url));
// Historical evaluation tools use workspace-relative input/output paths.
// They must never inherit the app or a sibling research working directory.
process.chdir(workspaceRoot);
