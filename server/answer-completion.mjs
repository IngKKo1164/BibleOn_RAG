// Preserve the prose-ending check without rejecting complete Markdown blocks.
// This is an ending guard, not a sanitizer or a factual verifier.
export function hasCompleteAnswerEnding(text) {
  if (typeof text !== 'string' || !text.trim()) return false;
  // Decorative trailing emoji may follow a complete sentence. Stripping only
  // that suffix still leaves unfinished prose subject to the ending check.
  const body = text.trim().replace(/[\p{Extended_Pictographic}\p{Emoji_Modifier}\uFE0F\u200D\s]+$/gu, '').trim();
  if (!body) return false;
  const lines = body.split(/\r?\n/);
  let fence = null, closedAt = -1, hasCode = false;
  for (const [index, line] of lines.entries()) {
    const marker = line.match(/^ {0,3}(`{3,}|~{3,})(.*)$/);
    if (!fence && marker && !(marker[1][0] === '`' && marker[2].includes('`'))) {
      fence = marker[1]; hasCode = false;
    } else if (fence) {
      if (marker && marker[1][0] === fence[0] && marker[1].length >= fence.length && !marker[2].trim()) {
        closedAt = hasCode ? index : -1; fence = null;
      } else if (line.trim()) hasCode = true;
    }
  }
  if (fence) return false;
  if (closedAt === lines.length - 1) return true;
  const last = lines.at(-1);
  if (/[.!?。！？][\s"'”’)*\]]*$/.test(last)) return true;
  if (/^\s*(?:[-+*]|\d+[.)])\s+\S(?:.*\S)?$/.test(last) && /[\p{L}\p{N}]/u.test(last)) {
    // Bare labels are valid lists; dangling Korean clauses are not.
    return !/(?:\s때|[가-힣](?:하며|하면서|하지만|때문에|위해서|통해서|하므로|한다면|있다면|있으며))[*_\s]*$/.test(last);
  }
  const cells = (line) => line.trim().replace(/^\||\|$/g, '').split(/(?<!\\)\|/).map((cell) => cell.trim());
  // Require an actual table header/delimiter, not merely a trailing pipe.
  for (let i = lines.length - 2; i > 0 && lines[i].trim(); i--) {
    const delimiter = cells(lines[i]);
    if (delimiter.length >= 2 && delimiter.every((cell) => /^:?-{3,}:?$/.test(cell))
      && cells(lines[i - 1]).length === delimiter.length
      && lines.slice(i + 1).every((line) => cells(line).length === delimiter.length && cells(line).every(Boolean))) return true;
  }
  return false;
}
