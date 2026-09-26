export class BibleProviderError extends Error {
  constructor(code, status = 502) { super(code); this.code = code; this.status = status; }
}
