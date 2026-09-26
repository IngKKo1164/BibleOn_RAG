// Bounded factual checks supplement source membership. They are NOT a general
// entailment judge: passing them does not prove all prose/interpretations true.
export function hasApparentDeathClaim(text) {
  for(const match of String(text).matchAll(/(?:죽은\s*것으로|죽었다고)\s*(?:여[가-힣]*|보[가-힣]*|생각[가-힣]*|추정[가-힣]*|간주[가-힣]*)/g)){
    const tail=String(text).slice(match.index+match[0].length).split(/[.!?]/)[0];
    if(!/(?:다는|라는)\s*(?:뜻|말|표현|주장)?(?:은|이)?\s*(?:아니|아닙)/.test(match[0]+tail))return true;
  }
  return false;
}
export function outcomeConflicts(answer,evidence) {
  // High-precision contradiction only: a named person in a narrator's explicit
  // death statement, paraphrased as apparent death in that person's paragraph.
  // Speech/metaphor and unrelated people are not adjudicated by this rule.
  const found=new Map(), blocks=String(answer).split(/\n\s*\n/);
  for(const verse of evidence){
    if(!/일으켜\s*보니\s*죽었|죽었는지라/.test(verse.text))continue;
    const subject=verse.text.match(/([가-힣]{2,12})라\s*하는\s*(?:청년|사람|남자|여자)/)?.[1];
    if(!subject)continue;
    for(const [blockIndex,original] of blocks.entries()){
      const block=original.replace(/\*\*/g,'');
      if(!block.includes(subject))continue;
      if(hasApparentDeathClaim(block)){
        found.set(blockIndex,{rule:'declared-death',verseId:verse.id,quote:verse.text,subject,blockIndex,original});
      }
    }
  }
  return [...found.values()];
}
export const explicitOutcomeConflict=(answer,evidence)=>outcomeConflicts(answer,evidence)[0]||null;
export function groundedClaimFailure(answer, evidence) {
  const text = String(answer).replace(/\*\*/g, '');
  const canonical = evidence.map(v => v.text).join('\n');
  // Personal offices/professions need textual support. In particular, a real
  // verse attached to an invented biography must not make that biography pass.
  for (const sentence of text.split(/(?<=[.!?])\s+|\n/)) {
    const patterns = [
      /([가-힣]{2,15})(?:은|는|이|가)\s+[^.!?\n]{0,50}?(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:이었|였|입니|이다|로\s*일했)/g,
      /(황제|총독|대제사장|제사장|목수|어부|세리|군인)(?:인|였던|이었던)\s+([가-힣]{2,15}?)(?=은|는|이|가|을|를|[\s,.!?]|$)/g,
    ];
    for (const [order, pattern] of patterns.entries()) for (const match of sentence.matchAll(pattern)) {
      const subject=match[order===0?1:2], role=match[order===0?2:1];
      // Only an explicit rejection of this predicate is exempt. A later
      // unrelated negation must not excuse an asserted invented profession.
      const tail=sentence.slice(match.index+match[0].length);
      if (/^(?:다는|라는)\s*(?:뜻|말|주장|설명)(?:은|이)?\s*(?:아닙|아니|틀렸|잘못)/.test(tail)) continue;
      if (!evidence.some(v => v.text.includes(subject) && v.text.includes(role))) return 'AI_FACTS_UNSUPPORTED';
    }
  }
  // Literal matching is suitable only for claims of a verbatim quotation,
  // not ordinary Korean paraphrases of a biblical speaker's words. Also check
  // quoted unconditional financial/salvation promises against the source.
  const quotes = text.matchAll(/[‘“]([^’”\n]{6,300})[’”](?:라고|고)\s*(?:말씀|말했|말합|기록|선포|외쳤)/g);
  for (const match of quotes) {
    const quote = match[1], before = text.slice(Math.max(0, match.index - 50), match.index);
    const verbatim = /원문|그대로\s*인용|직접\s*인용/.test(before);
    const promisedBenefits = /(?:헌금|기부|돈).{0,30}(?:언제나|항상|반드시|무조건|보장).{0,30}(?:부자|부유|구원|치유)/.test(quote);
    if (!verbatim && !promisedBenefits) continue;
    const normalized = quote.replace(/[\s.,!?]/g, '');
    // A modern sentence ending does not change the quoted proposition, e.g.
    // '놓였다' versus KRV '놓였으니'. Keep all negation/content words intact.
    const stem = normalized.replace(/(?:습니다|니다|느니라|으니|니라|으라|라|다)$/, '');
    const source = canonical.replace(/[\s.,!?]/g, '');
    if (!source.includes(normalized) && !(stem.length >= 6 && source.includes(stem))) return 'AI_QUOTE_INVALID';
  }
  return null;
}
