// Presentation-only normalization. Keep each paragraph and its citations
// intact; no generated claims, summaries, or semantic deduplication.
export function formatAnswerPart(text,{continuation=false}={}){
  let value=text.trim().replace(/\r\n?/g,'\n');
  // A continuation is not a second independent answer. If the model put a
  // preamble before its own heading, move that existing heading above it.
  if(continuation){
    const heading=/^## [^\n]+$/m.exec(value);
    if(heading?.index>0){
      const lead=value.slice(0,heading.index).trim();
      if(lead&&!/^#{1,6} /m.test(lead))value=[heading[0],lead,value.slice(heading.index+heading[0].length).trim()].filter(Boolean).join('\n\n');
    }
  }
  return value.split(/\n\s*\n/).map(block=>{
    const allBold=/^\*\*([^*\n]+)\*\*$/.exec(block);
    return allBold&&allBold[1].length>80?allBold[1]:block;
  }).join('\n\n');
}
