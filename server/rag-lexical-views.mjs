import {createBibleRetriever,retrievalScope,ragTerms} from './bible-rag.mjs';

// General vocabulary hints only. No answer locations, corpus facts, generated
// evaluation questions or automatically inverted dictionary definitions.
const hints=[
 [/기도|간구/,'기도 간구 빌매 구하'],[/빌려|대여/,'빌리 빌려'],[/도끼/,'도끼 부'],
 [/맹인|시각\s*장애|보지\s*못/,'소경 눈 맹인'],[/피부병|나병/,'문둥병'],
 [/잘라|절단|자르/,'찍어 버리 빼어'],[/회복/,'돌이키 곤경 회복'],
 [/치료|낫게|고치/,'고치 낫 나으'],[/자녀|아이들/,'자녀 자식'],
 [/정직/,'정직 진실'],[/화해/,'화목 화평'],[/도움|돕/,'도우 구제'],
 [/걱정|불안/,'염려 근심'],[/돈|재산/,'재물 은 돈'],[/병사|군인/,'군사'],
 [/죽음|사망/,'죽 사망'],[/부활|다시\s*살/,'살리 부활'],
 [/노예/,'종 노예'],[/감옥/,'옥 갇히'],[/세금/,'세리 세금 세'],
];
export function lexicalHints(question) {
  return [...new Set(hints.filter(([pattern])=>pattern.test(question)).map(([,terms])=>terms))].join(' ');
}
const views=new WeakMap();
function windowView(index,corpus) {
  if(views.has(index))return views.get(index);
  const parents=new Map(),inverse=new Map();
  index.chunks.forEach((c,i)=>{
    parents.set(c.sectionId||c.id,{...c,start:c.sectionStart??c.start,end:c.sectionEnd??c.end});
    c.verses.forEach(v=>{const id=`${c.bookId}:${c.chapter}:${v.verse}`;inverse.set(id,[...(inverse.get(id)||[]),i]);});
  });
  const docs=[];
  for(const p of parents.values())for(let v=p.start;v<=p.end;v++){
    if(!corpus[p.bookId]?.chapters[p.chapter]?.[v])continue;
    const verses=[];
    for(let n=Math.max(p.start,v-2);n<=Math.min(p.end,v+2);n++){
      const text=corpus[p.bookId].chapters[p.chapter][n];if(text)verses.push({verse:n,text});
    }
    docs.push({...p,id:`${p.bookId}:${p.chapter}:${v}`,verses,text:verses.map(x=>x.text).join('\n')});
  }
  const result={index:createBibleRetriever(corpus,null,null,docs),docs,inverse};views.set(index,result);return result;
}
export function lexicalViewBranches(index,corpus,question,{retry=false,scopeQuestion}={}) {
  const scope=retrievalScope(corpus,question,{coordinatedBooks:true}),
    books=new Set(scopeQuestion===undefined?scope.bookIds:retrievalScope(corpus,scopeQuestion,{coordinatedBooks:true}).bookIds),query=scope.contentQuery||question;
  const hint=lexicalHints(query),augmented=hint?`${query} ${hint}`:query,limit=retry?32:16;
  const eligible=i=>!books.size||books.has(index.chunks[i].bookId),branches=[];
  if(hint)branches.push(['modern-vocabulary',index.lexical(augmented).filter(r=>eligible(r.index)).slice(0,limit)]);
  const view=windowView(index,corpus),seen=new Set(),rows=[];
  for(const r of view.index.lexical(augmented)){
    for(const i of view.inverse.get(view.docs[r.index].id)||[]){
      if(seen.has(i)||!eligible(i))continue;seen.add(i);rows.push({index:i,score:r.score});
    }
    if(rows.length>=limit)break;
  }
  branches.push(['five-verse-window',rows.slice(0,limit)]);
  if(retry){
    const important=[...new Set(ragTerms(query).filter(t=>t.startsWith('w:')).map(t=>t.slice(2)))].filter(w=>w.length>1).join(' ');
    if(important)branches.push(['missing-part',index.focused(`${important} ${hint}`).filter(r=>eligible(r.index)).slice(0,16)]);
  }
  return branches;
}
