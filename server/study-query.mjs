// Bilingual retrieval aliases, not definitions or claims of word alignment.
// Strong IDs are accepted only when the user actually supplies them.
const aliases=[
 [/바리새/,'Pharisees'],[/사두개/,'Sadducees'],[/회당/,'Synagogue'],[/안식일/,'Sabbath'],
 [/유월절/,'Passover'],[/담무스/,'Tammuz'],[/니느웨/,'Nineveh'],[/사마리아/,'Samaria'],
 [/로마/,'Rome'],[/드로아/,'Troas'],[/유두고/,'Eutychus'],[/세례/,'Baptism'],
 [/성전/,'Temple'],[/성막/,'Tabernacle'],[/할례/,'Circumcision'],[/제사장/,'Priest'],
 [/대제사장/,'High priest'],[/구속|속량/,'Redemption'],[/구원/,'Salvation'],[/은혜/,'Grace'],
 [/믿음/,'Faith'],[/양선|선행/,'Goodness'],[/거룩|성화/,'Sanctification'],
 [/칭의/,'Justification'],[/화목/,'Reconciliation'],[/부활/,'Resurrection'],
 [/언약/,'Covenant'],[/바벨론/,'Babylon'],[/앗수르|아시리아/,'Assyria'],
 [/가나안/,'Canaan'],[/벧엘/,'Bethel'],[/욥/,'Job'],[/니고데모/,'Nicodemus'],
];
export function studyQueryTerms(question,modelHints=[]) {
  const terms=[...aliases.filter(([p])=>p.test(question)).map(([,t])=>t),...modelHints];
  for(const m of question.matchAll(/\b[A-Za-z][A-Za-z'-]{2,30}\b/g))if(!/^[GH]\d+$/i.test(m[0]))terms.push(m[0]);
  return [...new Set(terms.filter(t=>typeof t==='string'&&/^[a-zA-Z][a-zA-Z '-]{0,47}$/.test(t)).map(t=>t.toLowerCase().trim()))].slice(0,8);
}
const tokenCache=new WeakMap();
export function searchIndependentStudy(index,terms,question) {
  if(!tokenCache.has(index)){
    const titles=new Map();
    for(const row of index.records.values()){
      if(!['easton','mhcc'].includes(row.source))continue;
      for(const word of new Set(row.title.toLowerCase().match(/[a-z]+/g)||[])){
        if(!titles.has(word))titles.set(word,[]);titles.get(word).push(row);
      }
    }
    tokenCache.set(index,titles);
  }
  const matches=new Map(),titles=tokenCache.get(index);
  for(const phrase of terms){
    const tokens=phrase.match(/[a-z]+/g)||[];
    for(const token of tokens)for(const row of titles.get(token)||[]){
      const title=row.title.toLowerCase(),words=new Set(title.match(/[a-z]+/g)||[]);
      if(!tokens.every(t=>words.has(t)))continue;
      const score=title===phrase?100:50+tokens.length;
      if(!matches.has(row.id)||matches.get(row.id).score<score)matches.set(row.id,{row,score,method:'independent-title'});
    }
  }
  for(const match of question.matchAll(/\bG(\d{1,5})\b/gi)){
    const key='G'+String(Number(match[1])).padStart(4,'0'),id=index.data.aliases?.['strong:'+key]||'strong:'+key,row=index.records.get(id);
    if(row?.source==='strongGreek')matches.set(row.id,{row,score:120,method:'explicit-lexicon-id'});
  }
  return [...matches.values()].sort((a,b)=>b.score-a.score||a.row.id.localeCompare(b.row.id)).slice(0,32);
}
