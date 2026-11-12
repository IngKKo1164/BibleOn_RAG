"""Bounded one-hop references and separately attributed historical sources."""
from __future__ import annotations

import re
import unicodedata
import math
from collections import Counter

from .common import canonical_id_sort_key, text_length, unique
from .passages import verse_at
from .prompt_text import STUDY_SOURCES
from .retrieval import rag_terms
from .validation import has_apparent_death_claim, has_complete_answer_ending, korean_only

STUDY_LIMITS = {"roots": 6, "linkedVerses": 12, "linkedCharacters": 1500, "sources": 3,
                "sourceCharacters": 4000, "excerptCharacters": 2400, "notes": 2}
ALIASES = [(r"바리새", "Pharisees"), (r"사두개", "Sadducees"), (r"회당", "Synagogue"), (r"안식일", "Sabbath"),
           (r"유월절", "Passover"), (r"담무스", "Tammuz"), (r"니느웨", "Nineveh"), (r"사마리아", "Samaria"),
           (r"로마", "Rome"), (r"드로아", "Troas"), (r"유두고", "Eutychus"), (r"세례", "Baptism"),
           (r"성전", "Temple"), (r"성막", "Tabernacle"), (r"할례", "Circumcision"), (r"제사장", "Priest"),
           (r"대제사장", "High priest"), (r"구속|속량", "Redemption"), (r"구원", "Salvation"), (r"은혜", "Grace"),
           (r"믿음", "Faith"), (r"양선|선행", "Goodness"), (r"거룩|성화", "Sanctification"),
           (r"칭의", "Justification"), (r"화목", "Reconciliation"), (r"부활", "Resurrection"),
           (r"언약", "Covenant"), (r"바벨론", "Babylon"), (r"앗수르|아시리아", "Assyria"),
           (r"가나안", "Canaan"), (r"벧엘", "Bethel"), (r"욥", "Job"), (r"니고데모", "Nicodemus")]


def _indexed(data, key, default):
    if isinstance(data, list):
        return data[key] if isinstance(key, int) and 0 <= key < len(data) else default
    return data.get(str(key), data.get(key, default))


class StudyIndex:
    def __init__(self, corpus, data):
        canonical = [f"{b}:{c}:{v}" for b, book in corpus.items() for c in sorted(book["chapters"], key=int)
                     for v in sorted(book["chapters"][c], key=int)]
        if canonical != data["verseIds"]:
            raise ValueError("STUDY_CORPUS_MISMATCH")
        self.data, self.ids = data, {key: i for i, key in enumerate(canonical)}
        self.records, self.by_verse, self.titles = {}, {}, {}
        self.postings, self.lengths = {}, {}
        for row in data["records"]:
            self.records[row["id"]] = row
            for v in row["refs"]:
                self.by_verse.setdefault(v, []).append(row)
            if row["source"] in ("easton", "mhcc"):
                words = Counter(search_words(row["title"]+" "+row["text"]))
                self.lengths[row["id"]] = sum(words.values())
                for word, count in words.items():
                    self.postings.setdefault(word, {})[row["id"]] = count
                for word in unique(re.findall(r"[a-z]+", row["title"].lower())):
                    self.titles.setdefault(word, []).append(row)
        self.average_length = sum(self.lengths.values()) / max(1, len(self.lengths))
        # Supplementary chapter metadata never enters baseline postings, title
        # terms or verse-linked study candidates. It is supplied explicitly.
        self.superscriptions = {}
        for row in data.get('superscriptions', []):
            book, chapter = row.get('bookId'), row.get('chapter')
            anchor = f'{book}:{chapter}:1'
            if (row.get('source') != 'web_superscriptions' or anchor not in self.ids
                    or row.get('refs') != [self.ids[anchor]] or row.get('id') in self.records
                    or (book, chapter) in self.superscriptions or not isinstance(row.get('text'), str)
                    or not 4 <= len(row['text']) <= 1000):
                raise ValueError('INVALID_SUPERSCRIPTION_RECORD')
            self.superscriptions[book, chapter] = row
            self.records[row['id']] = row


def add_superscriptions(primary, sources, index):
    """Supply selected-chapter metadata within the three-source/4000-char caps.

    These are translated titles, not new verses, retrieval hits or authorship
    conclusions. Recovery's existing merge preserves accepted S-id bindings.
    """
    if not index.superscriptions:
        raise ValueError('SUPERSCRIPTION_DATA_UNAVAILABLE')
    chosen, seen = [], set()
    for verse in primary:
        key = verse['bookId'], verse['chapter']
        row = index.superscriptions.get(key)
        if row is None or key in seen:
            continue
        source = STUDY_SOURCES[row['source']]
        url = f"https://ebible.org/engwebp/PSA{row['chapter']:03}.htm"
        chosen.append({'recordId': row['id'], 'source': row['source'], 'title': row['title'],
                       'kind': source['kind'], 'excerpt': row['text'],
                       'retrieval': 'selected-chapter-metadata', 'license': source['license'],
                       'sourceUrl': url, 'context': {'publicationYear': None,
                           'eventTime': [], 'compositionTime': [], 'place': [],
                           'interpretiveStatus': 'translated_superscription', 'sourceUrl': url}})
        reading = index.data.get('superscriptionReadings', {}).get(row['id'])
        if reading is not None:
            from .common import sha256
            if (not isinstance(reading, dict) or reading.get('sourceTextSha256') != sha256(row['text'])
                    or not isinstance(reading.get('note'), str) or not 1 <= len(reading['note']) <= 300):
                raise ValueError('SUPERSCRIPTION_READING_STALE')
            # A separately reviewed reading aid, not a new verse, English
            # quotation, historical date or metadata-only citation target.
            chosen[-1]['context']['sourceReadingNote'] = reading['note']
        seen.add(key)
        # One chapter title cannot consume all three source slots.
        break
    if not chosen:
        return sources
    chars = sum(text_length(s['excerpt']) for s in chosen)
    for source in sources:
        if len(chosen) == 3 or chars + text_length(source['excerpt']) > 4000:
            continue
        chosen.append(source)
        chars += text_length(source['excerpt'])
    return [{**s, 'id': f'S{i}'} for i, s in enumerate(chosen, 1)]


def search_words(text):
    stop = {"the", "and", "with", "that", "this", "from", "was", "were", "for", "into", "are", "his", "her", "their"}
    return [w for w in re.findall(r"[a-z]{3,}", text.lower()) if w not in stop]


def source_context(row, source):
    """Unspecified historical facts stay unknown; publication year is not event time.

    Optional curated claims must carry an exact supporting span. They are never
    generated from titles, model memory or dates in a bibliography.
    """
    context = {"publicationYear": {"easton": 1897, "mhcc": 1706, "strongGreek": 1890}.get(row["source"]),
               "eventTime": [], "compositionTime": [], "place": [],
               "interpretiveStatus": "historical-source-view", "sourceUrl": source["url"]}
    for claim in row.get("contextClaims", []):
        if not isinstance(claim, dict):
            continue
        field, quote, value = claim.get("field"), claim.get("quote"), claim.get("value")
        if (field in {"eventTime", "compositionTime", "place"} and isinstance(quote, str)
                and 4 <= len(quote) <= 500 and quote in row["text"] and isinstance(value, str)
                and value and value in quote and claim.get("certainty") in {"stated", "traditional", "proposed"}):
            context[field].append({"value": value, "quote": quote, "certainty": claim["certainty"]})
    return context


def _term_score(text, terms):
    return sum(t in terms for t in rag_terms(text))


def provided_reference_roots(primary, supplied, question, index):
    """Share six existing one-hop roots between anchors and supplied passages.

    No invented coordinates or unseen retrieval hits. A secondary quotation can
    link to its original passage; neither a link nor lexical rank proves that
    relationship. Keep the same downstream target/character caps and review.
    """
    terms = set(rag_terms(question))
    def ranked(rows):
        seen, result = set(), []
        for pos, v in sorted(enumerate(rows), key=lambda p: (-_term_score(p[1]['text'], terms), p[0])):
            if v['id'] not in seen and _indexed(index.data['graph'], index.ids.get(v['id']), []):
                result.append(v); seen.add(v['id'])
        return result
    anchors = ranked(primary)
    anchored = {v['id'] for v in primary}
    others = ranked([v for v in supplied if v['id'] not in anchored])
    chosen = anchors[:3] if others else anchors[:6]
    # First admit a distinct chapter per secondary root; then fill spare roots.
    groups = set()
    for v in others:
        group = (v['bookId'], v['chapter'])
        if group not in groups and len(chosen) < 6:
            chosen.append(v); groups.add(group)
    ids = {v['id'] for v in chosen}
    for v in others + anchors:
        if len(chosen) == 6: break
        if v['id'] not in ids:
            chosen.append(v); ids.add(v['id'])
    return chosen


def expand_one_hop(corpus, primary, question, index, limits=None):
    limits = limits or STUDY_LIMITS
    existing, terms = {v["id"] for v in primary}, set(rag_terms(question))
    roots = sorted(enumerate(primary), key=lambda row: (-_term_score(row[1]["text"], terms), row[0]))[:limits["roots"]]
    candidates = {}
    for rank, (_, root) in enumerate(roots):
        for target in _indexed(index.data["graph"], index.ids.get(root["id"]), []):
            key = index.data["verseIds"][target]
            if not key or key in existing:
                continue
            book, chapter, verse = key.split(":")
            try:
                canonical = verse_at(corpus, {"bookId": book, "chapter": int(chapter), "verse": int(verse)})
            except ValueError:
                continue
            row = candidates.setdefault(key, {"verse": canonical, "roots": [], "score": 0})
            row["roots"].append(root["id"])
            row["score"] += 2/(rank+1) + _term_score(canonical["text"], terms)/8
    linked, relations, chars = [], [], 0
    for row in sorted(candidates.values(), key=lambda r: (-r["score"], index.ids[r["verse"]["id"]])):
        if len(linked) >= limits["linkedVerses"]:
            break
        if chars + text_length(row["verse"]["text"]) > limits["linkedCharacters"]:
            continue
        linked.append(row["verse"]); chars += text_length(row["verse"]["text"])
        relations.extend({"from": root, "to": row["verse"]["id"]} for root in row["roots"])
    return {"evidence": primary+linked, "linked": linked, "relations": relations, "roots": [v["id"] for _, v in roots]}


def normalize_study_terms(value):
    if not isinstance(value, list):
        return []
    return unique(s.lower().strip() for s in value if isinstance(s, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z '-]{0,47}", s))[:3]


def study_query_terms(question, model_hints=()):
    terms = [t for p, t in ALIASES if re.search(p, question)] + list(model_hints)
    terms += [m[0] for m in re.finditer(r"\b[A-Za-z][A-Za-z'-]{2,30}\b", question, re.A) if not re.fullmatch(r"[GH]\d+", m[0], re.I)]
    return unique(t.lower().strip() for t in terms if isinstance(t, str) and re.fullmatch(r"[a-zA-Z][a-zA-Z '-]{0,47}", t))[:8]


def search_independent_study(index, terms, question, *, fulltext=True, preserve_scores=False):
    matches = {}
    # BM25 over whole records complements exact dictionary-title matching.
    scores = {}
    count = len(index.lengths)
    for token in set(search_words(" ".join(terms))) if fulltext else ():
        posting = index.postings.get(token, {})
        idf = math.log(1+(count-len(posting)+.5)/(len(posting)+.5))
        for key, tf in posting.items():
            norm = 1.2*(.25+.75*index.lengths[key]/max(1, index.average_length))
            scores[key] = scores.get(key, 0) + idf*tf*2.2/(tf+norm)
    for key, score in sorted(scores.items(), key=lambda r: (-r[1], r[0]))[:32]:
        matches[key] = {"row": index.records[key], "score": score*3 if preserve_scores else min(45, score*3), "method": "independent-bm25"}
    for phrase in terms:
        tokens = re.findall(r"[a-z]+", phrase)
        for token in tokens:
            for row in index.titles.get(token, []):
                title = row["title"].lower()
                words = set(re.findall(r"[a-z]+", title))
                if not all(t in words for t in tokens):
                    continue
                score = 100 if title == phrase else 50+len(tokens)
                if row["id"] not in matches or matches[row["id"]]["score"] < score:
                    matches[row["id"]] = {"row": row, "score": score, "method": "independent-title"}
    for match in re.finditer(r"\bG(\d{1,5})\b", question, re.I | re.A):
        key = f"G{int(match[1]):04d}"
        record_id = index.data.get("aliases", {}).get("strong:"+key, "strong:"+key)
        row = index.records.get(record_id)
        if row and row["source"] == "strongGreek":
            matches[row["id"]] = {"row": row, "score": 120, "method": "explicit-lexicon-id"}
    return sorted(matches.values(), key=lambda r: (-r["score"], canonical_id_sort_key(r["row"]["id"])))[:32]


def _english_score(text, terms):
    return sum(term in text.lower() for term in terms)


def study_excerpt(text, terms, maximum=2400, *, sentence_fallback=False, preserve_lead=False):
    paragraphs = [p.strip() for p in text.split("\n") if text_length(p.strip()) > 25]
    if sentence_fallback:
        paragraphs = [piece for p in paragraphs for piece in (
            re.split(r"(?<=[.!?])\s+(?=[A-Z\"'])", p) if text_length(p) > maximum else [p])]
    tokens = set(search_words(" ".join(terms)))
    score = lambda p: _english_score(p, terms) + (len(tokens & set(search_words(p)))/max(1,len(tokens)) if sentence_fallback else 0)
    ranked = sorted(enumerate(paragraphs), key=lambda row: (-score(row[1]), row[0]))
    if preserve_lead and paragraphs and text_length(paragraphs[0]) <= maximum:
        # Retain the record's definition/location as well as query-matched facts.
        ranked = [(0, paragraphs[0])] + [(i,p) for i,p in ranked if i != 0]
    selected, chars = [], 0
    for i, paragraph in ranked:
        length = text_length(paragraph)
        if length > maximum or chars + length + 1 > maximum:
            continue
        selected.append((i, paragraph)); chars += length+1
    return "\n".join(p for _, p in sorted(selected))


def prefer_anchor_titles(ordered, anchor_ids, terms, index, requested, protected_ids=()):
    """Retain one actual verse-linked dictionary definition matching a query title.

    Explicit lexicon requests keep priority. This is bounded candidate routing,
    not factual approval; no generated query becomes a new historical source.
    """
    eligible = {}
    phrases = [set(re.findall(r'[a-z]+', t.lower())) for t in terms]
    for key in unique(v for v in anchor_ids if isinstance(v, str))[:6]:
        for row in index.by_verse.get(index.ids.get(key), []):
            words = set(re.findall(r'[a-z]+', row['title'].lower()))
            if row['source'] == 'easton' and any(p and p <= words for p in phrases):
                eligible[row['id']] = row
    if not eligible:
        return ordered
    ranking = {c['row']['id']: i for i, c in enumerate(ordered)}
    leader = min(eligible, key=lambda k: (ranking.get(k, len(ordered)), canonical_id_sort_key(k)))
    existing = next((c for c in ordered if c['row']['id'] == leader), None)
    selected = {**(existing or {'row': eligible[leader], 'score': 1}), 'method': 'anchor-title'}
    requested_ids = {index.data.get('aliases', {}).get('strong:'+k, 'strong:'+k) for k in requested}
    lexical = [c for c in ordered if c['row']['id'] in requested_ids]
    protected = lexical + [c for c in ordered if c['row']['id'] in protected_ids
                           and c['row']['id'] not in requested_ids]
    reserved = {c['row']['id'] for c in protected}
    # Distinct question-specific branches may already consume all three slots.
    # An extra anchor candidate must not evict their leaders to look relevant.
    if leader in reserved or len(protected) >= STUDY_LIMITS['sources']:
        return protected + [c for c in ordered if c['row']['id'] not in reserved]
    rest = [c for c in ordered if c['row']['id'] != leader and c['row']['id'] not in reserved]
    return protected + [selected] + rest


def retrieve_study_sources(primary, study_terms, question, index, *, independent=False, enhanced=False, multilingual_queries=(), balanced=False, anchor_ids=()):
    terms = study_query_terms(question, normalize_study_terms(study_terms)) if independent else normalize_study_terms(study_terms)
    candidates = {}
    requested = {f"G{int(s[1:]):04d}" for s in re.findall(r"\bG\d{1,5}\b", question, re.I | re.A)}
    korean_terms = set(rag_terms(question))
    roots = sorted(primary, key=lambda v: -_term_score(v["text"], korean_terms))[:12]
    for rank, v in enumerate(roots):
        n = index.ids.get(v["id"])
        for row in index.by_verse.get(n, []):
            candidate = candidates.setdefault(row["id"], {"row": row, "score": 0})
            candidate["score"] += (2/(rank+1))/(max(1, len(row["refs"]))**.5)
        for key in _indexed(index.data["greekTags"], n, []):
            if key not in requested:
                continue
            record_id = index.data["aliases"].get("strong:"+key, "strong:"+key)
            row = index.records.get(record_id)
            if row:
                candidates[record_id] = {"row": row, "score": 30}
    if independent:
        for match in search_independent_study(index, terms, question, fulltext=enhanced, preserve_scores=balanced):
            existing = candidates.get(match["row"]["id"])
            if existing is not None:
                existing["score"] += match["score"]; existing["method"] = match["method"]
            else:
                candidates[match["row"]["id"]] = dict(match)
    for candidate in candidates.values():
        row = candidate["row"]
        candidate["score"] += _english_score(row["title"], terms)*25 + _english_score(row["text"], terms)*2
    chosen, seen, chars = [], set(), 0
    protected_source_ids = []
    ordered = sorted(candidates.values(), key=lambda r: (-r["score"], canonical_id_sort_key(r["row"]["id"])))
    if multilingual_queries and independent and enhanced:
        # Original retrieval is one branch. English query branches search the
        # same source records, not generated answers or evaluation aliases.
        branches = [ordered[:32]] + [search_independent_study(index, [q], "", fulltext=True, preserve_scores=balanced)
                                     for q in multilingual_queries[:3]]
        fused, records = {}, {}
        for branch in branches:
            for rank, candidate in enumerate(branch, 1):
                key = candidate["row"]["id"]
                records[key] = candidate
                fused[key] = fused.get(key, 0) + 1/(60+rank)
        ordered = [{**records[key], "score": score, "method": "bilingual-rrf"}
                   for key, score in sorted(fused.items(), key=lambda r: (-r[1], canonical_id_sort_key(r[0])))]
        terms = unique([*terms, *multilingual_queries[:3]])
        if balanced:
            # Keep one leader per distinct question-specific branch before RRF
            # consensus can crowd out a rarer background need. Still only three
            # sources/4000 characters; review decides whether they are evidence.
            leaders = unique(b[0]["row"]["id"] for b in branches[1:] if b)
            protected_source_ids = leaders
            rank = {key: i for i, key in enumerate(leaders)}
            ordered.sort(key=lambda c: (c["row"]["id"] not in rank,
                                       rank.get(c["row"]["id"], 0)))
            branch_terms = {}
            for query, branch in zip(multilingual_queries[:3], branches[1:]):
                for candidate in branch[:8]:
                    branch_terms.setdefault(candidate["row"]["id"], []).append(query)
    if anchor_ids and independent and enhanced:
        ordered = prefer_anchor_titles(ordered, anchor_ids, terms, index, requested, protected_source_ids)
    for position, candidate in enumerate(ordered):
        row = candidate["row"]
        if len(chosen) >= 3:
            break
        if candidate["score"] <= 0 or (row["source"] in seen and not enhanced):
            continue
        remaining = min(3-len(chosen),sum(c["score"]>0 for c in ordered[position:]))
        budget = min(2400, (4000-chars)//max(1,remaining) if enhanced else 4000-chars)
        excerpt_terms = branch_terms.get(row["id"], terms) if balanced and multilingual_queries and independent and enhanced else terms
        excerpt = study_excerpt(row["text"], excerpt_terms, budget, sentence_fallback=enhanced, preserve_lead=balanced)
        if not excerpt:
            continue
        source = STUDY_SOURCES[row["source"]]
        item = {"id": f"S{len(chosen)+1}", "recordId": row["id"], "source": row["source"],
                "title": row["title"], "kind": source["kind"], "excerpt": excerpt}
        if independent:
            item.update(retrieval=candidate.get("method", "verse-linked"), license=source["license"],
                        sourceUrl=source["url"])
        if enhanced:
            item["context"] = source_context(row, source)
        chosen.append(item); chars += text_length(excerpt); seen.add(row["source"])
    return chosen


def original_language_support(question, sources=()):
    requested = bool(re.search(r"원어|어원|헬라어|히브리어|아람어|\b[GH]\d{1,5}\b", question, re.I | re.A))
    available = not re.search(r"히브리어|아람어|\bH\d{1,5}\b", question, re.I | re.A) and any(s["kind"] == "lexicon_1890" for s in sources)
    return {"requested": requested, "available": bool(available), "limitation": "이 답변에서 확인한 자료만으로는 해당 한국어 표현에 대응하는 원어와 그 정확한 뜻을 확인하기 어렵습니다. 본문에서 드러나는 의미를 중심으로 이해해 주세요." if requested and not available else ""}


def preserve_original_language_limit(answer, support):
    if not support["limitation"]:
        return answer
    import regex
    covered = any(re.search(r"원어|어원|헬라어|히브리어|아람어", s) and re.search(r"(?:확인|단정|설명|특정).{0,25}(?:어렵|없|못)|자료.{0,20}(?:없|않)", s)
                  for s in regex.split(r"(?<=[.!?](?:\*\*)?)\s+", answer))
    return answer if covered else answer.strip()+"\n\n"+support["limitation"]


def validate_study_notes(notes, sources, *, legacy_semantic_filter=True):
    if not isinstance(notes, list) or len(notes) > 2:
        return []
    by_id, accepted, seen = {s["id"]: s for s in sources}, [], set()
    normalize = lambda s: re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(s or ""))).strip()
    for note in notes:
        if not isinstance(note, dict):
            continue
        source, quote = by_id.get(note.get("sourceId")), normalize(note.get("quote"))
        if not source:
            continue
        minimum = (min(16, text_length(normalize(source['excerpt'])))
                   if source.get('kind') == 'scripture_superscription_translation' else 16)
        if source["id"] in seen or not minimum <= text_length(quote) <= 500 or quote not in normalize(source["excerpt"]):
            continue
        text = note.get("text", "")
        text = text.strip() if isinstance(text, str) else ""
        if not text or text_length(text) > 1200 or not korean_only(text) or not has_complete_answer_ending(text):
            continue
        if re.search(r"https?:|\]\(|<|>|\b(?:S\d|G\d{3,})\b", text, re.A):
            continue
        if legacy_semantic_filter and re.search(r"taken up dead|was dead|restored him to life|brought to life again", quote, re.I) and has_apparent_death_claim(text):
            continue
        accepted.append({"source": source, "text": text, "quote": quote}); seen.add(source["id"])
    return accepted


def append_study_notes(answer, notes, used_crossrefs=False):
    blocks = [answer.strip()]
    titles = [n for n in notes if n['source'].get('kind') == 'scripture_superscription_translation']
    notes = [n for n in notes if n['source'].get('kind') != 'scripture_superscription_translation']
    if titles:
        blocks.append('### 성경 표제 자료')
        for note in titles:
            source = note['source']
            blocks.append(f"**{source['title']}**\n\n{note['text']}\n\n[자료 출처]({source['sourceUrl']})")
    if notes:
        blocks.append("### 참고 자료로 살펴본 배경")
        for note in notes:
            source = STUDY_SOURCES[note["source"]["source"]]
            blocks.append(f"**{source['title']}의 설명**\n\n{note['text']}\n\n[자료 출처]({source['url']})")
            if note["source"]["source"] == "strongGreek":
                step = STUDY_SOURCES["step"]
                blocks.append(f"원어 연결자료: [{step['title']}]({step['url']}) · [저작자표시 4.0]({step['licenseUrl']}). {step['changes']}")
        blocks.append("참고 자료의 설명과 해석은 성경 본문의 진술과 구분해 읽어 주세요.")
    if used_crossrefs:
        source = STUDY_SOURCES["crossrefs"]
        blocks.append(f"관주 자료: [{source['title']}]({source['url']}).")
    return "\n\n".join(blocks)
