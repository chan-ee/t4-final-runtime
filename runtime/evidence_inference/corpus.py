from __future__ import annotations
import datetime as dt
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

ID_RE = re.compile(r'^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$')
STOP = set('the a an and of to for in on is are as with by from that this at be its it or will per each using only predict prediction provide support frozen corpus company inc corporation percent'.split())

def tokens(text):
    return [s for s in re.findall(r'[a-z0-9]+', str(text).lower()) if len(s)>1 and s not in STOP]

def calendar(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):
        raise ValueError('invalid calendar date')
    return dt.date.fromisoformat(value)

@dataclass(frozen=True)
class Document:
    doc_id: str
    date: str
    text: str
    metadata: dict
    entity_ids: tuple[str,...] = ()
    shared: bool = True

@dataclass(frozen=True)
class Chunk:
    doc_id: str
    start: int
    end: int
    text: str

@dataclass
class Corpus:
    docs: dict[str,Document]
    chunks: list[Chunk]
    counters: list[Counter]
    idf: dict[str,float]
    avg_len: float

def chunk_text(doc, width=1600, overlap=240):
    text=doc.text
    if len(text)<=5200:
        return [Chunk(doc.doc_id,0,len(text),text)] if text.strip() else []
    result=[]; start=0
    while start<len(text):
        stop=min(start+width,len(text))
        if stop<len(text):
            # Preserve sentence/table boundaries whenever available; offsets always refer to original text.
            lo=start+width//2
            candidates=[text.rfind('\n',lo,stop),text.rfind('. ',lo,stop),text.rfind('; ',lo,stop)]
            cut=max(candidates)
            if cut>=lo:stop=cut+1
        if text[start:stop].strip():result.append(Chunk(doc.doc_id,start,stop,text[start:stop]))
        if stop==len(text):break
        nxt=max(start+1,stop-overlap)
        boundary=text.find(' ',nxt,min(stop,nxt+80))
        start=boundary+1 if boundary>=0 else nxt
    return result

def load_corpus(corpus_dir,cutoff,task=None):
    corpus_dir=Path(corpus_dir);limit=calendar(cutoff)
    manifest_path=corpus_dir.parent/'manifest.json'
    if not manifest_path.is_file():manifest_path=corpus_dir/'manifest.json'
    if manifest_path.is_symlink():raise ValueError('manifest symlink')
    manifest=json.loads(manifest_path.read_text())
    docs={};declared_corpus=False
    for entry in manifest.get('files',[]):
        if entry.get('role')!='corpus':continue
        name=entry.get('path','')
        if name=='corpus/manifest.json':continue
        declared_corpus=True
        parts=name.split('/')
        if len(parts)!=2 or parts[0]!='corpus' or not parts[1].endswith('.json'):raise ValueError('invalid corpus path')
        doc_id=parts[1][:-5]
        if not ID_RE.fullmatch(doc_id) or doc_id in docs:raise ValueError('invalid/duplicate corpus id')
        path=corpus_dir/parts[1]
        if path.is_symlink() or not path.is_file():raise ValueError('unsafe corpus file')
        data=path.read_bytes()
        if len(data)>8*1024*1024 or hashlib.sha256(data).hexdigest()!=entry.get('sha256'):raise ValueError('corpus integrity mismatch')
        d=json.loads(data)
        try:date=calendar(d.get('doc_date'))
        except (ValueError,TypeError):continue
        # Filter BEFORE tokenization and retrieval; future text cannot shape a prompt.
        if date>limit:continue
        text=d.get('text')
        if not isinstance(text,str):text=' '.join(s.get('text','') for s in d.get('spans',[]) if isinstance(s,dict) and isinstance(s.get('text',''),str))
        if not text.strip():continue
        entity_ids=tuple(entry.get('entity_ids') or ())
        shared=entry.get('shared') is True
        # Missing organizer scope is not permission to share a document.
        if not shared and not entity_ids:continue
        docs[doc_id]=Document(doc_id,str(date),text,{k:v for k,v in d.items() if k not in ('text','spans')},entity_ids,shared)
    if not declared_corpus:raise ValueError('no declared corpus documents')
    # Recognized observation-table dates are checked before indexing as well as
    # before prompting. Keep original document bytes for verifiable citations.
    from .packets import safe_source_chunks
    boundaries=[e['latest_published_ref_month'] for e in (task or {}).get('entities',[]) if isinstance(e.get('latest_published_ref_month'),str) and re.fullmatch(r'\d{4}-\d{2}',e['latest_published_ref_month'])]
    latest_ref=min(boundaries) if boundaries else None
    chunks=[safe for doc_id in sorted(docs) for c in chunk_text(docs[doc_id])
            for safe in safe_source_chunks(c,docs[doc_id],cutoff,latest_ref) if safe.text.strip()]
    counters=[Counter(tokens(c.text)) for c in chunks]
    df=Counter(t for c in counters for t in c)
    n=len(chunks);idf={t:math.log(1+(n-v+.5)/(v+.5)) for t,v in df.items()}
    return Corpus(docs,chunks,counters,idf,sum(sum(c.values()) for c in counters)/max(n,1))

def supplement_credit_context(corpus,task,entity,selected,budget_chars):
    """Bring qualifying risk and mitigation disclosures into the same entity context.

    This static lexical dictionary is a retrieval aid, not a credit-event label.
    It reads only manifest-authorized pre-cutoff chunks and leaves exact offsets.
    """
    if task.get('cutoff_date','')<'2023-03-31':return selected
    from .credit_context import is_credit_task
    if not is_credit_task(task):return selected
    groups=(
        (('substantial doubt',5),('going concern',2),('alleviat',4),('management plan',3),('management believes',2)),
        (('events of default',6),('event of default',5),('payment default',5),('covenant',2),('waiver',3)),
        (('cash and cash equivalents',4),('liquidity',2),('maturit',2),('available cash',3),('operating cash flow',3)),
    )
    chosen=[];used=0
    candidates=[c for c in corpus.chunks if corpus.docs[c.doc_id].shared or entity['entity_id'] in corpus.docs[c.doc_id].entity_ids]
    def add(chunk):
        nonlocal used
        if len(chosen)>=10 or used+len(chunk.text)>budget_chars:return False
        if any(chunk.doc_id==x.doc_id and max(0,min(chunk.end,x.end)-max(chunk.start,x.start))>.55*min(chunk.end-chunk.start,x.end-x.start) for x in chosen):return False
        chosen.append(chunk);used+=len(chunk.text);return True
    for group in groups:
        ranked=[]
        for chunk in candidates:
            low=chunk.text.lower()
            score=sum(weight*min(2,low.count(phrase)) for phrase,weight in group)
            # Distinct cue words in one passage keep a qualification close to a risk signal.
            hits=sum(phrase in low for phrase,_ in group)
            if score:ranked.append((score+3*hits,hits,-chunk.start,chunk.doc_id,chunk))
        for _,_,_,_,chunk in sorted(ranked,reverse=True):
            if add(chunk):break
    for chunk in selected:add(chunk)
    return chosen or selected

def select_evidence(corpus,task,entity,budget_chars=14000):
    identity=' '.join(str(entity.get(k,'')) for k in ('entity_id','name','series_name','series_id','series_fred','tenor','sector','industry','description'))
    target=task.get('target',{}).get('name','')+' '+task.get('prompt','')
    terms=Counter(tokens(identity)+tokens(task.get('target',{}).get('name',''))*3+tokens(target))
    lowered=target.lower()
    additions=[]
    if 'eps' in lowered or 'earnings' in lowered:additions+='diluted earnings share net income outlook guidance margin revenue provisions expenses'.split()
    from .credit_context import is_credit_task, supplement_credit_evidence
    if is_credit_task(task):additions+='liquidity covenant waiver default bankruptcy going concern maturities debt cash refinancing'.split()
    if 'yield' in lowered or 'rate' in lowered:additions+='policy inflation rates yield expectations projected treasury maturity'.split()
    if 'revision' in lowered:additions+='vintage revisions preliminary revised estimate benchmark'.split()
    terms.update(additions)
    identity_tokens=set(tokens(identity));cik=str(entity.get('cik','')).lstrip('0')
    scores=[]
    for i,(chunk,count) in enumerate(zip(corpus.chunks,corpus.counters)):
        doc=corpus.docs[chunk.doc_id]
        if not doc.shared and entity['entity_id'] not in doc.entity_ids:continue
        length=sum(count.values());denom=.4+1.2*length/max(corpus.avg_len,1)
        score=sum(corpus.idf.get(t,0)*min(w,3)*count.get(t,0)*2.2/(count.get(t,0)+denom) for t,w in terms.items() if count.get(t,0))
        doc=corpus.docs[chunk.doc_id]
        meta=' '.join(str(v) for v in doc.metadata.values() if isinstance(v,(str,int)))+' '+chunk.doc_id
        mt=set(tokens(meta));score+=3*len(identity_tokens & mt)
        if cik and chunk.doc_id.startswith('EDGAR_'):
            match=re.match(r'EDGAR_(\d+)_',chunk.doc_id)
            if match:score+=30 if match.group(1).lstrip('0')==cik else -100
        # Equal scores prefer recent evidence, never future evidence.
        scores.append((score,doc.date,chunk.doc_id,chunk.start,i))
    ranked=sorted(scores,key=lambda x:(-x[0],-int(x[1].replace('-','')),x[2],x[3]))
    selected=[];used=0;per_doc=Counter()
    for score,_,_,_,i in ranked:
        c=corpus.chunks[i]
        if score<0 and selected:continue
        if any(c.doc_id==p.doc_id and max(0,min(c.end,p.end)-max(c.start,p.start))>.55*min(c.end-c.start,p.end-p.start) for p in selected):continue
        if per_doc[c.doc_id]>=5:continue
        if used+len(c.text)>budget_chars:continue
        selected.append(c);used+=len(c.text);per_doc[c.doc_id]+=1
        if len(selected)>=10:break
    if not selected and ranked:selected=[corpus.chunks[ranked[0][4]]]
    selected=supplement_credit_context(corpus,task,entity,selected,budget_chars)
    selected=supplement_credit_evidence(corpus,task,entity,selected,budget_chars)
    selected.sort(key=lambda c:corpus.docs[c.doc_id].shared)
    return selected
