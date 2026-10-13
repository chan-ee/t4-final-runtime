"""Preserved v5 fallback/quote helpers; the old House pipeline is not included."""
from __future__ import annotations
import json
import math
import os
import re
from pathlib import Path
from .corpus import load_corpus,select_evidence
from .numerical import fallback_estimate,finite,evaluate_expression
from .native_bounds import native_bounds
from .task_evidence import own_row_claim

_QUOTE_TRANSLATION=str.maketrans({'‘':"'",'’':"'",'“':'"','”':'"','–':'-','—':'-','\u00a0':' '})

def quote_matches(text,quote):
    """Exact lexical text after typography and whitespace normalization, with source offsets."""
    words=list(re.finditer(r'\S+',text))
    normalized=[];starts=[];ends=[]
    for i,word in enumerate(words):
        if i:
            normalized.append(' ');starts.append(words[i-1].end());ends.append(word.start())
        for offset,char in enumerate(word.group()):
            normalized.append(char.translate(_QUOTE_TRANSLATION))
            starts.append(word.start()+offset);ends.append(word.start()+offset+1)
    needle=' '.join(quote.translate(_QUOTE_TRANSLATION).split())
    haystack=''.join(normalized)
    matches=[];at=haystack.find(needle)
    while needle and at>=0:
        matches.append((starts[at],ends[at+len(needle)-1]))
        at=haystack.find(needle,at+1)
    return matches

def _fallback_claim(chunks):
    c=chunks[0]
    text=c.text
    marker=text.find('NOTES (derived')
    if marker<0:marker=text.find('NOTE (derived')
    start=marker if marker>=0 else 0
    while start<len(text) and text[start].isspace():start+=1
    end=min(len(text),start+350)
    if end<len(text):
        boundary=max(text.rfind('. ',start+80,end),text.rfind('\n',start+80,end))
        if boundary>=start+80:end=boundary+1
    quote=text[start:end].strip()
    end=start+len(quote)
    if not quote:raise ValueError('No citable pre-cutoff passage')
    return {'doc_id':c.doc_id,'span_start':c.start+start,'span_end':c.start+end,
            'claim':quote}

def ground_evidence(items,chunks,corpus):
    claims=[];seen=set()
    for item in items if isinstance(items,list) else []:
        if not isinstance(item,dict):continue
        doc_id=item.get('doc_id')
        if doc_id not in corpus.docs:
            try:doc_id=chunks[int(item.get('evidence_id','-1'))].doc_id if int(item.get('evidence_id','-1'))>=0 else None
            except (ValueError,IndexError,TypeError):continue
        quote=item.get('quote')
        if not isinstance(quote,str) or not quote.strip() or len(quote)>450:continue
        doc=corpus.docs.get(doc_id)
        if doc is None:continue
        # Bind explicit evidence index to that exact original context. Repeated text
        # in different quarters must never silently jump to the first occurrence.
        candidates=chunks
        if 'evidence_id' in item:
            try:
                index=int(item['evidence_id'])
                if index<0 or index>=len(chunks):continue
                candidates=[chunks[index]]
                if candidates[0].doc_id!=doc_id:continue
            except (TypeError,ValueError):continue
        matches=[]
        for chunk in candidates:
            if chunk.doc_id!=doc_id:continue
            for start,end in quote_matches(chunk.text,quote):
                matches.append((chunk,start,end))
        unique={(c.doc_id,c.start+start,c.start+end) for c,start,end in matches}
        if len(unique)!=1:continue
        doc_id,start,end=next(iter(unique));key=(doc_id,start,end)
        if end-start>4000:continue
        if key not in seen:
            claims.append({'doc_id':doc_id,'span_start':start,'span_end':end,'claim':doc.text[start:end]});seen.add(key)
        if len(claims)>=3:break
    return claims

def normalize_prediction(raw,task,entity,chunks,corpus,previous=None):
    base=dict(previous) if previous is not None else fallback_estimate(task,entity,chunks,corpus)
    base['interval']=dict(base['interval'])
    if not base.get('claims'):base['claims']=[_fallback_claim(chunks) if chunks else own_row_claim(task,entity)]
    if not isinstance(raw,dict):return base
    typ=task['target']['type'];valid=True
    point=finite(raw.get('point_forecast'))
    expression=raw.get('point_expression')
    calculated=False
    if expression:
        try:
            point=evaluate_expression(expression);calculated=True
        except (ValueError,SyntaxError,ZeroDivisionError,OverflowError,RecursionError):valid=False
    if point is None and typ=='classification':point=finite(base.get('point_forecast'))
    if point is None:valid=False
    interval=raw.get('interval',{})
    lo=finite(interval.get('lo')) if isinstance(interval,dict) else None
    hi=finite(interval.get('hi')) if isinstance(interval,dict) else None
    if lo is None or hi is None or lo>hi:valid=False
    label=raw.get('label')
    if typ=='classification' and label not in task['target'].get('labels',[]):valid=False
    claims=ground_evidence(raw.get('evidence',[]),chunks,corpus)
    if not claims:valid=False
    if not valid:return base
    row={'entity_id':entity['entity_id'],'point_forecast':point,'interval':{'level':float(task.get('interval_level',.9)),'lo':lo,'hi':hi},'claims':claims,
         '_reasoning':str(raw.get('reasoning',''))[:1100]+(' Arithmetic recomputed from supplied expression: '+str(expression) if calculated else ''),'_fallback':False}
    if typ=='classification':row['label']=label
    # Task-specific probability semantics; never rewrite other numeric classification outputs into probabilities.
    prompt=task.get('prompt','').lower();name=task['target'].get('name','').lower()
    bounds=native_bounds(task,entity)
    if bounds is not None and not(bounds[0]<=point<=bounds[1]and bounds[0]<=lo<=hi<=bounds[1]):return base
    return row

def _rows(response):
    result={}
    rows=response.get('predictions',response.get('entity_predictions',[])) if isinstance(response,dict) else []
    for r in rows if isinstance(rows,list) else []:
        if isinstance(r,dict) and isinstance(r.get('entity_id'),str):
            eid=r['entity_id']
            if eid in result:result[eid]=None
            else:result[eid]=r
    return result

def _write(path,answer):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(answer,ensure_ascii=False,allow_nan=False,separators=(',',':'))+'\n')
    os.chmod(temp,0o644)
    os.replace(temp,path)
