"""Source-bound observation packets and stable citation handles.

Derived statistics and candidate forecasts are never represented as source
quotes. A House-selected handle binds only to the original frozen document.
"""
from __future__ import annotations
from collections import Counter
import math
import re
from .corpus import Chunk, tokens
from .typed_eps import historical_eps_facts, conditional_eps_bridge
from .numerical_context import build_numeric_context
from .credit_context import credit_context_windows


def _eligible(doc, task, entity):
    return doc.date <= task['cutoff_date'] and (doc.shared or entity['entity_id'] in doc.entity_ids)


def observation_exclusions(doc,cutoff,latest_ref=None):
    """Only recognized OBSERVATION tables; future schedules/projections survive."""
    cutoff=min(cutoff,doc.date)
    excluded=[];mode=None;future_columns=[];offset=0;size=0
    for raw in doc.text.splitlines(keepends=True):
        line=raw.rstrip('\r\n');matches=list(re.finditer(r'[^|]+',line));cells=[m.group().strip() for m in matches]
        if len(cells)>1:
            first=cells[0].lower()
            is_vintage=first=='reference_month' and any(re.fullmatch(r'as_of_\d{4}-\d{2}-\d{2}',x) for x in cells[1:])
            known=(first=='report_date' and 'open_interest' in cells) or (first=='week_ending' and 'usd_per_gallon' in cells) or (first=='auction_date' and 'bid_to_cover' in cells)
            nearby=doc.text[max(0,offset-240):offset]+line
            caption=doc.text[:offset].rstrip().split('\n')[-1]
            treasury_observations=(first=='date' and len(cells)>1 and
                all(re.fullmatch(r'DGS\d+(?:MO)?',column,re.I)for column in cells[1:]) and
                not re.search(r'forecast|projection|projected|expected|scheduled|guidance',caption,re.I))
            known=known or treasury_observations
            monthly=first=='month' and not re.search(r'forecast|projection|projected|expected|scheduled|guidance',nearby,re.I)
            if is_vintage or known or monthly:
                mode='vintage' if is_vintage else 'observations';size=len(cells)
                future_columns=[i for i,x in enumerate(cells) if i and is_vintage and x.removeprefix('as_of_')>cutoff]
                for i in future_columns:
                    excluded.append((offset+matches[i].start(),offset+matches[i].end()))
            elif mode and len(cells)==size and re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?',first):
                boundary=cutoff if len(first)==10 else cutoff[:7]
                if len(first)==7 and mode=='observations' and isinstance(latest_ref,str) and re.fullmatch(r'\d{4}-\d{2}',latest_ref):
                    boundary=min(boundary,latest_ref)
                if first>boundary:excluded.append((offset,offset+len(line)))
                else:
                    for i in future_columns:excluded.append((offset+matches[i].start(),offset+matches[i].end()))
            elif mode and line.strip():mode=None;future_columns=[]
        elif line.strip():mode=None;future_columns=[]
        offset+=len(raw)
    return sorted(excluded)


def safe_source_chunks(chunk,doc,cutoff,latest_ref=None):
    cursor=chunk.start
    for start,end in observation_exclusions(doc,cutoff,latest_ref):
        if end<=cursor or start>=chunk.end:continue
        if cursor<start:
            stop=min(start,chunk.end)
            yield Chunk(doc.doc_id,cursor,stop,doc.text[cursor:stop])
        cursor=max(cursor,end)
    if cursor<chunk.end:yield Chunk(doc.doc_id,cursor,chunk.end,doc.text[cursor:chunk.end])


def _pieces(chunk, width=360):
    """Contiguous original excerpts, keeping sentence/line boundaries where possible."""
    start=0
    while start<len(chunk.text):
        while start<len(chunk.text) and chunk.text[start].isspace():start+=1
        if start>=len(chunk.text):break
        end=min(len(chunk.text),start+width)
        if end<len(chunk.text):
            stops=[chunk.text.rfind('\n',start+60,end),chunk.text.rfind('. ',start+60,end),chunk.text.rfind('; ',start+60,end)]
            boundary=max(stops)
            if boundary>=start+60:end=boundary+1
            else:
                boundary=chunk.text.rfind(' ',start+80,end)
                if boundary>start:end=boundary
        finish=end
        while finish>start and chunk.text[finish-1].isspace():finish-=1
        if finish>start:
            yield Chunk(chunk.doc_id,chunk.start+start,chunk.start+finish,chunk.text[start:finish])
        start=max(end,start+1)


def add_source(packet,chunk,corpus,task,entity,*,role='evidence'):
    doc=corpus.docs.get(chunk.doc_id)
    if not doc or not _eligible(doc,task,entity):return []
    if not (0<=chunk.start<chunk.end<=len(doc.text)) or doc.text[chunk.start:chunk.end]!=chunk.text:
        raise ValueError('Source packet does not match frozen document')
    added=[]
    pieces=(piece for safe in safe_source_chunks(chunk,doc,task['cutoff_date'],entity.get('latest_published_ref_month')) for piece in _pieces(safe))
    for piece in pieces:
        key=(piece.doc_id,piece.start,piece.end)
        existing=next((c for c in packet['cards'] if (c['doc_id'],c['span_start'],c['span_end'])==key),None)
        if existing:
            added.append(existing['id']);continue
        identifier=packet['source_prefix']+'F'+str(len(packet['cards']))
        card={'id':identifier,'doc_id':piece.doc_id,'doc_date':doc.date,
              'span_start':piece.start,'span_end':piece.end,'role':role,'text':piece.text,
              'citable':len(piece.text)>=8 and bool(re.search(r'[A-Za-z0-9]',piece.text))}
        packet['cards'].append(card);added.append(identifier)
    return added


def prepare_packet(task,entity,corpus,selected):
    index=next(i for i,e in enumerate(task['entities']) if e['entity_id']==entity['entity_id'])
    packet={'entity_id':entity['entity_id'],'source_prefix':'E'+str(index),'cards':[],'observations':{},'warnings':[]}
    common=[doc for doc in corpus.docs.values() if doc.shared and doc.date<=task['cutoff_date']]
    if sum(len(doc.text) for doc in common)<=26000:
        for doc in common:
            add_source(packet,Chunk(doc.doc_id,0,len(doc.text),doc.text),corpus,task,entity,role='shared_original')
    # These are transparent summaries of the current input, not learned target
    # answers. Preserve their observed/derived/candidate status in the prompt.
    numeric=build_numeric_context(task,entity,corpus)
    if numeric is not None:packet['observations']['numeric_history']=numeric
    for window in credit_context_windows(corpus,task,entity):
        add_source(packet,Chunk(window.doc_id,window.start,window.end,window.text),corpus,task,entity,role=window.role)
    if 'eps' in task.get('target',{}).get('name','').lower():
        ledger=historical_eps_facts(corpus,task,entity)
        slim=[];group_ids={}
        for fact in ledger['facts']:
            doc=corpus.docs[fact['doc_id']]
            key=(doc.doc_id,fact['header_start'],fact['value_start'],fact['value_end'])
            if key not in group_ids:
                context_start=max(0,fact['header_start']-450)
                accounting=Chunk(doc.doc_id,context_start,fact['header_start'],doc.text[context_start:fact['header_start']])
                if accounting.text.strip():add_source(packet,accounting,corpus,task,entity,role='accounting_context')
                header=Chunk(doc.doc_id,fact['header_start'],fact['header_end'],fact['header'])
                value=Chunk(doc.doc_id,fact['value_start'],fact['value_end'],fact['value_quote'])
                group_ids[key]=add_source(packet,header,corpus,task,entity,role='period_header')+add_source(packet,value,corpus,task,entity,role='historical_eps_row')
            slim.append({k:fact[k] for k in ('metric','value','unit','period_end','period_months')} | {'source_ids':group_ids[key],
                        'basis_scope_status':'Reported EPS in the entity document; GAAP and consolidated scope require original-context verification.'})
        packet['observations']['historical_eps']=slim
        packet['warnings'].extend(ledger['parser_limitations'])
        if ledger['conflicts']:packet['observations']['eps_conflicts']=ledger['conflicts']
        bridge=conditional_eps_bridge(ledger['facts'],entity.get('prior_year_q_eps'))
        if bridge:
            packet['observations']['unselected_eps_scenarios']={k:bridge[k] for k in ('scenarios_eps','unverified_assumptions','calibrated_interval','selected_forecast')}
    for chunk in selected:add_source(packet,chunk,corpus,task,entity)
    return packet


def retrieve_queries(corpus,task,entity,queries,*,limit=5,budget=9000):
    """Model queries are words for a local read-only index, never instructions."""
    if not isinstance(queries,list):return []
    queries=[q[:400] for q in queries[:6] if isinstance(q,str) and q.strip()]
    ranked=[]
    for query in queries:
        terms=Counter(tokens(query));terms=Counter(dict(terms.most_common(32)))
        if not terms:continue
        query_scores=[]
        for index,(chunk,count) in enumerate(zip(corpus.chunks,corpus.counters)):
            doc=corpus.docs[chunk.doc_id]
            if not _eligible(doc,task,entity):continue
            denom=.4+1.2*sum(count.values())/max(corpus.avg_len,1)
            value=sum(corpus.idf.get(t,0)*min(n,2)*count.get(t,0)*2.2/(count.get(t,0)+denom) for t,n in terms.items() if count.get(t,0))
            if value>0:query_scores.append((value,doc.date,-chunk.start,index))
        ranked.extend(sorted(query_scores,reverse=True)[:2])
    result=[];used=0
    for _,_,_,index in sorted(ranked,reverse=True):
        chunk=corpus.chunks[index]
        if any(chunk.doc_id==c.doc_id and max(0,min(chunk.end,c.end)-max(chunk.start,c.start))>.6*min(len(chunk.text),len(c.text)) for c in result):continue
        if used+len(chunk.text)>budget:continue
        result.append(chunk);used+=len(chunk.text)
        if len(result)>=limit:break
    return [safe for chunk in result for safe in safe_source_chunks(chunk,corpus.docs[chunk.doc_id],task['cutoff_date'],entity.get('latest_published_ref_month')) if safe.text.strip()]


def claims_from_ids(packet,identifiers,corpus,task,entity,*,limit=3):
    if not isinstance(identifiers,list):return []
    lookup={card['id']:card for card in packet['cards']};result=[];seen=set()
    for identifier in identifiers:
        card=lookup.get(identifier) if isinstance(identifier,str) else None
        if card is None or not card.get('citable',True):continue
        doc=corpus.docs.get(card['doc_id'])
        if not doc or not _eligible(doc,task,entity):continue
        start,end=card['span_start'],card['span_end']
        if doc.text[start:end]!=card['text']:continue
        key=(doc.doc_id,start,end)
        if key in seen:continue
        # Citation text is an actual selected fact. No generated numerical
        # statistic or future forecast can be smuggled in as an observed quote.
        result.append({'doc_id':doc.doc_id,'span_start':start,'span_end':end,'claim':card['text']})
        seen.add(key)
        if len(result)>=limit:break
    return result


def public_packet(packet,*,compact=False,selected_ids=()):
    result={'entity_id':packet['entity_id'],'observations':packet['observations'],'warnings':packet['warnings']}
    cards=packet['cards']
    if compact:
        wanted=set(selected_ids)
        critical_roles=('historical_eps_row','period_header','accounting_context','operating_cash_flow','liquidity_capacity','debt_horizon','covenant_status','going_concern_and_qualification')
        primary=[c for c in cards if c['id'] in wanted or c['role'] in critical_roles]
        extra=[c for c in cards if c['role']=='followup' and c not in primary]
        cards=primary+extra[:10]
        if not cards:cards=packet['cards'][:12]
    result['source_facts']=[{k:c[k] for k in ('id','doc_id','doc_date','role','text','citable')} for c in cards]
    return result


def shared_documents(task,corpus,*,budget=26000):
    docs=[d for d in corpus.docs.values() if d.shared and d.date<=task['cutoff_date']]
    if sum(len(d.text) for d in docs)>budget:return []
    result=[]
    boundaries=[e['latest_published_ref_month'] for e in task.get('entities',[]) if isinstance(e.get('latest_published_ref_month'),str) and re.fullmatch(r'\d{4}-\d{2}',e['latest_published_ref_month'])]
    latest_ref=min(boundaries) if boundaries else None
    for doc in docs:
        pieces=list(safe_source_chunks(Chunk(doc.doc_id,0,len(doc.text),doc.text),doc,task['cutoff_date'],latest_ref))
        text='\n[post-cutoff observation omitted]\n'.join(piece.text for piece in pieces)
        result.append({'doc_id':doc.doc_id,'doc_date':doc.date,'text':text})
    return result
