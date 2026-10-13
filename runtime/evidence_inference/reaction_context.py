"""Recover current-source operating outlook for an earnings-reaction task.

Management expectations, historical performance and accounting items are
observations about a business. They are not market consensus or return forecasts.
No issuer lookup, target-period answer or outside document is imported.
"""
from __future__ import annotations
import re
from .corpus import Chunk
from .packets import add_source,safe_source_chunks

_INVISIBLE='\u200b\u200c\u200d\ufeff\xa0'
_FINANCIAL=re.compile(r'net sales|revenues?|operating income|operating (?:margin|expenses?)|effective tax rates?|earnings per share|capital expenditures?|depreciation|restructuring charges?',re.I)
_FORWARD=re.compile(r'\bexpect(?:s|ed)?\b|\bguidance\b|\boutlook\b|\banticipat(?:e|es|ed)\b|\bforecast',re.I)
_QUARTER=re.compile(r'\b(?:(?:first|second|third|fourth)\s+(?:fiscal\s+)?quarter|Q[1-4])\s+(?:of\s+)?\d{4}\b',re.I)
_NUMBER=re.compile(r'\$\s*[\d,.]+|\d(?:[\d,.]*\d)?\s*(?:%|percent|billion|million|basis points)',re.I)
_FINANCIAL_ASSERTION=re.compile(r'(?:net sales|revenues?|operating income|expenses?|effective tax rates?|capital expenditures?)\s+(?:is\s+|are\s+)?expected\b|(?:\bwe|\bthe company)\s+(?:also\s+)?expects?\b.{0,150}(?:'+_FINANCIAL.pattern+r')',re.I)


def _normal(text):return text.translate(str.maketrans({c:' 'for c in _INVISIBLE}))


def _window(text,start,end):
    start=max(0,start);end=min(len(text),end)
    if start and not text[start-1].isspace():
        at=text.find(' ',start,min(end,start+100))
        if at>=0:start=at+1
    if end<len(text)and not text[end-1].isspace()and not text[end].isspace():
        at=text.rfind(' ',max(start,end-100),end)
        if at>start:end=at
    while start<end and text[start].isspace():start+=1
    while end>start and text[end-1].isspace():end-=1
    return start,end


def _overlap_fraction(doc_id,start,end,cards):
    intervals=sorted((max(start,c['span_start']),min(end,c['span_end']))for c in cards if c['doc_id']==doc_id and c['span_end']>start and c['span_start']<end)
    total=0;last=start
    for a,b in intervals:
        a=max(a,last)
        if b>a:total+=b-a;last=b
    return total/max(1,end-start)


def _candidates(doc):
    text=_normal(doc.text);result=[]
    def add(kind,start,end,priority,score):
        a,b=_window(doc.text,start,end)
        if b-a>=80:result.append({'kind':kind,'doc':doc,'start':a,'end':b,'priority':priority,'score':score})
    # Explicit dated source-period guidance. The event date alone does not tell
    # us which financial/fiscal quarter is being reported, so retain the source
    # period and mark event mapping unverified instead of inventing a quarter.
    for match in _QUARTER.finditer(text):
        year=int(re.search(r'\d{4}',match.group()).group())
        if year<int(doc.date[:4]):continue
        left=text.rfind('. ',max(0,match.start()-500),match.start())
        right=text.find('. ',match.end(),min(len(text),match.end()+1000))
        sentence_start=left+2 if left>=0 else max(0,match.start()-500)
        local=text[sentence_start:right if right>=0 else min(len(text),match.end()+1000)]
        a=max(0,match.start()-1600);b=min(len(text),match.end()+1200);section=text[a:b]
        # A past quarter next to a product expected next year is not a dated
        # financial outlook. Tie the forward financial assertion to the cue.
        heading=bool(re.match(r'\s*(?:Guidance|Outlook)\b',text[match.end():match.end()+60],re.I))
        forward_before_period=bool(_FINANCIAL_ASSERTION.search(text[sentence_start:match.start()]))
        if (heading or forward_before_period)and _FINANCIAL_ASSERTION.search(local)and _NUMBER.search(section):
            direct_guidance=bool(re.search(r'guidance|outlook',local,re.I))
            add('explicit_period_operating_outlook',a,b,0,50*direct_guidance+len(_FINANCIAL.findall(local)))
    for match in re.finditer(r'\bGuidance\b|\bOutlook\b',text,re.I):
        section=text[match.start():match.start()+1800]
        if _FINANCIAL.search(section)and len(_NUMBER.findall(section))>=2 and _FORWARD.search(section[match.end()-match.start():]):
            add('management_outlook_with_conditions',match.start()-120,match.start()+1800,1,10+len(_NUMBER.findall(section)))
    for match in _FINANCIAL_ASSERTION.finditer(text):
        section=text[max(0,match.start()-160):match.start()+850]
        if _FINANCIAL.search(section)and _NUMBER.search(section)and re.search(r'quarter|\d{4}|next year|full.year',section,re.I):
            if re.search(r'ASC\s*\d|accounting guidance',section,re.I)and not re.search(r'we expect|expects? (?:our|total|net)|operating income is expected',section,re.I):continue
            add('forward_operating_condition_with_context',match.start()-400,match.start()+1100,2,len(_NUMBER.findall(section)))
    patterns=(
        ('historical_operating_driver',r'(?:net sales|revenues?|operating income|gross margin)\s+(?:increased|decreased|grew|declined)',230,1100,3),
        ('historical_demand_condition',r'cost optimization|(?:volume|units? sold|average price per ad|ad impressions)\s+(?:increased|decreased)',350,1000,3),
        ('accounting_or_unusual_driver',r'(?:change(?:d|s)?|extend(?:ed)?)\s+(?:in\s+)?(?:the\s+)?(?:estimated\s+)?useful li(?:fe|ves)|restructuring charges|foreign.exchange (?:gain|loss)|unrealized (?:gain|loss)|one.time (?:charge|benefit)',400,1000,4),
        ('valuation_measure_or_expectation_uncertainty',r'price.to.earnings|earnings multiple|(?:quarterly|financial) results.{0,90}(?:analysts|market expectations)',250,850,5),
    )
    for kind,pattern,before,after,priority in patterns:
        for match in re.finditer(pattern,text,re.I):
            section=text[max(0,match.start()-before):match.start()+after]
            if not _NUMBER.search(section)and kind!='valuation_measure_or_expectation_uncertainty':continue
            add(kind,match.start()-before,match.start()+after,priority,len(_NUMBER.findall(section)))
    return result


def enrich_reaction_information(task,entity,corpus,packet,*,max_additional_chars=9000):
    """Append source-bound cards and uncertainty metadata, returning same packet."""
    name=str(task.get('target',{}).get('name','')).lower()
    if not('reaction'in name and'earnings'in name):return packet
    if packet.get('entity_id')!=entity.get('entity_id'):return packet
    observations=packet.setdefault('observations',{})
    if 'reaction_information'in observations:return packet
    budget=max(0,min(9000,int(max_additional_chars)))
    docs=[d for d in corpus.docs.values()if d.date<=task['cutoff_date']and not d.shared and entity['entity_id']in d.entity_ids]
    candidates=[item for doc in docs for item in _candidates(doc)]
    candidates.sort(key=lambda x:(x['priority'],-x['score'],-int(x['doc'].date.replace('-','')),x['doc'].doc_id,x['start']))
    selected=[];used=0;roles={};added_cards_start=len(packet['cards']);contexts=[]
    for candidate in candidates:
        role=candidate['kind'];doc=candidate['doc'];a,b=candidate['start'],candidate['end']
        if roles.get(role,0)>=(2 if role=='explicit_period_operating_outlook'else 1):continue
        if any(x['doc_id']==doc.doc_id and min(b,x['end'])-max(a,x['start'])>.45*min(b-a,x['end']-x['start'])for x in selected):continue
        # Existing partial context still needs its assumptions and qualifications.
        # A wholly already supplied window need not be duplicated.
        if _overlap_fraction(doc.doc_id,a,b,packet['cards'])>=.98:continue
        safe=list(safe_source_chunks(Chunk(doc.doc_id,a,b,doc.text[a:b]),doc,task['cutoff_date'],entity.get('latest_published_ref_month')))
        size=sum(len(chunk.text)for chunk in safe)
        if not size or used+size>budget:continue
        identifiers=[];before=len(packet['cards'])
        for chunk in safe:identifiers.extend(add_source(packet,chunk,corpus,task,entity,role='reaction_'+role))
        actual=sum(len(c['text'])for c in packet['cards'][before:]);used+=actual
        if not identifiers:continue
        text=' '.join(chunk.text for chunk in safe)
        contexts.append({'role':role,'source_ids':list(dict.fromkeys(identifiers)),
            'source_available_date':doc.date,'source_period_cues':list(dict.fromkeys(m.group()for m in _QUARTER.finditer(_normal(text))))[:8],
            'financial_event_period_mapping':'unverified; an earnings date is not a fiscal quarter declaration',
            'status':'reported management expectation with conditions'if'outlook'in role or'forward'in role else'historical reported premise; future recurrence and surprise unknown'})
        roles[role]=roles.get(role,0)+1;selected.append({'doc_id':doc.doc_id,'start':a,'end':b})
    observations['reaction_information']={'contexts':contexts,
        'additional_source_chars':used,'additional_card_count':len(packet['cards'])-added_cards_start,
        'requested_event':entity.get('report_datetime'),'requested_event_window':entity.get('event_window'),
        'requested_benchmark':entity.get('benchmark'),'source_budget_chars':budget,
        'market_consensus_measurement_computed':False,'current_valuation_multiple_computed':False,
        'future_earnings_result_known':False,'future_return_forecast_returned':False,
        'limitations':['Management guidance is not contemporaneous market consensus or evidence of a future earnings surprise.',
            'Historical growth, an accounting benefit or financing capacity does not determine the sign of a one-day abnormal return.',
            'Do not substitute the task cutoff stock price for the later event-window starting close.',
            'Financial/calendar/fiscal period mapping, recurrence, valuation and what was already priced require separate reasoning.',
            'An 8-K may concern credit agreements or directors; a link to an earnings exhibit does not include its contents.']}
    return packet
