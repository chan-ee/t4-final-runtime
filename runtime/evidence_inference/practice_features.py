"""Identity-free frozen-input features for a declared practice calibration."""
import datetime as dt
import math
import re
import statistics as st
from .numerical_context import build_numeric_context
from .typed_eps import _normal,_HEADER,_DATE,_MONTHS
from .financial_measurement import _cells

def cpi_features(task,entity,corpus):
    ctx=build_numeric_context(task,entity,corpus)
    if not ctx or ctx['family']!='cpi_component' or not ctx['source_monthly_percent_explicit']:raise ValueError('CPI source')
    if ctx['observed_adjustment']!=ctx['target_adjustment_from_literal_text'] or ctx['observed_adjustment']!='seasonally_adjusted':raise ValueError('adjustment')
    vals=[x[1]for x in ctx['observed_history']]
    if len(vals)<6 or abs(vals[-1]-entity['latest_published_mom_pct'])>1e-8:raise ValueError('history')
    if entity.get('unit')!='mom_pct_change_sa':raise ValueError('native CPI unit')
    latest=dt.date.fromisoformat(ctx['observed_history'][-1][0]+'-01')
    following=(latest+dt.timedelta(days=32)).strftime('%Y-%m')
    if entity.get('ref_month')!=following or entity.get('latest_published_ref_month')!=latest.strftime('%Y-%m'):raise ValueError('one-step CPI horizon')
    if any(not math.isfinite(v)for v in vals):raise ValueError('nonfinite')
    return dict(intercept=1.,last=vals[-1],**{'lag'+str(i):vals[-i]for i in range(2,7)}),ctx

def reaction_features(task,entity,corpus):
    if task['target']['name']!='earnings_reaction' or entity.get('benchmark')!='SPY':raise ValueError('reaction contract')
    candidates=[]
    for doc in corpus.docs.values():
        if doc.date>task['cutoff_date'] or doc.shared or entity['entity_id']not in doc.entity_ids:continue
        text=_normal(doc.text)
        # An explicitly quantified consolidated annual narrative is a fallback
        # when a comparable quarter revenue table is absent. It is not relabeled
        # as a quarterly observation and never becomes a future return fact.
        pattern=r"(?:the company[’']s\s+)?total net sales\s+(increased|decreased|grew|declined)\s+(\d+(?:\.\d+)?)\s*%.{0,200}?during\s+(20\d{2})\s+compared to\s+(20\d{2})"
        for match in re.finditer(pattern,text,re.I):
            if int(match[3])-int(match[4])!=1 or int(match[3])>int(doc.date[:4]):continue
            g=float(match[2])*(-1 if match[1].lower()in('decreased','declined')else 1)
            candidates.append({'growth':g,'duration':12,'year':int(match[3]),'doc_date':doc.date,
                'sources':[{'doc_id':doc.doc_id,'span_start':match.start(),'span_end':match.end()}]})
        for match in re.finditer(r'(?:Total net sales|Total revenue|Revenue)(?=\s*\$?\s*\d)',text,re.I):
            if match.group().lower()=='revenue' and re.search(r'(?:other|of)\s*$',text[max(0,match.start()-30):match.start()],re.I):continue
            heads=list(_HEADER.finditer(text,max(0,match.start()-5000),match.start()))
            if not heads:continue
            nearest=heads[-1];adjacent=[h for h in heads if nearest.start()-h.start()<=200]
            if len(adjacent)>2 or not adjacent[0].group().lower().startswith('three'):continue
            start=adjacent[0].start();fragment=text[start:match.start()]
            if re.search(r'as a percentage of revenue',text[max(0,start-500):start],re.I):continue
            years=list(re.finditer(r'(?<!\d)20\d{2}(?!\d)',fragment[:450]))[:2*len(adjacent)]
            if len(years)!=2*len(adjacent):continue
            yy=[int(m.group())for m in years]
            if abs(yy[0]-yy[1])!=1 or any(yy[i:i+2]!=yy[:2]for i in range(0,len(yy),2)):continue
            if re.search(r'%\s*change|change\s*%',fragment[:years[-1].end()],re.I):continue
            cells=_cells(text,match.end(),len(yy))
            if not cells or any(c['value']is None or c['value']<=0 for c in cells[:2]):continue
            newest=0 if yy[0]>yy[1]else 1;old=1-newest
            dates=list(_DATE.finditer(fragment[:years[0].start()]))
            if not dates:continue
            month=_MONTHS[dates[0][1].lower()];day=int(dates[0][2])if dates[0][2]else None
            if day is None:continue
            end=str(dt.date(max(yy),month,day))
            if end>min(task['cutoff_date'],doc.date):continue
            growth=100*(cells[newest]['value']/cells[old]['value']-1)
            candidates.append({'growth':growth,'duration':3,'end':end,
                'doc_date':doc.date,'sources':[{'doc_id':doc.doc_id,'span_start':start,'span_end':start+years[-1].end()},
                    {'doc_id':doc.doc_id,'span_start':match.start(),'span_end':cells[-1]['end']}]})
    quarter=[c for c in candidates if c['duration']==3]
    selected=quarter or candidates
    if not selected:raise ValueError('no comparable revenue growth')
    newest=max(c['doc_date']for c in selected);selected=[c for c in selected if c['doc_date']==newest]
    if max(c['growth']for c in selected)-min(c['growth']for c in selected)>1e-7:raise ValueError('ambiguous revenue')
    row=selected[0]
    return {'intercept':1.,'reported_revenue_growth_pct':row['growth']},row
