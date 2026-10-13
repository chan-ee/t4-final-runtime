"""Eligible pre-cutoff linear CPI model; inputs/claims only from frozen corpus."""
import calendar
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import re
import statistics as st
from .component_graph import _role
from .corpus import Chunk
from .numerical_context import build_numeric_context,_table,number
from .packets import add_source,claims_from_ids

PATH=Path(__file__).resolve().parent/'assets/cpi-pump-linear.json'


def predict(task,entity,corpus,packet):
    role=_role(entity)
    if role not in ('gasoline','energy') or task.get('target',{}).get('type')!='regression':return None
    if entity.get('unit')!='mom_pct_change_sa':return None
    name=task.get('target',{}).get('name','').lower()
    if 'cpi' not in name or 'mom' not in name or any(x in name for x in ('revision','yoy','annual','log')):return None
    units=str(task.get('target',{}).get('unit',task.get('target',{}).get('units',''))).lower().replace('_',' ')
    if re.search(r'fraction|basis.?points?|\bbps\b|index|dollars?|annual|year.over.year|\byoy\b|\blog',units):return None
    if re.search(r'annuali[sz]ed|year.over.year|\byoy\b|logarithm',task.get('prompt',''),re.I):return None
    raw=PATH.read_bytes();model=json.loads(raw)
    cutoff=dt.date.fromisoformat(task['cutoff_date'])
    if task['cutoff_date']<model['available_by'] or float(task.get('interval_level',.9))!=.9:return None
    ref=entity.get('ref_month')
    if not isinstance(ref,str) or not re.fullmatch(r'\d{4}-\d{2}',ref):return None
    year,month=map(int,ref.split('-'))
    if cutoff!=dt.date(year,month,calendar.monthrange(year,month)[1]):return None
    resolution=dt.date.fromisoformat(task['resolution_date'])
    release=dt.date.fromisoformat(entity.get('release_date',task['resolution_date']))
    next_month=cutoff+dt.timedelta(days=1)
    if release!=resolution or resolution<=cutoff or (resolution.year,resolution.month)!=(next_month.year,next_month.month):return None
    previous=(dt.date(year,month,1)-dt.timedelta(days=1)).strftime('%Y-%m')
    ctx=build_numeric_context(task,entity,corpus)
    if not ctx or ctx.get('family')!='cpi_component' or not ctx.get('source_monthly_percent_explicit'):return None
    if ctx.get('observed_adjustment')!=ctx.get('target_adjustment_from_literal_text') or ctx.get('observed_adjustment')!='seasonally_adjusted':return None
    history=ctx.get('observed_history',[])
    if not history or history[-1][0]!=previous or entity.get('latest_published_ref_month')!=previous:return None
    latest=number(entity.get('latest_published_mom_pct'))
    if latest is None or not math.isclose(latest,history[-1][1],abs_tol=1e-9,rel_tol=1e-9):return None
    chosen=None
    for doc in sorted(corpus.docs.values(),key=lambda d:(d.date,d.doc_id),reverse=True):
        if doc.date>task['cutoff_date'] or not(doc.shared or entity['entity_id'] in doc.entity_ids):continue
        table=_table(doc,{'week_ending'})
        if not table or table['header']!=['week_ending','usd_per_gallon']:continue
        preface=doc.text[:table['source']['span_start']]
        if not re.search(r'(?:U\.S\.|US|United States)\s+regular\s+gasoline',preface,re.I):continue
        if re.search(r'forecast|projection|projected|scheduled',preface[-240:],re.I):continue
        grouped={previous:[],ref:[]};selected=[]
        for values,a,b in table['rows']:
            date=dt.date.fromisoformat(values[0]);value=number(values[1]);key=values[0][:7]
            if value is None or value<=0:return None
            if date.weekday()!=0:return None
            if key not in grouped or values[0]>min(doc.date,task['cutoff_date']):continue
            if key==ref and date+dt.timedelta(days=2)>cutoff:continue
            grouped[key].append(value);selected.append((a,b))
        if min(len(grouped[k]) for k in (previous,ref))<3:continue
        chosen=(doc,grouped,selected);break
    if chosen is None:return None
    doc,grouped,selected=chosen
    change=100*(st.mean(grouped[ref])/st.mean(grouped[previous])-1)
    x=[latest,math.sin(2*math.pi*month/12),math.cos(2*math.pi*month/12),change]
    parameters=model['targets'][role]
    point=parameters['intercept']+sum(a*b for a,b in zip(parameters['coef'],x))
    lo=point+parameters['residual_lo'];hi=point+parameters['residual_hi']
    if len(parameters['coef'])!=4 or not all(math.isfinite(v) for v in (point,lo,hi)) or point<=-100 or lo>=hi:return None
    if lo<=-100:return None
    source_ids=add_source(packet,Chunk(doc.doc_id,selected[0][0],selected[-1][1],doc.text[selected[0][0]:selected[-1][1]]),corpus,task,entity,role='local_model_pump_observations')
    ids=list(source_ids)
    cpi_doc=corpus.docs.get(ctx['source']['doc_id'])
    cpi_table=_table(cpi_doc,{'month'}) if cpi_doc else None
    if cpi_table:
        past_row=next(((a,b) for values,a,b in cpi_table['rows'] if values[0]==previous),None)
        if past_row:
            a,b=past_row
            ids+=add_source(packet,Chunk(cpi_doc.doc_id,a,b,cpi_doc.text[a:b]),corpus,task,entity,role='local_model_latest_cpi_observation')
    claims=claims_from_ids(packet,ids[:3],corpus,task,entity,limit=3)
    if not claims:return None
    exact={(c['doc_id'],c['span_start'],c['span_end'])for c in claims}
    selected_ids=list(dict.fromkeys(c['id']for c in packet['cards']
        if(c['doc_id'],c['span_start'],c['span_end'])in exact))
    return {'entity_id':entity['entity_id'],'point_forecast':point,'interval':{'level':.9,'lo':lo,'hi':hi},'claims':claims,
        '_fallback':False,'_views':['local_predictor'],'_field_sources':{'point_forecast':'local_cpi_pump_median_linear','interval':'local_cpi_pre_cutoff_residual_quantiles','claims':'original_source_spans'},
        '_source_ids':selected_ids,
        '_forecast_decision':{'kind':'statistical_cpi_anchor','method':'pump_price_monthly_proxy',
            'role':role,'features':x,'coefficients':parameters['coef'],'intercept':parameters['intercept'],
            'latest_mom_pct':latest,'latest_ref_month':previous,'target_ref_month':ref,
            'pump_mean_previous':st.mean(grouped[previous]),'pump_mean_current':st.mean(grouped[ref]),
            'source_ids':selected_ids,'artifact_available_by':model['available_by'],
            'native_output':{'point_forecast':point,'interval':{'level':.9,'lo':lo,'hi':hi}}},
        '_local_model':{'name':'evidence-inference-cpi-pump-median-linear','revision':'sha256:'+hashlib.sha256(raw).hexdigest(),
            'information_available_by':model['available_by'],'features':x,'current_pump_weeks':len(grouped[ref]),'previous_pump_weeks':len(grouped[previous]),
            'source_doc_id':doc.doc_id,'interval_calibration_verified':False,
            'interpretation':'Regular-grade NSA price is a predictive proxy for all-types SA CPI; learned seasonality and residuals are assumptions, not observed future values.'}}
