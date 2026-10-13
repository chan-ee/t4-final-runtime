"""Current-source gas/energy association and an optional conditional link.

No entity-ID routing, added static weights, external event lookup, forecast row,
label or hard override. Regression is descriptive current-input arithmetic.
"""
from __future__ import annotations
import datetime as dt
import math
import re
import statistics as st
from .numerical_context import build_numeric_context,number

_SERIES={'CUSR0000SETB01':'gasoline','CPIENGSL':'energy'}
_NAMES={'gasolinealltypes':'gasoline','gasoline':'gasoline','energy':'energy','energyindex':'energy'}

def _key(value):return re.sub(r'[^a-z0-9]','',str(value).lower())

def _role(entity):
    by_series=_SERIES.get(entity.get('series_fred'));by_name=_NAMES.get(_key(entity.get('name','')))
    if entity.get('series_fred') and by_series is None:return None
    if by_series and by_name and by_series!=by_name:return None
    return by_series or by_name


def _date(value):
    if not isinstance(value,str)or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):return None
    try:return dt.date.fromisoformat(value)
    except ValueError:return None


def _month(value):return _date(value+'-01')if isinstance(value,str)and re.fullmatch(r'\d{4}-\d{2}',value)else None


def _eligible_doc(corpus,doc_id,entity,cutoff):
    doc=corpus.docs.get(doc_id)
    if doc is None or _date(doc.date)is None or doc.date>cutoff:return None
    return doc if doc.shared or entity.get('entity_id')in doc.entity_ids else None


def fit_observed_relation(gas,energy):
    """Fixed OLS with intercept. Not causal weights or calibrated future errors."""
    if len(gas)!=len(energy)or len(gas)<3 or any(number(v)is None for v in[*gas,*energy]):return None
    xbar,ybar=st.mean(gas),st.mean(energy);xx=sum((x-xbar)**2 for x in gas);yy=sum((y-ybar)**2 for y in energy)
    if xx<=1e-12 or yy<=1e-12:return None
    xy=sum((x-xbar)*(y-ybar)for x,y in zip(gas,energy));slope=xy/xx;intercept=ybar-slope*xbar
    residual=[y-intercept-slope*x for x,y in zip(gas,energy)]
    return {'method':'ordinary least squares with intercept on paired observed SA monthly percent changes',
        'n_paired_months':len(gas),'slope':slope,'intercept_pct':intercept,
        'observed_correlation':xy/math.sqrt(xx*yy),
        'in_sample_residual_sd_pct':math.sqrt(sum(r*r for r in residual)/(len(gas)-2)),
        'observed_child_min_pct':min(gas),'observed_child_max_pct':max(gas),
        'causal_weight_established':False,'future_interval_calibration_verified':False}


def build_component_graph(task,entity,corpus):
    """Return source-only component association and optional conditional hint."""
    target=task.get('target',{});name=str(target.get('name','')).lower();prompt=task.get('prompt','')
    if target.get('type')!='regression'or'cpi'not in name or'mom'not in name:return None
    if re.search(r'(?:^|_)(?:log|ln|revision)(?:_|$)',name):return None
    target_units=str(target.get('unit',target.get('units',''))).lower().replace('_',' ')
    if re.search(r'fraction|basis.?points?|annual|year.over.year|\byoy\b|index|dollars?',target_units):return None
    if re.search(r'annuali[sz]ed|year.over.year|\byoy\b|\blogarithm|\blog\s*\(|\bln\s*\(',name.replace('_',' ')+' '+prompt,re.I):return None
    cutoff=_date(task.get('cutoff_date'))
    if cutoff is None or _role(entity)not in('gasoline','energy'):return None
    roles={role:[e for e in task.get('entities',[])if _role(e)==role]for role in('gasoline','energy')}
    if any(len(v)!=1 for v in roles.values()):return None
    child,parent=roles['gasoline'][0],roles['energy'][0]
    if entity.get('entity_id')not in(child.get('entity_id'),parent.get('entity_id')):return None
    own=child if entity['entity_id']==child['entity_id'] else parent
    if any(entity.get(k)!=own.get(k)for k in('series_fred','name','unit','ref_month','latest_published_ref_month','latest_published_mom_pct','release_date')):return None
    if child.get('release_date',task.get('resolution_date'))!=parent.get('release_date',task.get('resolution_date')):return None
    ref=_month(child.get('ref_month'))
    if ref is None or child.get('ref_month')!=parent.get('ref_month')or ref.year!=cutoff.year or ref.month!=cutoff.month:return None
    contexts={};histories={};sources={}
    for role,node in(('gasoline',child),('energy',parent)):
        if node.get('unit')!='mom_pct_change_sa':return None
        release=_date(node.get('release_date',task.get('resolution_date')))
        if release is None or release<=cutoff:return None
        try:ctx=build_numeric_context(task,node,corpus)
        except (ValueError,TypeError,KeyError,IndexError):return None
        if not ctx or ctx.get('family')!='cpi_component'or not ctx.get('source_monthly_percent_explicit'):return None
        if ctx.get('observed_adjustment')!='seasonally_adjusted'or ctx.get('target_adjustment_from_literal_text')!='seasonally_adjusted':return None
        if _NAMES.get(_key(ctx.get('column')))!=role:return None
        doc=_eligible_doc(corpus,ctx.get('source',{}).get('doc_id'),entity,task['cutoff_date'])
        if doc is None:return None
        rows=ctx.get('observed_history',[]);months=[m for m,_ in rows]
        if not rows or len(months)!=len(set(months))or months!=sorted(months):return None
        latest=node.get('latest_published_ref_month');latest_value=number(node.get('latest_published_mom_pct'))
        if _month(latest)is None or latest!=months[-1]or latest>=child['ref_month']or latest_value is None:return None
        bound=min(task['cutoff_date'],doc.date)[:7]
        if any(_month(m)is None or m>bound or m>=child['ref_month']or number(v)is None for m,v in rows):return None
        if not math.isclose(latest_value,rows[-1][1],rel_tol=1e-9,abs_tol=1e-9):return None
        contexts[role]=ctx;histories[role]=rows;sources[role]=dict(ctx['source'])
    if [m for m,_ in histories['gasoline']]!=[m for m,_ in histories['energy']]:return None
    relation=fit_observed_relation([v for _,v in histories['gasoline']],[v for _,v in histories['energy']])
    if relation is None:return None
    out={'family':'observed_cpi_component_link','entity_id':entity['entity_id'],'entity_role':_role(entity),
        'child_entity_id':child['entity_id'],'parent_entity_id':parent['entity_id'],'target_reference_month':child['ref_month'],
        'source_provenance':sources,'observed_months':[m for m,_ in histories['gasoline']],
        'relation':relation,'conditional_linked_candidate':None,'conditional_candidate_status':'no_verified_child_candidate',
        'hard_override':False,'forecast_row_returned':False,'label_returned':False,'citable_as_source_claim':False,
        'limitations':['This observed statistical association is not a published aggregation weight or a causal gasoline pass-through coefficient.',
            'Gasoline is one component of energy; electricity, utility gas and other fuels remain uncertain.',
            'A short frozen vintage does not establish a stable future relation or conditional coverage.',
            'Derived coefficients and linked candidates are not source quotes. Do not use them as claim text.']}
    candidate=contexts['gasoline'].get('calendar_candidate');pump=contexts['gasoline'].get('pump_observation',{})
    if not isinstance(candidate,dict)or candidate.get('hard_override')is not False or candidate.get('forecast_interval_supplied')is not False:return out
    value=number(candidate.get('conditional_point_pct'))
    if value is None or pump.get('current_month')!=child['ref_month']:return out
    if not _eligible_doc(corpus,candidate.get('frozen_input_source_doc_id'),entity,task['cutoff_date']):return out
    out['conditional_candidate_status']='child_condition_available'
    if _role(entity)=='energy':
        linked=relation['intercept_pct']+relation['slope']*value
        if not math.isfinite(linked):return out
        out['conditional_linked_candidate']={'status':'forecast-control hypothesis, not an observed energy value or a submitted forecast',
            'child_conditional_value_pct':value,'parent_conditional_value_pct':linked,
            'formula':'observed_intercept + observed_slope * existing_conditional_gasoline_candidate',
            'child_outside_observed_range':not(relation['observed_child_min_pct']<=value<=relation['observed_child_max_pct']),
            'assumptions':['The child seasonal/proxy transformation is adequate.','The observed gas-energy relation remains relevant for the target month.'],
            'forecast_interval_supplied':False,'interval_calibration_verified':False,'hard_override':False}
    return out
