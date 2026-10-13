"""Deterministic arithmetic and conservative no-service estimates; no fitted external artifacts."""
import ast
import math
import operator
import re
import statistics
from .priors import vintage_prior
from .native_bounds import native_bounds
from .native_reference import matching_historical_values,task_reference,known_vintage_contract,change_target

OPS={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}

def evaluate_expression(expression):
    if not isinstance(expression,str) or len(expression)>240:raise ValueError('expression length')
    def walk(node):
        if isinstance(node,ast.Constant) and type(node.value) in (int,float):v=float(node.value)
        elif isinstance(node,ast.BinOp) and type(node.op) in OPS:v=OPS[type(node.op)](walk(node.left),walk(node.right))
        elif isinstance(node,ast.UnaryOp) and isinstance(node.op,(ast.UAdd,ast.USub)):v=walk(node.operand)*(1 if isinstance(node.op,ast.UAdd) else -1)
        else:raise ValueError('arithmetic only')
        if not math.isfinite(v) or abs(v)>1e15:raise ValueError('arithmetic range')
        return v
    return walk(ast.parse(expression,mode='eval').body)

def finite(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)):return None
    return float(value) if math.isfinite(value) and abs(value)<1e100 else None

def historical_values(task,chunks,entity=None,corpus=None):
    return matching_historical_values(task,entity or{},chunks,corpus)

def cot_rank_fallback(task,entity,chunks):
    """Causal five-report trend from a supplied COT table; never an outcome lookup.

    This rule was selected on 2024 pre-cutoff public history and is only eligible
    for tasks with cutoff at or after that history. Earlier tasks use generic fallback.
    """
    if task.get('cutoff_date','') < '2024-10-31':
        return None
    name=task.get('target',{}).get('name','').lower()
    if task.get('target',{}).get('type')!='ranking' or 'positioning_change_pct_oi' not in name:
        return None
    for chunk in chunks:
        if 'report_date | open_interest' not in chunk.text or 'noncomm_net' not in chunk.text:
            continue
        rows=[]
        for line in chunk.text.splitlines():
            cells=[x.strip() for x in line.split('|')]
            if len(cells)<7 or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',cells[0]):
                continue
            try:
                oi=float(cells[1].replace(',',''));net=float(cells[4].replace(',',''))
            except ValueError:
                continue
            if oi>0:rows.append((oi,net))
        if len(rows)<8:continue
        oi,net=rows[-1]
        forecast=0.1*100*5*(net-rows[-2][1])/oi
        residuals=[]
        for i in range(8,len(rows)-5):
            base_oi,base_net=rows[i]
            prediction=0.1*100*5*(base_net-rows[i-1][1])/base_oi
            realized=100*(rows[i+5][1]-base_net)/base_oi
            residuals.append(abs(realized-prediction))
        half=sorted(residuals)[min(len(residuals)-1,int(.9*len(residuals)))] if residuals else 10.
        return forecast,max(half,0.5)
    return None

def fallback_estimate(task,entity,chunks,corpus=None):
    name=task.get('target',{}).get('name','').lower()
    prompt=task.get('prompt','').lower()
    target_type=task.get('target',{}).get('type','classification')
    labels=task.get('target',{}).get('labels',[])
    # Report the organizer's requested level without reselecting the existing
    # uncalibrated fallback point or endpoints.
    level=float(task.get('interval_level',.9))
    # This numerical prior was selected with publicly available vintage outcomes
    # through 2024-09-30. It must not be routed to earlier-cutoff tasks.
    vintage=vintage_prior(task,entity,chunks,corpus) if task.get('cutoff_date','')>='2024-09-30'and known_vintage_contract(task,entity)else None
    if vintage is not None:
        row={'entity_id':entity['entity_id'],'point_forecast':vintage['point'],
             'interval':{'level':level,'lo':vintage['lo'],'hi':vintage['hi']},
             '_reasoning':f"current vintage-table revision prior from {vintage['observed_revisions']} observed revisions over {vintage['horizon_releases']} release steps; empirical tails, not guaranteed calibration",
             '_fallback':True}
        if target_type=='classification' and vintage['label'] in labels:row['label']=vintage['label']
        elif target_type=='classification':row['label']=labels[0]
        return row
    neutral=next((v for v in ('inline','flat','no_event','unchanged','stable','neutral') if v in labels),labels[0] if labels else None)
    # Prediction meaning comes from target and prompt, never task_id or remembered outcomes.
    point=0.0;half=1.0;method='unresolved native-level numeric placeholder; no baseline or calibrated interval established'
    rank_prior=cot_rank_fallback(task,entity,chunks)
    if rank_prior is not None:
        point,half=rank_prior
        method='pre-cutoff one-week positioning trend, conservatively shrunk with historical residual band'
    is_change=change_target(task)
    history=historical_values(task,chunks,entity,corpus) if not is_change else []
    reference=task_reference(task,entity,chunks,corpus)if not is_change else None
    if history and rank_prior is None:
        recent=history[-6:];point=statistics.mean(recent)
        deviation=statistics.stdev(history) if len(history)>1 else abs(point)*.2
        half=max(1.645*deviation,abs(point)*.05,.01);method='recent historical mean with uncalibrated historical-variation band'
    elif rank_prior is not None:
        pass
    elif reference is not None:
        point=reference['value'];half=max(abs(point)*.25,.05);method=reference['method']+'; uncalibrated band'
    elif native_bounds(task,entity)is not None:
        bounds=native_bounds(task,entity);point=(bounds[0]+bounds[1])/2;half=(bounds[1]-bounds[0])/2
        method='uninformative native-range midpoint; not a learned risk estimate'
    elif 'bps' in name or 'basis point' in prompt:
        point=0;half=100;method=('zero-change fallback; heuristic uncertainty band'if is_change else'unresolved native-level numeric placeholder; heuristic uncertainty band')
    elif 'growth' in name:
        point=0;half=50;method='zero-growth fallback; heuristic 50 percentage-point band'
    elif is_change:
        point=0;half=10;method='zero-change fallback; heuristic interval, no ordering information'
    elif target_type=='ranking':
        point=0;half=10;method='unresolved native ranking-metric placeholder; heuristic interval, no ordering information'
    else:
        if labels:point=.5;half=.5;method='label fallback; numeric target not established'
    result={'entity_id':entity['entity_id'],'point_forecast':point,'interval':{'level':level,'lo':point-half,'hi':point+half},'_reasoning':method,'_fallback':True}
    if target_type=='classification':result['label']=neutral
    bounds=native_bounds(task,entity)
    if bounds is not None:
        result['interval']['lo']=max(bounds[0],result['interval']['lo'])
        result['interval']['hi']=min(bounds[1],result['interval']['hi'])
    # Native probability ranges were already resolved from the target and unit.
    # Incidental probability words cannot overwrite a percent, signed-change,
    # or bps interval with the dimensionless credit-event range.
    return result
