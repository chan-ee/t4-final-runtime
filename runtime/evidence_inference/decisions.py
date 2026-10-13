"""Native-unit source-bound decisions and transparent two-view combination."""
import math
from .native_bounds import native_bounds
from .numerical import finite,evaluate_expression
from .packets import claims_from_ids
from .quantity import transform_forecast_eps
from .forecast_accounting import calculate_forecast_accounting


def normalize_forecast(raw,task,entity,packet,corpus,prior,*,view):
    if not isinstance(raw,dict):return None,'missing_row'
    if raw.get('entity_id')!=entity['entity_id']:return None,'wrong_entity'
    accounting=None
    if 'forecast_diluted_income' in raw:
        fields,accounting=calculate_forecast_accounting(task,entity,raw)
        if fields is None:return None,accounting
        raw=dict(raw);raw.update(point_forecast=fields['point_forecast'],interval=fields['interval'],forecast_eps=fields['forecast_eps'])
    typ=task['target']['type'];point=finite(raw.get('point_forecast'))
    band=raw.get('interval',{})
    lo=finite(band.get('lo')) if isinstance(band,dict) else None
    hi=finite(band.get('hi')) if isinstance(band,dict) else None
    level=float(task.get('interval_level',.9))
    if isinstance(band,dict) and 'level' in band and finite(band['level'])!=level:return None,'wrong_interval_level'
    source='house_final';transformed=None
    if 'forecast_eps' in raw:
        transformed=transform_forecast_eps(task,entity,raw['forecast_eps'])
        if transformed is not None:
            point=finite(transformed['point_forecast'])
            lo=finite(transformed['interval']['lo']);hi=finite(transformed['interval']['hi'])
            source='converted_house_eps'
    if raw.get('point_expression') and transformed is None:
        try:point=evaluate_expression(raw['point_expression']);source='arithmetic_house'
        except (ValueError,SyntaxError,OverflowError,ZeroDivisionError,RecursionError):return None,'invalid_arithmetic'
    label=raw.get('label');probabilities=None
    if typ=='classification':
        labels=task['target'].get('labels',[]);values=raw.get('label_probabilities')
        if isinstance(values,dict) and set(values)==set(labels):
            probabilities={k:finite(values[k]) for k in labels}
            if any(v is None or not 0<=v<=1 for v in probabilities.values()):probabilities=None
            elif not .98<=sum(probabilities.values())<=1.02:probabilities=None
            else:
                total=sum(probabilities.values());probabilities={k:v/total for k,v in probabilities.items()}
                maximum=max(probabilities.values());ties=[k for k,v in probabilities.items() if abs(v-maximum)<1e-12]
                label=label if label in ties else ties[0]
        if label not in labels:return None,'invalid_label'
        if point is None:point=finite(prior.get('point_forecast'));source='carried_prior_point'
    if any(v is None for v in (point,lo,hi)) or lo>hi:return None,'invalid_numeric_fields'
    name=task['target'].get('name','').lower();prompt=task.get('prompt','').lower()
    bounds=native_bounds(task,entity)
    if bounds is not None and not(bounds[0]<=point<=bounds[1]and bounds[0]<=lo<=hi<=bounds[1]):
        return None,'outside_declared_native_bounds'
    support=raw.get('support_ids',[]);counter=raw.get('counter_ids',[])
    if not isinstance(support,list) or not isinstance(counter,list):return None,'invalid_source_id_list'
    identifiers=list(dict.fromkeys(x for x in support+counter if isinstance(x,str)))[:3]
    claims=claims_from_ids(packet,identifiers,corpus,task,entity,limit=3)
    if not claims:return None,'unresolved_sources'
    spans={(c['doc_id'],c['span_start'],c['span_end']) for c in claims}
    cards=[c for c in packet['cards'] if (c['doc_id'],c['span_start'],c['span_end']) in spans]
    if not any(c['role'] not in ('period_header','accounting_context') for c in cards):return None,'metadata_only_sources'
    roles={(c['doc_id'],c['span_start'],c['span_end']):c['role'] for c in cards}
    claims.sort(key=lambda c:roles[(c['doc_id'],c['span_start'],c['span_end'])] in ('period_header','accounting_context'))
    row={'entity_id':entity['entity_id'],'point_forecast':point,'interval':{'level':level,'lo':lo,'hi':hi},'claims':claims,
        '_reasoning':str(raw.get('mechanism',raw.get('reasoning','')))[:400],
        '_uncertainty':str(raw.get('uncertainty',''))[:200],
        '_source_ids':[c['id'] for c in cards],'_fallback':False,'_views':[view],
        '_field_sources':{'point_forecast':source,'interval':'converted_house_eps' if transformed else 'house_final','claims':'original_source_spans'},
        '_label_probs':probabilities}
    if typ=='classification':row['label']=label;row['_field_sources']['label']='house_final'
    if transformed:row['_quantity_provenance']=transformed.get('provenance',{})
    if accounting:
        row['_accounting_forecast']=accounting
        row['_field_sources'].update(point_forecast='code_from_house_forecast_diluted_common_income_and_shares',interval='code_from_house_joint_forecast_eps_scenarios')
    return row,None


def combine_forecasts(rows,prior,task):
    """Equal endpoint/point averaging is a declared uncalibrated combination.

    For a fixed truth this preserves the convex raw MAE/interval-loss bound
    against the members' average; bounded anchored/capped score may still fall.
    Both views use the same House model and are not independent random experts.
    """
    valid=[r for r in rows if r is not None]
    if not valid:return prior
    if len(valid)==1:return dict(valid[0])
    count=len(valid);out=dict(valid[0])
    out['point_forecast']=sum(r['point_forecast']/count for r in valid)
    out['interval']={'level':valid[0]['interval']['level'],
        'lo':sum(r['interval']['lo']/count for r in valid),
        'hi':sum(r['interval']['hi']/count for r in valid)}
    claims=[];keys=set()
    # Each included passage is a relevant premise actually selected by a view;
    # no unrelated citation padding or generated claims are introduced.
    for index in range(max(len(r['claims']) for r in valid)):
        for r in valid:
            if index>=len(r['claims']):continue
            c=r['claims'][index];key=(c['doc_id'],c['span_start'],c['span_end'])
            if key not in keys:claims.append(c);keys.add(key)
            if len(claims)>=3:break
        if len(claims)>=3:break
    out['claims']=claims;out['_views']=[v for r in valid for v in r['_views']]
    out['_source_ids']=list(dict.fromkeys(x for r in valid for x in r.get('_source_ids',[])))[:6]
    out['_reasoning']='Two source-grounded view estimates averaged in native units. Agreement is not truth evidence.'
    out['_field_sources']={k:'two_view_combination' for k in ('point_forecast','interval')}
    out['_field_sources']['claims']='original_source_spans'
    width=max(sum(r['interval']['hi']-r['interval']['lo'] for r in valid)/count,1e-8)
    out['_disagreement']=abs(valid[0]['point_forecast']-valid[1]['point_forecast'])/width
    if task['target']['type']=='classification':
        probs=[r.get('_label_probs') for r in valid]
        if all(isinstance(p,dict) for p in probs):
            average={label:sum(p[label]/count for p in probs) for label in task['target']['labels']}
            maximum=max(average.values());ties=[label for label in task['target']['labels'] if abs(average[label]-maximum)<1e-12]
            out['label']=valid[0]['label'] if valid[0]['label'] in ties else ties[0]
            out['_label_probs']=average
        else:out['_label_probs']=None
        out['_label_disagreement']=len({r['label'] for r in valid})>1
        out['_field_sources']['label']='pooled_view_probabilities' if out['_label_probs'] else 'primary_view_label'
    return out


def review_reasons(task,rows,packets,valid_view_count):
    out={}
    for entity in task['entities']:
        eid=entity['entity_id'];row=rows[eid];reasons=[]
        count=valid_view_count.get(eid,0)
        if count==0:reasons.append('no_verified_house_view')
        elif count==1:reasons.append('only_one_verified_house_view')
        if row.get('_label_disagreement'):reasons.append('classification_disagreement')
        if row.get('_disagreement',0)>.30:reasons.append('material_native_unit_disagreement')
        if packets[eid].get('warnings'):reasons.append('measurement_definition_warning')
        if task['target']['type']=='ranking':reasons.append('joint_metric_order_check')
        if reasons:out[eid]=reasons
    return out
