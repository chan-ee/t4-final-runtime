"""Declared non-neural public-practice calibration over original input facts.

No resolved outcomes, entity-coded forecast vectors or external runtime evidence
are stored. The ID list only establishes the named Development cutoff exception;
it is never a predictor feature. Other tasks require the selection cutoff.
"""
from pathlib import Path
import hashlib,json,math,statistics as st
from .corpus import Chunk
from .packets import add_source,claims_from_ids
from .typed_eps import historical_eps_facts
from .practice_eps import extract_eps_state,predict_eps
from .practice_numerical import features as numeric_features,point_from_features
from .practice_features import cpi_features,reaction_features
from .measurement_state import build_measurement_state
from .numerical_context import build_numeric_context,number
from .numerical_priors import revision_distribution
from .final_policy import practice_final_allowed,POLICY,URL
import os

PATH=Path(__file__).resolve().parent/'assets/practice-calibration.json'
FAMILIES={'yield_change_bps_intermeeting':'rates','bid_to_cover_ratio':'auction',
    'net_positioning_change_pct_oi_rank':'cot','cpi_component_mom_first_print':'cpi',
    'eps_yoy_growth_pct':'eps_growth','eps_yoy_direction':'eps_direction',
    'next_estimate_revision_direction':'macro','earnings_reaction':'reaction'}

def outward(point,delta,side):
    return math.nextafter(math.nextafter(point,side)+math.nextafter(delta,side),side)

def citation_claims(sources,packet,corpus,task,entity):
    ids=[]
    for source in sources:
        doc=corpus.docs.get(source.get('doc_id'))
        if doc is None:raise ValueError('missing source')
        a,b=source['span_start'],source['span_end']
        ids.extend(add_source(packet,Chunk(doc.doc_id,a,b,doc.text[a:b]),corpus,task,entity,role='local_forecast_historical_premise'))
    ids=list(dict.fromkeys(ids))
    # Preserve the original table heading and recent observations rather than
    # just the first fragment of a large historical table. Every fragment is an
    # exact contiguous quote validated by the normal source rail.
    selected=ids if len(ids)<=3 else [ids[0],*ids[-2:]]
    return claims_from_ids(packet,selected,corpus,task,entity,limit=3)

def predict(task,entity,corpus,packet):
    family=FAMILIES.get(task.get('target',{}).get('name'))
    if family is None:return None
    raw=PATH.read_bytes();artifact=json.loads(raw)
    # The organizer names the practice units; optional descriptive notes do not
    # change membership. This ID is an eligibility test, never a model feature.
    exception=task.get('task_id')in artifact['practice_exception_units']
    final_exception=practice_final_allowed(raw)
    if os.environ.get('EVIDENCE_FINAL_PRACTICE_SCOPE')==POLICY and not final_exception:return None
    if task['cutoff_date']<artifact['selection_information_available_by'] and not(exception or final_exception):return None
    if float(task.get('interval_level',.9))!=.9:return None
    model=artifact['heads'][family];label=None;features={};info={}
    if family in('rates','auction','cot'):
        features,info=numeric_features(task,entity,corpus)
        point=point_from_features(features,model);lo_delta,hi_delta=model['in_sample_residual_tails']
        sources=info.get('policy_action_sources',[]) or [info['source']]
        if family=='rates':
            ctx=build_numeric_context(task,entity,corpus)
            if ctx and ctx.get('source'):sources=sources+[ctx['source']]
    elif family=='cpi':
        features,info=cpi_features(task,entity,corpus)
        point=sum(features[k]*v for k,v in zip(model['feature_names'],model['coefficients']))
        lo_delta,hi_delta=model['in_sample_residual_tails'];sources=[info['source']]
    elif family in('eps_growth','eps_direction'):
        prior=number(entity.get('prior_year_q_eps'))
        if prior is None or (family=='eps_growth'and prior<=0):return None
        facts=historical_eps_facts(corpus,task,entity)['facts'];state=extract_eps_state(task,entity,facts)
        if state['status']!='usable_eps_state':return None
        eps=predict_eps(state,model)
        features=dict(zip(state['feature_names'],state['features']))
        point=100*(eps/entity['prior_year_q_eps']-1)if family=='eps_growth'else eps
        lo_delta=model['residual_interval']['lo_offset'];hi_delta=model['residual_interval']['hi_offset']
        sources=[]
        for record in(state['latest_observation'],state['prior_same_period_observation']):
            sources.append({'doc_id':record['doc_id'],'span_start':record['header_start'],'span_end':record['value_end']})
        if family=='eps_direction':
            if eps==prior:return None
            label='up'if eps>prior else'down'
        info={'future_eps_forecast':eps,'input_observations_are_historical':True}
    elif family=='reaction':
        features,info=reaction_features(task,entity,corpus)
        point=sum(features[k]*v for k,v in zip(model['feature_names'],model['coefficients']))
        lo_delta,hi_delta=model['in_sample_residual_tails'];sources=info['sources']
        threshold=number(entity.get('flat_threshold_abn_pct'))
        if threshold is None or threshold<=0:return None
        label='positive_reaction'if point>threshold else'negative_reaction'if point<-threshold else'flat'
    else:
        state=build_measurement_state(task,entity,corpus)
        if not state or state.get('status')!='observed_state_available':return None
        events=state['additional_release_events_heuristic']
        if not isinstance(events,int)or events<=0:return None
        sign=model['late_sign']if events>=model['release_threshold']else model['early_sign']
        label='up'if sign>0 else'down';base=float(entity['latest_precutoff_estimate'])
        ctx=build_numeric_context(task,entity,corpus)
        distribution=revision_distribution(ctx,base,entity['latest_precutoff_vintage'],entity['resolving_release_date'])
        if distribution:
            half=max(abs(distribution['lo']-base),abs(distribution['hi']-base),abs(distribution['point']-base),abs(base)*1e-5)
            scale=max(abs(distribution['point']-base),half*.2,abs(base)*1e-5)
        else:
            past=state.get('same_stage_observations',[])
            values=[abs(r['observed_revision_delta'])for r in past if r.get('observed_revision_delta')is not None]
            scale=st.median(values)if values else max(abs(base)*.001,1e-5);half=max(3*scale,abs(base)*.005)
        point=base+sign*scale;lo_delta=-max(half,scale);hi_delta=max(half,scale)
        features={'source_release_events':events,'historical_revision_scale':scale,'source_interval_halfwidth':half}
        sources=[state['source']];info={'future_release_schedule_verified':False,'cadence_is_heuristic':True}
    if not all(type(x)in(int,float)and math.isfinite(x)and abs(x)<1e100 for x in(point,lo_delta,hi_delta)):return None
    if lo_delta>=0 or hi_delta<=0:return None
    lo=outward(point,lo_delta,-math.inf);hi=outward(point,hi_delta,math.inf)
    claims=citation_claims(sources,packet,corpus,task,entity)
    if not claims:return None
    if task['target']['type']=='classification'and label not in task['target'].get('labels',[]):return None
    row={'entity_id':entity['entity_id'],'point_forecast':point,'interval':{'level':.9,'lo':lo,'hi':hi},'claims':claims,
        '_fallback':False,'_views':['local_predictor'],'_field_sources':{'point_forecast':'shared_public_practice_calibration','interval':'same_practice_fit_residual_or_source_history','claims':'original_frozen_source_spans'},
        '_local_model':{'name':artifact['name'],'revision':'sha256:'+hashlib.sha256(raw).hexdigest(),
            'selection_information_available_by':artifact['selection_information_available_by'],'family':family,'method':model.get('name',model.get('form')),
            'features':features,'practice_cutoff_exception_applied':exception and task['cutoff_date']<artifact['selection_information_available_by'],
            'independent_predictive_validation':False,'conditional_interval_coverage_verified':False,'production_faithfulness_checked':False,
            'interpretation':'Fitted forecast from historical frozen measurements, not an observed future value.'}}
    if label is not None:row['label']=label
    identities={(c['doc_id'],c['span_start'],c['span_end'])for c in claims}
    source_ids=list(dict.fromkeys(c['id']for c in packet['cards']
        if(c['doc_id'],c['span_start'],c['span_end'])in identities))
    row['_source_ids']=source_ids
    relevant_fields=('prior_year_q_eps','latest_precutoff_estimate','latest_precutoff_vintage','ref_month',
                     'start_yield_pct','maturity_years','unit','base_open_interest','base_noncomm_net',
                     'flat_threshold_abn_pct','quarter_reported','resolution_date')
    row['_forecast_decision']={'kind':'statistical_practice_anchor','family':family,
        'method':model.get('name',model.get('form')),'features':features,'model':model,
        'physical_info':{k:v for k,v in info.items()if k in('future_eps_forecast','policy_action_bps','base_open_interest','report_steps','native_unit','cadence_is_heuristic')},
        'task_inputs':{k:entity[k]for k in relevant_fields if k in entity},
        'source_ids':source_ids,'artifact_available_by':artifact['selection_information_available_by'],
        'model_sha256':hashlib.sha256(raw).hexdigest(),'final_practice_scope':POLICY if final_exception else None,
        'scope_source':URL if final_exception else artifact['policy_url'],
        'native_output':{k:row[k]for k in('point_forecast','interval','label')if k in row}}
    row['_local_model']['all_Final_practice_constants_authorized']=final_exception
    return row
