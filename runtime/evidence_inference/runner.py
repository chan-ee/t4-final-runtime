"""Complete-roster checkpoints, two forecast views and bounded factual review."""
from __future__ import annotations
import argparse
import concurrent.futures
import json
import math
import os
from pathlib import Path
import time
from .client import HouseClient
from .corpus import load_corpus,select_evidence
from .decisions import normalize_forecast,combine_forecasts,review_reasons
from .forecast_context import enrich_forecast_context
from .forecast_prompts import SYSTEM,build_forecast_prompt
from .legacy import normalize_prediction,_write,_rows
from .numerical_priors import build_numerical_prior
from .packets import prepare_packet,claims_from_ids
from .target_context import enrich_target_information
from .reaction_context import enrich_reaction_information
from .financial_measurement import build_financial_measurement,public_financial_measurement
from .measurement_links import attach_measurement_citations
from .measurement_state import build_measurement_state
from .component_graph import build_component_graph
from .contract_flows import enrich_contract_flow_information
from .deliberation import execute_deliberation
from .delivery_gate import delivery_findings,ModelDeliveryUnverified
from .cpi_predictor import predict as local_cpi_predict
from .practice_forecast import predict as practice_predict
from .harness.coordinator import execute_collaboration
from .single import execute_single,safe_delivery_summary
from .delivery_precision import compact_house_answers
from .reason_task_inputs import reason_task_inputs
from .decision_ledger import build_ledger
from .reason_compiler import compile_reasons


CORRECTIONS={'initial_source_forecast','arithmetic','unit','period','accounting','scope','classification_assessment','cross_entity_order'}


def review_decision(raw,task,entity,packet,corpus,previous,flags):
    """Require a flagged issue and specific current source facts before revision.

    These gates verify protocol and provenance, not economic correctness/NLI.
    """
    if not isinstance(raw,dict):return None,'missing_review_row'
    if raw.get('correction_kind')=='retain':return None,'retained'
    if not flags:return None,'unflagged_change'
    if raw.get('correction_kind') not in CORRECTIONS:return None,'invalid_correction_kind'
    if raw.get('correction_kind')=='initial_source_forecast' and 'no_verified_house_view' not in flags:return None,'initial_forecast_requires_zero_views'
    reason=raw.get('correction_reason')
    if not isinstance(reason,str) or not reason.strip() or len(reason)>160:return None,'invalid_correction_reason'
    references=raw.get('correction_source_ids')
    if not isinstance(references,list) or not 1<=len(references)<=3:return None,'missing_correction_sources'
    if len(set(x for x in references if isinstance(x,str)))!=len(references):return None,'invalid_correction_sources'
    citations=claims_from_ids(packet,references,corpus,task,entity)
    if len(citations)!=len(references):return None,'unresolved_correction_sources'
    request=dict(raw)
    # Changed estimates must actually cite the declared correction premises.
    request['support_ids']=references;request['counter_ids']=[]
    row,error=normalize_forecast(request,task,entity,packet,corpus,previous,view='review')
    if row is None:return None,error
    row['_correction']={'kind':raw['correction_kind'],'reason':reason,'source_ids':references}
    return row,None


def run(task_path,corpus_dir,out_path,*,transport=None,max_calls=25,deadline_seconds=480,review=True,protocol=None,require_verified_house=None):
    began=time.monotonic();task=json.loads(Path(task_path).read_text())
    protocol=protocol or os.environ.get('EVIDENCE_PROTOCOL','single')
    requested_protocol=protocol
    if protocol in ('collaborative','single') and task['target'].get('name')=='credit_event_12m' and os.environ.get('EVIDENCE_HARNESS_PRESERVE_CREDIT','1')=='1':
        protocol='direct_control'
    if protocol not in ('information_control','deliberative','direct_control','collaborative','single'):raise ValueError('unknown inference protocol')
    required=bool(require_verified_house)if require_verified_house is not None else os.environ.get('EVIDENCE_REQUIRE_VERIFIED_HOUSE','0')=='1'
    single_delivery_required=os.environ.get('EVIDENCE_SINGLE_REQUIRE_MODEL_DELIVERY','0')=='1'
    if required and protocol=='information_control':raise ValueError('verified delivery trial requires a single-decision protocol')
    if required and Path(out_path).exists():raise ValueError('verified delivery trial requires a fresh final-output path')
    checkpoint_path=Path(out_path).with_name(Path(out_path).stem+'.checkpoint'+Path(out_path).suffix)if required or single_delivery_required else out_path
    if single_delivery_required and Path(out_path).exists():raise ValueError('Single delivery diagnostic requires fresh final output')
    def checkpoint(payload):
        _write(checkpoint_path,payload)
        summary=safe_delivery_summary(payload)
        _write(Path(out_path).with_name('delivery-diagnostics.json'),summary)
    entities=task['entities'];ids=[e['entity_id'] for e in entities]
    if not ids or len(ids)!=len(set(ids)):raise ValueError('invalid trusted roster')
    if task.get('target',{}).get('type') not in ('classification','regression','ranking'):raise ValueError('unknown target type')
    corpus=load_corpus(corpus_dir,task['cutoff_date'],task=task)
    evidence={e['entity_id']:select_evidence(corpus,task,e) for e in entities}
    priors={e['entity_id']:normalize_prediction(None,task,e,evidence[e['entity_id']],corpus) for e in entities}
    results=dict(priors);packets={};views={eid:{} for eid in ids};rejections={eid:[] for eid in ids}
    reviews={};flags={};enrichment_errors={};local_models={};stage='initial_fallback';client=None;inference_plan={}

    def answer():
        counts={eid:len(views[eid]) for eid in ids}
        runtime=client.snapshot() if client is not None else {'requests_attempted':0,'preprocessing':True}
        runtime.update(elapsed_seconds=round(time.monotonic()-began,3),judge_evaluated=False,intervals_calibrated=False)
        payload={'task_id':task['task_id'],'schema_version':'3','target_type':task['target']['type'],
            'entity_predictions':[{k:v for k,v in results[eid].items() if not k.startswith('_')} for eid in ids],
            'evidence_trace':'Request-bound single forecast with locally resolved original citations and explicit native quantities. Delivery diagnostics do not establish predictive quality, model identity, calibration or production judge acceptance.',
                'notes':{'agent':'evidence-inference-final-general','protocol':protocol,'requested_protocol':requested_protocol,'protected_control':requested_protocol!=protocol,'stage':stage,'inference_plan':inference_plan,'verified_house_delivery_required':required,
                'fallback_entities':sum(bool(results[eid].get('_fallback',True)) for eid in ids),
                'verified_views_per_entity':counts,'row_rejections':rejections,'review_results':reviews,'review_flags':flags,
                'field_sources':{eid:results[eid].get('_field_sources',{'point_forecast':'simple_prior','interval':'uncalibrated_prior','claims':'original_source_spans'}) for eid in ids},
                'prior_methods':{eid:priors[eid].get('_prior_method','v5_fallback') for eid in ids},
                'forecast_accounting':{eid:results[eid]['_accounting_forecast'] for eid in ids if '_accounting_forecast' in results[eid]},
                'local_models':local_models,
                'enrichment_errors':enrichment_errors,'runtime':runtime}}
        if inference_plan.get('submitted_reasons'):payload['submitted_reasons']=inference_plan['submitted_reasons']
        return payload

    # There is an entire valid roster before source enrichment or any network call.
    checkpoint(answer())
    stage='source_enrichment'
    for entity in entities:
        eid=entity['entity_id']
        try:
            if not evidence[eid]:
                enrichment_errors[eid]='no_scoped_pre_cutoff_corpus_evidence'
                packets[eid]={'entity_id':eid,'source_prefix':'E'+str(ids.index(eid)),
                    'cards':[], 'observations':{},'warnings':['retained canonical own-row task fallback; no scoped pre-cutoff corpus evidence']}
                continue
            packet=prepare_packet(task,entity,corpus,evidence[eid])
            packet=enrich_forecast_context(task,entity,corpus,packet)
            packet=enrich_target_information(task,entity,corpus,packet)
            packet=enrich_reaction_information(task,entity,corpus,packet)
            try:
                packet=enrich_contract_flow_information(task,entity,corpus,packet)
            except (ValueError,KeyError,TypeError,OverflowError) as exc:
                # Optional observations cannot erase an already valid source
                # packet. The helper commits source additions atomically.
                packet.setdefault('optional_measurement_diagnostics',{})['contract_flow_error']=type(exc).__name__
            if os.environ.get('EVIDENCE_FINANCIAL_MEASUREMENT','1')=='1':
                financial=build_financial_measurement(task,entity,corpus)
                if financial is not None:
                    compact=public_financial_measurement(financial)
                    packet['observations']['financial_measurement']=attach_measurement_citations(task,entity,corpus,packet,financial,compact)
            measurement=build_measurement_state(task,entity,corpus)
            if measurement is not None:packet['observations']['measurement_state']=measurement
            graph=build_component_graph(task,entity,corpus) if os.environ.get('EVIDENCE_COMPONENT_GRAPH','1')=='1' else None
            if graph is not None:packet['observations']['component_graph']=graph
            packet['observations'].pop('unselected_eps_scenarios',None)
            packets[eid]=packet
            numeric=build_numerical_prior(task,entity,corpus,context=packet['observations'].get('numeric_history'))
            if numeric is not None:
                packet['observations']['alternative_numerical_control']={k:v for k,v in numeric.items() if k not in ('claims','_fallback')}
                # CPI wider bands and macro revised heuristics did not establish
                # score gains. Expose them as controls, retain v5 fallbacks.
                if numeric.get('_prior_method') in ('auction_same_kind_mean','auction_recent_six_mean','cot_existing_shrunk_weekly_trend'):
                    numeric['claims']=priors[eid]['claims']
                    source_docs={s['doc_id'] for s in numeric.get('_prior_sources',[]) if 'doc_id' in s}
                    source_cards=[c['id'] for c in packet['cards'] if c['doc_id'] in source_docs and c['role'] not in ('period_header','accounting_context') and c.get('citable')]
                    resolved=claims_from_ids(packet,source_cards[:2],corpus,task,entity)
                    if resolved:numeric['claims']=resolved
                    priors[eid]=numeric;results[eid]=numeric
        except Exception as exc:
            enrichment_errors[eid]=type(exc).__name__
            packets[eid]={'entity_id':eid,'source_prefix':'E'+str(ids.index(eid)),'cards':[], 'observations':{},'warnings':['source_enrichment_failed; retained earlier fallback']}
    for entity in entities:
        eid=entity['entity_id']
        if eid in enrichment_errors:continue
        try:
            local=practice_predict(task,entity,corpus,packets[eid]) if os.environ.get('EVIDENCE_PRACTICE_HEADS','1')=='1'else None
            if local is None:local=local_cpi_predict(task,entity,corpus,packets[eid])
            if local is not None:
                results[eid]=local;views[eid]['local_predictor']=local;local_models[eid]=local['_local_model']
        except (ValueError,KeyError,TypeError,OverflowError):
            rejections[eid].append({'stage':'local_predictor','reason':'local_model_input_ineligible'})
    checkpoint(answer())
    client=HouseClient(transport=transport,max_calls=max_calls,deadline_seconds=max(0,deadline_seconds-(time.monotonic()-began)),max_retries=1 if protocol=='information_control'else 0)
    batch_size=max(2,math.ceil(len(entities)/5))
    active=[e for e in entities if e['entity_id'] not in enrichment_errors and e['entity_id'] not in local_models]
    batches=[active[i:i+batch_size] for i in range(0,len(active),batch_size)]

    def execute(batch,view,drafts=None,issues=None):
        prompt=build_forecast_prompt(task,batch,packets,corpus,priors,view=view,all_drafts=drafts,review_flags=issues)
        # Character limit is a conservative engineering bound, not a serving-token guarantee.
        if len(prompt)>145000:return batch,{},'prompt_character_bound'
        response=client.complete(SYSTEM,prompt,thinking=False,stage=view)
        return batch,_rows(response),None

    if active and protocol=='single':
        stage='single_forecast';checkpoint(answer())
        execute_single(task,active,packets,corpus,priors,results,views,rejections,client,
            lambda:checkpoint(answer()),inference_plan)
    elif client.enabled and not client.disabled and active and protocol=='collaborative':
        stage='collaborative';checkpoint(answer())
        execute_collaboration(task,active,packets,corpus,priors,results,views,rejections,client,
            lambda:checkpoint(answer()),inference_plan,
            write_reasons=os.environ.get('EVIDENCE_HARNESS_REASONS','1')=='1')
    elif client.enabled and not client.disabled and batches and protocol!='information_control':
        stage='deliberative' if protocol=='deliberative' else 'direct_control';checkpoint(answer())
        inference_plan=execute_deliberation(task,active,packets,corpus,priors,results,views,rejections,client,lambda:checkpoint(answer()),thinking=protocol=='deliberative',require_model_fields=required)
    elif client.enabled and not client.disabled and batches:
        stage='two_forecast_views';checkpoint(answer())
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures={pool.submit(execute,batch,view):(batch,view) for view in ('fundamental','challenge') for batch in batches}
            for future in concurrent.futures.as_completed(futures):
                batch,view=futures[future]
                try:_,raw_rows,error=future.result()
                except Exception as exc:raw_rows={};error=type(exc).__name__
                for entity in batch:
                    eid=entity['entity_id']
                    row,reason=normalize_forecast(raw_rows.get(eid),task,entity,packets[eid],corpus,priors[eid],view=view)
                    if row is not None:views[eid][view]=row
                    else:rejections[eid].append({'stage':view,'reason':error or reason})
                    # Fixed view order, independent of response completion order.
                    results[eid]=combine_forecasts([views[eid].get(v) for v in ('fundamental','challenge')],priors[eid],task)
                checkpoint(answer())
        flags=review_reasons(task,results,packets,{eid:len(views[eid]) for eid in ids})
        flags={eid:issues for eid,issues in flags.items() if eid not in enrichment_errors}
        if review and flags and not client.disabled and time.monotonic()+25<client.deadline and client.attempts<client.max_calls:
            stage='targeted_review';checkpoint(answer())
            if task['target']['type']=='ranking':review_batches=[active]
            else:review_batches=[[e for e in batch if e['entity_id'] in flags] for batch in batches]
            review_batches=[b for b in review_batches if b]
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,len(review_batches))) as pool:
                futures={pool.submit(execute,batch,'review',[dict(results[e['entity_id']],_individual_views=[views[e['entity_id']][v] for v in ('fundamental','challenge') if v in views[e['entity_id']]]) for e in batch],{e['entity_id']:flags.get(e['entity_id'],[]) for e in batch}):batch for batch in review_batches}
                for future in concurrent.futures.as_completed(futures):
                    batch=futures[future]
                    try:_,raw_rows,error=future.result()
                    except Exception as exc:raw_rows={};error=type(exc).__name__
                    for entity in batch:
                        eid=entity['entity_id']
                        row,reason=review_decision(raw_rows.get(eid),task,entity,packets[eid],corpus,results[eid],flags.get(eid))
                        reviews[eid]={'status':'changed' if row is not None else (error or reason)}
                        if row is not None:results[eid]=row
                    checkpoint(answer())
        elif flags:
            reviews['unit']={'status':'skipped','review_enabled':bool(review),'route_disabled':client.disabled,'remaining_seconds':round(max(0,client.deadline-time.monotonic()),3)}
    precision=compact_house_answers(results,ids)
    stage='complete';final=answer()
    final['notes']['delivery_precision']=precision
    for entity in entities:
        eid=entity['entity_id'];row=results[eid]
        if '_forecast_decision'not in row and not row.get('_fallback',True)and row.get('_reasoning'):
            row['_forecast_decision']={'kind':'house_same_response','mechanism':row['_reasoning'],
                'assumptions':[row['_uncertainty']] if row.get('_uncertainty') else [],
                'source_ids':row.get('_source_ids',[]),
                'native_output':{k:row[k]for k in('point_forecast','interval','label')if k in row}}
            if row.get('_premise_fact'):
                row['_forecast_decision']['premise_fact']=row['_premise_fact']
            task_inputs=reason_task_inputs(task,entity,row.get('_quantity_contract',{}),
                conversion_applied=row.get('_response_coordinate_kind')!='native',
                used_fields=row.get('_used_task_fields'))
            if task_inputs:row['_forecast_decision']['task_inputs']=task_inputs
    ledger=build_ledger(task,results,packets,corpus)
    reasons,diagnostics=compile_reasons(task,final,ledger,corpus)
    final.pop('submitted_reasons',None)
    if reasons:final['submitted_reasons']=reasons
    final['notes']['decision_reason_validation']=diagnostics
    _write(Path(out_path).with_name('decision-ledger.json'),ledger)
    if required:
        findings=delivery_findings(task,results,views,client,protocol)
        final['notes']['delivery_gate']={'passed':not findings,'findings':findings,
             'certificate_scope':'Final row adoption from HTTP route only; model identity/proxy enforcement external, no prediction/NLI/calibration certificate.'}
        checkpoint(final)
        if findings:raise ModelDeliveryUnverified('No final answer: verified HTTP forecast required for every roster entity; complete diagnostic checkpoint retained.')
    checkpoint(final)
    if single_delivery_required and protocol=='single':
        delivery_entities=[e for e in entities if e['entity_id']not in local_models]
        missing_count=sum('single'not in views[e['entity_id']]for e in delivery_entities)
        final['notes']['single_delivery_gate']={'required_active_entities':len(delivery_entities),'missing_validated_entities':missing_count,
            'scope':'All single nonlocal rows, including preprocessing failures; local CPI/protected Credit excluded. HTTP identity and quality unverified.'}
        checkpoint(final)
        if missing_count:
            print(json.dumps(safe_delivery_summary(final),allow_nan=False,separators=(',',':')),flush=True)
            raise ModelDeliveryUnverified('No final answer: required single model rows missing; diagnostic checkpoint preserved')
    _write(out_path,final)
    if os.environ.get('EVIDENCE_DIAGNOSTIC_STDOUT','1')=='1':
        print(json.dumps(safe_delivery_summary(final),allow_nan=False,separators=(',',':')),flush=True)
    return final


def main():
    parser=argparse.ArgumentParser();parser.add_argument('verb',nargs='?',default='analyze',choices=['analyze'])
    parser.add_argument('--task',required=True);parser.add_argument('--corpus',required=True);parser.add_argument('--out',required=True)
    args=parser.parse_args()
    import signal
    deadline=max(1,min(480,float(os.environ.get('EVIDENCE_DEADLINE_SECONDS','480'))))
    def finish_on_deadline(signum,frame):os._exit(0 if Path(args.out).is_file() and Path(args.out).stat().st_size else 1)
    signal.signal(signal.SIGALRM,finish_on_deadline);signal.alarm(math.ceil(deadline+15))
    try:run(args.task,args.corpus,args.out,deadline_seconds=deadline)
    finally:signal.alarm(0)


if __name__=='__main__':main()
