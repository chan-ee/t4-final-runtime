"""A single verified decision, with bounded missing-final recovery.

Thinking is a computation hypothesis, not a correctness certificate. No
unverified transcript/provisional reasoning is ever adopted as a prediction.
"""
import concurrent.futures
import math
import time
from .decisions import normalize_forecast
from .forecast_prompts import SYSTEM,build_forecast_prompt
from .legacy import _rows
from .client import Unavailable,BudgetExhausted


def plan_batches(entities,*,request_cap=25):
    # Reserve one possible recovery for every primary batch. Unlike entity-only
    # plans, this still fits larger rosters without silently omitting entities.
    if not entities or request_cap<=0:return [],{'primary_batches':0,'recovery_reserve':0,'maximum_attempts_without_retry':0,'request_cap':request_cap}
    primary_cap=max(1,min(12,max(1,request_cap//2)))
    size=max(1,math.ceil(len(entities)/primary_cap))
    batches=[entities[i:i+size]for i in range(0,len(entities),size)]
    recovery_reserve=min(len(batches),max(0,request_cap-len(batches)))
    return batches,{'primary_batches':len(batches),'recovery_reserve':recovery_reserve,
        'maximum_attempts_without_retry':len(batches)+recovery_reserve,'request_cap':request_cap,
        'retry_reserve':0,'unverified_reasoning_budget_parameter':False,
        'entities_per_primary_batch':size,'stage_workers':4,
        'limitation':'Large batches may exceed final-output/context capacity; physical cap and complete fallback still apply.'}


def recovery_feedback(task,batch,packets,rejections,*,require_model_fields):
    """Local validation metadata, never rejected model prose or predictions."""
    safe_codes={'missing_or_duplicate_row','wrong_final_shape','invalid_final_json',
        'unfinished_reasoning','reasoning_only','no_final_content','invalid_numeric_fields',
        'wrong_interval_level','invalid_probability_target','invalid_label','unresolved_sources',
        'invalid_source_id_list','metadata_only_sources','invalid_arithmetic','missing_house_point',
        'conflicting_final_envelopes','incomplete_final_envelope','Unavailable','unavailable',
        'remaining_time','primary_time_reserve','prompt_character_bound','BudgetExhausted',
        'TimeoutError','ValueError','TypeError','OSError','HTTP_401','HTTP_403','HTTP_404',
        'HTTP_429','HTTP_500','HTTP_502','HTTP_503','HTTP_504'}
    feedback={}
    for entity in batch:
        eid=entity['entity_id'];history=rejections.get(eid,[])
        code=history[-1].get('reason') if history else 'unavailable'
        code=code if code in safe_codes else 'validation_failed'
        fields=['interval.lo','interval.hi','support_ids']
        if task['target']['type']!='classification' or require_model_fields:fields.append('point_forecast')
        if task['target']['type']=='classification':fields.append('label')
        ids=[c['id'] for c in packets[eid]['cards'] if c.get('citable') and c['role'] not in ('period_header','accounting_context')]
        feedback[eid]={'last_rejection_code':code,'required_response_fields':fields,
            'substantive_source_ids':list(dict.fromkeys(ids))[:32]}
    return feedback


def decision_prompt(task,batch,packets,corpus,priors,*,recovery=False,feedback=None):
    base=build_forecast_prompt(task,batch,packets,corpus,priors,view='fundamental')
    instruction='''Work through the target dependencies before choosing a forecast. For EPS, start from the requested future quarter and the supplied operating outlook, revenues, provision versus charge-offs, taxes, common-income allocation and diluted shares. Reconcile the published baseline where possible; separate management guidance, unobserved future GAAP items and explicit assumptions. A sensitivity envelope is not a90% prediction interval. For revisions, use the supplied publication-state ledger: source reference age, distinct release stages and ordinary versus candidate common-update histories cannot be pooled without justification. A statistic's poor same-stage support is genuine uncertainty. General financial knowledge may interpret observations but cannot introduce a realized post-cutoff answer. Select substantive source cards containing the actual driver, amount or condition. A filing title, issuer name, date or bare column header alone is not a predictive premise. Return the requested final JSON with all supplied entities, exact native quantities and source IDs.'''
    if recovery:
        instruction+=''' A prior computation did not produce a verified complete forecast for these entities. This is a fresh final-answer completion from the same original evidence, not endorsement of a missing answer. No provisional analysis or incomplete thought is supplied as evidence. Prior fallbacks remain controls only. Give concise final JSON.'''
        if feedback:
            import json
            instruction+='\nLOCAL_VALIDATION_FEEDBACK\n'+json.dumps(feedback,allow_nan=False,separators=(',',':'))+'\nThese are validator codes and response-field requirements, not observed economic facts. Source IDs are valid syntax options, not evidence that any arbitrary source supports your forecast. Use relevant supplied facts.'
    return instruction+'\n'+base


def execute_deliberation(task,entities,packets,corpus,priors,results,views,rejections,client,checkpoint,*,thinking=True,require_model_fields=False):
    active=[e for e in entities if packets[e['entity_id']]['cards']]
    if not active:return {'primary_batches':0,'recovery_batches':0,'status':'no_active_source_packets'}
    batches,plan=plan_batches(active,request_cap=client.max_calls)
    if task['target']['type']=='ranking' and len(active)<=10:
        batches=[active];plan.update(primary_batches=1,recovery_reserve=min(1,max(0,client.max_calls-1)),maximum_attempts_without_retry=min(2,client.max_calls),entities_per_primary_batch=len(active),joint_ranking=True)
    if not batches:return plan
    start=time.monotonic();available=max(0,client.deadline-start)
    # Per-request sockets alone do not prove total runtime. The client body
    # clock and outer process watchdog are additional guards.
    waves=max(1,math.ceil(len(batches)/4))
    primary_timeout=min(80,max(1,(available*.55-5)/waves))
    primary_deadline=min(client.deadline,start+available*.58)
    plan.update(primary_request_timeout_seconds=round(primary_timeout,3),
        primary_time_share=.58,thinking=bool(thinking),recovery_batches=0)

    def call(batch,phase,timeout):
        if time.monotonic()+5>=client.deadline:return batch,{},'remaining_time'
        if phase=='deliberative' and time.monotonic()+5>=primary_deadline:return batch,{},'primary_time_reserve'
        feedback=recovery_feedback(task,batch,packets,rejections,require_model_fields=require_model_fields) if phase=='final_recovery' else None
        prompt=decision_prompt(task,batch,packets,corpus,priors,recovery=phase=='final_recovery',feedback=feedback)
        if len(prompt)>145000:return batch,{},'prompt_character_bound'
        try:
            response=client.complete(SYSTEM,prompt,thinking=thinking if phase=='deliberative'else False,
                stage=phase,request_timeout_seconds=timeout)
        except (Unavailable,BudgetExhausted) as exc:
            return batch,{},getattr(exc,'reason_code',type(exc).__name__)
        return batch,_rows(response),None

    def apply(batch,raw_rows,phase,error=None):
        for entity in batch:
            eid=entity['entity_id']
            row,reason=normalize_forecast(raw_rows.get(eid),task,entity,packets[eid],corpus,priors[eid],view=phase)
            if row is not None and require_model_fields and row.get('_field_sources',{}).get('point_forecast')=='carried_prior_point':
                row=None;reason='missing_house_point'
            if row is None:rejections[eid].append({'stage':phase,'reason':error or reason})
            else:results[eid]=row;views[eid][phase]=row
        checkpoint()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,len(batches)))as pool:
        futures={pool.submit(call,b,'deliberative',primary_timeout):b for b in batches}
        for future in concurrent.futures.as_completed(futures):
            batch=futures[future]
            try:_,raw,error=future.result()
            except Exception as exc:raw={};error=type(exc).__name__
            apply(batch,raw,'deliberative',error)

    failed=[[e for e in batch if not views[e['entity_id']]]for batch in batches]
    failed=[batch for batch in failed if batch]
    plan['recovery_batches']=len(failed)
    if failed and not client.disabled and client.attempts<client.max_calls and time.monotonic()+10<client.deadline:
        remaining=max(0,client.deadline-time.monotonic()-8)
        waves=max(1,math.ceil(len(failed)/4));timeout=min(50,max(1,remaining/waves))
        plan['recovery_request_timeout_seconds']=round(timeout,3)
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(4,len(failed)))as pool:
            futures={pool.submit(call,b,'final_recovery',timeout):b for b in failed}
            for future in concurrent.futures.as_completed(futures):
                batch=futures[future]
                try:_,raw,error=future.result()
                except Exception as exc:raw={};error=type(exc).__name__
                apply(batch,raw,'final_recovery',error)
    elif failed:plan['recovery_skipped']='time_or_call_or_route_limit'
    plan['elapsed_seconds']=round(time.monotonic()-start,3)
    return plan
