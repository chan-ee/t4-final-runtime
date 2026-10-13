"""Observed revision measurement state, never a forecast or label recommendation.

Only the current task and its authorized, publication-bounded numerical context
are used. No external calendar, static event lookup or model parameter is read.
"""
from __future__ import annotations
from collections import Counter
import datetime as dt
import math
import re
from .numerical_context import build_numeric_context, number


def _date(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):return None
    try:return dt.date.fromisoformat(value)
    except ValueError:return None


def _month(value):
    if not isinstance(value,str) or not re.fullmatch(r'\d{4}-\d{2}',value):return None
    return _date(value+'-01')


def _month_distance(after,before):
    return (after.year-before.year)*12+after.month-before.month


def _reference_at_age(date,age):
    serial=date.year*12+date.month-1-age
    return f'{serial//12:04}-{serial%12+1:02}'


def _cadence(dates):
    groups={}
    for date in dates:groups.setdefault((date.year,date.month),[]).append(date)
    counts=[len(values)for values in groups.values()]
    twice=(sum(n==2 for n in counts)>=3 and sum(n==2 for n in counts)>=len(groups)/2
           and all(any(d.day<=15 for d in values)and any(d.day>15 for d in values)
                   for values in groups.values()if len(values)==2))
    if twice:return {'kind':'two_observed_early_late_phases','ordinary_columns_per_month':2,'heuristic':True}
    if len(groups)>=2 and sum(n==1 for n in counts)>=len(groups)/2:
        return {'kind':'approximately_monthly_with_possible_extra_columns','ordinary_columns_per_month':1,'heuristic':True}
    return {'kind':'not_established','ordinary_columns_per_month':None,'heuristic':True}


def _transition_observations(dates,matrix,cadence,schedule):
    result=[]
    for j,(start,end)in enumerate(zip(dates,dates[1:])):
        old=[]
        for ref,values in matrix.items():
            if _month_distance(start,_month(ref))<4:continue
            a,b=values[j],values[j+1]
            if a is not None and b is not None:old.append(b-a)
        changed=[v for v in old if abs(v)>1e-10]
        positive=sum(v>0 for v in changed);negative=sum(v<0 for v in changed)
        broad=bool(len(changed)>=3 and (positive==len(changed)or negative==len(changed)))
        extra=(cadence['ordinary_columns_per_month']==1 and start.year==end.year and start.month==end.month)
        flags=[]
        if broad:flags.append('multiple_older_reference_months_shift_together')
        if extra:flags.append('additional_same_month_vintage_column')
        result.append({'from_vintage':str(start),'to_vintage':str(end),
            'older_reference_definition':'reference month at least four calendar months before starting vintage',
            'older_reference_rows_comparable':len(old),'older_reference_rows_changed':len(changed),
            'positive_old_row_changes':positive,'negative_old_row_changes':negative,
            'candidate_flags':flags,'event_kind_confirmed':False,
            'interpretation':'Observed pattern only. Ordinary revisions, source corrections or broader updates may produce it; this does not establish an annual event.'})
    return result


def build_measurement_state(task,entity,corpus):
    """Return a publication-stage observation ledger for a revision task, else None.

    Structural ambiguity returns None. A valid source with a conflicting task
    baseline returns an explicit untrusted state and no comparable-transition
    calculations. Caller must not convert this ledger into source claim text.
    """
    if not isinstance(task,dict)or not isinstance(entity,dict):return None
    target=task.get('target',{})
    if not isinstance(target,dict)or'revision'not in str(target.get('name','')).lower():return None
    cutoff=_date(task.get('cutoff_date'));ref=_month(entity.get('ref_month'))
    baseline=_date(entity.get('latest_precutoff_vintage'));resolving=_date(entity.get('resolving_release_date'))
    if None in(cutoff,ref,baseline,resolving):return None
    if baseline>cutoff or resolving<=cutoff or baseline<ref:return None
    try:context=build_numeric_context(task,entity,corpus)
    except (ValueError,TypeError,KeyError,IndexError):return None
    if not context or context.get('family')!='vintage_revision':return None
    source=context.get('source',{});doc=corpus.docs.get(source.get('doc_id'))
    if doc is None or doc.shared or entity.get('entity_id')not in doc.entity_ids:return None
    published=_date(doc.date)
    if published is None or published>cutoff:return None
    source_series=doc.metadata.get('series_id')
    if source_series and entity.get('series_id') and source_series!=entity['series_id']:return None
    bound=min(cutoff,published)
    dates=[_date(value)for value in context.get('vintage_dates',[])]
    if not dates or any(date is None or date>bound for date in dates)or dates!=sorted(set(dates)):return None
    matrix=context.get('history_by_reference_month',{})
    if not isinstance(matrix,dict)or entity['ref_month']not in matrix:return None
    for month,values in matrix.items():
        measured=_month(month)
        if measured is None or measured>bound or not isinstance(values,list)or len(values)!=len(dates):return None
        if any(value is not None and(not isinstance(value,(int,float))or number(value)is None or isinstance(value,bool))for value in values):return None
        if any(value is not None and dates[index]<measured for index,value in enumerate(values)):return None
    supplied=number(entity.get('latest_precutoff_estimate'));values=matrix[entity['ref_month']]
    observed=values[dates.index(baseline)]if baseline in dates else None
    issues=[]
    if baseline not in dates:issues.append('task_baseline_vintage_absent_from_visible_columns')
    elif baseline!=dates[-1]:issues.append('task_latest_vintage_is_not_latest_visible_source_column')
    if supplied is None:issues.append('task_baseline_value_is_not_finite_numeric')
    if observed is None:issues.append('source_value_missing_at_task_baseline')
    elif supplied is not None and not math.isclose(supplied,observed,rel_tol=1e-10,abs_tol=1e-8):issues.append('task_and_source_baseline_values_disagree')
    age=_month_distance(baseline,ref);horizon=_month_distance(resolving,baseline)
    cadence=_cadence(dates);schedule=context.get('source_revision_schedule','')
    explicit_annual=bool(re.search(r'annual\s+(?:updates?|revisions?)|benchmark\s+(?:updates?|revisions?)',schedule,re.I))
    units=re.search(r'\bUnits:\s*([^\n]+)',doc.text,re.I)
    visible=[str(date)for date,value in zip(dates,values)if value is not None and date<=baseline]
    first_age=_month_distance(_date(visible[0]),ref)if visible else None
    events=None
    if cadence['ordinary_columns_per_month']==2:
        events=2*horizon+int(resolving.day>15)-int(baseline.day>15)
    elif cadence['ordinary_columns_per_month']==1 and horizon>0:events=horizon
    if events is not None and events<=0:events=None
    transitions=_transition_observations(dates,matrix,cadence,schedule)
    out={'family':'revision_measurement_state','entity_id':entity['entity_id'],
        'status':'baseline_conflict'if issues else'observed_state_available',
        'trusted_for_analogue_matching':not issues,'baseline_issues':issues,
        'source':dict(source),'observation_information_bound':str(bound),
        'source_units_literal':units.group(1).strip()if units else None,
        'task_units_literal':entity.get('units',entity.get('unit')),
        'unit_equivalence_automatically_verified':False,
        'target_reference_month':entity['ref_month'],'task_baseline_vintage':str(baseline),
        'task_baseline_value':supplied,'source_value_at_task_baseline':observed,
        'target_resolving_release_date':str(resolving),
        'reference_month_age_at_baseline':age,'reference_month_age_at_resolving_date':_month_distance(resolving,ref),
        'elapsed_days_baseline_to_resolution':(resolving-baseline).days,
        'calendar_month_horizon':horizon,
        'first_visible_estimate_date':visible[0]if visible else None,
        'visible_baseline_estimate_ordinal':len(visible),
        'first_release_identity':'not_established_from_truncated_table',
        'first_visible_reference_age_months':first_age,
        'cadence_from_observed_columns':cadence,'additional_release_events_heuristic':events,
        'future_release_schedule_verified':False,
        'source_schedule_literal':schedule,'source_mentions_annual_or_benchmark_update':explicit_annual,
        'observed_transition_patterns':transitions,
        'external_calendar_used':False,'forecast_returned':False,'label_recommendation_returned':False,
        'citable_as_source_claim':False,
        'limitations':['Same calendar horizon does not imply the same reference-month publication stage.',
            'Visible columns may omit earlier releases and can include unusual updates; visible ordinal is not a proved first/second/third agency release.',
            'Candidate common-shift or extra-column flags do not establish event type and do not automatically exclude observations.',
            'This is computed state and source provenance, not a source quote or a calibrated future distribution.']}
    if issues:return out
    if len(dates)>120 or len(matrix)>240:
        out['analogue_status']='not_computed_source_size_limit';return out
    matches=[];two_phase=cadence['ordinary_columns_per_month']==2
    for i,start in enumerate(dates):
        if two_phase and(start.day>15)!=(baseline.day>15):continue
        month=_reference_at_age(start,age);past=matrix.get(month)
        if past is None or past[i]is None:continue
        for j,end in enumerate(dates):
            if j<=i or _month_distance(end,start)!=horizon:continue
            if two_phase and(end.day>15)!=(resolving.day>15):continue
            if past[j]is None:continue
            flagged=[index for index in range(i,j)if transitions[index]['candidate_flags']]
            matches.append({'reference_month':month,'from_vintage':str(start),'to_vintage':str(end),
                'reference_age_at_start_months':age,'start_value':past[i],'end_value':past[j],
                'observed_revision_delta':past[j]-past[i],
                'delta_units':'same native units as the source table; no annualization or scale conversion applied',
                'crossed_candidate_transition_indices':flagged})
    out.update(analogue_status='comparable_observations_available'if matches else'no_comparable_observations',
        same_stage_observation_count=len(matches),same_stage_distinct_start_vintages=len({m['from_vintage']for m in matches}),
        same_stage_without_candidate_transition_count=sum(not m['crossed_candidate_transition_indices']for m in matches),
        same_stage_observations=matches[:12],same_stage_observations_omitted=max(0,len(matches)-12),
        analogue_definition='Same calendar reference-month age and calendar horizon; match early/late phases only when the observed source establishes a recurring two-phase pattern.',
        analogue_count_is_independent_sample_size=False,
        sample_warning=f'{len(matches)} matched historical transitions from {len({m["from_vintage"]for m in matches})} starting vintages; no minimum count establishes direction accuracy or interval calibration.')
    return out
