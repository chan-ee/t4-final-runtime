"""Pure numeric EPS-state features and tiny linear predictions; no file/network I/O.

Input observations must originate from the caller's authorized frozen-corpus
parser. Issuer/task IDs, hashes, absolute dates and target labels are never
model features. Dates select comparable source periods only.
"""
import datetime as dt
import math
import re


def finite(value):return type(value)in(int,float)and math.isfinite(value)


def extract_eps_state(task,entity,historical_records):
    result={'status':'missing_eligible_history','features':None,'fallback_eps':None,'missing':[],
            'source_basis_independently_verified':False,'future_target_observed':False}
    prior=entity.get('prior_year_q_eps')
    if not finite(prior):return result|{'missing':['task_prior_year_q_eps']}
    result['fallback_eps']=prior
    result['task_prior_target_eps']=prior
    try:
        cutoff=dt.date.fromisoformat(task['cutoff_date'])
        match=re.search(r'20\d{2}-\d{2}-\d{2}',str(entity.get('quarter_reported','')))
        target_end=dt.date.fromisoformat(match[0])if match else None
    except (KeyError,TypeError,ValueError):return result|{'missing':['valid_period_contract']}
    if target_end is None:return result|{'missing':['target_quarter_end']}
    quarter=[]
    for record in historical_records:
        if record.get('entity_id')!=entity.get('entity_id'):continue
        if record.get('period_months')!=3 or record.get('unit')!='USD/share' or not finite(record.get('value')):continue
        try:end=dt.date.fromisoformat(record['period_end']);published=dt.date.fromisoformat(record['doc_date'])
        except (KeyError,TypeError,ValueError):continue
        if published>cutoff or end>published or end>=target_end:continue
        quarter.append((end,record))
    if not quarter:return result|{'missing':['latest_three_month_EPS']}
    latest_date=max(date for date,_ in quarter);latest_rows=[r for date,r in quarter if date==latest_date]
    if len({r['value']for r in latest_rows})!=1:return result|{'missing':['unambiguous_latest_EPS']}
    latest=latest_rows[0];current=latest['value'];result['fallback_eps']=current
    matches=[r for date,r in quarter if 351<=(latest_date-date).days<=379]
    if len({(r['period_end'],r['value'])for r in matches})!=1:
        return result|{'status':'latest_only','latest_observation':latest,'missing':['unambiguous_comparable_prior_year_quarter']}
    old=matches[0];past=old['value']
    return result|{'status':'usable_eps_state','missing':[],'latest_observation':latest,'prior_same_period_observation':old,
        'task_prior_target_eps':prior,'latest_eps':current,'prior_same_period_eps':past,
        'observed_yoy_delta_eps':current-past,'past_seasonal_difference_eps':prior-past,
        'historical_ratio_eligible':past>0,'feature_names':['task_prior_target_eps','latest_eps','past_seasonal_difference_eps'],
        'features':[prior,current,prior-past],
        'limitations':['Quarter/fiscal duration and source GAAP/total/consolidation meaning still require source verification.',
            'Historical negative EPS remains negative; no primitive clipping is performed.',
            'A same-period historical trend is a forecasting hypothesis, not the future realized EPS.']}


def predict_eps(state,artifact):
    """No entity ID/date/hash branches; None when the stated form lacks inputs."""
    form=artifact['form']
    if form=='prior':return state.get('task_prior_target_eps',state.get('fallback_eps'))
    if form=='latest':return state.get('latest_eps',state.get('fallback_eps'))
    if state.get('features')is None:return state.get('fallback_eps')
    prior,latest,seasonal=state['features']
    if form=='seasonal_additive':return latest+seasonal
    if form=='seasonal_ratio':
        return prior*latest/state['prior_same_period_eps']if state['historical_ratio_eligible']else latest
    if form=='bias':return latest+artifact['coefficient']
    if form=='scale':return latest*artifact['coefficient']
    if form=='linear3':
        coefficients=artifact['coefficients']
        if len(coefficients)!=3 or not all(finite(x)for x in coefficients):return None
        output=sum(a*b for a,b in zip(coefficients,state['features']))
        return output if finite(output)else None
    raise ValueError('unsupported model form')
