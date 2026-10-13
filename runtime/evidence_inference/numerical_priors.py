"""Small native-unit baselines from the current authorized observation tables.

No fitted external artifact, task-ID lookup, House output override or citation text.
The caller may use these complete row-shaped candidates as fallbacks/draft priors.
"""
from __future__ import annotations
import datetime as dt
import math
import re
import statistics as st
from .numerical_context import build_numeric_context,number,quantile,selected_method_eligible


def _normal_multiplier(level):
    return st.NormalDist().inv_cdf((1+level)/2)


def _auction_kind(value):
    value=str(value).strip().lower().replace('_',' ').replace('-',' ')
    if value in('new','new issue'):return 'new'
    if value in('reopen','reopening','re open','re opened'):return 'reopen'
    return None


def _row(task,entity,context,point,lo,hi,method,reasoning,*,label=None,limitations=()):
    values=(point,lo,hi)
    if not all(isinstance(v,(int,float))and not isinstance(v,bool)and math.isfinite(v)for v in values)or lo>=hi:return None
    result={'entity_id':entity['entity_id'],'point_forecast':float(point),
            'interval':{'level':float(task.get('interval_level',.9)),'lo':float(lo),'hi':float(hi)},
            '_fallback':True,'_reasoning':reasoning,'_prior_method':method,
            '_prior_sources':[dict(context['source'])],
            '_prior_limitations':list(limitations)+['Source ranges identify observations, not ready-made claims. Use the normal cutoff-safe citation path.',
                'Historical variation or empirical tails are not a verified conditional coverage guarantee.']}
    if task.get('target',{}).get('type')=='classification':
        if label not in task.get('target',{}).get('labels',[]):return None
        result['label']=label
    return result


def _auction(task,entity,context,level):
    target=task.get('target',{});name=target.get('name','').lower()
    if target.get('type')!='regression' or 'bid_to_cover'not in name:return None
    if any(word in name for word in('log','growth','change')):return None
    units=' '.join(str(value)for value in(target.get('unit',''),target.get('units',''),entity.get('unit',''),entity.get('units',''))).lower()
    if re.search(r'percent|\bpct\b',name.replace('_',' ')+' '+units.replace('_',' ')):return None
    history=context.get('observed_history',[]);values=[r['bid_to_cover']for r in history]
    if len(values)<3:return None
    kind=_auction_kind(entity.get('new_or_reopening'))
    same=[r['bid_to_cover']for r in history if _auction_kind(r.get('kind'))==kind]if kind else[]
    if len(same)>=2 and selected_method_eligible(task,'auction_same_kind_preference'):
        point=st.mean(same);method='auction_same_kind_mean';n=len(same)
        reason=f'recent historical same-{kind} auction mean from {n} source observations; native bid-to-cover ratio'
    else:
        point=st.mean(values[-6:]);method='auction_recent_six_mean';n=min(6,len(values))
        reason=f'recent historical auction mean from {n} source observations; native bid-to-cover ratio'
    half=max(_normal_multiplier(level)*st.stdev(values),abs(point)*.05,.01)
    return _row(task,entity,context,point,point-half,point+half,method,reason,
                limitations=['New/reopening kind is used only when supplied by the task; unannounced offering sizes remain unknown.',
                             'This simple historical baseline does not anticipate future auction demand or event shocks.'])


def _cpi(task,entity,context,level):
    target=task.get('target',{});name=target.get('name','').lower();prompt=task.get('prompt','').lower()
    if target.get('type')!='regression' or 'cpi'not in name or 'mom'not in name:return None
    if not context.get('source_monthly_percent_explicit'):return None
    if context.get('observed_adjustment')!=context.get('target_adjustment_from_literal_text')or context.get('observed_adjustment')=='not_established':return None
    if re.search(r'(?:^|_)(?:log|ln)(?:_|$)',name)or re.search(r'annuali[sz]ed|year.over.year|\byoy\b|\blogarithm|\blog\s*\(|\bln\s*\(',name.replace('_',' ')+' '+prompt):return None
    target_units=str(target.get('unit',target.get('units',''))).lower().replace('_',' ')
    literal_monthly_percent=bool(re.search(r'(?:month.over.month|\bmom\b).{0,100}(?:percent|\bpct\b)|(?:percent|\bpct\b).{0,100}(?:month.over.month|\bmom\b)',prompt))
    if not(literal_monthly_percent or 'mom'in target_units and('pct'in target_units or'percent'in target_units)):return None
    history=context.get('observed_history',[])
    if len(history)<3:return None
    values=[v for _,v in history];point=values[-1]
    supplied=number(entity.get('latest_published_mom_pct'))
    if supplied is not None and not math.isclose(supplied,point,rel_tol=1e-9,abs_tol=1e-9):return None
    half=max(_normal_multiplier(level)*st.stdev(values),.05)
    return _row(task,entity,context,point,point-half,point+half,'cpi_last_print_history_variation',
                f'carry-forward of the latest matching CPI component observation ({history[-1][0]}); uncertainty uses {len(values)} same-column historical monthly observations',
                limitations=['Point is a transparent persistence baseline, not an optimized future component forecast.',
                             'Regular gasoline pump-price NSA change is not automatically all-types CPI gasoline SA change.',
                             'Source and requested seasonal-adjustment conventions must match.'])


def revision_release_horizon(vintage_dates,last_vintage,resolving_date):
    """Calendar-month/observed-phase heuristic; never a guarantee of future publication dates."""
    dates=[dt.date.fromisoformat(x)for x in vintage_dates]
    last=dt.date.fromisoformat(last_vintage);target=dt.date.fromisoformat(resolving_date)
    if target<=last:return None
    counts={}
    for date in dates:counts[(date.year,date.month)]=counts.get((date.year,date.month),0)+1
    months=(target.year-last.year)*12+target.month-last.month
    twice=len(counts)>=2 and st.median(counts.values())>=2 and all(any(d.day<=15 for d in dates if(d.year,d.month)==m)and any(d.day>15 for d in dates if(d.year,d.month)==m)for m,c in counts.items()if c>=2)
    if twice:
        steps=max(1,2*months+int(target.day>15)-int(last.day>15));basis='calendar months with two observed early/late release phases'
    else:steps=max(1,months);basis='calendar months with one ordinary monthly release'
    return steps,basis


def revision_distribution(context,current,last,resolve,*,level=.9,annual_target=False):
    """Pure table arithmetic, separate from deployment eligibility checks.

    Historical diagnostics can apply this function to an explicit prefix. The
    runtime separately checks source dates and the literal target contract.
    """
    dates=context.get('vintage_dates',[]);matrix=context.get('history_by_reference_month',{})
    if len(dates)<2 or last!=dates[-1]:return None
    try:inferred=revision_release_horizon(dates,last,resolve)
    except (ValueError,TypeError):return None
    if inferred is None:return None
    steps,basis=inferred
    if steps>=len(dates):return None
    dated=[dt.date.fromisoformat(d)for d in dates];deltas=[];excluded=[]
    annual_cue='annual'in context.get('source_revision_schedule','').lower()
    for j in range(len(dates)-steps):
        old=[]
        for month,values in matrix.items():
            age=(dated[j].year-int(month[:4]))*12+dated[j].month-int(month[5:])
            a,b=values[j],values[j+steps]
            if a is not None and b is not None and age>=4:old.append(b-a)
        nonzero=[x for x in old if abs(x)>1e-10]
        broad=len(old)>=3 and len(nonzero)>=.6*len(old)
        coherent=bool(nonzero)and max(sum(x>0 for x in nonzero),sum(x<0 for x in nonzero))>=.9*len(nonzero)
        if annual_cue and broad and coherent and not annual_target:
            excluded.append((dates[j],dates[j+steps]));continue
        available=sorted(m for m,values in matrix.items()if values[j]is not None)
        for month in available[-3:]:
            a,b=matrix[month][j],matrix[month][j+steps]
            if a is not None and b is not None:deltas.append(b-a)
    if len(deltas)<3:return None
    center=st.median(deltas);tail=(1-level)/2
    low=min(quantile(deltas,tail),center,0.);high=max(quantile(deltas,1-tail),center,0.)
    if low==high:
        spread=max(abs(current)*1e-5,1e-6);low-=spread;high+=spread
    up=sum(x>1e-10 for x in deltas);down=sum(x<-1e-10 for x in deltas);label='up'if up>down else'down'
    return {'point':current+center,'lo':current+low,'hi':current+high,'label':label,
            'observed_revisions':len(deltas),'horizon_steps':steps,'horizon_basis':basis,'excluded_transitions':excluded}


def _vintage(task,entity,context,level):
    target=task.get('target',{});name=target.get('name','').lower();prompt=task.get('prompt','').lower()
    if 'revision'not in name:return None
    if not selected_method_eligible(task,'vintage_revision_prior'):return None
    if re.search(r'(?:^|_)(?:log|ln|delta|amount)(?:_|$)',name):return None
    if not re.search(r'(?:point\s+)?forecast of (?:the )?revised (?:value|level|estimate)|revised level',prompt):return None
    if re.search(r'logarithm|log\s*\(|ln\s*\(',name+' '+prompt):return None
    if target.get('type')=='classification'and not {'up','down'}<=set(target.get('labels',[])):return None
    current=number(entity.get('latest_precutoff_estimate'));dates=context.get('vintage_dates',[]);matrix=context.get('history_by_reference_month',{})
    ref=entity.get('ref_month');last=entity.get('latest_precutoff_vintage');resolve=entity.get('resolving_release_date')
    if current is None or len(dates)<2 or ref not in matrix or last!=dates[-1]or not isinstance(resolve,str)or resolve<=task['cutoff_date']:return None
    observed=matrix[ref][-1]
    if observed is None or not math.isclose(current,observed,rel_tol=1e-10,abs_tol=1e-8):return None
    distribution=revision_distribution(context,current,last,resolve,level=level,
        annual_target=any(x in prompt for x in('annual update target','benchmark revision target')))
    if distribution is None:return None
    result=_row(task,entity,context,distribution['point'],distribution['lo'],distribution['hi'],'vintage_level_observed_revision_prior',
        f"current vintage-table revised-level baseline using {distribution['observed_revisions']} past revision deltas over {distribution['horizon_steps']} inferred release steps; current level is not itself the revision amount",label=distribution['label'],
        limitations=[distribution['horizon_basis']+'; future intermediate release dates are not established.',
                     'A source annual-update cue plus broad coherent old-month shifts identifies a candidate regime change, not a proved event date.',
                     'Ordinary mixed-sign/small monthly revisions are not discarded merely because several older months changed.',
                     'The up/down baseline uses past nonzero sign counts; unchanged historical rows are retained for numerical tails.'])
    if result is not None:result['_prior_horizon_steps']=distribution['horizon_steps'];result['_prior_excluded_candidate_regime_transitions']=distribution['excluded_transitions']
    return result


def _cot(task,entity,context,level):
    target=task.get('target',{});name=target.get('name','').lower()
    if target.get('type')!='ranking'or'positioning_change'not in name:return None
    # Recheck here even when a caller supplies an otherwise matching context.
    # The .1 shrink was selected on a dated panel, not an a-priori identity.
    if not selected_method_eligible(task,'cot_shrink_0_1'):return None
    if 'pct'not in name or context.get('observed_report_spacing_days')!=[7]:return None
    if not all(context.get(k)for k in('source_futures_only','target_futures_only','target_base_open_interest_explicit','task_base_open_interest_matches_source','task_base_net_matches_source')):return None
    horizon=context.get('target_report_steps');data=context.get('observed_history',[])
    if not isinstance(horizon,int)or horizon<=0 or len(data)<8:return None
    _,oi,net=data[-1];point=.1*100*horizon*(net-data[-2][2])/oi
    residuals=[]
    for j in range(8,len(data)-horizon):
        _,base_oi,base_net=data[j]
        prediction=.1*100*horizon*(base_net-data[j-1][2])/base_oi
        realized=100*(data[j+horizon][2]-base_net)/base_oi
        residuals.append(abs(realized-prediction))
    half=sorted(residuals)[min(len(residuals)-1,int(level*len(residuals)))]if residuals else 10.
    half=max(half,.5)
    return _row(task,entity,context,point,point-half,point+half,'cot_existing_shrunk_weekly_trend',
         f'pre-cutoff one-week positioning trend, shrunk as in v5 over the literal {horizon}-report target horizon; base-report open interest denominator',
         limitations=['This preserves the old simple trend; the unsuccessful ridge, rank-head and NIW artifacts are not loaded.',
                      'Report-as-of date and public release date differ; later market prices do not reveal unreported positions.'])


def build_numerical_prior(task,entity,corpus,*,context=None):
    """Return a claim-free v5-shaped fallback candidate, or None for an unknown contract."""
    level=number(task.get('interval_level',.9))
    if level is None or not 0<level<1:return None
    context=build_numeric_context(task,entity,corpus)if context is None else context
    if not isinstance(context,dict)or context.get('entity_id')!=entity.get('entity_id'):return None
    if context.get('cutoff_date')!=task.get('cutoff_date')or context.get('target_name')!=task.get('target',{}).get('name'):return None
    source=context.get('source',{});doc=corpus.docs.get(source.get('doc_id'))if isinstance(source,dict)else None
    if doc is None or doc.date>task['cutoff_date']or not(doc.shared or entity['entity_id']in doc.entity_ids):return None
    family=context.get('family')
    if family=='auction':return _auction(task,entity,context,level)
    if family=='cpi_component':return _cpi(task,entity,context,level)
    if family=='vintage_revision':return _vintage(task,entity,context,level)
    if family=='positioning':return _cot(task,entity,context,level)
    return None
