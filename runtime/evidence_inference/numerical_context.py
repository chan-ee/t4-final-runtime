"""Typed past-data context; standard library, no requests or fitted weights.

This module returns observed values, deterministic arithmetic and explicitly named
candidate priors. It does not declare any future answer, interval calibration or score.
Every source coordinate addresses the original frozen document text.
A disclosed static calendar coefficient supplies one explicitly conditional candidate.
"""
from __future__ import annotations
import datetime as dt
import json
import math
from pathlib import Path
import re
import statistics as st


# These are information-availability bounds for methods chosen with task-specific
# historical outcomes, not task-shape/date routing. Observed table summaries do
# not need the gates; a selected forecast method does. COT/auction source
# editions used for selection were frozen on October 31; macro selection data
# ended at the September 30 frozen edition. The COT final report was as of
# October 22 / ordinarily released October 25, but its original release bytes
# and correction history were not independently archived, so retain the more
# conservative verified frozen-edition bound from v5 ARTIFACT_PROVENANCE.md.
METHOD_SELECTION_AVAILABLE_BY={
    'cot_shrink_0_1':'2024-10-31',
    'auction_same_kind_preference':'2024-10-31',
    'vintage_revision_prior':'2024-09-30',
}


def selected_method_eligible(task,method):
    cutoff=task.get('cutoff_date');available=METHOD_SELECTION_AVAILABLE_BY.get(method)
    if not isinstance(cutoff,str)or available is None:return False
    try:return dt.date.fromisoformat(cutoff)>=dt.date.fromisoformat(available)
    except ValueError:return False


def number(value):
    if isinstance(value, bool): return None
    if isinstance(value, (int,float)):
        return float(value) if math.isfinite(value) else None
    try:
        v=float(str(value).replace(',','').strip().rstrip('%'))
        return v if math.isfinite(v) else None
    except (ValueError,TypeError): return None


def quantile(values,p):
    values=sorted(values);x=(len(values)-1)*p;i=int(x);j=min(i+1,len(values)-1)
    return values[i]+(x-i)*(values[j]-values[i])


def _stats(values):
    return {'n':len(values),'latest':values[-1],'mean_all':st.mean(values),
            'median_all':st.median(values),'mean_last3':st.mean(values[-3:]),
            'mean_last6':st.mean(values[-6:]),'min':min(values),'max':max(values),
            'sample_sd':st.stdev(values) if len(values)>1 else None,
            'historical_q05':quantile(values,.05),'historical_q95':quantile(values,.95)}


def _authorized_docs(task,entity,corpus):
    # Mirror manifest ownership used by the runtime; no other entity's document
    # may be silently substituted for this entity's own history.
    for doc in sorted(corpus.docs.values(),key=lambda d:(d.date,d.doc_id),reverse=True):
        if doc.date>task['cutoff_date']:continue
        if doc.shared or entity['entity_id'] in doc.entity_ids:yield doc


def _table(doc,header_starts):
    header=None;rows=[];offset=0;start=None;end=None;header_source=None
    for raw in doc.text.splitlines(keepends=True):
        text=raw.rstrip('\r\n');parts=[x.strip() for x in text.split('|')]
        if len(parts)>1 and parts[0] in header_starts:
            header=parts;start=offset;header_source={'start':offset,'end':offset+len(text)}
        elif header is not None and len(parts)==len(header) and re.fullmatch(r'\d{4}-\d{2}(?:-\d{2})?',parts[0]):
            rows.append((parts,offset,offset+len(text)));end=offset+len(text)
        elif rows and text.strip():
            break
        offset+=len(raw)
    if not header or not rows:return None
    # Source byte order is not evidence of chronological order. Refuse duplicate
    # observation keys rather than silently count or overwrite them.
    keys=[row[0][0] for row in rows]
    if len(keys)!=len(set(keys)):return None
    try:
        for key in keys:dt.date.fromisoformat(key+'-01' if len(key)==7 else key)
    except ValueError:return None
    rows.sort(key=lambda row:row[0][0])
    return {'header':header,'rows':rows,'source':{'doc_id':doc.doc_id,'doc_date':doc.date,
            'span_start':start,'span_end':end,'header_span':header_source}}


def _key(value):return re.sub(r'[^a-z0-9]','',str(value).lower())


def _auction_kind(value):
    value=str(value).strip().lower().replace('_',' ').replace('-',' ')
    if value in('new','new issue'):return 'new'
    if value in('reopen','reopening','re open','re opened'):return 'reopen'
    return None


def _series_label(text,series):
    """Read a source-declared SERIES (column label) link, including nested parentheses."""
    if not isinstance(series,str)or not series:return None
    match=re.search(r'(?<![A-Za-z0-9])'+re.escape(series)+r'\s*\(',text)
    if not match:return None
    start=match.end();depth=1
    for i in range(start,min(len(text),start+300)):
        if text[i]=='(':depth+=1
        elif text[i]==')':
            depth-=1
            if depth==0:return text[start:i]
    return None


def _adjustment(text):
    text=text.lower()
    if re.search(r'not seasonally adjusted|non[- ]seasonally adjusted|\bunadjusted\b|\bnsa\b',text):return 'not_seasonally_adjusted'
    if re.search(r'seasonally[- ]adjusted|\bsa\b',text):return 'seasonally_adjusted'
    return 'not_established'


def _calendar_candidate(task,entity,context,docs):
    """Prompt-only measurement candidate; never changes a submitted forecast or interval."""
    target=task.get('target',{});name=target.get('name','').lower()
    if target.get('type')!='regression' or 'cpi'not in name or'mom'not in name:return None
    if re.search(r'(?:^|_)(?:log|ln)(?:_|$)',name)or re.search(r'annuali[sz]ed|year.over.year|\byoy\b|logarithm|\blog\s*\(|\bln\s*\(',task.get('prompt',''),re.I):return None
    if entity.get('unit')!='mom_pct_change_sa'or not context.get('source_monthly_percent_explicit')or context.get('observed_adjustment')!='seasonally_adjusted'or context.get('target_adjustment_from_literal_text')!='seasonally_adjusted':return None
    if 'first_print'not in name and not re.search(r'first[ -]published',task.get('prompt',''),re.I):return None
    try:config=json.loads((Path(__file__).parent/'assets/cpi-calendar-2024.json').read_text())
    except (OSError,ValueError):return None
    if task['cutoff_date']<config['selection_not_before']or task['cutoff_date']<config['available_by']:return None
    if entity.get('ref_month')!=config['valid_reference_month']or entity.get('series_fred')!=config['series_fred']:return None
    release=entity.get('release_date',task.get('resolution_date'))
    if not isinstance(release,str)or not re.fullmatch(r'\d{4}-\d{2}-\d{2}',release)or release<=task['cutoff_date']:return None
    try:dt.date.fromisoformat(release)
    except ValueError:return None
    pump=context.get('pump_observation',{})
    if pump.get('current_month')!=config['valid_reference_month']or pump.get('previous_month')!=config['previous_reference_month']:return None
    previous,current=number(pump.get('previous_mean')),number(pump.get('current_mean'))
    if previous is None or current is None or min(previous,current)<=0:return None
    if pump.get('unit')!='nominal dollars per gallon, not seasonally adjusted':return None
    doc=next((d for d in docs if d.doc_id==pump.get('source',{}).get('doc_id')),None)
    if doc is None:return None
    metadata=' '.join(str(v)for v in doc.metadata.values()if isinstance(v,str))+' '+doc.text[:450]
    if'GASREGW'not in metadata or not re.search(r'\bregular\s+gasoline\b',metadata,re.I):return None
    ratio=current/previous;point=100*(ratio*config['gross_change_multiplier']-1)
    bounds=[100*(ratio*x-1)for x in config['rounding_only_multiplier_bounds']]
    if not all(math.isfinite(x)for x in[point,*bounds]):return None
    return {'status':'conditional numerical candidate, not an observed corpus fact or verified future value',
            'conditional_point_pct':point,'frozen_input_source_doc_id':doc.doc_id,
            'measurement':'regular-grade weekly pump-price proxy with pre-cutoff calendar transformation to all-types CPI SA change',
            'coefficient_available_by':config['available_by'],'method_selection_not_before':config['selection_not_before'],
            'gross_change_multiplier':config['gross_change_multiplier'],
            'rounding_only_bounds_NOT_prediction_interval':bounds,
            'hard_override':False,'forecast_interval_supplied':False,'interval_calibration_verified':False,
            'unresolved_uncertainty':['regular versus all-types grade mix','outlet/geographic weighting','within-month sampling','first-print measurement error'],
            'instruction':'Use only as an explicitly conditional forecast candidate. Do not treat coefficient-rounding bounds as a 90% forecast interval, or the computed value as a source quote.'}


def _cpi_context(task,entity,docs):
    for doc in docs:
        table=_table(doc,{'month'})
        if not table:continue
        headers=table['header']
        desired=_key(entity.get('name',''))
        matches=[i for i,h in enumerate(headers) if i and _key(h)==desired]
        if not matches:
            linked=_series_label(doc.text[:table['source']['span_start']],entity.get('series_fred'))
            if linked:matches=[i for i,h in enumerate(headers)if i and _key(h)==_key(linked)]
        if len(matches)!=1:continue
        column=matches[0]
        limit=min(entity.get('latest_published_ref_month',task['cutoff_date'][:7]),task['cutoff_date'][:7],doc.date[:7])
        points=[(r[0],number(r[column])) for r,_,_ in table['rows'] if r[0]<=limit]
        if not points or any(v is None for _,v in points):return None
        values=[v for _,v in points]
        source_adjustment=_adjustment(doc.text[:doc.text.find('\n\n')])
        source_preface=doc.text[:table['source']['span_start']].lower()
        source_monthly_percent=bool(re.search(r'(?:month.over.month|\bmom\b).{0,100}(?:percent|\bpct\b)|(?:percent|\bpct\b).{0,100}(?:month.over.month|\bmom\b)',source_preface))
        target_adjustment=_adjustment(task.get('prompt','')+' '+task.get('target',{}).get('name','').replace('_',' '))
        out={'family':'cpi_component','observed_quantity':'month-over-month percent change only if stated by the source table',
             'observed_adjustment':source_adjustment,'target_adjustment_from_literal_text':target_adjustment,
             'source_monthly_percent_explicit':source_monthly_percent,
             'column':headers[column],'source':table['source'],'observed_history':points,
             'derived_history_summary':_stats(values),
             'limitations':['These are past observations. Verify the target adjustment and vintage contract independently; a CPI/mom name alone does not establish it.',
                            'Past rows are one frozen vintage; original first-print identity is not proven for every month.',
                            'Historical q05/q95 summarize observed rows; they are not future forecast quantiles.',
                            'Nine or fewer monthly values do not establish calibrated 90% future quantiles.']}
        if source_monthly_percent and source_adjustment==target_adjustment=='seasonally_adjusted':
            out['candidate_points']={'last_month_observation_baseline':values[-1]}
        if any(x in desired for x in ('gasoline','energy')):
            for gasdoc in docs:
                gas=_table(gasdoc,{'week_ending'})
                if not gas or gas['header'][1]!='usd_per_gallon':continue
                grouped={}
                for r,_,_ in gas['rows']:
                    value=number(r[1])
                    if r[0]<=min(task['cutoff_date'],gasdoc.date) and value is not None:grouped.setdefault(r[0][:7],[]).append(value)
                months=sorted(grouped)
                if len(months)>=2:
                    a,b=months[-2:]
                    reference=entity.get('ref_month')
                    if isinstance(reference,str)and re.fullmatch(r'\d{4}-\d{2}',reference)and reference in grouped:
                        previous=(dt.date.fromisoformat(reference+'-01')-dt.timedelta(days=1)).strftime('%Y-%m')
                        if previous in grouped:a,b=previous,reference
                    x=st.mean(grouped[a]);y=st.mean(grouped[b])
                    out['pump_observation']={'source':gas['source'],'unit':'nominal dollars per gallon, not seasonally adjusted',
                         'previous_month':a,'current_month':b,'previous_mean':x,'current_mean':y,
                         'unweighted_weekly_mean_change_pct':100*(y/x-1),
                         'previous_weeks':len(grouped[a]),'current_weeks':len(grouped[b])}
                    out['limitations'].append('Pump-price NSA change is not automatically the requested CPI target. If the target is SA, a seasonal transformation is additionally required. Monthly retail weights and seasonal factors are absent from this table.')
        calendar_candidate=_calendar_candidate(task,entity,out,docs)
        if calendar_candidate is not None:out['calendar_candidate']=calendar_candidate
        return out
    return None


def _auction_context(task,entity,docs):
    for doc in docs:
        table=_table(doc,{'auction_date'})
        if not table or 'bid_to_cover' not in table['header']:continue
        # A shared all-tenor document must expose matching tenor rows. Bound
        # single-tenor tables are already manifest-authorized for this entity.
        if doc.shared or len(doc.entity_ids)!=1:continue
        headers=table['header'];index=headers.index('bid_to_cover')
        kind_index=headers.index('new/reopen') if 'new/reopen' in headers else None
        values=[];same=[];history=[]
        for r,_,_ in table['rows']:
            if r[0]>min(task['cutoff_date'],doc.date):continue
            value=number(r[index])
            if value is None:continue
            kind=r[kind_index] if kind_index is not None else None
            values.append(value);history.append({'date':r[0],'kind':kind,'bid_to_cover':value})
            if _auction_kind(kind)==_auction_kind(entity.get('new_or_reopening')) and _auction_kind(kind) is not None:same.append(value)
        if not values:return None
        selected_kind=selected_method_eligible(task,'auction_same_kind_preference')
        point=st.mean(same) if same and selected_kind else st.mean(values[-6:])
        spread=max(1.645*st.stdev(values),abs(point)*.05,.01) if len(values)>1 else abs(point)*.1
        result={'family':'auction','quantity':'competitive and noncompetitive bids tendered / total accepted, ratio',
                'source':table['source'],'observed_history':history,'derived_history_summary':_stats(values),
                'same_new_or_reopening':{'kind':entity.get('new_or_reopening'),'n':len(same),
                                         'mean':st.mean(same) if same else None,'median':st.median(same) if same else None},
                'limitations':['A new issue can have a different demand ratio from a smaller reopening.',
                               'Offering sizes not announced by cutoff remain unknown; do not copy later actual sizes.',
                               'Historical q05/q95 summarize observed rows; they are not future forecast quantiles.',
                               'Historical variation is not verified 90% conditional coverage.']}
        result['candidate_points']={'recent6_mean':st.mean(values[-6:])}
        if selected_kind:result['candidate_points']['same_kind_mean']=point
        result['candidate_interval']={'lo':point-spread,'hi':point+spread,'method':'historical variation band, uncalibrated'}
        return result
    return None


def _cot_context(task,entity,docs):
    for doc in docs:
        table=_table(doc,{'report_date'})
        if not table or not {'open_interest','noncomm_net'}<=set(table['header']):continue
        if doc.shared or len(doc.entity_ids)!=1:continue
        headers=table['header'];oi_index=headers.index('open_interest');net_index=headers.index('noncomm_net')
        data=[]
        for r,_,_ in table['rows']:
            oi,net=number(r[oi_index]),number(r[net_index])
            if r[0]<=min(task['cutoff_date'],doc.date) and oi and oi>0 and net is not None:data.append((r[0],oi,net))
        if len(data)<5:return None
        latest,oi,net=data[-1]
        spacings=sorted({(dt.date.fromisoformat(b[0])-dt.date.fromisoformat(a[0])).days for a,b in zip(data,data[1:])})
        source_futures_only=bool(re.search(r'futures[\s-]+only',doc.text[:650],re.I))
        prompt=task.get('prompt','')
        target_futures_only=bool(re.search(r'futures[\s-]+only',prompt,re.I))
        base_denominator=bool(re.search(re.escape(latest)+r'\s+open interest',prompt,re.I))
        supplied_oi=number(entity.get('open_interest_'+latest.replace('-','')))
        supplied_net=number(entity.get('net_noncommercial_'+latest.replace('-','')))
        changes={str(lag):100*(net-data[-1-lag][2])/oi for lag in (1,4) if len(data)>lag}
        history_pct=[100*n/o for _,o,n in data[-12:]]
        deviation=st.mean(history_pct)-100*net/oi
        match=re.search(r'from the (\d{4}-\d{2}-\d{2}) COT report to the (\d{4}-\d{2}-\d{2}) COT report',task.get('prompt',''))
        horizon=None
        if match:
            try:
                days=(dt.date.fromisoformat(match.group(2))-dt.date.fromisoformat(match.group(1))).days
                if match.group(1)==latest and days>0 and days%7==0:horizon=days//7
            except ValueError:pass
        out={'family':'positioning','observed_quantity':'noncommercial net contracts and open interest; percentage changes require the requested denominator',
             'source_futures_only':source_futures_only,'target_futures_only':target_futures_only,
             'target_base_open_interest_explicit':base_denominator,
             'task_base_open_interest_matches_source':supplied_oi is None or math.isclose(supplied_oi,oi,rel_tol=1e-10,abs_tol=1e-8),
             'task_base_net_matches_source':supplied_net is None or math.isclose(supplied_net,net,rel_tol=1e-10,abs_tol=1e-8),
             'source':table['source'],'history_columns':['report_date','open_interest','noncommercial_net'],
             'observed_history':data,'base_report':latest,'base_open_interest':oi,'base_net_position':net,
             'base_net_pct_oi':100*net/oi,'trailing_net_changes_pct_current_oi':changes,
             'trailing_change_lag_unit':'observed reports','observed_report_spacing_days':spacings,
             'latest_12_report_mean_net_pct_oi_minus_current':deviation,'target_report_steps':horizon,
             'limitations':['Use the literal task denominator. A base-date/open-interest phrase is checked explicitly before offering the existing baseline.',
                            'Report dates precede publication dates; latest supplied report is the knowledge boundary.',
                            'Election/FOMC outcomes are unknown at cutoff. Historical weak rank signal is not a known ordering.']}
        if horizon is not None and spacings==[7] and source_futures_only and target_futures_only and base_denominator and selected_method_eligible(task,'cot_shrink_0_1'):
            trend=.1*horizon*changes['1']
            out['candidate_points']={'v5_shrunk_weekly_trend_baseline':trend}
            residuals=[]
            for j in range(8,len(data)-horizon):
                _,o,n=data[j]
                past_pred=.1*horizon*100*(n-data[j-1][2])/o
                realized=100*(data[j+horizon][2]-n)/o
                residuals.append(realized-past_pred)
            if residuals:
                out['past_residual_summary']={'n':len(residuals),'q05':quantile(residuals,.05),
                                              'q95':quantile(residuals,.95),'median':st.median(residuals),
                                              'known_target_latest_date':data[-1][0],
                                              'calibration_verified':False}
        return out
    return None


def _vintage_context(task,entity,docs):
    for doc in docs:
        table=_table(doc,{'reference_month'})
        if not table or not any(h.startswith('as_of_') for h in table['header']):continue
        if doc.shared:continue
        columns=[(i,h.removeprefix('as_of_')) for i,h in enumerate(table['header']) if i and h.startswith('as_of_') and h.removeprefix('as_of_')<=min(task['cutoff_date'],doc.date)]
        if len(columns)<2 or len({d for _,d in columns})!=len(columns):continue
        try:
            for _,date in columns:dt.date.fromisoformat(date)
        except ValueError:continue
        columns.sort(key=lambda pair:pair[1])
        matrix={r[0]:[number(r[i]) for i,_ in columns] for r,_,_ in table['rows'] if r[0]<=min(task['cutoff_date'],doc.date)[:7]}
        ref=entity.get('ref_month')
        if ref not in matrix:continue
        transitions=[]
        dates=[dt.date.fromisoformat(date) for _,date in columns]
        for j in range(len(columns)-1):
            available=sorted(month for month,vs in matrix.items() if vs[j] is not None)
            for age,month in enumerate(reversed(available[-3:])):
                a,b=matrix[month][j],matrix[month][j+1]
                if a is not None and b is not None:
                    transitions.append({'from':str(dates[j]),'to':str(dates[j+1]),'reference_month':month,
                                        'age_rank_at_origin':age,'delta':b-a})
        return {'family':'vintage_revision','observed_quantity':entity.get('units',entity.get('currency','series level')),
                'source':table['source'],'vintage_dates':[str(d) for d in dates],
                'history_by_reference_month':matrix,'target_reference_month':ref,
                'source_revision_schedule':next((line.strip() for line in doc.text.splitlines() if line.lower().startswith('revision schedule:')),''),
                'target_latest_value':entity.get('latest_precutoff_estimate'),
                'target_resolving_release_date':entity.get('resolving_release_date'),
                'recent_month_revision_transitions':transitions,
                'limitations':['The table contains historical levels and computed revision deltas. The target can request a level, delta or direction: follow its literal prompt instead of assuming from the word revision.',
                               'A 1st-to-2nd estimate and a 2nd-to-3rd estimate need not share a revision distribution.',
                               'Release intervals can be irregular; dividing calendar days by the median gap is only a heuristic.',
                               'Distinguish a broad annual level update from an ordinary next release; do not repeat a past annual shift mechanically.',
                               'Small, overlapping histories do not establish calibrated 90% intervals.']}
    return None


def _rates_context(task,entity,docs):
    """Yield levels are observations; a future basis-point change is a different quantity."""
    name=task.get('target',{}).get('name','').lower();prompt=task.get('prompt','').lower()
    if 'yield'not in name or 'change'not in name or not('bps'in name or'basis point'in prompt):return None
    start=number(entity.get('start_yield_pct'));series=entity.get('series_fred');maturity=number(entity.get('maturity_years'))
    if start is None or not isinstance(series,str):return None
    for doc in docs:
        table=_table(doc,{'date'})
        if table and series in table['header'] and re.search(r'yields?[^\n]{0,100}(?:percent|%)',doc.text,re.I):
            column=table['header'].index(series)
            history=[(r[0],number(r[column]))for r,_,_ in table['rows']if r[0]<=min(task['cutoff_date'],doc.date)]
            if history and all(value is not None for _,value in history):
                return {'family':'treasury_yield_change','observed_quantity':'constant-maturity yield level in percent',
                        'requested_quantity':'future yield change in basis points, as literally declared by target/prompt',
                        'source':table['source'],'observed_history':history,'target_start_yield_pct':start,
                        'target_start_asof':entity.get('as_of'),'observed_last_level_pct':history[-1][1],
                        'observed_last_date':history[-1][0],
                        'arithmetic_contract':'target_bps = 100 * (unobserved_target_yield_pct - task_start_yield_pct)',
                        'limitations':['The displayed historical levels are not future basis-point changes.',
                                       'No seven-week predictive distribution is estimated from this short daily snapshot.',
                                       'A policy rate cut/rise does not mechanically determine the change of a maturity yield; prior market pricing matters.']}
        if maturity is None:continue
        term=str(int(maturity))if maturity.is_integer()else str(maturity)
        pattern=r'\b(?:The\s+)?'+re.escape(term)+r'-year Treasury yield was\s+([+-]?\d+(?:\.\d+)?)\s+percent'
        matches=list(re.finditer(pattern,doc.text,re.I))
        if len(matches)==1 and math.isclose(float(matches[0].group(1)),start,rel_tol=1e-9,abs_tol=1e-9):
            match=matches[0]
            return {'family':'treasury_yield_change','observed_quantity':'constant-maturity yield level in percent',
                    'requested_quantity':'future yield change in basis points, as literally declared by target/prompt',
                    'source':{'doc_id':doc.doc_id,'doc_date':doc.date,'span_start':match.start(),'span_end':match.end()},
                    'observed_history':[(entity.get('as_of',task['cutoff_date']),start)],
                    'target_start_yield_pct':start,'target_start_asof':entity.get('as_of'),
                    'arithmetic_contract':'target_bps = 100 * (unobserved_target_yield_pct - task_start_yield_pct)',
                    'limitations':['A single snapshot cannot establish a calibrated forecast or interval.',
                                   'Historical yield level and the requested future bps change are separate quantities.']}
    return None


def build_numeric_context(task,entity,corpus):
    docs=list(_authorized_docs(task,entity,corpus))
    target=task.get('target',{}).get('name','').lower()
    if 'cpi' in target and 'mom' in target:result=_cpi_context(task,entity,docs)
    elif 'bid_to_cover' in target:result=_auction_context(task,entity,docs)
    elif 'positioning_change' in target:result=_cot_context(task,entity,docs)
    elif 'revision' in target:result=_vintage_context(task,entity,docs)
    elif 'yield' in target and 'change' in target:result=_rates_context(task,entity,docs)
    else:return None
    if result is None:return None
    result['entity_id']=entity['entity_id']
    result['target_name']=task.get('target',{}).get('name')
    result['target_type']=task.get('target',{}).get('type')
    result['cutoff_date']=task.get('cutoff_date')
    result['provenance_note']='Observed rows are from the cited frozen source. Summaries/candidate points are transparent arithmetic, not observed future values and not claim text.'
    return result
