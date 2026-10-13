"""Small shared linear research predictors on semantic frozen-input features.

No task/entity ID or response lookup enters a design vector. Fitted artifacts in
this study use post-cutoff practice labels and are NOT certified for deployment.
"""
import datetime as dt,math,re,statistics as st
from .numerical_context import build_numeric_context,_table,number,_auction_kind,quantile

def fraction(text):
    if '/'in text:
        a,b=text.split('/');return float(a)/float(b)
    return float(text)

def policy_action(task,entity,corpus):
    found=[]
    for doc in corpus.docs.values():
        if doc.date>task['cutoff_date']or(not doc.shared and entity['entity_id']not in doc.entity_ids):continue
        text=doc.text
        pattern=r'Committee decided to (raise|lower) the target range for the federal funds rate by\s+([\d.]+(?:/[\d.]+)?)\s+percentage point'
        for m in re.finditer(pattern,text,re.I):found.append((100*fraction(m[2])*(1 if m[1].lower()=='raise'else-1),doc.doc_id,m.span(),m.group()))
        pattern=r'its\s+(?:second consecutive\s+)?(\d+(?:\.\d+)?)\s+basis point\s+(increase|reduction|cut)'
        for m in re.finditer(pattern,text,re.I):found.append((float(m[1])*(1 if m[2].lower()=='increase'else-1),doc.doc_id,m.span(),m.group()))
        pattern=r'Federal Open Market Committee\s+(raised|lowered)\s+the target range for the federal funds rate by\s+(\d+(?:\.\d+)?)\s+basis points'
        for m in re.finditer(pattern,text,re.I):found.append((float(m[2])*(1 if m[1].lower()=='raised'else-1),doc.doc_id,m.span(),m.group()))
    values={r[0]for r in found}
    if len(values)!=1:raise ValueError('unresolved_or_conflicting_literal_policy_action')
    return found[0][0],[{'doc_id':r[1],'span_start':r[2][0],'span_end':r[2][1],'text':r[3]}for r in found]

def features(task,entity,corpus):
    name=task.get('target',{}).get('name','');info={}
    if name=='yield_change_bps_intermeeting':
        maturity=number(entity.get('maturity_years'));start=number(entity.get('start_yield_pct'))
        if not maturity or maturity<=0 or start is None or entity.get('as_of')!=task['cutoff_date']or entity.get('unit')!='bps_change':raise ValueError('rates_contract')
        action,sources=policy_action(task,entity,corpus);a=action/100.;logm=math.log(maturity)
        ns2=-math.expm1(-maturity/2)/(maturity/2);ns10=-math.expm1(-maturity/10)/(maturity/10)
        horizon=(dt.date.fromisoformat(task['resolution_date'])-dt.date.fromisoformat(task['cutoff_date'])).days
        if not 25<=horizon<=90:raise ValueError('outside_intermeeting_horizon_contract')
        f={'intercept':1.,'policy_action_pct':a,'log_maturity':logm,'action_log_maturity':a*logm,'ns_tau2':ns2,'ns_tau10':ns10,'action_ns_tau2':a*ns2,'action_ns_tau10':a*ns10}
        info={'family':'rates','native_unit':'bps_change','task_baseline_yield_pct':start,'policy_action_bps':action,'policy_action_sources':sources,'maturity_years':maturity,'horizon_days':horizon,'context_point':0.,'context_lo_delta':-100.,'context_hi_delta':100.,'secondary_anchor_point':0.,'secondary_anchor_lo_delta':-50.,'secondary_anchor_hi_delta':50.}
    elif name=='bid_to_cover_ratio':
        ctx=build_numeric_context(task,entity,corpus)
        if not ctx or ctx['family']!='auction':raise ValueError('auction_source')
        history=ctx['observed_history'];values=[r['bid_to_cover']for r in history];mean6=st.mean(values[-6:]);last=values[-1];kind=_auction_kind(entity.get('new_or_reopening'))
        same=[r['bid_to_cover']for r in history if _auction_kind(r['kind'])==kind]
        if len(values)<6 or not same or not kind:raise ValueError('auction_history')
        m=re.fullmatch(r'(\d+)-Year',str(entity.get('tenor','')))
        if not m:raise ValueError('maturity_units')
        maturity=float(m[1]);window=values[-6:];x=[i-2.5 for i in range(6)];trend=sum(a*(v-mean6)for a,v in zip(x,window))/sum(a*a for a in x)
        doc=corpus.docs[ctx['source']['doc_id']];table=_table(doc,{'auction_date'});headers=table['header'];target_amount=number(entity.get('offering_amount_usd_bn'));prior_amount=None
        if 'offering_$B'in headers and 'new/reopen'in headers:
            for row,_,_ in table['rows']:
                if row[0]<=min(task['cutoff_date'],doc.date)and _auction_kind(row[headers.index('new/reopen')])==kind:prior_amount=number(row[headers.index('offering_$B')])
        amount_known=entity.get('size_announced_precutoff')is True and target_amount is not None and prior_amount is not None and prior_amount>0
        amount_contrast=target_amount/prior_amount-1 if amount_known else 0.
        half=max(1.645*st.stdev(values),abs(mean6)*.05,.01)
        f={'intercept':1.,'mean6':mean6,'last_minus_mean6':last-mean6,'same_kind_minus_mean6':st.mean(same)-mean6,'trend6':trend,'log_maturity':math.log(maturity),'known_announced_size_contrast':amount_contrast}
        info={'family':'auction','native_unit':'bid_to_cover_ratio','maturity_years':maturity,'known_announced_amount_feature':amount_known,'amount_missing_means_no_size_effect_not_future_imputation':not amount_known,'context_point':mean6,'context_lo_delta':-half,'context_hi_delta':half,'secondary_anchor_point':st.mean(same),'secondary_anchor_lo_delta':-half,'secondary_anchor_hi_delta':half,'source':ctx['source']}
    elif name=='net_positioning_change_pct_oi_rank':
        ctx=build_numeric_context(task,entity,corpus)
        if not ctx or ctx['family']!='positioning'or not all(ctx.get(k)for k in('source_futures_only','target_futures_only','target_base_open_interest_explicit','task_base_open_interest_matches_source','task_base_net_matches_source')):raise ValueError('COT_source_contract')
        doc=corpus.docs[ctx['source']['doc_id']];table=_table(doc,{'report_date'});headers=table['header'];required=['open_interest','noncomm_long','noncomm_short','noncomm_net']
        if any(k not in headers for k in required):raise ValueError('missing_COT_columns')
        rows=[]
        for row,_,_ in table['rows']:
            if row[0]<=min(task['cutoff_date'],doc.date):rows.append([number(row[headers.index(k)])for k in required])
        if len(rows)!=26 or any(any(v is None for v in row)for row in rows):raise ValueError('exact26COT_history')
        if any(abs(row[1]-row[2]-row[3])>1e-6 or row[0]<=0 for row in rows):raise ValueError('COT_net_definition')
        oi,long,short,net=rows[-1];h=ctx['target_report_steps']
        if h is None or ctx['observed_report_spacing_days']!=[7]:raise ValueError('COT_report_horizon')
        weekly=[100*(b[3]-a[3])/b[0]for a,b in zip(rows[-14:],rows[-13:])];weak=.1*h*100*(net-rows[-2][3])/oi
        f={'intercept':1.,'net_pct_base_oi':100*net/oi,'weak_weekly_trend':weak,'horizon_scaled_change4':h/4*100*(net-rows[-5][3])/oi,'deviation26':100*net/oi-st.mean(100*r[3]/r[0]for r in rows),'horizon_weekly_vol':math.sqrt(h)*st.stdev(weekly),'gross_pct_base_oi':100*(long+short)/oi}
        # Fixed contextual scenarios use observed five-report changes; they do not use practice labels.
        changes=[];residual=[]
        for i in range(1,len(rows)-h):
            y=100*(rows[i+h][3]-rows[i][3])/rows[i][0];p=.1*h*100*(rows[i][3]-rows[i-1][3])/rows[i][0];changes.append(y);residual.append(y-p)
        lo,hi=min(-1.,quantile(changes,.05)),max(1.,quantile(changes,.95));rlo,rhi=min(-1.,quantile(residual,.05)),max(1.,quantile(residual,.95))
        info={'family':'cot','native_unit':'pct_of_base_open_interest','base_open_interest':oi,'base_net_contracts':net,'report_steps':h,'context_point':0.,'context_lo_delta':lo,'context_hi_delta':hi,'secondary_anchor_point':weak,'secondary_anchor_lo_delta':rlo,'secondary_anchor_hi_delta':rhi,'source':ctx['source'],'historical_scenarios':len(changes)}
    else:raise ValueError('unsupported_native_contract')
    if not all(math.isfinite(v)for v in f.values()):raise ValueError('nonfinite_features')
    return f,info

def point_from_features(values,model):
    if len(model['feature_names'])!=len(model['coefficients'])or len(set(model['feature_names']))!=len(model['feature_names']):raise ValueError('invalid_equation_shape')
    if any(isinstance(c,bool)or not isinstance(c,(int,float))or not math.isfinite(c)for c in model['coefficients']):raise ValueError('nonfinite_coefficient')
    point=sum(values[n]*c for n,c in zip(model['feature_names'],model['coefficients']))
    if model.get('offset_feature'):point+=values[model['offset_feature']]
    return point

def predict_research(task,entity,corpus,model,*,interval_mode='context'):
    if interval_mode not in('context','in_sample_residual'):raise ValueError('unknown_interval_mode')
    values,info=features(task,entity,corpus)
    if model['family']!=info['family']:raise ValueError('wrong_family')
    point=point_from_features(values,model)
    tails=(info['context_lo_delta'],info['context_hi_delta'])if interval_mode=='context'else model['in_sample_residual_tails']
    if len(tails)!=2 or not all(math.isfinite(x)for x in(point,*tails))or tails[0]>=tails[1]:raise ValueError('invalid_native_forecast')
    result={'point_forecast':point,'interval':{'lo':point+tails[0],'hi':point+tails[1],'level':.9},'unit':info['native_unit'],'research_only':True,'calibrated_90pct_claim':False,'task_cutoff_eligible_for_training_labels':task['cutoff_date']>=model['label_available_by'],'deployment_certified':False}
    if info['family']=='rates':result['future_yield_pct_coordinates']={'point':info['task_baseline_yield_pct']+point/100,'lo':info['task_baseline_yield_pct']+(point+tails[0])/100,'hi':info['task_baseline_yield_pct']+(point+tails[1])/100}
    return result
