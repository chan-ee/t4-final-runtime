"""Compact observed financial context; no forecasts, fitted weights or API calls.

This layer preserves periods, accounting ambiguity and financing conditions.
An exact source quote establishes a premise, not a future outcome distribution.
"""
from __future__ import annotations
import datetime as dt
import math
import re
from .corpus import Chunk
from .quantity import eps_growth_contract
from .credit_context import is_credit_task


_MONTH_NAMES=('January','February','March','April','May','June','July','August','September','October','November','December')
_MONTH_MAP={month.lower():i+1 for i,month in enumerate(_MONTH_NAMES)}
_MONTH_MAP.update({month[:3].lower():i+1 for i,month in enumerate(_MONTH_NAMES)})
_DATE=r'(?:'+ '|'.join(sorted(_MONTH_MAP,key=len,reverse=True))+r')\.?\s+\d{1,2},?\s+20\d{2}'
_MONEY=r'\$\s*(?P<amount>\d[\d,]*(?:\.\d+)?)\s*(?P<scale>thousand|million|billion)s?\b'
_CASH_PATTERNS=(
    r'As\s+of\s+(?P<date>'+_DATE+r'),?\s+(?:we|the\s+company)\s+(?:had|reported)\s+'+_MONEY+r'\s+in\s+cash\s+and\s+cash\s+equivalents',
    r'(?P<date>'+_DATE+r'),?\s+(?:and\s+(?:'+_DATE+r'|20\d{2}),?\s+)?(?:our|the\s+company.s)\s+cash\s+and\s+cash\s+equivalents\s+(?:was|were)\s+'+_MONEY,
    _MONEY+r'\s+in\s+cash\s+and\s+cash\s+equivalents\s+as\s+of\s+(?P<date>'+_DATE+r')',
    r'cash\s+and\s+cash\s+equivalents\s+of\s+'+_MONEY+r'\s+as\s+of\s+(?P<date>'+_DATE+r')',
)


def _own_docs(corpus,task,entity):
    # Shared financial prose may cover several issuers. Keep it as ordinary
    # evidence, but do not assign its numbers automatically to every row.
    return [d for d in corpus.docs.values() if d.date<=task['cutoff_date'] and not d.shared and entity['entity_id'] in d.entity_ids]


def _normal(text):
    return text.translate(str.maketrans({'\u200b':' ','\u200c':' ','\u200d':' ','\ufeff':' '}))


def _date(value):
    parts=re.fullmatch(r'([A-Za-z]+)\.?\s+(\d{1,2}),?\s+(20\d{2})',value.strip())
    if not parts:return None
    try:return dt.date(int(parts[3]),_MONTH_MAP[parts[1].lower()],int(parts[2]))
    except (KeyError,ValueError):return None


def _window(doc,start,end,before,after,*,first_sentence=False):
    left=max(0,start-before);right=min(len(doc.text),end+after)
    # Do not turn an adjacent '$250 million' into '$2' at a raw character cap.
    # All coordinates continue to address one unchanged contiguous source span.
    if left and not doc.text[left-1].isspace():
        boundary=doc.text.find(' ',left,min(start,left+100))
        if boundary>=0:left=boundary+1
    endings=list(re.finditer(r'[.!?]\s+(?=[A-Z])',doc.text[end:right]))
    if endings:
        stop=end+(endings[0].start() if first_sentence else endings[-1].start())+1
        right=stop
    elif right<len(doc.text) and not doc.text[right-1].isspace() and not doc.text[right].isspace():
        boundary=re.search(r'\s',doc.text[right:min(len(doc.text),right+100)])
        if boundary:right+=boundary.start()
        else:
            boundary=max(doc.text.rfind(' ',end,right),doc.text.rfind('\n',end,right))
            if boundary>end:right=boundary
    return Chunk(doc.doc_id,left,right,doc.text[left:right])


def _cash_observations(corpus,task,entity):
    found=[];seen=set();cutoff=dt.date.fromisoformat(task['cutoff_date'])
    for doc in _own_docs(corpus,task,entity):
        normalized=_normal(doc.text)
        for pattern in _CASH_PATTERNS:
            for match in re.finditer(pattern,normalized,re.I):
                measured=_date(match.group('date'))
                if measured is None or str(measured)>doc.date or measured>cutoff:continue
                before=normalized[max(0,match.start()-80):match.start()].lower()
                if re.search(r'minimum|required|must\s+maintain|at\s+least|expect|project',before):continue
                value=float(match.group('amount').replace(',',''))
                multiplier={'thousand':.001,'million':1.,'billion':1000.}[match.group('scale').lower()]
                if not math.isfinite(value) or value<0:continue
                key=(doc.doc_id,str(measured),value*multiplier)
                if key in seen:continue
                seen.add(key)
                found.append({'quantity':'reported cash and cash equivalents amount',
                    'value_usd_millions':value*multiplier,'measurement_date':str(measured),
                    'source_available_date':doc.date,'days_before_task_cutoff':(cutoff-measured).days,
                    'status':'historical reported amount candidate; not current unrestricted cash',
                    'scope_verified':False,'accessibility_verified':False,
                    '_window':_window(doc,match.start(),match.end(),80,500,first_sentence=True)})
    return sorted(found,key=lambda x:(x['measurement_date'],x['source_available_date']),reverse=True)[:2]


def _ranked_windows(corpus,task,entity,kind):
    if kind=='eps':
        facets=('per_share_effect','valuation_or_accounting_change','income_gain_or_loss','explicit_special_item','recurring_charge')
        rules=(
            (r'(?:decreased|reduced|increased)\s+(?:diluted\s+)?earnings\s+per\s+(?:common\s+)?share.{0,80}\$',16),
            (r'net\s+unrealized\s+(?:gain|loss)|bargain\s+purchase\s+gain|change\s+in\s+(?:the\s+)?method\s+of\s+accounting',11),
            (r'(?:net\s+revenue|net\s+income|other\s+income)\s+included.{0,100}(?:gain|loss)',10),
            (r'(?:one[- ]time|non[- ]?recurring|notable|special)\s+(?:items?|charges?|assessments?|gains?|losses?)',7),
            (r'amortization\s+of\s+acquisition[- ]related\s+intangibles|workforce\s+rebalancing\s+charges?',7),
        )
    else:
        facets=('cash_scope','financing_capacity','cash_restriction','known_cash_change','conditional_financing','lease_flow_classification')
        rules=(
            (r'(?:cash\s+and\s+cash\s+equivalents.{0,70}(?:included|of).{0,100}(?:VIE|variable\s+interest)|(?:VIE|variable\s+interest).{0,150}cash)',12),
            (r'unissued\s+(?:secured\s+)?notes|(?:secured\s+)?debt\s+covenant\s+capacity|managed\s+accessibility',12),
            (r'(?:short[- ]term\s+|long[- ]term\s+)?restricted\s+cash',7),
            (r'reducing\s+(?:our|the\s+company.s)\s+cash\s+and\s+cash\s+equivalents\s+by',12),
            (r'if\s+and\s+(?:once|when)\s+implemented|subject\s+to.{0,40}(?:financing|credit\s+approval|lender\s+consent)',8),
            (r'(?:cash\s+paid|operating\s+cash\s+flows).{0,110}(?:operating\s+leases?|finance\s+leases?).{0,100}(?:operating\s+activities|\$)',8),
        )
    candidates=[]
    for doc in _own_docs(corpus,task,entity):
        normalized=_normal(doc.text)
        for index,(pattern,base) in enumerate(rules):
            for match in re.finditer(pattern,normalized,re.I):
                source_window=_window(doc,match.start(),match.end(),300,650)
                start,end=source_window.start,source_window.end
                near=normalized[max(0,match.start()-180):min(len(doc.text),match.end()+380)]
                text=doc.text[start:end]
                money=len(re.findall(r'\$\s*\(?\s*[\d,.]+',near))
                if not money:continue
                if kind=='eps' and not re.search(r'earnings|\beps\b|income|revenue',near,re.I):continue
                if kind=='eps':
                    # Other comprehensive income is not automatically an EPS
                    # item. It requires an explicit reclassification/earnings
                    # link; a true quote about an AOCI mark is insufficient.
                    if re.search(r'other\s+comprehensive\s+income|\baoci\b|accumulated\s+oci',near,re.I) and not re.search(r'reclassif.{0,90}(?:earnings|net\s+income)|recogniz.{0,70}(?:in\s+earnings|in\s+net\s+income)',near,re.I):
                        continue
                    if re.search(r'(?:drivers|gains?|losses?).{0,120}(?:not\s+material|not\s+significant)',near,re.I):continue
                score=base+2*min(money,3)+3*bool(re.search(r'prior[- ]year|prior year|current quarter|20\d{2}',near,re.I))
                if kind=='eps':score+=5*bool(re.search(r'net[- ]of[- ]tax|per\s+(?:common\s+)?share\s+by',text,re.I))
                candidates.append((score,doc.date,-match.start(),doc.doc_id,index,source_window))
    ranked=[{'focus':facets[x[-2]],'window':x[-1]} for x in sorted(candidates,key=lambda x:x[:-1],reverse=True)]
    if kind=='credit':
        # Repeated capacity clauses must not crowd out a completed cash event.
        # This is coverage of distinct source roles, not a score or probability.
        order=('known_cash_change','cash_scope','financing_capacity','cash_restriction','lease_flow_classification','conditional_financing')
        first=[]
        for focus in order:
            first.extend([x for x in ranked if x['focus']==focus][:1])
        return first
    return ranked


def _eps_comparison(packet):
    facts=packet.get('observations',{}).get('historical_eps',[])
    quarter=[f for f in facts if f.get('period_months')==3 and f.get('unit')=='USD/share' and isinstance(f.get('value'),(int,float)) and not isinstance(f.get('value'),bool)]
    if not quarter:return []
    latest=max(quarter,key=lambda f:f['period_end'])
    try:
        when=dt.date.fromisoformat(latest['period_end'])
        prior=next((f for f in quarter if 351<=(when-dt.date.fromisoformat(f['period_end'])).days<=379),None)
    except (ValueError,KeyError):return []
    if prior is None:return []
    now,old=float(latest['value']),float(prior['value'])
    if not math.isfinite(now) or not math.isfinite(old):return []
    return [{'status':'arithmetic on historical observations only; not a future forecast',
             'current_reported_period_end':latest['period_end'],'prior_reported_period_end':prior['period_end'],
             'absolute_eps_difference':now-old,'unit':'USD/share',
             'defined_relative_change_pct':100*(now/old-1) if old>0 else None,
             'source_ids':list(dict.fromkeys(latest.get('source_ids',[])+prior.get('source_ids',[]))),
             'limitations':['Verify equal accounting/share basis and total-versus-continuing scope.',
                 'Current and prior periods can both contain unusual items; their after-tax EPS effects are not automatically known.',
                 'These are past comparative quarters, not the task target quarter.']}]


def enrich_forecast_context(task,entity,corpus,packet,*,max_additional_chars=3200):
    """Add one compact deterministic context; mutate and return the same packet.

    Existing source IDs remain stable. Additional text is original, scoped and
    bounded; observations, arithmetic and unresolved assumptions stay separate.
    """
    target_name=task.get('target',{}).get('name','').lower()
    eps='eps' in target_name or 'earnings_per_share' in target_name
    credit=is_credit_task(task)
    if not eps and not credit:return packet
    # Imported at call time because packets owns the single source-ID registry.
    from .packets import add_source
    budget=max(0,int(max_additional_chars));used=0;selected=[]
    context={'kind':'eps' if eps else 'credit','observations':[],
             'derived_relations':[],'source_checks':[],'unresolved_future_quantities':[],
             'status':'forecast input context; no predicted labels, probabilities or calibrated intervals'}
    def add(window,role):
        nonlocal used
        if used+len(window.text)>budget:return []
        if any(window.doc_id==old.doc_id and max(0,min(window.end,old.end)-max(window.start,old.start))>.65*min(len(window.text),len(old.text)) for old in selected):return []
        identifiers=add_source(packet,window,corpus,task,entity,role=role)
        if identifiers:selected.append(window);used+=len(window.text)
        return identifiers
    if eps:
        context['quantity_contract']=eps_growth_contract(task,entity)
        context['derived_relations']=_eps_comparison(packet)
        context['source_checks']=[
            'Read the exact target quarter and match source reporting currency, GAAP/adjusted, basic/diluted and total/continuing definitions.',
            'Compare unusual items in BOTH periods; gross gains, net-of-tax amounts and per-share effects are different.',
            'Noncash amortization may recur in GAAP. Core/adjusted EPS is not automatically the requested target.',
            'Half-year EPS minus quarter EPS need not equal an exact prior-quarter EPS because weighted-average share counts differ.']
        context['unresolved_future_quantities']=['target-period revenue and margin','target-period taxes and diluted shares','recurrence or reversal of unusual items']
        for candidate in _ranked_windows(corpus,task,entity,'eps'):
            window=candidate['window']
            ids=add(window,'earnings_comparability')
            if ids:context['observations'].append({'kind':'published comparability disclosure','source_ids':ids,
                'focus':candidate['focus'],
                'period_and_tax_basis':'must be read from the source; no automatic adjustment made'})
            if len(context['observations'])>=2:break
    else:
        context['source_checks']=[
            'Separate measurement date from document date and cutoff; bridge missing elapsed cash flows explicitly.',
            'Cash including restricted/VIE balances, invested cash, undrawn commitments and covenant capacity are different quantities.',
            'A gross financing issuance is not automatically net cash. Conditional future financing is not completed cash.',
            'Subtract a known post-balance repayment once. Never add overlapping cash pools or capacities twice.',
            'Operating cash flow already includes the disclosed operating lease/interest components; deduct only genuinely incremental future outflows.',
            'A cash shortfall projection is not the credit-event label; future financing, legal events and distressed exchanges need separate reasoning.']
        context['unresolved_future_quantities']=['net cash changes since the last measured balance','future working-capital reversals or persistence',
            'future usable borrowing and financing completion','incremental capex/principal/interest','task-defined event process and probability']
        for observed in _cash_observations(corpus,task,entity):
            window=observed.pop('_window');ids=add(window,'cash_scope_and_conditions')
            if ids:context['observations'].append({**observed,'source_ids':ids})
        for candidate in _ranked_windows(corpus,task,entity,'credit'):
            window=candidate['window']
            ids=add(window,'cash_scope_and_conditions')
            if ids:context['observations'].append({'kind':'cash scope or financing condition context','source_ids':ids,
                'focus':candidate['focus'],
                'status':'original disclosure; no capacity-to-cash conversion or default probability inferred'})
            if used>=budget-300:break
    context['additional_original_text_chars']=used
    packet.setdefault('observations',{})['forecast_context']=context
    return packet
