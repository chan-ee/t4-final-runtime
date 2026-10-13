"""Conservative original-document contract-flow measurement, never a forecast.

One supported disclosure grammar combines a complete new-note table, an
amount-matched issuance date, a dated interest-bearing-asset table and explicit
historical financing termination/extinguishment flows. No issuer names,
document IDs, numeric answers, source offsets or external data are embedded.
Unrecognized or ambiguous structures return None instead of filling values.
"""
from __future__ import annotations
import datetime as dt
import math
import re

_MONTHS=('January','February','March','April','May','June','July','August','September','October','November','December')
_DATE=r'(?:'+'|'.join(_MONTHS)+r')\s+\d{1,2},\s+20\d{2}'
_AMOUNT=r'[0-9][0-9,]*(?:\.[0-9]+)?'


def _date(literal):
    try:return dt.datetime.strptime(' '.join(literal.split()),'%B %d, %Y').date()
    except (TypeError,ValueError):return None


def _number(literal):
    try:value=float(literal.replace(',',''))
    except (TypeError,ValueError):return None
    return value if math.isfinite(value)else None


def _span(doc,start,end,role):
    return {'doc_id':doc.doc_id,'doc_date':doc.date,'span_start':start,'span_end':end,
            'text':doc.text[start:end],'role':role,'status':'original source observation; not a forecast'}


def _one(pattern,text,flags=re.I):
    matches=list(re.finditer(pattern,text,flags))
    return matches[0]if len(matches)==1 else None


def _document_bundle(doc,cutoff,target_end):
    text=doc.text
    # Require a table explicitly declaring its period and millions scale.
    header=_one(r'Debt issuances and acquisition-related financing\s+During the three months ended\s+('+_DATE+r')\s*,?\s*.{0,500}?'
                r'we issued the following series of notes\s*\(in millions\)\s*:\s*Principal Amount',text,re.I|re.S)
    if header is None:return None
    observed_end=_date(header[1])
    if observed_end is None or observed_end.isoformat()>min(doc.date,cutoff):return None
    if target_end is not None and observed_end>=target_end:return None
    tail=text[header.end():header.end()+2400]
    total=re.search(r'\bTotal\s*\$\s*('+_AMOUNT+r')',tail)
    if total is None:return None
    table=tail[:total.start()]
    row_pattern=r'('+_AMOUNT+r')\s*%\s*(20\d{2})\s+Notes\s*\$?\s*('+_AMOUNT+r')'
    rows=list(re.finditer(row_pattern,table,re.I))
    if not 2<=len(rows)<=32:return None
    # Every table character must belong to a parsed row, not an omitted clause.
    residue=re.sub(row_pattern,'',table,flags=re.I)
    if residue.strip():return None
    tranches=[]
    for match in rows:
        coupon=_number(match[1]);principal=_number(match[3]);year=int(match[2])
        if coupon is None or not 0<=coupon<=100 or principal is None or principal<=0:return None
        tranches.append({'coupon_percent':coupon,'maturity_year':year,'principal_source_millions':principal,
                         'source':_span(doc,header.end()+match.start(),header.end()+match.end(),'issued_note_tranche')})
    stated_total=_number(total[1]);sum_total=sum(r['principal_source_millions']for r in tranches)
    if stated_total is None or not math.isclose(stated_total,sum_total,abs_tol=1e-6,rel_tol=1e-12):return None
    if len({(r['coupon_percent'],r['maturity_year'])for r in tranches})!=len(tranches):return None
    # Link issuance timing by matching the disclosed aggregate principal, not by
    # proximity to some other note, court date, or incorporated exhibit date.
    dates=[]
    pattern=r'funds received from our\s*\$\s*('+_AMOUNT+r')\s*(million|billion)\s+of senior notes issued on\s+('+_DATE+r')'
    for match in re.finditer(pattern,text,re.I):
        amount=_number(match[1])
        if amount is None:continue
        amount*=1000 if match[2].lower()=='billion'else 1
        issued=_date(match[3])
        if math.isclose(amount,stated_total,abs_tol=1e-6,rel_tol=1e-12)and issued is not None:
            dates.append((match,issued))
    if len(dates)!=1:return None
    issuance,issued=dates[0]
    if not issued<=observed_end or issued.isoformat()>min(doc.date,cutoff):return None
    if any(r['maturity_year']<issued.year for r in tranches):return None
    # The supported grammar certifies a common dated asset stock. It does not
    # certify future balance, yield, availability, cash-flow classification or FX.
    cash=_one(r'The fair values of interest-bearing securities by location.{0,120}?\(in millions\)\s*:\s*'
              r'Condensed Consolidated Balance Sheets locations\s+('+_DATE+r')\s+('+_DATE+r')\s+'
              r'Cash and cash equivalents\s*\$\s*('+_AMOUNT+r')\s*\$\s*('+_AMOUNT+r')\s+'
              r'Marketable securities\s+('+_AMOUNT+r')\s+('+_AMOUNT+r')\s+'
              r'Total interest-bearing securities\s*\$\s*('+_AMOUNT+r')\s*\$\s*('+_AMOUNT+r')',text,re.I|re.S)
    if cash is None:return None
    asset_date=_date(cash[1]);comparison_date=_date(cash[2])
    if asset_date!=observed_end or comparison_date is None or comparison_date>=asset_date:return None
    values=[_number(cash[i])for i in range(3,9)]
    if any(v is None or v<0 for v in values):return None
    cash_now,cash_old,market_now,market_old,total_now,total_old=values
    if not math.isclose(cash_now+market_now,total_now,abs_tol=1e-6,rel_tol=1e-12):return None
    if not math.isclose(cash_old+market_old,total_old,abs_tol=1e-6,rel_tol=1e-12):return None
    fee=_one(r'Subsequent to our\s+[A-Za-z]+\s+20\d{2}\s+debt issuance described above,\s*we terminated the bridge credit agreement\.\s*'
             r'Accordingly, during the three months ended\s+('+_DATE+r')\s*,?\s*we recognized\s*\$\s*('+_AMOUNT+r')\s*million of financing cost associated with the bridge credit agreement,\s*primarily in Other income\s*\(expense\),\s*net',text,re.I)
    gain=_one(r'During the three months ended\s+('+_DATE+r')\s*,?\s*we repurchased portions.{0,1200}?'
              r'which resulted in the recognition of a\s*\$\s*('+_AMOUNT+r')\s*million\s+(gain|loss)\s+on extinguishment of debt',text,re.I|re.S)
    if fee is None or gain is None or _date(fee[1])!=observed_end or _date(gain[1])!=observed_end:return None
    fee_value=_number(fee[2]);gain_value=_number(gain[2])
    if fee_value is None or gain_value is None or min(fee_value,gain_value)<0:return None
    full_table_end=header.end()+total.end()
    annual=sum(r['principal_source_millions']*r['coupon_percent']/100 for r in tranches)
    # Preserve adjacent qualifications without treating them as numeric rows.
    # They govern whether the contract remains outstanding or earning assets
    # remain available; dropping them would encourage a gross-only forecast.
    cash_context_end=cash.end()
    maturity_clause=re.search(r'All interest-bearing securities.{0,200}?mature in one year or less\.',text[cash.end():cash.end()+1200],re.I|re.S)
    if maturity_clause is not None:cash_context_end=cash.end()+maturity_clause.end()
    note_conditions=None
    following=text[full_table_end:full_table_end+2600]
    if re.match(r'\s*If the consummation of the proposed acquisition',following,re.I):
        next_paragraph=re.search(r'\bIn December\s+20\d{2},\s*in connection',following,re.I)
        if next_paragraph is not None:
            note_conditions=_span(doc,full_table_end,full_table_end+next_paragraph.start(),'note_redemption_and_acquisition_conditions')
    source_contexts=[_span(doc,header.start(),full_table_end,'complete_new_note_contract_table'),
        _span(doc,issuance.start(),issuance.end(),'amount_matched_issuance_date'),
        _span(doc,cash.start(),cash_context_end,'dated_interest_bearing_asset_offset'),
        _span(doc,fee.start(),fee.end(),'historical_terminated_financing_cost'),
        _span(doc,gain.start(),gain.end(),'historical_debt_extinguishment_flow')]
    if note_conditions is not None:source_contexts.append(note_conditions)
    return {'measurement_period_end':observed_end.isoformat(),'measurement_period_months':3,
        'source_amount_unit':'millions of source-denominated dollars ($)',
        'currency_independently_verified':False,'consolidation_scope_independently_verified':False,
        'observed_issuance_date':issued.isoformat(),'principal_total_source_millions':stated_total,'tranches':tranches,
        'interest_bearing_assets':{'stock_date':asset_date.isoformat(),'cash_equivalents_source_millions':cash_now,
            'marketable_securities_source_millions':market_now,'total_source_millions':total_now,
            'previous_stock_date':comparison_date.isoformat(),'previous_total_source_millions':total_old,
            'future_cash_yield_known':False,'future_balance_known':False},
        'historical_flows':[{'kind':'terminated_bridge_financing_cost','amount_source_millions':fee_value,
            'period_end':observed_end.isoformat(),'period_months':3,'source_income_location':'primarily Other income (expense), net',
            'future_recurrence_assumed':False},{'kind':'debt_extinguishment_'+gain[3].lower(),
            'amount_source_millions':gain_value,'period_end':observed_end.isoformat(),'period_months':3,
            'future_recurrence_assumed':False}],
        'derived_contract_arithmetic':{'annual_stated_coupon_run_rate_source_millions':annual,
            'formula':'sum(observed principal * stated annual coupon /100)',
            'meaning':'Contract coupon rate on the observed new principal, conditional on remaining outstanding; not future GAAP net interest expense.',
            'observed_source_quote':False,'forecast_EPS_returned':False},
        'source_contexts':source_contexts,
        'limitations':['Only this complete disclosure grammar is measured; no absent contract rows are filled.',
            'Maturity years are not exact maturity dates, and stated coupons are not effective-interest rates.',
            'No day-count, next-quarter allocation, cash reinvestment yield, tax, hedge result or future shares is inferred.',
            'Historical gain and financing cost are separate flows; the parser does not automatically remove them from GAAP.',
            'A contractual stock/run-rate is not the realized future target or a prediction interval.']}


def build_contract_flow_measurements(task,entity,corpus):
    """Return one complete source-bound bundle, otherwise None.

    Caller may add each source_context through its existing exact-span source
    ID registry. Measurement records and derived arithmetic are never citations
    themselves, and this helper supplies no submission forecast or label.
    """
    if not isinstance(task,dict)or not isinstance(entity,dict):return None
    if 'eps'not in str(task.get('target',{}).get('name','')).lower():return None
    try:cutoff=dt.date.fromisoformat(task['cutoff_date']).isoformat()
    except (KeyError,TypeError,ValueError):return None
    eid=entity.get('entity_id')
    if not isinstance(eid,str):return None
    target_match=re.search(r'\b(20\d{2}-\d{2}-\d{2})\b',str(entity.get('quarter_reported','')))
    try:target_end=dt.date.fromisoformat(target_match[1])if target_match else None
    except ValueError:return None
    if target_end is None:return None
    bundles=[]
    for doc in corpus.docs.values():
        if doc.shared or eid not in doc.entity_ids or not isinstance(doc.text,str):continue
        try:published=dt.date.fromisoformat(doc.date).isoformat()
        except (TypeError,ValueError):continue
        if published>cutoff:continue
        item=_document_bundle(doc,cutoff,target_end)
        if item is not None:
            item.update(entity_id=eid,doc_id=doc.doc_id,doc_date=doc.date,
                        task_cutoff=cutoff,target_period_literal=entity['quarter_reported'],
                        task_currency_literal=entity.get('currency'),manifest_entity_scope_verified=True)
            bundles.append(item)
    if len(bundles)!=1:return None
    return {'family':'observed_contract_flow_measurement','status':'one_complete_supported_disclosure_bundle',
            'observed_bundle':bundles[0],'forecast_returned':False,'label_returned':False,
            'source_bound_only':True,'citable_as_a_generated_claim':False}


def enrich_contract_flow_information(task,entity,corpus,packet):
    """Atomically add complete original contract/offset sources, never EPS.

    A failure to establish any required source leaves the existing packet
    unchanged. The parsed annual coupon is disclosed arithmetic and cannot
    itself be emitted as a quoted source claim.
    """
    import copy
    from .corpus import Chunk
    from .packets import add_source
    measured=build_contract_flow_measurements(task,entity,corpus)
    if measured is None:return packet
    enriched=copy.deepcopy(packet)
    previous_ids={card['id'] for card in packet['cards']}
    contexts=measured['observed_bundle']['source_contexts']
    for context in contexts:
        identifiers=add_source(enriched,Chunk(context['doc_id'],context['span_start'],
            context['span_end'],context['text']),corpus,task,entity,role='contract_flow_measurement')
        if not identifiers:raise ValueError('contract_flow_original_source_unavailable')
        context['source_ids']=identifiers
        context['dependency_role']='contract_flow_measurement'
    measured['interpretation_instruction']=(
        'These are observed pre-cutoff contractual stocks, timing and historical flows. '
        'The annual coupon calculation is conditional on principal remaining outstanding. '
        'Future cash balances/yield, effective interest, hedge results, taxes, shares and '
        'recurrence are unknown. Evaluate offsets and period exposure jointly; do not '
        'subtract gross coupon directly from EPS or automatically remove a historical flow.')
    measured['source_cards_added']=[card['id'] for card in enriched['cards'] if card['id'] not in previous_ids]
    enriched['observations']['contract_flow_measurement']=measured
    return enriched
