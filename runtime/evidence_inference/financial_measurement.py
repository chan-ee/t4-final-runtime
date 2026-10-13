"""Source-bound financial cells and historical EPS identities, never forecasts.

No issuer, source-offset, target-answer or fitted-parameter lookup is stored.
Values keep their source sign, column period and semantic qualifications.
Arithmetic compatibility with rounded EPS does not prove the numerator's role.
"""
from __future__ import annotations
import math
import re
import json
import copy
from .typed_eps import historical_eps_facts,_periods,_normal,_AMOUNT
from .target_context import _candidates

_NUMBER=re.compile(r'\(?\s*[-−+]?\s*(?:\d[\d,]*(?:\.\d+)?|\.\d+)\s*\)?|[—–]')
_FOOTNOTE=re.compile(r'\s*\(\s*[a-z]\s*\)\s*',re.I)
_FAMILY=re.compile(r'(?:earnings|loss|income)(?:\s*\(loss\))?\s+per\s+(?:common\s+)?share|per common share (?:data|information)|(?:weighted[- ]average|average)\s+(?:number of\s+)?(?:diluted\s+)?(?:common\s+)?shares|shares used in.{0,35}(?:calculation|earnings)|share count information',re.I)
_LABELS=[
 ('net_interest_income',r'(?:total\s+)?net interest (?:income|revenue)'),
 ('noninterest_income',r'(?:total\s+)?non[- ]?interest (?:income|revenue)'),
 ('total_other_income',r'total other income'),
 ('fee_income',r'(?:total\s+)?fee income'),
 ('revenue_component',r'asset management and brokerage|capital markets and advisory|card and cash management|lending and deposit services|residential and commercial mortgage|investment banking fees|asset management fees|commissions and other fees|mortgage fees and related income'),
 ('revenue',r'(?:total\s+(?:net\s+)?revenues?(?:,?\s*net)?|net (?:revenues?|sales))'),
 ('provision_for_credit_losses',r'(?:total\s+)?provision for (?:credit|loan) losses'),
 ('net_chargeoffs',r'net (?:loan )?charge[- ]offs'),
 ('noninterest_expense',r'(?:total\s+)?non[- ]?interest expenses?'),
 ('operating_expense',r'total (?:costs and )?operating expenses?'),
 ('pretax_income',r'(?:income|earnings)(?:\s*\(loss\))? before (?:provision for )?(?:income )?tax(?:es| expense)(?: and (?:equity income|noncontrolling interests))?|pre[- ]tax earnings'),
 ('continuing_pretax_income',r'income from continuing operations before income taxes'),
 ('income_tax',r'(?:applicable income taxes|income tax (?:expense|provision)|provision(?:\s*\(credit\))? for income taxes|provision for taxes|income taxes|tax expense)'),
 ('interest_income',r'(?:(?:total|other)\s+)?interest income'),
 ('interest_expense',r'(?:(?:total|other|net)\s+)?interest expense(?: and amortization of debt discount|,?\s*net)?'),
 ('operating_income',r'operating income(?:\s*\(loss\))?'),
 ('cost_of_sales',r'(?:total\s+)?cost of (?:sales|(?:product|service) revenues?|products and services sold)'),
 ('gross_profit',r'gross profit'),
 ('acquisition_amortization',r'amortization of acquisition[- ]related intangibles'),
 ('restructuring_expense',r'restructuring(?: and (?:asset related charges|other costs))?(?:\s*[-–]\s*net)?'),
 ('preferred_dividends',r'preferred(?: stock)? dividends(?: and other)?'),
 ('preferred_conversion_addback',r'preferred stock dividends due to assumed conversions'),
 ('preferred_redemption_accretion',r'preferred stock discount accretion and redemptions'),
 ('participating_allocation',r'(?:dividends and undistributed earnings|earnings|net income) (?:attributable|allocated) to (?:participating securities|participating stock awards|nonvested restricted shares)'),
 ('attributed_income',r'(?:net\s+)?(?:income|earnings)(?:\s*\(loss\))?\s+(?:attributable|applicable|allocated|available)\s+(?:to|for)\s+[^$\d]{1,150}?'),
 ('net_income',r'net (?:income|earnings)(?:\s*\(loss\))?'),
 ('shares',r'(?:total\s+|adjusted\s+)?(?:diluted\s+)?(?:weighted[- ]average|average)\s+[^$\d]{0,90}?shares[^$\d]{0,100}?'),
 ('shares',r'(?:basic|diluted)\s+weighted[- ]average\s+(?:common\s+)?shares(?: outstanding)?'),
 ('bare_family_row',r'\b(?:Basic|Diluted|Assuming dilution)\b'),
]
# A label must actually be followed by a financial number, not prose.
_ROW_PATTERNS=[(kind,re.compile(r'(?<![A-Za-z])(?:'+label+r')(?=\s*(?:\([^)]{1,14}\)\s*)?\$?\s*(?:\(?\s*[-−+]?\s*(?:\d|\.\d)|[—–]))',re.I)) for kind,label in _LABELS]


def _cells(text,start,count):
    cursor=start;footnote=_FOOTNOTE.match(text,cursor)
    if footnote:cursor=footnote.end()
    # Unit annotations attached to a row are metadata, not numeric footnotes.
    unit=re.match(r'\s*\(\s*(?:in\s+)?(?:millions|thousands)\s*\)\s*',text[cursor:],re.I)
    if unit:cursor+=unit.end()
    result=[]
    for _ in range(count):
        ws=re.match(r'[\s$]*',text[cursor:]);cursor+=ws.end()
        match=_NUMBER.match(text,cursor)
        if match is None:return None
        raw=match.group().strip()
        if raw in ('—','–'):value=None;precision=None
        else:
            negative=raw.startswith(('(','-','−'))
            cleaned=re.sub(r'[\s(),+−-]','',raw)
            try:value=float(cleaned)*(-1 if negative else 1)
            except ValueError:return None
            if not math.isfinite(value):return None
            precision=len(cleaned.split('.')[1]) if '.' in cleaned else 0
        result.append({'value':value,'printed':raw,'start':match.start(),'end':match.end(),'decimal_places':precision})
        cursor=match.end()
    return result


def _active_family(text,start):
    matches=list(_FAMILY.finditer(text,max(0,start-650),start))
    if not matches:return None
    last=matches[-1].group().lower()
    return 'shares' if ('average' in last or 'shares used' in last or 'share count' in last) else 'eps'


def _classify(kind,label,text,start):
    low=label.lower();family=_active_family(text,start)
    if kind=='bare_family_row':
        if family!='shares':return None
        return 'diluted_shares' if low in ('diluted','assuming dilution') else 'basic_shares'
    if kind=='shares':
        if 'antidilut' in low or 'excluding' in low:return None
        if 'dilut' in low:return 'diluted_shares'
        return 'basic_shares'
    if kind=='attributed_income':
        if 'noncontrolling' in low:return 'noncontrolling_income'
        if 'participating' in low or 'restricted shares' in low:return 'participating_allocation'
        if 'common' in low:
            if 'diluted' in low:return 'diluted_common_income'
            if 'basic' in low:return 'basic_common_income'
            if 'allocated' in low:return 'allocated_common_income'
            return 'common_income'
        return 'income_attributable_named_owner'
    if kind=='net_income' and family=='eps':
        # Plain 'Net income $1.52' can be the TOTAL EPS row in a basic/diluted
        # subsection. Do not relabel it millions of dollars.
        return None
    if kind=='interest_income' and low.startswith('other'):return 'other_interest_income'
    if kind=='interest_expense' and low.startswith('other'):return 'other_interest_expense'
    if kind=='interest_expense' and ('net' in low):return 'net_interest_expense'
    return kind


def _unit(text,header_start,header_end,label_start,label_end,metric):
    # Some headings place the unit AFTER the final year, as in 13-week fiscal
    # tables. Stop at the row, not at the last year token.
    prefix=text[max(0,header_start-500):label_start]
    own=text[label_start:label_end]
    matches=list(re.finditer(r'\b(?:in|and shares in)\s+(millions|thousands)\b',prefix,re.I))
    local=list(re.finditer(r'\bin\s+(millions|thousands)\b',own,re.I))
    literal=(local[-1].group() if local else matches[-1].group() if matches else None)
    scale=1e6 if literal and 'million' in literal.lower() else 1e3 if literal else None
    return {'kind':'shares' if metric.endswith('_shares') else 'currency_amount','scale':scale,
            'literal':literal,'scale_verified_from_local_heading':scale is not None}


def _windows(facts,corpus,task,entity):
    bydoc={}
    for fact in facts:
        for source in fact.get('corroborating_contexts',[]) or [fact]:
            bydoc.setdefault(source['doc_id'],[]).append((max(0,source['header_start']-1800),source['value_end']+1900))
    allowed={doc:{'eps_headers':set(),'primary_ranges':[],'eps_ranges':[]} for doc in bydoc}
    for fact in facts:
        for source in fact.get('corroborating_contexts',[]) or [fact]:
            allowed[source['doc_id']]['eps_headers'].add(source['header_start'])
    for doc in corpus.docs.values():
        if doc.date>task['cutoff_date'] or doc.shared or entity['entity_id'] not in doc.entity_ids:continue
        candidates=_candidates(doc,task,entity)
        primary=sorted((c for c in candidates if c['role']=='historical_earnings_components'),key=lambda c:(-c['score'],c['start']))[:1]
        notes=[c for c in candidates if c['role']=='common_income_and_dilution']
        for candidate in primary+notes:
            a,b=candidate['start'],candidate['end']
            if candidate in primary:
                stop=re.search(r'consolidated\s+(?:comprehensive income statement|statements? of comprehensive|balance sheets?)',doc.text[a+300:b],re.I)
                if stop:b=a+300+stop.start()
            bydoc.setdefault(doc.doc_id,[]).append((a,b))
            state=allowed.setdefault(doc.doc_id,{'eps_headers':set(),'primary_ranges':[],'eps_ranges':[]})
            state['primary_ranges' if candidate in primary else'eps_ranges'].append((a,b))
    merged={}
    for doc,values in bydoc.items():
        out=[]
        for a,b in sorted(set(values)):
            if out and a<=out[-1][1]:out[-1]=(out[-1][0],max(b,out[-1][1]))
            else:out.append((a,b))
        merged[doc]=out
    return merged,allowed


def _extract_rows(doc,windows,allowed,task,entity,diagnostics):
    text=_normal(doc.text);rows=[];seen=set()
    for low,high in windows:
        high=min(high,len(text))
        matches=[]
        for kind,pattern in _ROW_PATTERNS:
            for match in pattern.finditer(text,low,high):matches.append((match.start(),-len(match.group()),kind,match))
        occupied=[]
        for _,_,kind,match in sorted(matches,key=lambda item:item[:3]):
            if any(a<=match.start()<b for a,b in occupied):continue
            metric=_classify(kind,match.group(),text,match.start())
            if metric is None:continue
            periods=_periods(text,match.start())
            if periods is None:continue
            columns,header_start,header_end=periods
            in_primary=any(a<=header_start<b for a,b in allowed.get('primary_ranges',[]))
            in_eps=any(a<=header_start<b for a,b in allowed.get('eps_ranges',[]))
            if not(in_primary or in_eps or header_start in allowed.get('eps_headers',set())):continue
            values=_cells(text,match.end(),len(columns))
            if not values:continue
            after=text[values[-1]['end']:values[-1]['end']+80].lstrip()
            if _NUMBER.match(after) and not re.match(r'\d{1,3}\.\s*[A-Za-z]|\(\d{1,3}\)\s*[A-Za-z]',after):
                diagnostics.append({'reason':'extra_numeric_cell_or_bare_footnote_ambiguous','doc_id':doc.doc_id,'row_start':match.start()});continue
            # The source operation (Less/Add) is retained. A negative expense
            # remains negative as printed; no silent normalization is applied.
            before=text[max(0,match.start()-14):match.start()]
            op='subtract_printed' if re.search(r'Less:\s*$',before,re.I) else 'add_printed' if re.search(r'Add(?: back)?:\s*$',before,re.I) else 'not_established'
            unit=_unit(text,header_start,header_end,match.start(),match.end(),metric)
            around=text[max(0,header_start-300):values[-1]['end']]
            if re.search(r'\bnon[ -]gaap\b|adjusted earnings',around,re.I):
                diagnostics.append({'reason':'non_gaap_table_declined','doc_id':doc.doc_id,'row_start':match.start()});continue
            heading=text[max(0,header_start-200):header_end+200]
            if re.search(r'\b(?:Canadian|Australian|Singapore|Hong Kong) dollars|\b(?:CAD|AUD|EUR|JPY|CNY|RMB|yen|euros)\b|[€£¥]',heading,re.I):
                diagnostics.append({'reason':'non_usd_or_conflicting_currency_heading','doc_id':doc.doc_id,'row_start':match.start()});continue
            table_id=f'{doc.doc_id}:{header_start}'
            for index,(period,cell) in enumerate(zip(columns,values)):
                _,end,months=period
                if end>min(doc.date,task['cutoff_date']):continue
                key=(doc.doc_id,match.start(),index,metric)
                if key in seen:continue
                seen.add(key)
                rows.append({'metric':metric,'value':cell['value'],'value_as_printed':cell['value'],
                    'printed':cell['printed'],'decimal_places':cell['decimal_places'],'unit':dict(unit),
                    'currency_from_task':entity.get('currency',entity.get('unit')),'currency_independently_verified':False,
                    'period_end':end,'period_months':months,'period_start':None,
                    'period_identity':'source table columns; calendar month-end representation if heading omits day',
                    'table_id':table_id,'column_index':index,'label':match.group(),'source_operation':op,
                    'table_context_role':'eps_calculation_context' if in_eps else'primary_income_statement' if in_primary else'reported_eps_table_context',
                    'status':'historical observed source cell; not a forecast or normalized component contribution',
                    'gaap_basis_verified':False,'consolidation_scope_verified':False,
                    'source':{'doc_id':doc.doc_id,'doc_date':doc.date,'span_start':match.start(),'span_end':values[-1]['end'],
                       'quote':doc.text[match.start():values[-1]['end']], 'cell_start':cell['start'],'cell_end':cell['end'],
                       'cell_quote':doc.text[cell['start']:cell['end']],
                       'header_start':header_start,'header_end':header_end,'header':doc.text[header_start:header_end]}})
            occupied.append((match.start(),values[-1]['end']))
    return rows


def _ratio_bounds(numerator,shares):
    n=numerator['value'];s=shares['value']
    if n is None or s is None or s<=0:return None
    if numerator['unit']['scale'] is None or shares['unit']['scale'] is None:return None
    ratio_scale=numerator['unit']['scale']/shares['unit']['scale']
    nd=.5*10**(-numerator['decimal_places']);sd=.5*10**(-shares['decimal_places'])
    if s-sd<=0:return None
    corners=[(n+dn)/(s+ds)*ratio_scale for dn in (-nd,nd) for ds in (-sd,sd)]
    return n/s*ratio_scale,min(corners),max(corners)


def _identities(rows,eps_facts):
    results=[]
    order={'diluted_common_income':0,'allocated_common_income':1,'common_income':2,
           'income_attributable_named_owner':3,'net_income':4}
    for fact in eps_facts:
        same=[r for r in rows if r['period_end']==fact['period_end'] and r['period_months']==fact['period_months']]
        nums=[r for r in same if r['metric'] in order and r['value'] is not None]
        shares=[r for r in same if r['metric']=='diluted_shares' and r['value'] is not None and r['value']>0]
        combinations=[]
        for numerator in nums:
            for denominator in shares:
                if numerator['source']['doc_id']!=denominator['source']['doc_id']:continue
                distance=abs(numerator['source']['span_start']-denominator['source']['span_start'])
                if distance>3000:continue
                same_table=numerator['table_id']==denominator['table_id']
                score=(order[numerator['metric']],numerator['table_context_role']!='eps_calculation_context',not same_table,distance)
                combinations.append((score,numerator,denominator))
        out={'period_end':fact['period_end'],'period_months':fact['period_months'],'reported_diluted_eps':fact['value'],
             'reported_eps_source':{k:fact[k] for k in ('doc_id','header_start','header_end','value_start','value_end','header','value_quote')},
             'status':'missing_numerator_or_diluted_shares','semantic_numerator_verified':False,
             'forecast_validated':False,'formula':'observed numerator / observed weighted diluted shares, respecting source scales'}
        if not combinations:results.append(out);continue
        combinations.sort(key=lambda x:x[0]);_,numerator,denominator=combinations[0]
        bounds=_ratio_bounds(numerator,denominator)
        out.update(numerator_measurement_id=numerator['measurement_id'],shares_measurement_id=denominator['measurement_id'],
                   numerator_metric=numerator['metric'],same_table=numerator['table_id']==denominator['table_id'],
                   numerator_role_evidence='explicit diluted-common-income label' if numerator['metric']=='diluted_common_income' else'common/allocation/source context; independent semantic verification remains open',
                   numerator_selection_rule='semantic label, EPS-computation context, same table, distance; never minimum EPS error')
        if bounds is None:out['status']='unit_scale_or_numeric_value_missing';results.append(out);continue
        value,lo,hi=bounds;eps=fact['value'];precisions=set()
        for printed in _AMOUNT.finditer(fact['value_quote']):
            raw=printed.group().strip();negative=raw.startswith(('(','-'))
            cleaned=re.sub(r'[\s(),+-]','',raw)
            if math.isclose(float(cleaned)*(-1 if negative else 1),eps,rel_tol=0,abs_tol=1e-12):
                precisions.add(len(cleaned.split('.')[1]))
        if len(precisions)!=1:
            out['status']='reported_eps_precision_ambiguous';results.append(out);continue
        eps_half=.5*10**(-next(iter(precisions)))
        compatible=max(lo,eps-eps_half)<=min(hi,eps+eps_half)
        out.update(status='rounding_compatible' if compatible else'identity_mismatch',
                   computed_historical_eps=value,source_rounding_ratio_bounds=[lo,hi],
                   reported_eps_rounding_bounds=[eps-eps_half,eps+eps_half],absolute_difference=abs(value-eps),
                   warning='Rounded identity compatibility does not establish two-class/conversion allocations, GAAP basis or issuer consolidation scope.')
        results.append(out)
    return results


def _guidance(doc,task,entity):
    """Explicit published percentage/range clauses; no quarterly allocation."""
    results=[];seen=set()
    metric_pattern=r'(?P<metric>core noninterest expense|noninterest expense|net interest income(?:,?\s+and net interest income excluding [A-Za-z]+,?)?|fee income|other noninterest income|noninterest income|net loan charge-offs|(?:GAAP )?effective tax rate|adjusted expense)'
    patterns=[
        ('percent_change',re.compile(metric_pattern+r'\s*,?\s*(?:to be|will be|is expected to be)\s+(?P<direction>up|down)\s+(?:approximately\s+)?(?P<a>\d+(?:\.\d+)?)\s*%(?:\s+to\s+(?:(?P<b>\d+(?:\.\d+)?)\s*%|(?P<stable>stable|flat)))?',re.I)),
        ('amount',re.compile(metric_pattern+r'\s*,?\s*(?:to be|will be|is expected to be)\s+(?:approximately\s+)?\$\s*(?P<a>\d+(?:\.\d+)?)\s*(?P<scale>million|billion)(?:\s+to\s+\$?\s*(?P<b>\d+(?:\.\d+)?)\s*(?:million|billion))?',re.I)),
        ('rate_level',re.compile(metric_pattern+r'(?:\s+in\s+(?P<guide_year>20\d{2}))?\s+(?:to be|will be|is expected to be)\s+(?:approximately\s+|between\s+)?(?P<a>\d+(?:\.\d+)?)\s*%(?:\s+(?:and|to)\s+(?P<b>\d+(?:\.\d+)?)\s*%)?',re.I)),
    ]
    contexts=[c for c in _candidates(doc,task,entity) if c['role'] in ('target_quarter_guidance','annual_financial_guidance','future_cost_or_tax')]
    for context in contexts:
        segment=_normal(doc.text[context['start']:context['end']])
        for kind,pattern in patterns:
            for match in pattern.finditer(segment):
                start=context['start']+match.start();end=context['start']+match.end()
                if (start,end) in seen:continue
                seen.add((start,end));groups=match.groupdict();a=float(groups['a']);b=float(groups['b']) if groups.get('b') else a
                if groups.get('stable'):b=0.
                if groups.get('direction')=='down':a=-a;b=-b
                scale=1e9 if groups.get('scale','').lower()=='billion' else 1e6 if groups.get('scale','').lower()=='million' else 1.
                prefix=segment[:match.start()]
                # 'compared to the second quarter' is the BASE, not the target.
                horizons=list(re.finditer(r'for\s+(?:the\s+)?(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?20\d{2}|(?:for the )?full[- ]year\s+20\d{2}',prefix,re.I))
                local_prefix=prefix[horizons[-1].start():] if horizons else prefix
                base=re.search(r'compared to\s+(?:the\s+)?(?P<period>(?:first|second|third|fourth)\s+quarter\s+(?:of\s+)?20\d{2}|full[- ]year\s+(?:of\s+)?20\d{2})',local_prefix,re.I)
                horizon=horizons[-1].group() if horizons else ('in '+groups['guide_year'] if groups.get('guide_year') else None)
                if horizon is None:
                    annual=list(re.finditer(r'20\d{2}\s+annual',prefix[-100:],re.I))
                    if annual:horizon=annual[-1].group()
                results.append({'metric_literal':groups['metric'],'measurement_kind':kind,'lower':min(a,b),'upper':max(a,b),
                    'unit':'percentage_change' if kind=='percent_change' else 'percent_rate_level' if kind=='rate_level' else 'USD_amount',
                    'unit_scale':scale,'horizon_literal':horizon,
                    'base_period_literal':base.group('period') if base else None,
                    'base_relation':'percentage_change_of_named_base' if kind=='percent_change' else'literal_context_only; no change formula implied',
                    'approximate_word_present':bool(re.search(r'approximately',match.group(),re.I)),
                    'range_semantics':'management_range_not_probability_interval' if groups.get('b') or groups.get('stable') else'management_point_guide_with_unmodeled_tolerance',
                    'source_context_role':context['role'],'period_basis_unknown':True,'base_period_automatically_verified':False,
                    'gaap_basis_verified':False,'consolidation_scope_verified':False,'forecast_or_guidance':'published management guidance; not realized or selected agent forecast',
                    'probability_coverage':None,'quarterly_allocation_performed':False,
                    'source':{'doc_id':doc.doc_id,'doc_date':doc.date,'span_start':start,'span_end':end,'quote':doc.text[start:end],
                              'context_start':context['start'],'context_end':context['end'],'context':doc.text[context['start']:context['end']]}})
        schedule=re.search(r'estimated annual amortization expense.{0,160}follows\s*:\s*(?P<table>.+?)(?:\bTotal\b|\bNOTE\s+\d)',segment,re.I)
        if schedule:
            table=schedule.group('table');unit=re.search(r'\bin\s+(millions|thousands)\b',table,re.I)
            if unit:
                scale=1e6 if unit.group(1).lower()=='millions' else 1e3
                for match in re.finditer(r'(?P<period>Remainder of 20\d{2}|20\d{2}(?:\s+and thereafter)?)\s+\$?\s*(?P<value>\d[\d,]*(?:\.\d+)?)',table,re.I):
                    start=context['start']+schedule.start('table')+match.start();end=context['start']+schedule.start('table')+match.end()
                    if (start,end) in seen:continue
                    seen.add((start,end));value=float(match.group('value').replace(',',''))
                    results.append({'metric_literal':'acquisition-related intangible amortization schedule',
                        'measurement_kind':'published_cost_schedule','lower':value,'upper':value,
                        'unit':'USD_amount','unit_scale':scale,'horizon_literal':match.group('period'),
                        'base_period_literal':None,'base_relation':'named future fiscal period; not a quarter allocation',
                        'source_context_role':'future_cost_or_tax','period_basis_unknown':True,'base_period_automatically_verified':False,
                        'gaap_basis_verified':False,'consolidation_scope_verified':False,
                        'forecast_or_guidance':'published conditional cost schedule; not realized or selected agent forecast',
                        'range_semantics':'management_point_schedule_with_unmodeled_tolerance',
                        'probability_coverage':None,'quarterly_allocation_performed':False,
                        'source':{'doc_id':doc.doc_id,'doc_date':doc.date,'span_start':start,'span_end':end,
                            'quote':doc.text[start:end],'context_start':context['start'],'context_end':context['end'],
                            'context':doc.text[context['start']:context['end']]}})
    return results


def _aggregates(rows):
    """Only explicit historical table totals; semantic fee naming stays open."""
    groups={};result=[]
    for row in rows:groups.setdefault((row['table_id'],row['column_index']),[]).append(row)
    for group in groups.values():
        totals=[r for r in group if r['metric']=='noninterest_income' and r['label'].lower().startswith('total')]
        others=[r for r in group if r['metric']=='total_other_income']
        if len(totals)!=1 or len(others)!=1:continue
        a,b=totals[0],others[0]
        if a['value'] is None or b['value'] is None or a['unit']!=b['unit']:continue
        components=[r for r in group if r['metric']=='revenue_component' and r['value'] is not None]
        value=a['value']-b['value'];component_sum=sum(r['value'] for r in components)
        result.append({'metric':'reported_noninterest_income_less_total_other_income','value':value,'unit':a['unit'],
            'period_end':a['period_end'],'period_months':a['period_months'],'table_id':a['table_id'],
            'formula':'reported total noninterest income - reported total other income',
            'source_measurement_ids':[a['measurement_id'],b['measurement_id']],
            'selected_component_measurement_ids':[r['measurement_id'] for r in components],
            'selected_component_sum':component_sum,'sum_matches_reported_difference':math.isclose(component_sum,value,abs_tol=1e-8),
            'fee_interpretation_requires_source_definition':True,'status':'derived historical amount, not a source quote or future guidance realization'})
    return result


def build_financial_measurement(task,entity,corpus):
    """Return conservative financial measurements for EPS tasks, else None."""
    if 'eps' not in str(task.get('target',{}).get('name','')).lower():return None
    currency=re.sub(r'\s+','',str(entity.get('currency',entity.get('unit',''))).lower())
    if currency not in ('usd/share','usd/pershare','usdpershare','usd/commonshare','usdpercommonshare','usd/dilutedshare'):
        return {'family':'source_financial_measurement','entity_id':entity['entity_id'],
                'status':'unsupported_or_unspecified_currency','measurements':[],'guidance':[],
                'forecast_returned':False,'label_returned':False,'calibrated_interval_returned':False,
                'reason':'This parser requires an explicit task USD per-share declaration; it does not relabel another reporting currency.'}
    history=historical_eps_facts(corpus,task,entity);windows,allowed=_windows(history['facts'],corpus,task,entity);rows=[];diagnostics=[];guidance=[]
    for doc in corpus.docs.values():
        if doc.date>task['cutoff_date'] or doc.shared or entity['entity_id'] not in doc.entity_ids:continue
        rows.extend(_extract_rows(doc,windows.get(doc.doc_id,[]),allowed.get(doc.doc_id,{}),task,entity,diagnostics))
        guidance.extend(_guidance(doc,task,entity))
    rows.sort(key=lambda r:(r['source']['doc_id'],r['source']['span_start'],r['column_index']))
    entity_index=next((i for i,e in enumerate(task.get('entities',[])) if e.get('entity_id')==entity['entity_id']),0)
    for i,row in enumerate(rows):row['measurement_id']=f'E{entity_index}M{i}'
    identities=_identities(rows,history['facts'])
    latest=max((f['period_end'] for f in history['facts'] if f['period_months']==3),default=None)
    latest_rows=[r for r in rows if r['period_end']==latest and r['period_months']==3]
    metrics=sorted({r['metric'] for r in latest_rows})
    return {'family':'source_financial_measurement','entity_id':entity['entity_id'],
        'status':'observations_available' if rows else'no_unambiguous_financial_cells',
        'latest_observed_quarter_end':latest,'measurements':rows,'guidance':guidance,'historical_eps_identities':identities,
        'derived_historical_aggregates':_aggregates(rows),
        'latest_quarter_metrics_present':metrics,
        'latest_quarter_metrics_missing':[m for m in ('pretax_income','income_tax','net_income','diluted_shares') if m not in metrics],
        'diagnostics':diagnostics,'historical_eps_conflicts':history['conflicts'],
        'forecast_returned':False,'label_returned':False,'calibrated_interval_returned':False,'citable_as_source_claim':False,
        'limitations':['Rows retain the source sign; an expense printed negative is not silently normalized to a positive deduction.',
          'A dash is missing/unspecified, never an assumed zero.',
          'Rounded EPS reconstruction alone does not establish a correct common numerator: different allocations can round to the same EPS.',
          'NCO, provision expense and reserve movement are different quantities; no interchange is performed.',
          'FY guidance is not a quarterly amount or quarterly tax rate; fiscal/calendar mapping is unverified.',
          'Only documented local column units are scaled; missing units remain unavailable.',
          'All periods are historical source observations or explicitly published guidance; no future target outcome is read or generated.']}


def public_financial_measurement(ledger,*,max_chars=20000):
    """Compact source cells without deleting identity inputs or inventing values.

Full research ledgers remain unchanged. Optional noncurrent component cells
may be omitted with explicit diagnostics; required identity cells never are.
The size bound uses the same compact JSON encoding as runtime prompts.
"""
    if ledger is None:return None
    cap=max(1000,int(max_chars));records=ledger.get('measurements',[])
    required={i.get(key) for i in ledger.get('historical_eps_identities',[]) for key in ('numerator_measurement_id','shares_measurement_id') if i.get(key)}
    latest=ledger.get('latest_observed_quarter_end')
    out={'family':ledger.get('family'),'entity_id':ledger.get('entity_id'),'status':ledger.get('status'),
         'latest_observed_quarter_end':latest,'cell_fields':['measurement_id','column_index','value_as_printed','cell_start','cell_end','decimal_places'],
         'all_measurements_inherit':{'gaap_basis_verified':False,'consolidation_scope_verified':False,'currency_independently_verified':False,
             'currency_from_task':records[0].get('currency_from_task') if records else None,
             'source_operation_when_omitted':'not_established',
             'period_identity':'source table columns; period_start unknown; month-only headings represented at calendar month-end',
             'sign_rule':'numeric cells retain printed sign; consult source_operation before aggregating expenses'},
         'unit_definitions':{},
         'all_guidance_inherit':{'period_basis_unknown':True,'base_period_automatically_verified':False,
             'gaap_basis_verified':False,'consolidation_scope_verified':False,'probability_coverage':None,
             'quarterly_allocation_performed':False,
             'meaning':'published management guidance/schedule; not realized or selected agent forecast'},
         'all_identities_inherit':{'semantic_numerator_verified':False,'forecast_validated':False,
             'formula':'observed numerator / observed weighted diluted shares with source scales',
             'warning':'Rounding compatibility is not semantic verification.'},
         'tables':[],'guidance':[],'guidance_contexts':[],'guidance_kinds':{},'guidance_periods':{},'reported_eps_sources':[],
         'historical_eps_identities':[],'derived_historical_aggregates':[],
         'latest_quarter_metrics_missing':ledger.get('latest_quarter_metrics_missing',[]),
         'historical_eps_conflicts':ledger.get('historical_eps_conflicts',[]),
         'diagnostics':ledger.get('diagnostics',[]),
         'limitations':['Keep source signs; missing/dash is not zero.','Rounded EPS compatibility does not prove allocations, GAAP or scope.',
             'NCO is not provision; FY is not Q; fiscal mapping is unverified.','No future EPS, label or calibrated interval is returned.'],
         'forecast_returned':False,'label_returned':False,'calibrated_interval_returned':False,'citable_as_source_claim':False}
    table_lookup={};row_lookup={};units={}
    for record in records:
        source=record['source'];key=record['table_id']
        if key not in table_lookup:
            table={'table_id':key,'doc_id':source['doc_id'],'doc_date':source['doc_date'],
                'header_start':source['header_start'],'header_end':source['header_end'],'header':source['header'],
                'table_context_role':record['table_context_role'],'columns':{},'rows':[]}
            out['tables'].append(table);table_lookup[key]=table
        table=table_lookup[key];column=record['column_index']
        table['columns'][str(column)]={k:record[k] for k in ('period_end','period_months','period_start')}
        row_key=(key,source['span_start'],record['metric'])
        if row_key not in row_lookup:
            unitkey=json.dumps(record['unit'],sort_keys=True,separators=(',',':'))
            if unitkey not in units:
                units[unitkey]=f'U{len(units)}';out['unit_definitions'][units[unitkey]]=dict(record['unit'])
            row={'metric':record['metric'],'label':record['label'],'unit_ref':units[unitkey],
                 'row_span':[source['span_start'],source['span_end']],'cells':[]}
            if record['source_operation']!='not_established':row['source_operation']=record['source_operation']
            for flag in ('gaap_basis_verified','consolidation_scope_verified','currency_independently_verified'):
                if record.get(flag) is not False:row[flag]=record.get(flag)
            table['rows'].append(row);row_lookup[row_key]=row
        row_lookup[row_key]['cells'].append([record['measurement_id'],column,record['value_as_printed'],source['cell_start'],source['cell_end'],record['decimal_places']])
    eps_lookup={}
    for identity in ledger.get('historical_eps_identities',[]):
        item=copy.deepcopy(identity);source=item.pop('reported_eps_source',None)
        for key in ('formula','numerator_selection_rule','warning'):
            item.pop(key,None)
        for key in ('semantic_numerator_verified','forecast_validated'):
            if item.get(key) is False:item.pop(key,None)
        if source:
            key=(source['doc_id'],source['value_start'],source['value_end'])
            if key not in eps_lookup:
                ref=f'R{len(eps_lookup)}';eps_lookup[key]=ref
                out['reported_eps_sources'].append({'id':ref,**source})
            item['reported_eps_source_ref']=eps_lookup[key]
        out['historical_eps_identities'].append(item)
    context_lookup={};guidance_kinds={};guidance_periods={}
    for guidance in ledger.get('guidance',[]):
        item=copy.deepcopy(guidance);source=item['source'];key=(source['doc_id'],source['context_start'],source['context_end'])
        for flag in ('period_basis_unknown','base_period_automatically_verified','gaap_basis_verified','consolidation_scope_verified','probability_coverage','quarterly_allocation_performed'):
            if item.get(flag)==out['all_guidance_inherit'][flag]:item.pop(flag,None)
        item.pop('forecast_or_guidance',None)
        if key not in context_lookup:
            ref=f'G{len(context_lookup)}';context_lookup[key]=ref
            out['guidance_contexts'].append({'id':ref,'doc_id':source['doc_id'],'doc_date':source['doc_date'],
                'span_start':source['context_start'],'span_end':source['context_end'],'text':source['context']})
        item['source']={k:v for k,v in source.items() if k not in ('context_start','context_end','context','doc_id','doc_date')}
        kind_fields=('measurement_kind','unit','unit_scale','base_relation','range_semantics')
        kind={k:item.pop(k) for k in kind_fields if k in item};kindkey=json.dumps(kind,sort_keys=True)
        if kindkey not in guidance_kinds:
            guidance_kinds[kindkey]=f'K{len(guidance_kinds)}';out['guidance_kinds'][guidance_kinds[kindkey]]=kind
        period={k:item.pop(k) for k in ('horizon_literal','base_period_literal','source_context_role') if k in item}
        period['context_ref']=context_lookup[key];periodkey=json.dumps(period,sort_keys=True)
        if periodkey not in guidance_periods:
            guidance_periods[periodkey]=f'P{len(guidance_periods)}';out['guidance_periods'][guidance_periods[periodkey]]=period
        item['kind_ref']=guidance_kinds[kindkey];item['period_ref']=guidance_periods[periodkey]
        out['guidance'].append(item)
    # Keep current-quarter decomposition; full historical arrays remain in the
    # research ledger and no future value is substituted when a cell is omitted.
    out['derived_historical_aggregates']=[copy.deepcopy(x) for x in ledger.get('derived_historical_aggregates',[]) if x['period_end']==latest and x['period_months']==3]
    for aggregate in out['derived_historical_aggregates']:
        required.update(aggregate.get('source_measurement_ids',[]));required.update(aggregate.get('selected_component_measurement_ids',[]))
    stats={'full_measurement_cells':len(records),'retained_measurement_cells':len(records),
           'full_guidance_records':len(ledger.get('guidance',[])),'distinct_guidance_contexts':len(out['guidance_contexts']),
           'distinct_reported_eps_sources':len(out['reported_eps_sources']),'omitted_measurement_ids':[],
           'required_measurement_count':len(required),'max_chars':cap,'budget_exceeded':False}
    out['compaction']=stats
    def size():return len(json.dumps(out,ensure_ascii=False,allow_nan=False,separators=(',',':')))
    # Reserve space for final omission/size diagnostics added below.
    work_budget=max(0,cap-384)
    if size()>work_budget:
        # YTD components first, then noncurrent quarterly components. All
        # numerator/share references and current-quarter measurements survive.
        optional=sorted((r for r in records if r['measurement_id'] not in required and not(r['period_end']==latest and r['period_months']==3)),
                        key=lambda r:(r['period_months']==3,r['period_end'],r['source']['span_start']))
        for record in optional:
            source=record['source'];key=(record['table_id'],source['span_start'],record['metric'])
            row=row_lookup[key];row['cells']=[cell for cell in row['cells'] if cell[0]!=record['measurement_id']]
            if not row['cells']:
                table_lookup[record['table_id']]['rows']=[r for r in table_lookup[record['table_id']]['rows'] if r is not row]
            stats['omitted_measurement_ids'].append(record['measurement_id']);stats['retained_measurement_cells']-=1
            if size()<=work_budget:break
        for table in out['tables']:table['rows']=[row for row in table['rows'] if row['cells']]
    stats['omission_reason']='noncurrent_nonidentity_cells_over_budget' if stats['omitted_measurement_ids'] else None
    retained={cell[0] for table in out['tables'] for row in table['rows'] for cell in row['cells']}
    stats['required_identity_and_current_aggregate_cells_preserved']=required<=retained
    # Do not silently discard indispensable fields to force a size claim.
    stats['serialized_chars']=size();stats['budget_exceeded']=size()>cap
    stats['serialized_chars']=size()
    return out
