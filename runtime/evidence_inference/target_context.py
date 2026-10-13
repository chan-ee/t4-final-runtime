"""Current-corpus financial dependencies, with no issuer or forecast lookup.

Retrieve explicit target-period guidance, its historical accounting base and
its qualifications. Original passages remain original; neither guidance nor
historical arithmetic is promoted to a known future outcome.
"""
from __future__ import annotations
import re
from .corpus import Chunk
from .packets import add_source,safe_source_chunks

_INVISIBLE='\u200b\u200c\u200d\ufeff'
_FINANCIAL=re.compile(r'net interest income|noninterest income|fee income|(?:noninterest|adjusted) expense|net loan charge.off|gross margin|net revenues?|operating (?:income|expenses?)|earnings per share',re.I)
_ROW_LABELS=(r'net interest income',r'total (?:net )?revenues?',r'net (?:revenues?|sales)',r'total noninterest income',
             r'provision for credit losses',r'total (?:noninterest|operating) expenses?',r'income(?:\s*\(loss\))? before (?:income )?taxes',
             r'(?:provision for (?:income )?|expense for |income )taxes',r'net (?:income|earnings)(?:\s*\(loss\))?',r'preferred stock dividends',r'cost of (?:sales|revenues?)',
             r'total costs?(?:,? (?:and |expenses and |costs and )?(?:operating )?expenses?| and other)?',
             r'total expense and other\s*\(income\)',r'gross profit',r'operating income(?:\s*\(loss\))?',r'tax expense')
_STATEMENT=re.compile(r'consolidated\s+(?:(?:statements?|condensed statement)\s+of\s+(?:income|earnings|operations)|income statement)',re.I)
_NEXT_STATEMENT=re.compile(r'consolidated\s+(?:statements?\s+of\s+(?:comprehensive|cash|equity)|balance sheets?)',re.I)
_FORWARD=re.compile(r'\bexpect(?:s|ed)?\b|\bguidance\b|\boutlook\b|\bestimat(?:e|ed)\b|\bforecast',re.I)
_FISCAL_MARKER=re.compile(r'\bfiscal\s+(?:(?:first|second|third|fourth)[- ]quarter|quarter|year\s+(?:end(?:ing|ed)?|ends?))|\b(?:first|second|third|fourth)[- ]fiscal\s+quarter|\b(?:52|53)[- ]week\s+(?:fiscal\s+)?year',re.I)


def _normal(text):return text.translate(str.maketrans({x:' ' for x in _INVISIBLE}))


def _window(text,start,end):
    start=max(0,start);end=min(len(text),end)
    # Never cut through an amount or word. Keep captions/numeric table rows.
    if start and not text[start-1].isspace():
        boundary=text.rfind(' ',max(0,start-80),start)
        if boundary>=0:start=boundary+1
    if end<len(text) and not text[end:end+1].isspace():
        boundary=text.rfind(' ',max(start,end-80),end)
        if boundary>start:end=boundary
    while start<end and text[start].isspace():start+=1
    while end>start and text[end-1].isspace():end-=1
    return start,end


def _period(task,entity):
    # A release/cutoff quarter is not necessarily the accounting target quarter.
    # Even an explicit accounting period-end DATE establishes a calendar cue,
    # not the issuer's fiscal quarter number or fiscal year.
    for value in (entity.get('quarter_reported'),task.get('target_period')):
        match=re.search(r'(20\d{2})-(\d{2})-\d{2}',str(value or ''))
        if match:
            year,month=map(int,match.groups())
            if 1<=month<=12:return year,(month-1)//3+1
    return None,None


def _candidates(doc,task,entity):
    text=_normal(doc.text);year,quarter=_period(task,entity);out=[]
    def add(role,start,end,priority,score=0,period='unknown'):
        start,end=_window(doc.text,start,end)
        if end-start>=80:out.append({'doc':doc,'start':start,'end':end,'role':role,'priority':priority,'score':score,'period':period})
    if year:
        ordinal=('first','second','third','fourth')[quarter-1]
        quarter_re=re.compile(r'\b(?:for\s+(?:the\s+)?)?(?:'+ordinal+r'[- ]quarter|Q'+str(quarter)+r')\s+(?:of\s+)?'+str(year)+r'\b',re.I)
        for match in quarter_re.finditer(text):
            nearby=text[max(0,match.start()-180):match.start()+1150]
            if _FORWARD.search(nearby) and len(set(m.group().lower() for m in _FINANCIAL.finditer(nearby)))>=2:
                # Includes the nearby annual outlook and basis qualifications.
                add('target_quarter_guidance',match.start()-150,match.start()+2700,0,20,'calendar_ordinal_cue_matched; fiscal_mapping_unverified')
        annual=re.compile(r'\b(?:full[- ]year\s+(?:of\s+)?'+str(year)+r'|for\s+(?:the\s+year\s+)?'+str(year)+r')\b',re.I)
        for match in annual.finditer(text):
            nearby=text[max(0,match.start()-120):match.start()+850]
            if _FORWARD.search(nearby) and len(set(m.group().lower() for m in _FINANCIAL.finditer(nearby)))>=2:
                end=match.start()+1400
                stop=re.search(r'\bBusiness Developments\b|\bRegulatory developments\b',text[match.start()+200:end],re.I)
                if stop:end=match.start()+200+stop.start()
                first=_FINANCIAL.search(nearby)
                score=50-min(first.start()/30,40)+2*len(set(m.group().lower() for m in _FINANCIAL.finditer(nearby)))
                add('annual_financial_guidance',match.start()-100,end,1,score,'date_year_cue_matched; fiscal_year_unknown; quarterly_allocation_unknown')
    for match in _STATEMENT.finditer(text):
        end=min(len(text),match.start()+3600)
        stop=_NEXT_STATEMENT.search(text,match.end()+100,end)
        if stop:end=stop.start()
        section=text[match.start():end]
        hits=sum(bool(re.search(label+r'\s*\$?\s*\(?\s*-?\d',section,re.I)) for label in _ROW_LABELS)
        if hits>=4 and section.count('$')>=2 and re.search(r'three months|quarter ended',section,re.I):
            # Registrant/subsidiary names often precede the statement caption.
            add('historical_earnings_components',match.start()-200,end,2,hits,'historical_table; verify_columns')
    # Two-class allocation and actual denominator, not a bare diluted EPS row.
    note=re.compile(r'\bnote\s*\d+\s*[–—.:-]?\s*(?:earnings\s+per\s+(?:common\s+)?share|income\s+per\s+share)',re.I)
    for match in note.finditer(text):
        section=text[match.start():match.start()+2500]
        if re.search(r'diluted',section,re.I) and re.search(r'weighted.average|average common shares',section,re.I):
            # A body cross-reference such as 'see Note 13' is not an end header.
            end=match.start()+2500
            number=re.search(r'\d+',match.group()).group()
            for nxt in re.finditer(r'\b(?:Note|NOTE)\s*(\d+)\s*[.—–:-]?\s+(?=[A-Z])',section[500:]):
                if nxt.group(1)!=number:
                    end=match.start()+500+nxt.start();break
            score=8+4*bool(re.search(r'participating|nonvested|preferred',section,re.I))
            add('common_income_and_dilution',match.start(),end,3,score,'historical_table; verify_columns')
    for match in re.finditer(r'(?:net income|net earnings)\s+(?:attributable|applicable)\s+to\s+(?:diluted\s+)?common\s+(?:shareholders|stockholders)',text,re.I):
        section=text[max(0,match.start()-900):match.start()+1300]
        if re.search(r'weighted.average|average common shares',section,re.I) and re.search(r'participating|nonvested',section,re.I):
            add('common_income_and_dilution',match.start()-900,match.start()+1300,3,11,'historical_table; verify_columns')
    # An explicit future schedule/condition is useful even without EPS guidance.
    patterns=(
      ('future_cost_or_tax',r'(?:\bexpect(?:s|ed)?\b.{0,70}(?:GAAP )?effective tax rate.{0,100}\d\s*(?:%|percent)|\bexpect(?:s|ed)?\b.{0,70}additional costs|estimated annual amortization expense|estimated amortization.{0,100}remaining)',1000,1050,4),
      ('forward_operating_condition',r'(?:expect\w*.{0,80}lower future net interest income|Investment banking fees backlog|To the extent these conditions continue)',350,1150,5),
      ('capital_plan_and_execution_condition',r'expects? to repurchase.{0,90}(?:common shares|common stock)|timing and amount of common stock repurchases are subject',180,750,6),
      ('current_operating_driver',r'Net interest income decreased.{0,100}(?:million|billion)|(?:increase|decrease) in (?:net )?interest expense.{0,120}(?:primarily due|driven by)|Net interest expense increased.{0,100}(?:million|billion)',140,800,7),
    )
    for role,pattern,before,after,priority in patterns:
        for match in re.finditer(pattern,text,re.I):
            end=match.start()+after
            if role=='future_cost_or_tax':
                stop=re.search(r'\bSeg\s*ment Assets(?: and Operating Results)?\b',text[match.end():end])
                if stop:end=match.end()+stop.start()
            add(role,match.start()-before,end,priority,5,'period_and_basis_require_verification')
    return out


def enrich_target_information(task,entity,corpus,packet,*,max_additional_chars=9000):
    """Add source-bound dependencies once; do not select an EPS or direction."""
    if 'target_information' in packet.setdefault('observations',{}):return packet
    target=str(task.get('target',{}).get('name','')).lower()
    if 'eps' not in target and 'earnings_per_share' not in target:return packet
    budget=max(0,min(9000,int(max_additional_chars)))
    eligible=[d for d in corpus.docs.values() if d.date<=task['cutoff_date'] and entity['entity_id'] in d.entity_ids]
    fiscal_docs={d.doc_id for d in eligible if _FISCAL_MARKER.search(_normal(d.text))}
    candidates=[item for doc in eligible for item in _candidates(doc,task,entity)]
    candidates.sort(key=lambda x:(x['priority'],-x['score'],-int(x['doc'].date.replace('-','')),x['doc'].doc_id,x['start']))
    selected=[];roles=set();used=0;skipped=[];contexts=[];new_card_start=len(packet['cards'])
    for candidate in candidates:
        doc=candidate['doc'];a=candidate['start'];b=candidate['end'];role=candidate['role']
        if role in roles:continue
        overlaps=[x for x in selected if x['doc'].doc_id==doc.doc_id and min(b,x['end'])>max(a,x['start'])]
        if any(max(a,x['start'])<=a+30 and min(b,x['end'])>=b-30 for x in overlaps):
            roles.add(role);continue
        # A nearly duplicate context does not consume a second role's budget.
        if any(min(b,x['end'])-max(a,x['start'])>.7*(b-a) for x in overlaps):continue
        chunks=list(safe_source_chunks(Chunk(doc.doc_id,a,b,doc.text[a:b]),doc,task['cutoff_date'],entity.get('latest_published_ref_month')))
        size=sum(len(c.text) for c in chunks)
        if used+size>budget:
            skipped.append({'role':role,'reason':'whole_context_exceeds_remaining_budget','chars':size,'remaining':budget-used});continue
        added=False
        for chunk in chunks:
            if not chunk.text.strip():continue
            source_ids=add_source(packet,chunk,corpus,task,entity,role='target_'+role)
            if not source_ids:continue
            contexts.append({'doc_id':doc.doc_id,'doc_date':doc.date,'span_start':chunk.start,'span_end':chunk.end,
                'text':chunk.text,'source_ids':source_ids,'dependency_role':role,
                'status':'original pre-cutoff disclosure; future values, if present, are published guidance or conditions, not realized outcomes',
                'period_status':candidate['period'],'period_basis_unknown':True,
                'source_fiscal_calendar_terms_present':doc.doc_id in fiscal_docs,
                'accounting_basis_verified':False,'entity_consolidation_verified':False,
                'contains_forward_looking_language':bool(_FORWARD.search(chunk.text))})
            used+=len(chunk.text);added=True
        if added:selected.append(candidate);roles.add(role)
    found={x['role'] for x in candidates};missing=sorted(found-roles)
    year,quarter=_period(task,entity)
    data={'original_contexts':contexts,'additional_original_chars':used,'max_additional_chars':budget,
          'quarter_mapping':{'calendar_date_cue':None if year is None else f'{year}-Q{quarter}',
              'fiscal_quarter_verified':False,'fiscal_year_verified':False,
              'source_fiscal_calendar_marker_docs':sorted(fiscal_docs),
              'status':'retrieval cue only; date-derived calendar ordinal is not an established fiscal period mapping'},
          'period_warnings':['Source uses fiscal-period/calendar terminology. Verify the issuer calendar before applying an ordinal-quarter or FY guide; a marker alone does not establish a mismatch.'] if fiscal_docs else [],
          'new_card_text_chars':sum(len(c['text']) for c in packet['cards'][new_card_start:]),
          'original_text_also_appears_in_source_cards':True,
          'detected_dependency_roles':sorted(found),'retained_dependency_roles':sorted(roles),
          'detected_but_omitted_roles':missing,'budget_diagnostics':skipped,
          'required_measurements_not_established':[role for role in ('historical_earnings_components','common_income_and_dilution') if role not in roles],
          'scope':'only documents explicitly assigned to this entity and dated by cutoff; no shared-amount assignment',
          'calculation_performed':False,'forecast_selected':False,
          'checks':['Match the target quarter, historical comparison columns and GAAP/consolidation scope.',
                    'A period-end date does not establish the issuer fiscal quarter or fiscal year. Verify source accounting-calendar definitions before applying guidance.',
                    'Net charge-offs are not the provision: reserve changes and other provision adjustments remain.',
                    'Annual guidance is not a quarterly value; an annual tax rate is not necessarily the quarter tax rate.',
                    'Use the common diluted-income numerator including preferred and participating allocations, and weighted diluted shares.',
                    'Guidance exclusions and cost schedules do not authorize removing recurring GAAP expenses.',
                    'Do not treat guidance midpoints or endpoint ranges as conditional medians or calibrated prediction intervals.']}
    packet['observations']['target_information']=data
    if missing:packet.setdefault('warnings',[]).append('target_information: detected dependencies omitted within bounded context: '+', '.join(missing))
    return packet
