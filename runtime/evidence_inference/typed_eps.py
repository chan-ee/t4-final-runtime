"""Conservative, source-bound historical EPS observations, never forecasts.

No issuer identifiers, fixed financial values, target outcomes, or model weights
are stored here. Ambiguous prose, percent-change columns, and multi-date header
layouts are refused. Parsed figures are historical observations, not forecasts.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import datetime as dt
import math
import re


_INVISIBLE = "\u200b\u200c\u200d\ufeff"
_AMOUNT = re.compile(r"(?<![\d.])\(?\s*[-+]?\s*(?:\d[\d,]*)?\.\d+\s*\)?")
_HEADER = re.compile(r"(?:(?:three|six|nine|twelve)\s+months|quarter)\s+ended\b", re.I)
_ANCHOR = re.compile(
    r"diluted\s+(?:earnings|loss)(?:\s*\(loss\))?\s+per\s+(?:common\s+)?share"
    r"|earnings\s+per\s+diluted\s+(?:common\s+)?share"
    r"|diluted\s+EPS\b"
    r"|(?:net\s+income|earnings(?:\s*\(loss\))?)\s+per\s+(?:common\s+)?share\s*[—–-]\s*diluted"
    r"|earnings\s+per\s+share\s+of\s+common\s+stock.{0,4}assuming\s+dilution"
    r"|assuming\s+dilution\b"
    r"|\bdiluted\b",
    re.I,
)
_MONTHS = {x.lower():i for i,x in enumerate(
    ("", "January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")) if x}
_MONTHS.update({name[:3]:number for name,number in list(_MONTHS.items())})
_DATE = re.compile(r"\b("+"|".join(sorted(_MONTHS,key=len,reverse=True))+r")\b\.?\s*(\d{1,2}(?!\d))?",re.I)


@dataclass(frozen=True)
class EPSFact:
    entity_id: str
    metric: str
    value: float
    unit: str
    period_start: str | None
    period_end: str
    period_months: int
    accounting_basis: str
    consolidation_scope: str
    doc_id: str
    doc_date: str
    header_start: int
    header_end: int
    value_start: int
    value_end: int
    header: str
    value_quote: str


def _normal(text):
    # Keep length exactly unchanged so original document offsets remain valid.
    return text.translate(str.maketrans({x:" " for x in _INVISIBLE}))


def _row(text, anchor):
    """Return numeric EPS-row span, refusing descriptive prose and share counts."""
    start, end = anchor.span()
    name = anchor.group().lower()
    before = text[max(0,start-500):start].lower()
    if re.search(r"(?:shares?|denominator|stock\s+equivalents)\s+(?:used\s+(?:in|for)\s+|for\s+|in\s+)(?:(?:the\s+)?calculation\s+of\s+)?(?:basic\s+(?:and|/)\s+)?$",before):
        # Explicit EPS words also appear in denominator labels, for example
        # 'Weighted-average shares for diluted EPS 538.3 551.5'.
        return None
    if name == "diluted":
        # The word 'Basic' alone is not an EPS heading: decimal weighted-average
        # share counts also have Basic/Diluted rows. Require the active row family
        # to be per-share earnings, with no later share-count heading.
        eps_family=list(re.finditer(r"(?:earnings|loss|net\s+income)(?:\s*\(loss\))?\s+per\s+(?:common\s+)?share|per\s+common\s+share\b|earnings\s+per\s+share\s+of\s+common\s+stock",before))
        if not eps_family:return None
        share_family=list(re.finditer(
            r"(?:weighted[-\s]+average|average|number\s+of)\s+(?:diluted\s+|basic\s+|common\s+|ordinary\s+)*(?:shares|share)"
            r"|shares?\s+(?:outstanding|used|denominator)|share\s+count|common\s+stock\s+equivalents",before))
        if share_family and share_family[-1].start()>eps_family[-1].start():return None
        # 'Shares used in earnings per share' contains the EPS phrase, but its
        # subject remains a share denominator, not an EPS value row.
        last=eps_family[-1]
        active_family=before[last.start():]
        if 'continuing operations' in active_family and not re.search(r'\btotal\b',active_family):
            # A continuing-operations EPS table is not a total EPS observation.
            # Preserve it as raw evidence rather than silently relabel its row.
            return None
        prefix=before[max(0,last.start()-100):last.start()]
        if re.search(r"(?:shares?|share\s+count).{0,65}(?:used\s+(?:in|for)|for\s+(?:basic|diluted)|calculat).{0,25}$",prefix):return None
    tail = text[end:end+600]
    if name == "assuming dilution":
        # IBM-style statements distinguish continuing EPS from TOTAL GAAP EPS.
        total = re.search(r"\bTotal\s+", tail, re.I)
        if not total or total.start()>350:return None
        numeric_start = end + total.end()
    else:
        numeric_start=end
        footnote = re.match(r"\s*\(\s*\d{1,2}\s*\)", text[numeric_start:numeric_start+15])
        if footnote:numeric_start += footnote.end()
    values=[];cursor=numeric_start
    # Only decimal per-share amounts; commas and brackets alone are not enough.
    for _ in range(8):
        whitespace = re.match(r"[\s$]*",text[cursor:cursor+100]);cursor += whitespace.end()
        number = _AMOUNT.match(text,cursor)
        if number is None:break
        raw=number.group().strip();negative=raw.startswith("(") or raw.startswith("-")
        value=float(re.sub(r"[\s(),+]", "",raw).lstrip("-"))*(-1 if negative else 1)
        if not math.isfinite(value):return None
        values.append(value);cursor=number.end()
    if len(values) not in (2,4):return None
    if 'loss' in name and 'earnings' not in name and any(value>0 for value in values):
        # Verbal loss labels with unsigned magnitudes use differing conventions.
        # Do not turn a stated loss of $0.13 into positive signed EPS.
        return None
    return values,start,cursor


def _periods(text, row_start):
    """Map only a table with explicit quarter (+ half-year) year columns."""
    low=max(0,row_start-5000)
    headings=list(_HEADER.finditer(text,low,row_start))
    for nearest in reversed(headings):
        # Join adjacent quarter/half-year headings, but never a preceding table.
        nearby=[nearest]
        for preceding in reversed([h for h in headings if h.start()<nearest.start()]):
            if nearest.start()-preceding.start()>200:break
            if re.search(r"20\d{2}",text[preceding.end():nearby[0].start()]):break
            nearby.insert(0,preceding)
        start=nearby[0].start();candidate=text[start:row_start]
        if len(candidate)>5000:continue
        expected=2*len(nearby)
        year_matches=list(re.finditer(r"(?<!\d)20\d{2}(?!\d)",candidate[:450]))[:expected]
        if len(year_matches)!=expected:continue
        header=candidate[:year_matches[-1].end()]
        if re.search(r"%\s*change|change\s*%",header,re.I):continue
        raw_headers=list(_HEADER.finditer(header))
        dates=[];durations=[]
        for i,h in enumerate(raw_headers):
            end=raw_headers[i+1].start() if i+1<len(raw_headers) else len(header)
            piece=header[h.end():end]
            names=list(_DATE.finditer(piece))
            if len(names) not in (1,2):break
            pair=[(_MONTHS[x.group(1).lower()],int(x.group(2)) if x.group(2) else None) for x in names]
            if len(pair)==1:pair*=2
            duration={"three":3,"six":6,"nine":9,"twelve":12,"quarter":3}[h.group().split()[0].lower()]
            dates.append(pair);durations.append(duration)
        else:
            years=[int(x.group()) for x in year_matches]
            if len(years)!=2*len(dates):continue
            if any(years[j]!=years[0] or years[j+1]!=years[1] for j in range(0,len(years),2)):continue
            if years[0]-years[1]!=1:continue
            periods=[]
            for duration,pair in zip(durations,dates):
                for year,(month,day) in zip(years[:2],pair):
                    try:
                        if day:last=dt.date(year,month,day)
                        else:last=dt.date(year+month//12,month%12+1,1)-dt.timedelta(days=1)
                    except ValueError:
                        return None
                    # A fiscal quarter can have 13 weeks; its start is not stated
                    # by a three-month column heading and must not be invented.
                    periods.append((None,str(last),duration))
            return periods,start,start+len(header)
    return None


def historical_eps_facts(corpus,task,entity):
    """Extract unambiguous historical EPS tables only from authorized documents.

    Returns accepted observations and conflict diagnostics. Repeated statements
    can corroborate a value, but conflicting values fail closed for that period.
    """
    declared_unit=re.sub(r'\s+','',str(entity.get('currency',entity.get('unit',''))).lower())
    if declared_unit not in ('usd/share','usd/pershare','usdpershare','usd/commonshare','usdpercommonshare','usd/dilutedshare'):
        return {'facts':[],'conflicts':[],'meaning':'historical extraction declined: per-share reporting currency is not established',
                'parser_limitations':['This parser supports explicitly task-declared USD per-share values only; inspect original evidence for other or unspecified currencies.']}
    accepted=[]
    for doc in corpus.docs.values():
        if doc.date>task['cutoff_date']:continue
        if doc.shared or entity['entity_id'] not in doc.entity_ids:continue
        text=_normal(doc.text)
        for anchor in _ANCHOR.finditer(text):
            row=_row(text,anchor)
            if row is None:continue
            values,row_start,row_end=row
            period_data=_periods(text,row_start)
            if period_data is None:continue
            periods,header_start,header_end=period_data
            if len(values)!=len(periods):continue
            local=text[max(header_start-160,0):row_start].lower()
            if re.search(r"non[ -]gaap|adjusted\s+earnings|operating\s+\(non",local):continue
            header_context=text[max(header_start-450,0):header_end].lower()
            if re.search(r'\b(?:canadian|australian|hong\s+kong|singapore|new\s+zealand)\s+dollars?\b|\b(?:eur|jpy|cny|rmb|cad|aud|yen|renminbi|euros?)\b|[€£¥]',header_context):continue
            # A standalone diluted row must remain in an earnings-per-share
            # subsection; never accept a weighted-share denominator as EPS.
            for value,(period_start,period_end,duration) in zip(values,periods):
                if period_end>doc.date or period_end>task['cutoff_date']:continue
                accepted.append(EPSFact(entity['entity_id'],'diluted_eps',value,'USD/share',period_start,
                    period_end,duration,'reported EPS; GAAP/adjusted requires context verification',
                    'manifest entity document; parent/subsidiary scope requires verification',doc.doc_id,doc.date,
                    header_start,header_end,row_start,row_end,
                    doc.text[header_start:header_end],doc.text[row_start:row_end]))
    groups={}
    for fact in accepted:groups.setdefault((fact.period_end,fact.period_months),[]).append(fact)
    result=[];conflicts=[]
    for period,values in groups.items():
        numbers={round(v.value,9) for v in values}
        if len(numbers)>1:
            conflicts.append({'period':period,'values':sorted(numbers),'reason':'multiple scoped statements disagree; no automatic issuer/subsidiary resolution'})
            continue
        best=min(values,key=lambda f:(f.value_end-f.header_start,-int(f.doc_date.replace('-',''))))
        record=asdict(best)
        record['status']='observed historical row, not the target-quarter forecast'
        record['basis_verified']=False
        record['consolidation_scope_verified']=False
        record['currency_verified']=False
        record['unit_basis']='USD/share explicitly declared by task; verify source reporting currency and share definition'
        record['earnings_scope']='total row explicitly located' if re.match(r'Assuming\s+dilution',best.value_quote,re.I) and re.search(r'\bTotal\b',best.value_quote) else 'reported row; verify total versus continuing operations'
        source_dates=list(_DATE.finditer(best.header))
        record['period_end_precision']='day in source heading' if source_dates and all(m.group(2) for m in source_dates) else 'source month; calendar month-end representation'
        seen=set();contexts=[]
        for value in values:
            key=(value.doc_id,value.header_start,value.value_start,value.value_end)
            if key in seen:continue
            seen.add(key)
            contexts.append({k:getattr(value,k) for k in ('doc_id','doc_date','header_start','header_end','value_start','value_end','header','value_quote')})
        record['corroborating_contexts']=contexts
        result.append(record)
    return {'facts':sorted(result,key=lambda f:(f['period_end'],f['period_months'])), 'conflicts':conflicts,
            'meaning':'historical fact extraction, not a forecast or accuracy claim',
            'parser_limitations':['GAAP basis is inferred from reported EPS with no non-GAAP marker in the local table; verify basis and issuer scope against the complete filing.',
                'A single issuer filing can discuss subsidiaries. Manifest authorization does not by itself establish consolidation scope.',
                'USD/share is task-declared; this does not independently establish the reporting currency of every source table.',
                'Period starts are left null because a fiscal quarter can be 13 weeks. A month-only header is represented at calendar month end.']}


def conditional_eps_bridge(facts,prior_year_target_eps):
    """Compare explicit current-corpus persistence assumptions, without choosing.

    These are forecast scenarios, never fitted or calibrated predictions. Their
    range is model disagreement, not a 90% predictive interval.
    """
    quarter=[f for f in facts if f['period_months']==3]
    if not quarter or not isinstance(prior_year_target_eps,(float,int)):return None
    latest=max(quarter,key=lambda f:f['period_end'])
    previous=next((f for f in quarter if 351<=(dt.date.fromisoformat(latest['period_end'])-dt.date.fromisoformat(f['period_end'])).days<=379),None)
    if previous is None:return None
    current=latest['value'];past=previous['value'];target=prior_year_target_eps
    scenarios={'same_quarter_last_year':target,'latest_reported_quarter':current,
               'persist_absolute_yoy_change':target+(current-past)}
    if abs(past)>.01 and current*past>0:
        scenarios['persist_relative_yoy_change']=target*(current/past)
    return {'scenarios_eps':scenarios,'observed_current_quarter':latest,
            'observed_same_quarter_prior_year':previous,
            'unverified_assumptions':['No change in future seasonality, revenue mix, taxes or diluted share count.',
             'One-off gains/losses have not been removed. The latest historical quarter is not the target quarter.'],
            'calibrated_interval':None,'selected_forecast':None}
