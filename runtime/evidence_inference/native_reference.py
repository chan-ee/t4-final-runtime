"""Fail-closed native observations, never field-name or magnitude conversions.

Generic observations require explicit organizer declarations of metric, unit,
period and basis. A returned task reference is a persistence hypothesis input,
not an observed future answer. Existing published physical contracts are narrow.
"""
import calendar
import datetime as dt
import math
import re
from .native_units import declared_native_unit, _normal
from .native_bounds import native_bounds


def finite(value):
    return type(value) in (int,float) and math.isfinite(value) and abs(value)<1e100


def normal(value):
    return ' '.join(value.casefold().replace('_',' ').replace('-',' ').split()) if isinstance(value,str) else None


def unit(value):
    if not isinstance(value,str) or not value.strip():return None
    key=_normal(value)
    return {'probability percent':'percent','percent probability':'percent',
            'probability pct':'percent','probability':'probability fraction',
            'probability 0 to 1':'probability fraction','decimal probability':'probability fraction',
            'usd/shares':'usd/share','us dollars per share':'usd/share'}.get(key,key)


def period_kind(value):
    return {'day':'day','days':'day','daily':'day','month':'month','months':'month','monthly':'month',
            'quarter':'quarter','quarters':'quarter','quarterly':'quarter',
            'year':'year','years':'year','annual':'year','annually':'year','yearly':'year'}.get(normal(value))


def period(value):
    """Only literal calendar periods; fiscal-period names stay unresolved."""
    if not isinstance(value,str):return None
    value=value.strip()
    try:
        if re.fullmatch(r'\d{4}-\d{2}-\d{2}',value):return('day',dt.date.fromisoformat(value))
        if re.fullmatch(r'\d{4}-\d{2}',value):
            year,month=map(int,value.split('-'));return('month',dt.date(year,month,calendar.monthrange(year,month)[1]))
        match=re.fullmatch(r'(\d{4})[- ]?Q([1-4])',value,re.I)
        if match:
            year,q=map(int,match.groups());month=3*q;return('quarter',dt.date(year,month,calendar.monthrange(year,month)[1]))
        if re.fullmatch(r'\d{4}',value):return('year',dt.date(int(value),12,31))
        match=re.fullmatch(r'(three|twelve) months ended (\d{4}-\d{2}-\d{2})',value,re.I)
        if match:return('quarter'if match[1].lower()=='three'else'year',dt.date.fromisoformat(match[2]))
    except (ValueError,TypeError):return None
    return None


def date(value):
    result=period(value)
    return result[1]if result and result[0]=='day'else None


def change_target(task):
    name=task.get('target',{}).get('name','')
    if re.search(r'(?:^|[_ -])(?:change|growth|return|reaction|delta|difference|log|ln)(?:[_ -]|$)',str(name),re.I):return True
    prompt=positive_definition(task.get('prompt',''))
    return bool(re.search(r'\b(?:forecast|predict|output|target|point forecast|native forecast)(?:\s+of)?\s+(?:the\s+)?(?:(?:year over year|month over month|quarter over quarter|annual|monthly|quarterly|percentage|percent|log|logarithmic|absolute|net|relative|signed)\s+)*(?:change|growth|return|delta|difference)\b',prompt))


def positive_definition(value):
    text=normal(value)or''
    # Only straightforward negative instructions are removed. Complex mixed
    # definitions remain ambiguous and are refused by the basis checks.
    return re.sub(r'\b(?:do not|never|avoid|exclude|without|not using|rather than|not)\s+(?:(?:use|include|report|using|rely on)\s+)?[^.;\n]*','',text)


def basis(task):
    target=task.get('target',{})
    explicit=normal(target.get('basis'))
    if 'basis'in target and not explicit:return None
    text=positive_definition(task.get('prompt',''))
    adjusted=bool(re.search(r'\badjusted\b|\bnon gaap\b',text))
    gaap=bool(re.search(r'\bgaap\b',re.sub(r'\bnon gaap\b','',text)))
    diluted=bool(re.search(r'\bdiluted\b',text));basic=bool(re.search(r'\bbasic\b',text))
    if (gaap and adjusted)or(diluted and basic):return None
    if explicit:
        if gaap and('adjusted'in explicit or'non gaap'in explicit):return None
        if adjusted and'gaap'in explicit and'non gaap'not in explicit:return None
        if diluted and'basic'in explicit or basic and'diluted'in explicit:return None
        return explicit
    tags=(['gaap']if gaap else['adjusted']if adjusted else[])+(['diluted']if diluted else['basic']if basic else[])
    return ' '.join(tags)or None


def generic_reference(task,entity,chunks=(),corpus=None):
    target=task.get('target',{})
    declaration=target.get('reference')
    if change_target(task)or not isinstance(declaration,dict):return None
    field=declaration.get('field')
    if not isinstance(field,str)or not finite(entity.get(field)):return None
    output,status=declared_native_unit(task,entity)
    expected_basis=basis(task)
    historical=period(declaration.get('period'));future=period(target.get('period'))
    observed=date(declaration.get('as_of'));cutoff=date(task.get('cutoff_date'))
    if unit(output)is None or unit(declaration.get('unit'))!=unit(output):return None
    if declaration.get('units')is not None and unit(declaration['units'])!=unit(output):return None
    if expected_basis is None or normal(declaration.get('basis'))!=expected_basis:return None
    if not historical or not future or historical[0]!=future[0]or historical[1]>=future[1]:return None
    for key in('period_type','frequency'):
        if target.get(key)is not None and period_kind(target[key])!=future[0]:return None
    if observed is None or cutoff is None or not historical[1]<=observed<=cutoff:return None
    row_as_of=date(entity.get('as_of'))if entity.get('as_of')is not None else observed
    if row_as_of is None or row_as_of!=observed:return None
    for suffix,expected in (('_unit',unit(output)),('_units',unit(output)),('_basis',expected_basis)):
        actual=entity.get(field+suffix)
        if actual is not None and(unit(actual)if suffix!='_basis'else normal(actual))!=expected:return None
    if entity.get(field+'_period')is not None and period(entity[field+'_period'])!=historical:return None
    if entity.get(field+'_as_of')is not None and date(entity[field+'_as_of'])!=observed:return None
    bounds=native_bounds(task,entity)
    if bounds is not None and not bounds[0]<=entity[field]<=bounds[1]:return None
    if corpus is not None and chunks:
        history=matching_historical_values(task,entity,chunks,corpus,with_periods=True,_check_reference=False)
        if isinstance(history,dict)and historical in history and history[historical]!=entity[field]:return None
    return {'value':float(entity[field]),'field':field,'method':'explicit native task-reference persistence assumption; not an observed future value'}


def known_vintage_contract(task,entity):
    target=task.get('target',{});prompt=task.get('prompt','').lower()
    if target.get('name')!='next_estimate_revision_direction' or not re.search(r'forecast of (?:the )?revised (?:value|level|estimate)|revised level',prompt):return False
    if change_target(task)or re.search(r'logarithm|log\s*\(|ln\s*\(|revision (?:amount|delta)|amount of (?:the )?revision',positive_definition(prompt)):return False
    current=entity.get('latest_precutoff_estimate');as_of=date(entity.get('latest_precutoff_vintage'))
    ref=period(entity.get('ref_month'));cutoff=date(task.get('cutoff_date'));resolve=date(entity.get('resolving_release_date'))
    output,_=declared_native_unit(task,entity)
    row=entity.get('units',entity.get('unit'))
    return bool(finite(current)and ref and ref[0]=='month'and as_of and cutoff and resolve
                and ref[1]<=as_of<=cutoff<resolve and unit(output)is not None and unit(output)==unit(row))


def task_reference(task,entity,chunks=(),corpus=None):
    target=task.get('target',{});name=target.get('name');prompt=positive_definition(task.get('prompt',''))
    if 'reference'in target:return generic_reference(task,entity,chunks,corpus)
    # The published consensus classifier defines its physical output through
    # the exact diluted-EPS/consensus comparison, not arbitrary row currency.
    if name=='eps_outcome' and re.search(r'\bdiluted (?:eps|earnings per share)\b',prompt)and'consensus'in prompt:
        if 'adjusted'in prompt or'non gaap'in prompt:return None
        output,status=declared_native_unit(task,entity)
        if output is not None and unit(output)!='usd/share':return None
        if finite(entity.get('consensus_eps')):
            declared=entity.get('consensus_eps_basis')
            if declared is not None and normal(declared)not in('diluted','gaap diluted'):return None
            row_unit=entity.get('consensus_eps_unit',entity.get('currency'))
            if row_unit is not None and unit(row_unit)!='usd/share':return None
            if entity.get('consensus_eps_as_of')is not None:
                observed=date(entity['consensus_eps_as_of']);cutoff=date(task.get('cutoff_date'))
                if observed is None or cutoff is None or observed>cutoff:return None
            if entity.get('consensus_eps_period')is not None:
                declared_period=normal(entity['consensus_eps_period'])
                exact=normal(target.get('period',entity.get('quarter_reported')))
                if exact is None:
                    match=re.search(r'\bq([1-4])\s*fy\s*(\d{4})\b',prompt)
                    if match:exact='q'+match[1]+' fy'+match[2]
                if exact is None or declared_period!=exact:return None
            return {'value':float(entity['consensus_eps']),'field':'consensus_eps','method':'task-defined diluted-EPS consensus anchor; uncalibrated forecast hypothesis'}
    if name=='eps_yoy_direction' and'prior year q eps'in prompt and'gaap'in prompt and'diluted'in prompt:
        if basis(task)!='gaap diluted':return None
        output,status=declared_native_unit(task,entity)
        if output is not None and unit(output)!='usd/share':return None
        current=period(entity.get('quarter_reported'));prior=period(entity.get('prior_year_quarter'))
        if current and prior and current[0]==prior[0]=='quarter'and current[1].year==prior[1].year+1 and(current[1].month,current[1].day)==(prior[1].month,prior[1].day):
            if unit(entity.get('currency'))=='usd/share'and finite(entity.get('prior_year_q_eps')):
                if entity.get('prior_year_q_eps_basis')is not None and normal(entity['prior_year_q_eps_basis'])!='gaap diluted':return None
                if entity.get('prior_year_q_eps_unit')is not None and unit(entity['prior_year_q_eps_unit'])!='usd/share':return None
                if entity.get('prior_year_q_eps_period')is not None and period(entity['prior_year_q_eps_period'])!=prior:return None
                return {'value':float(entity['prior_year_q_eps']),'field':'prior_year_q_eps','method':'task-defined same-quarter prior-year GAAP diluted-EPS persistence hypothesis'}
    if known_vintage_contract(task,entity):
        return {'value':float(entity['latest_precutoff_estimate']),'field':'latest_precutoff_estimate','method':'task-defined revised-level current-vintage persistence hypothesis'}
    return None


def matching_historical_values(task,entity,chunks,corpus=None,*,with_periods=False,_check_reference=True):
    """Exact declared columns only; token overlap is not a metric definition."""
    if change_target(task)or corpus is None:return[]
    target=task.get('target',{});metric=target.get('name')
    output,_=declared_native_unit(task,entity);expected_basis=basis(task)
    target_period=period(target.get('period'))
    frequency=period_kind(target.get('period_type',target.get('frequency')))
    if frequency is None and target_period:frequency=target_period[0]
    if target_period and frequency is not None and target_period[0]!=frequency:return[]
    if target.get('period_type')is not None and target.get('frequency')is not None and period_kind(target['period_type'])!=period_kind(target['frequency']):return[]
    cutoff=date(task.get('cutoff_date'))
    if not isinstance(metric,str)or unit(output)is None or expected_basis is None or frequency is None or cutoff is None:return[]
    source_doc=chunks[0].doc_id if chunks else None
    doc=corpus.docs.get(source_doc)
    if doc is None or not(getattr(doc,'shared',False)or entity.get('entity_id')in getattr(doc,'entity_ids',())):return[]
    doc_date=date(doc.date)
    if doc_date is None or doc_date>cutoff:return[]
    metadata=getattr(doc,'metadata',{})
    columns=metadata.get('columns')if isinstance(metadata,dict)else None
    declaration=columns.get(metric)if isinstance(columns,dict)else None
    if not isinstance(declaration,dict)or declaration.get('metric',metric)!=metric:return[]
    source_unit=unit(declaration.get('unit',declaration.get('units')))
    source_frequency=period_kind(declaration.get('period_type',declaration.get('frequency')))
    if source_unit!=unit(output)or source_frequency!=frequency or normal(declaration.get('basis'))!=expected_basis:return[]
    if declaration.get('unit')is not None and declaration.get('units')is not None and unit(declaration['unit'])!=unit(declaration['units']):return[]
    if declaration.get('period_type')is not None and declaration.get('frequency')is not None and period_kind(declaration['period_type'])!=period_kind(declaration['frequency']):return[]
    as_of=date(declaration.get('as_of'))
    if as_of is None or not as_of<=doc_date<=cutoff:return[]
    for key in('unit','units'):
        if metadata.get(key)is not None and unit(metadata[key])!=source_unit:return[]
    history={}
    bounds=native_bounds(task,entity)
    header=None;index=None
    for chunk in chunks:
        if chunk.doc_id!=source_doc:continue
        for line in chunk.text.splitlines():
            cells=[part.strip()for part in line.split('|')]
            if metric in cells:
                if cells.count(metric)!=1:return[]
                header=cells;index=cells.index(metric);continue
            observed_period=period(cells[0])if cells else None
            if header is None or observed_period is None or index>=len(cells):continue
            if observed_period[0]!=frequency or observed_period[1]>as_of:continue
            try:value=float(cells[index].replace(',',''))
            except (ValueError,TypeError):continue
            if not finite(value):return[]
            if bounds is not None and not bounds[0]<=value<=bounds[1]:return[]
            if observed_period in history and history[observed_period]!=value:return[]
            history[observed_period]=value
    if _check_reference and'reference'in target and generic_reference(task,entity,chunks,corpus)is None:return[]
    if with_periods:return history
    return [history[key]for key in sorted(history,key=lambda value:value[1])][-24:]
