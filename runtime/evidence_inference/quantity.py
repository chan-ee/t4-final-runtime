"""Explicit EPS forecast-level conversion, with no fitted or historical answers.

Native target-unit forecasts should be retained when this optional conversion
refuses. A returned number is a transformed forecast, never a source fact.
"""
from __future__ import annotations
import math
import re


def _finite(value):
    return isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value) and abs(value)<1e100


def eps_growth_contract(task,entity):
    """Recognize only an unambiguous usual positive-base YoY percent EPS target."""
    target=task.get('target',{});name=str(target.get('name','')).lower()
    prompt=' '.join(str(task.get('prompt','')).lower().split())
    words=set(re.findall(r'[a-z0-9]+',name))
    result={'eligible':False,'reason':'not an explicit EPS year-over-year percent-growth target',
            'target_name':target.get('name'),'target_type':target.get('type'),
            'observed_or_forecast':'forecast conversion only',
            'input_unit':'EPS on the task-requested accounting/share basis',
            'output_unit':'percent relative to task prior_year_q_eps'}
    if target.get('type') not in ('regression','ranking','classification'):return result
    if not ('eps' in words or 'earnings_per_share' in name):return result
    if not words.intersection({'growth','change'}):return result
    if words.intersection({'probability','prob','chance','odds','likelihood'}):
        result['reason']='the named target is probability rather than EPS growth amount';return result
    if not words.intersection({'pct','percent','percentage'}):
        result['reason']='target name does not establish percentage units';return result
    if not ('yoy' in words or re.search(r'year[- ]over[- ]year|same quarter (?:one|a) year earlier|same quarter (?:of|in) (?:the )?prior year',prompt)):
        result['reason']='year-over-year denominator comparison is not established';return result
    if 'prior_year_q_eps' not in entity:
        result['reason']='required prior_year_q_eps task feature is absent';return result
    # The field name by itself is not sufficient to override an explicitly
    # different formula or scale in the task text.
    percent_quantity=(
        r'express(?:ed)?\s+(?:as|in)\s+(?:a\s+)?percent',
        r'(?:growth|change)\s+(?:in|as)\s+percent',
        r'(?:percent(?:age)?|%)\s+(?:year[- ]over[- ]year\s+)?(?:eps\s+)?(?:growth|change)',
        r'(?:growth|change)\s*\(\s*%\s*\)',
    )
    if not any(re.search(pattern,prompt) for pattern in percent_quantity):
        result['reason']='percentage target is not explicit; a 90% interval alone does not establish units';return result
    incompatible=(
        r'\b(?:log\d*|log1p|ln)\s*\(',r'\blog(?:arithmic|arithm)?\s+(?:growth|change|return|rate)',
        r'continuous(?:ly)?\s+compound(?:ed)?',r'absolute[- ]+(?:change|difference|value)',
        r'percentage[- ]point',r'\bbasis\s+points?\b',r'\b(?:fraction|fractional|decimal)\s+(?:growth|change|return)',
        r'\bquarter[- ]over[- ]quarter\b',r'\b(?:relative|compared)\s+to\s+(?:the\s+)?consensus',
    )
    if any(re.fullmatch(r'log\d*|log1p|ln|logarithm|logarithmic',word) for word in words) or any(re.search(pattern,prompt) for pattern in incompatible):
        result['reason']='log, absolute, differently scaled or differently based definition requires native output';return result
    if re.search(r'(?:denominator|divid(?:e|ing)\s+by).{0,60}(?:consensus|current|average|absolute|prior[- ]quarter)',prompt):
        result['reason']='another denominator is named';return result
    denominator=entity.get('prior_year_q_eps')
    if not _finite(denominator) or denominator<=0:
        result['reason']='zero, negative or nonfinite prior EPS has ambiguous growth semantics';return result
    currency=entity.get('currency',entity.get('unit'))
    if currency is not None and not any(term in str(currency).lower() for term in ('share','eps')):
        result['reason']='task prior EPS unit is not a per-share unit';return result
    result.update(eligible=True,reason='explicit YoY percentage EPS target with finite positive task denominator',
                  formula='100 * (forecast_eps / prior_year_q_eps - 1)',
                  prior_field='prior_year_q_eps',prior_value=float(denominator),
                  basis_check='The forecast EPS must use exactly the accounting and share basis requested by the task.')
    return result


def transform_forecast_eps(task,entity,forecast_eps):
    """Return a target-unit decision, or None when conversion is not justified.

    This accepts only an explicitly supplied future forecast triple. It never
    chooses a historical EPS observation as a forecast and assigns no labels.
    """
    contract=eps_growth_contract(task,entity)
    if not contract['eligible'] or not isinstance(forecast_eps,dict):return None
    values=[forecast_eps.get(k) for k in ('point','lo','hi')]
    if not all(_finite(value) for value in values) or values[1]>values[2]:return None
    level=task.get('interval_level',.9)
    if not _finite(level) or not 0<level<1:return None
    denominator=contract['prior_value']
    transformed=[100*(float(value)/denominator-1) for value in values]
    if not all(_finite(value) for value in transformed):return None
    return {'point_forecast':transformed[0],
            'interval':{'level':float(level),'lo':transformed[1],'hi':transformed[2]},
            'provenance':{'status':'derived forecast, not observed source evidence',
                          'formula':contract['formula'],'prior_field':'prior_year_q_eps',
                          'prior_value':denominator,'input_forecast_eps':dict(zip(('point','lo','hi'),values)),
                          'forecast_assumptions_validated':False}}
