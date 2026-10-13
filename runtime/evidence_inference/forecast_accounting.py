"""Optional future diluted-common-income/share arithmetic, not a predictor.

All supplied scenarios are forecasts. They never become observed claims.
This contract cannot certify GAAP assumptions, quantiles or predictive quality.
"""
import datetime as dt
import math
import re
from .quantity import eps_growth_contract,transform_forecast_eps


def requested_eps_period(task,entity):
    name=str(task.get('target',{}).get('name','')).lower()
    prompt=' '.join(str(task.get('prompt','')).lower().split())
    if 'eps'not in name or not re.search(r'GAAP\s+diluted',prompt,re.I):return None
    value=str(entity.get('quarter_reported',task.get('target_period','')))
    match=re.fullmatch(r'(?:three months ended\s+)?(\d{4}-\d{2}-\d{2})',value,re.I)
    if not match:return None
    try:dt.date.fromisoformat(match.group(1))
    except ValueError:return None
    units=[entity[k]for k in ('currency','unit')if entity.get(k)is not None]
    supported={'usd/share','usd/pershare','usdpercommonshare','usd/commonshare','usd/dilutedshare','usdperdilutedshare'}
    if not units or any(re.sub(r'\s+','',str(u).lower())not in supported for u in units):return None
    level=task.get('interval_level',.9)
    if isinstance(level,bool)or not isinstance(level,(int,float))or level!=.9:return None
    return match.group(1)


def _number(x):
    return isinstance(x,(float,int))and not isinstance(x,bool)and math.isfinite(x)and abs(x)<1e90


def calculate_forecast_accounting(task,entity,raw):
    """Return (native fields, diagnostic) or (None, explicit refusal).

    Take an already forecast diluted-common numerator rather than silently
    subtracting preferred/noncontrolling/allocation items from ambiguous NI.
    No arbitrary non-GAAP addback is an accepted field in a scenario.
    """
    contract=raw.get('forecast_diluted_income')if isinstance(raw,dict)else None
    if contract is None:return None,'not_requested'
    period=requested_eps_period(task,entity)
    if period is None or not isinstance(contract,dict):return None,'unsupported_eps_accounting_target'
    required={'period_end':period,'accounting_basis':'GAAP','income_basis':'diluted_common_numerator',
              'share_basis':'weighted_average_diluted_shares','consolidation_scope':'issuer_consolidated',
              'currency':'USD','amount_scale':'millions','scenario_basis':'joint_forecast_for_eps_quantiles'}
    if set(contract)!=(set(required)|{'scenarios'}):return None,'invalid_accounting_contract_fields'
    for key,value in required.items():
        if contract.get(key)!=value:return None,'wrong_accounting_'+key
    scenarios=contract.get('scenarios')
    if not isinstance(scenarios,dict)or set(scenarios)!={'q05','q50','q95'}:return None,'missing_joint_eps_scenarios'
    eps={}
    for key,row in scenarios.items():
        if not isinstance(row,dict)or set(row)!={'diluted_common_income','weighted_diluted_shares'}:return None,'invalid_joint_scenario_fields'
        income=row['diluted_common_income'];shares=row['weighted_diluted_shares']
        if not _number(income)or not _number(shares)or shares<=0:return None,'invalid_joint_scenario_quantity'
        value=income/shares
        if not math.isfinite(value)or abs(value)>=1e90:return None,'invalid_eps_ratio'
        eps[key]=value
    if not eps['q05']<=eps['q50']<=eps['q95']:return None,'eps_quantile_order_conflict'
    forecast={'point':eps['q50'],'lo':eps['q05'],'hi':eps['q95']}
    if 'forecast_eps'in raw:
        other=raw['forecast_eps']
        if not isinstance(other,dict)or any(not _number(other.get(k))or not math.isclose(other[k],forecast[k],rel_tol=1e-8,abs_tol=1e-8)for k in forecast):
            return None,'conflicting_forecast_eps'
    if eps_growth_contract(task,entity)['eligible']:
        native=transform_forecast_eps(task,entity,forecast)
    elif task['target']['type']=='classification'and 'direction'in str(task['target'].get('name','')).lower()and 'point forecast'in str(task.get('prompt','')).lower():
        native={'point_forecast':forecast['point'],'interval':{'lo':forecast['lo'],'hi':forecast['hi'],'level':task.get('interval_level',.9)}}
    else:return None,'unsupported_native_eps_mapping'
    if native is None:return None,'accounting_conversion_failed'
    return {**native,'forecast_eps':forecast}, {'quantity_source':'code_divided_house_forecast_diluted_common_income_by_house_forecast_shares',
         'future_forecast_not_observed':True,'period_end':period,'basis_is_house_declaration_not_semantic_certificate':True,
         'quantiles_calibrated':False,'management_guidance_range_is_not_prediction_interval':True}


def accounting_prompt_contract(task,entity):
    period=requested_eps_period(task,entity)
    if period is None:return None
    return {'optional_field':'forecast_diluted_income','requested_period_end':period,
       'purpose':'If you construct a forecast of diluted-common-income and weighted diluted shares, return joint scenarios for the EPS5/50/95th quantiles. Code divides each numerator by its matching shares and converts all three EPS values to the requested native unit. These remain uncertain forecasts, not source facts.',
       'contract':{'period_end':period,'accounting_basis':'GAAP','income_basis':'diluted_common_numerator',
          'share_basis':'weighted_average_diluted_shares','consolidation_scope':'issuer_consolidated','currency':'USD',
          'amount_scale':'millions','scenario_basis':'joint_forecast_for_eps_quantiles',
          'scenarios':{q:{'diluted_common_income':'finite FORECAST millionsUSD','weighted_diluted_shares':'positive FORECAST millionsShares'}for q in ('q05','q50','q95')}},
       'limits':'Optional: do not fabricate missing accounting facts. Numerator is already after preferred/redemption/participating adjustments and any valid if-converted adjustment, never ambiguous parent net income or adjusted earnings. Component marginal quantiles cannot be independently added or divided to obtain EPS quantiles. No claim of calibrated coverage. Omit separate numeric forecast fields when this path is used; keep allowed label/probabilities, source IDs, mechanism and uncertainty.'}
