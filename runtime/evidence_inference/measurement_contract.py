"""Explicit response coordinates and positive affine conversion to task outputs.

Only task-supplied baselines define transformations. No inferred or external
future observation is used; these operations do not establish forecast skill.
"""
import math
import re
import datetime as dt
from .harness.contracts import quantity_contract as original_contract
from .native_bounds import native_bounds
from .native_units import _normal as native_unit_key


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) < 1e100


def yield_transform_eligible(task, entity):
    try:
        return (entity.get('unit') == 'bps_change' and finite(entity.get('start_yield_pct'))
                and entity.get('as_of') == task['cutoff_date']
                and dt.date.fromisoformat(task['resolution_date']) > dt.date.fromisoformat(task['cutoff_date']))
    except (KeyError, TypeError, ValueError):
        return False


def quantity_contract(task, entity):
    contract = original_contract(task, entity)
    name = task['target']['name']
    contract['response_field'] = 'forecast_eps' if contract['kind'] == 'future_eps' else 'native_forecast'
    contract['output_unit'] = contract['unit']
    contract['response_to_output'] = {'scale': 1., 'offset': 0.}
    if contract['kind'] == 'future_eps':
        base = contract['denominator']
        contract['response_to_output'] = {'scale': 100. / base, 'offset': -100.}
        contract['output_unit'] = 'yoy_growth_pct'
        contract['reference_observation'] = {'value': base, 'unit': contract['unit'],
            'period': entity.get('prior_year_quarter'), 'meaning': 'task supplied prior-year quarter EPS'}
    elif name == 'yield_change_bps_intermeeting' and yield_transform_eligible(task, entity):
        start = entity['start_yield_pct']
        contract.update(kind='future_yield', unit='yield_pct', basis='constant_maturity_yield',
            period=task.get('resolution_date', 'as_defined_in_task_prompt'),
            response_field='forecast_yield_pct', output_unit='bps_change',
            response_to_output={'scale': 100., 'offset': -100. * start},
            reference_observation={'value': start, 'unit': 'yield_pct', 'as_of': entity.get('as_of'),
                'maturity_years': entity.get('maturity_years'), 'meaning': 'task supplied cutoff yield'},
            conversion='output_bps = 100 * (future_yield_pct - start_yield_pct)')
    elif name == 'earnings_reaction':
        contract.update(unit='abnormal_return_pct', output_unit='abnormal_return_pct',
            basis='company_total_return_minus_benchmark_total_return',
            benchmark=entity.get('benchmark'), period=entity.get('event_window', contract['period']))
    elif name == 'credit_event_12m':
        bounds=native_bounds(task,entity)or(0.,1.)
        unit='probability_percent'if bounds==(0.,100.)else'probability'
        contract.update(unit=unit, output_unit=unit, bounds=list(bounds))
    elif name == 'bid_to_cover_ratio':
        contract.update(unit='bid_to_cover_ratio', output_unit='bid_to_cover_ratio', basis='total_bids_tendered_divided_by_accepted')
    elif name == 'net_positioning_change_pct_oi_rank':
        # Preserve the published native metric's more specific denominator
        # annotation instead of replacing pct_of_open_interest with percent.
        row_unit=entity.get('unit',entity.get('units',contract['unit']))
        contract.update(unit=row_unit,output_unit=row_unit)
        pairs = []
        for key, value in entity.items():
            match = re.fullmatch(r'net_noncommercial_(\d{8})', key)
            if match:
                oi = entity.get('open_interest_' + match.group(1))
                if finite(value) and finite(oi) and oi > 0:
                    pairs.append({'base_report': match.group(1), 'base_net_contracts': value,
                                  'base_open_interest': oi})
        if len(pairs) == 1:
            dates = re.search(r'from the (\d{4}-\d{2}-\d{2}) COT report to the (\d{4}-\d{2}-\d{2}) COT report',
                              task.get('prompt', ''), re.IGNORECASE)
            target_report = 'exact report date in task prompt; not resolution publication date'
            if dates and dates[1].replace('-', '') == pairs[0]['base_report']:
                try:
                    if dt.date.fromisoformat(dates[2]) > dt.date.fromisoformat(dates[1]):
                        target_report = dates[2]
                except ValueError: pass
            contract.update(basis='net_contract_change_divided_by_fixed_base_open_interest',
                reference_observation=pairs[0],
                conversion='native_forecast = 100 * (future_net_contracts - base_net_contracts) / base_open_interest',
                period={'base_report': pairs[0]['base_report'],
                        'target_report': target_report})
    elif name == 'next_estimate_revision_direction':
        contract.update(basis='revised_level_at_entity_resolving_release_date_not_revision_delta',
            period={'reference_month': entity.get('ref_month'),
                    'resolving_release_date': entity.get('resolving_release_date')},
            reference_observation={'value': entity.get('latest_precutoff_estimate'),
                'vintage': entity.get('latest_precutoff_vintage'), 'unit': contract['unit']})
    return contract


def convert_triple(values, contract, *, to_output=True):
    if not all(finite(v) for v in values):
        raise ValueError('nonfinite_forecast')
    affine = contract['response_to_output']; scale, offset = affine['scale'], affine['offset']
    if not finite(scale) or scale <= 0 or not finite(offset):
        raise ValueError('invalid_quantity_conversion')
    result = [(scale * v + offset) if to_output else ((v - offset) / scale) for v in values]
    if not all(finite(v) for v in result):
        raise ValueError('nonfinite_converted_forecast')
    return result


def response_control(prior, contract):
    """Any optional control must use the SAME physical quantity as model replies."""
    values = convert_triple([prior['point_forecast'], prior['interval']['lo'], prior['interval']['hi']],
                            contract, to_output=False)
    result = {'unit': contract['unit'], 'basis': contract['basis'], 'period': contract['period'],
              contract['response_field']: dict(zip(('point','lo','hi'), values)),
              'status': 'uncalibrated forecast hypothesis; not evidence or hidden scorer naive'}
    if 'label' in prior: result['label'] = prior['label']
    return result


def unit_matches(value, expected):
    if not isinstance(value, str): return False
    if isinstance(expected,str)and native_unit_key(value)==native_unit_key(expected):return True
    normal = lambda s: ' '.join(s.casefold().replace('_', ' ').split())
    percent={'percent','percentage','pct','%'}
    if isinstance(expected,str)and normal(expected)in percent:
        return normal(value)in percent
    aliases = {
        'probability_percent': {'probability_percent','probability_pct','percent','percentage','pct','%'},
        'yoy_growth_pct': {'yoy_growth_pct','percent','percentage','pct','%'},
        'bps_change': {'bps_change','bps','bp','basis_points','basis points','basis point change'},
        'USD/share': {'USD/share', 'USD/shares', 'USD per share'},
        'yield_pct': {'yield_pct', 'percent_yield', 'percent', '%'},
        'abnormal_return_pct': {'abnormal_return_pct', 'percent', '%', 'percentage_points'},
        'bid_to_cover_ratio': {'bid_to_cover_ratio', 'ratio', 'unitless'},
        'probability': {'probability', 'probability_0_to_1'},
    }
    return normal(value) in {normal(x) for x in aliases.get(expected, {expected})}
