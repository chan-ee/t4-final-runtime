"""Optional same-response arithmetic bridge for trusted EPS/yield references.

This checks a declared calculation, not actual model use, causality or truth.
Only caller-established task coordinates are supported; source numerals never
acquire model-authored unit/period/basis metadata. No prediction is changed.
"""
import copy
import datetime as dt
import json
import math
import re
from .harness.contracts import digest
from .measurement_contract import convert_triple, quantity_contract, unit_matches
from .native_reference import period as literal_period, basis as requested_basis, positive_definition, normal
from .native_units import declared_native_unit
from .quantity import eps_growth_contract


def finite(value):
    return type(value) in (int, float) and math.isfinite(value) and abs(value) < 1e100


def reference_contract(task, entity, contract=None):
    contract = contract or quantity_contract(task, entity)
    kind = contract.get('kind')
    if kind not in ('future_eps', 'future_yield'):
        raise ValueError('witness_reference_contract_unavailable')
    output_unit, output_status = declared_native_unit(task, entity)
    expected_output = 'yoy_growth_pct' if kind == 'future_eps' else 'bps_change'
    if (output_unit is None and output_status == 'native_unit_unresolved_in_task_prompt'
            and kind == 'future_eps' and eps_growth_contract(task, entity).get('eligible')):
        # This existing narrow recognizer verifies the explicit percent-growth
        # definition even when it is not phrased as an "output in" clause.
        output_unit = 'yoy_growth_pct'
    if output_unit is None or not unit_matches(output_unit, expected_output):
        raise ValueError('witness_declared_native_output_unit_conflict')
    field = 'prior_year_q_eps' if kind == 'future_eps' else 'start_yield_pct'
    if any(not isinstance(contract.get(key), str) or not contract[key]
           for key in ('unit', 'basis', 'output_unit')):
        raise ValueError('witness_quantity_metadata_unresolved')
    reference = contract.get('reference_observation', {})
    if not isinstance(reference, dict):
        raise ValueError('witness_reference_value_unbound')
    value = entity.get(field)
    if not finite(value) or reference.get('value') != value:
        raise ValueError('witness_reference_value_unbound')
    period = contract.get('period')
    prior_period = entity.get('prior_year_quarter') if kind == 'future_eps' else entity.get('as_of')
    if not isinstance(period, str) or not period or 'as_defined' in period:
        raise ValueError('witness_target_period_unresolved')
    if not isinstance(prior_period, str) or not prior_period:
        raise ValueError('witness_reference_period_unresolved')
    if kind == 'future_yield':
        positive = positive_definition(task.get('prompt', ''))
        declared = task.get('target', {}).get('basis')
        if declared is not None and normal(declared) not in (
                'constant maturity yield', 'treasury constant maturity yield', 'us treasury constant maturity yield'):
            raise ValueError('witness_yield_target_basis_conflict')
        if not ('treasury' in positive and 'constant maturity' in positive):
            raise ValueError('witness_treasury_cmt_definition_unresolved')
        if re.search(r'\b(?:forecast|predict|output)\s+(?:the\s+)?(?:future\s+)?(?:policy rate|policy interest rate|federal funds rate|fed funds rate)\b', positive):
            raise ValueError('witness_policy_rate_output_refused')
        future = literal_period(period)
        resolution = literal_period(task.get('resolution_date'))
        cutoff = literal_period(task.get('cutoff_date'))
        observed = literal_period(prior_period)
        if (not future or not resolution or not cutoff or not observed
                or any(p[0] != 'day' for p in (future, resolution, cutoff, observed))
                or future != resolution or observed != cutoff or future[1] <= cutoff[1]):
            raise ValueError('witness_yield_resolution_or_asof_mismatch')
    if kind == 'future_eps':
        if requested_basis(task) not in ('gaap diluted', 'gaap diluted quarter', 'gaap diluted quarterly'):
            raise ValueError('witness_eps_basis_unresolved')
        positive = positive_definition(task.get('prompt', ''))
        if re.search(r'\bytd\b|\byear to date\b|\bannual\b|\btwelve months\b', positive):
            raise ValueError('witness_eps_cumulative_basis_refused')
        current = literal_period(entity.get('quarter_reported'))
        prior = literal_period(prior_period)
        future = literal_period(period)
        if (not current or not prior or current[0] != 'quarter' or prior[0] != 'quarter'
                or future != current or current[1].year != prior[1].year + 1
                or (current[1].month, current[1].day) != (prior[1].month, prior[1].day)):
            raise ValueError('witness_eps_quarter_reference_mismatch')
    observed = literal_period(prior_period)
    if not observed:
        raise ValueError('witness_reference_period_unresolved')
    declared_basis = entity.get(field + '_basis')
    expected_bases = (('gaap diluted', 'gaap diluted quarter', 'gaap diluted quarterly') if kind == 'future_eps'
                      else ('constant maturity yield', 'treasury constant maturity yield', 'us treasury constant maturity yield'))
    if declared_basis is not None and normal(declared_basis) not in expected_bases:
        raise ValueError('witness_explicit_reference_basis_conflict')
    for suffix in ('_unit', '_units'):
        if entity.get(field + suffix) is not None and not unit_matches(entity[field + suffix], contract['unit']):
            raise ValueError('witness_explicit_reference_unit_conflict')
    if entity.get(field + '_period') is not None and literal_period(entity[field + '_period']) != observed:
        raise ValueError('witness_explicit_reference_period_conflict')
    declared_asof = entity.get(field + '_as_of')
    if declared_asof is not None:
        asof = literal_period(declared_asof); cutoff_period = literal_period(task.get('cutoff_date'))
        if (not asof or not cutoff_period or asof[0] != 'day' or cutoff_period[0] != 'day'
                or not observed[1] <= asof[1] <= cutoff_period[1]
                or kind == 'future_yield' and asof != cutoff_period):
            raise ValueError('witness_explicit_reference_asof_conflict')
    try:
        if observed[1] > dt.date.fromisoformat(task['cutoff_date']):
            raise ValueError('witness_reference_after_cutoff')
    except (KeyError, TypeError):
        raise ValueError('witness_cutoff_unresolved') from None
    if kind == 'future_yield' and entity.get('as_of') != task['cutoff_date']:
        raise ValueError('witness_reference_asof_mismatch')
    if kind == 'future_eps' and value <= 0:
        raise ValueError('witness_nonpositive_eps_reference')
    return {'field': field, 'value': value, 'unit': contract['unit'],
            'basis': contract['basis'], 'period': prior_period,
            'availability_cutoff': task['cutoff_date'], 'future_period': period,
            'output_unit': contract['output_unit'], 'kind': kind,
            'conversion': contract['response_to_output']}


def optional_context(task, entity, contract=None):
    try:
        ref = reference_contract(task, entity, contract)
    except (ValueError, TypeError, KeyError):
        return {'available': False, 'policy': 'Optional witness unavailable; retain ordinary unverified explanation policy.'}
    return {'available': True, 'reference': ref,
            'shape': {'reference': ref['field'], 'future': {'value': 'conditional physical point',
                'unit': ref['unit'], 'basis': ref['basis'], 'period': ref['future_period'],
                'condition': 'If/assuming/conditional hypothesis; at most 160 characters'}},
            'scope': 'Optional declared point arithmetic only; no interval, label or rank proof.'}


def validate_witness(raw, task, entity, native_point, *, contract=None, used_fields=None):
    ref = reference_contract(task, entity, contract)
    if not isinstance(raw, dict) or set(raw) != {'reference', 'future'}:
        raise ValueError('witness_shape')
    try:
        if len(json.dumps(raw, ensure_ascii=False, allow_nan=False, separators=(',', ':')).encode('utf-8')) > 600:
            raise ValueError('witness_byte_cap')
    except (TypeError, OverflowError):
        raise ValueError('witness_shape') from None
    if raw['reference'] != ref['field']:
        raise ValueError('witness_wrong_task_reference')
    if used_fields is not None and used_fields != [ref['field']]:
        raise ValueError('witness_unsupported_task_input_declaration')
    future = raw['future']
    if not isinstance(future, dict) or set(future) != {'value', 'unit', 'basis', 'period', 'condition'}:
        raise ValueError('witness_future_shape')
    if not finite(future['value']) or not finite(native_point):
        raise ValueError('witness_nonfinite')
    if not unit_matches(future['unit'], ref['unit']):
        raise ValueError('witness_future_unit_mismatch')
    if future['basis'] != ref['basis'] or future['period'] != ref['future_period']:
        raise ValueError('witness_future_basis_or_period_mismatch')
    condition = future['condition']
    if (not isinstance(condition, str) or not 8 <= len(condition) <= 160
            or not re.search(r'\b(if|assuming|conditional|uncertain|hypothesis)\b', condition, re.I)):
        raise ValueError('witness_future_condition_required')
    if any(phrase in condition.casefold() for phrase in (
            'known future', 'observed future', 'will definitely', 'will certainly',
            'is guaranteed', 'future outcome is known')):
        raise ValueError('witness_future_assertion_refused')
    contract = contract or quantity_contract(task, entity)
    computed = convert_triple([future['value']] * 3, contract)[0]
    if not math.isclose(computed, native_point, rel_tol=1e-12, abs_tol=1e-9):
        raise ValueError('witness_native_point_mismatch')
    physical = format(future['value'], '.12g')
    baseline = format(ref['value'], '.12g')
    formula = ('100*(' + physical + '/' + baseline + '-1)' if ref['kind'] == 'future_eps'
               else '100*(' + physical + '-' + baseline + ')')
    return {'entity_id': entity['entity_id'], 'task_digest': digest(task),
            'raw': copy.deepcopy(raw), 'reference': ref, 'native_point': native_point,
            'formula': formula, 'computed_native_point': computed,
            'arithmetic_match_tolerance': {'relative': 1e-12, 'absolute': 1e-9},
            'arithmetic_verified': True, 'actual_model_use_verified': False,
            'causality_or_future_truth_verified': False, 'class_rank_interval_verified': False}


def validate_record(record, task, entity, native_point):
    if not isinstance(record, dict) or record.get('task_digest') != digest(task):
        raise ValueError('witness_task_binding_changed')
    checked = validate_witness(record.get('raw'), task, entity, native_point)
    if checked != record:
        raise ValueError('witness_record_changed')
    return checked


def render_parts(record):
    ref = record['reference']; future = record['raw']['future']
    premise = ('Declared task operand ' + ref['field'] + '=' + format(ref['value'], '.12g')
               + ' ' + ref['unit'] + '; basis ' + ref['basis'] + '; period ' + ref['period']
               + '; supplied by task at cutoff ' + ref['availability_cutoff'] + '.')
    link = ('Conditional future quantity ' + format(future['value'], '.12g') + ' ' + ref['unit']
            + ' for ' + ref['future_period'] + ': ' + future['condition']
            + '. Declared arithmetic matches the submitted point; actual model use and causality are unverified.')
    consequence = record['formula'] + '=' + format(record['native_point'], '.12g') + ' ' + ref['output_unit']
    return premise, link, consequence
