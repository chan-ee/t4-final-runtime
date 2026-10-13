"""Typed role contracts and local provenance validation, not a truth judge."""
from __future__ import annotations
import hashlib
import json
import math
from ..decisions import normalize_forecast
from ..packets import claims_from_ids, observation_exclusions
from ..quantity import eps_growth_contract
from ..native_units import declared_native_unit


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                   ensure_ascii=False, allow_nan=False).encode()).hexdigest()


def quantity_contract(task, entity):
    target = task['target']; name = target['name']
    period = next((entity[k] for k in ('quarter_reported', 'ref_month', 'event_window',
                    'auction_date', 'resolving_release_date') if entity.get(k)), None)
    if period is None:
        # The exact task defines an otherwise unparsed horizon. Do not invent one.
        period = 'as_defined_in_task_prompt'
    unit, unit_status = declared_native_unit(task, entity)
    eps = eps_growth_contract(task, entity)
    contract = {'kind': 'native', 'unit': unit, 'unit_status': unit_status, 'period': period,
                'basis': 'exact_task_definition', 'output_target': name,
                'definition': task.get('prompt', ''), 'interval_level': task.get('interval_level', .9),
                'labels': target.get('labels', []),
                'label_assertions': target.get('label_assertions', {})}
    if eps['eligible']:
        contract.update(kind='future_eps', unit=entity.get('currency', 'USD/share'),
                        basis='GAAP_diluted_quarter', conversion=eps['formula'],
                        denominator=eps['prior_value'])
    elif name in ('eps_yoy_direction', 'eps_outcome'):
        contract.update(unit=entity.get('currency', 'USD/share'), basis='task_diluted_quarter')
    return contract


def binding(task, entity, packet):
    return {'entity_id': entity['entity_id'], 'task_digest': digest(task),
            'contract_digest': digest(quantity_contract(task, entity)),
            'evidence_digest': digest(packet)}


def bound(raw, expected):
    if not isinstance(raw, dict): return 'missing_row'
    for key, value in expected.items():
        if raw.get(key) != value: return 'binding_mismatch_' + key
    return None


def index_items(response, requested_ids):
    """Reject ambiguous/foreign roster responses; missing rows remain recoverable."""
    if not isinstance(response, dict) or not isinstance(response.get('items'), list):
        return {}, 'invalid_items'
    rows = {}; allowed = set(requested_ids)
    for row in response['items']:
        if not isinstance(row, dict) or not isinstance(row.get('entity_id'), str) or row.get('entity_id') not in allowed:
            return {}, 'foreign_entity'
        eid = row['entity_id']
        if eid in rows: return {}, 'duplicate_entity'
        rows[eid] = row
    return rows, None


def validate_ids(identifiers, packet, corpus, task, entity, *, minimum=1, maximum=3):
    if not isinstance(identifiers, list) or not minimum <= len(identifiers) <= maximum:
        return 'source_count'
    if any(not isinstance(x, str) for x in identifiers) or len(set(identifiers)) != len(identifiers):
        return 'duplicate_or_invalid_source_id'
    lookup = {c['id']: c for c in packet['cards']}
    if len(lookup) != len(packet['cards']): return 'ambiguous_source_handles'
    seen_spans = set()
    for identifier in identifiers:
        c = lookup.get(identifier)
        if c is None or not c.get('citable'): return 'unknown_or_uncitable_source'
        doc = corpus.docs.get(c['doc_id'])
        if doc is None or doc.date > task['cutoff_date']: return 'ineligible_source_date'
        if c.get('doc_date') != doc.date: return 'source_card_date_mismatch'
        if not doc.shared and entity['entity_id'] not in doc.entity_ids: return 'foreign_source_scope'
        start, end = c['span_start'], c['span_end']
        if any(not isinstance(x, int) or isinstance(x, bool) for x in (start, end)):
            return 'invalid_source_span_type'
        if not 0 <= start < end <= len(doc.text) or doc.text[start:end] != c['text']:
            return 'source_span_mismatch'
        exclusions = observation_exclusions(doc, task['cutoff_date'], entity.get('latest_published_ref_month'))
        if any(max(start, a) < min(end, b) for a, b in exclusions): return 'future_observation_span'
        key = (c['doc_id'], start, end)
        if key in seen_spans: return 'duplicate_source_span'
        seen_spans.add(key)
    if len(claims_from_ids(packet, identifiers, corpus, task, entity, limit=maximum)) != len(identifiers):
        return 'unresolved_source'
    return None


def normalize_typed(raw, task, entity, packet, corpus, prior, role):
    error = bound(raw, binding(task, entity, packet))
    if error: return None, error
    quantity = raw.get('quantity'); contract = quantity_contract(task, entity)
    if not isinstance(quantity, dict): return None, 'missing_quantity'
    for key in ('kind', 'unit', 'period', 'basis'):
        if quantity.get(key) != contract[key]: return None, 'quantity_mismatch_' + key
    values = [quantity.get(k) for k in ('point', 'lo', 'hi')]
    if any(not isinstance(x, (int, float)) or isinstance(x, bool) or
           abs(x) >= 1e100 or not math.isfinite(x) for x in values):
        return None, 'nonfinite_quantity'
    if values[1] > values[2]: return None, 'reversed_interval'
    support, counter = raw.get('support_ids'), raw.get('counter_ids', [])
    if not isinstance(support, list) or not support or not isinstance(counter, list):
        return None, 'invalid_source_id_list'
    error = validate_ids(support + counter, packet, corpus, task, entity)
    if error: return None, error
    for key, maximum in (('mechanism', 500), ('uncertainty', 300)):
        text = raw.get(key, '')
        if not isinstance(text, str) or len(text) > maximum: return None, 'invalid_' + key
        try: text.encode('utf-8')
        except UnicodeError: return None, 'invalid_unicode_' + key
    if task['target']['type'] == 'classification':
        labels = task['target']['labels']; probabilities = raw.get('label_probabilities')
        if raw.get('label') not in labels: return None, 'invalid_label'
        if probabilities is not None:
            if not isinstance(probabilities, dict) or set(probabilities) != set(labels):
                return None, 'invalid_label_probabilities'
            ps = list(probabilities.values())
            if any(not isinstance(p, (int, float)) or isinstance(p, bool) or
                   not 0 <= p <= 1 or not math.isfinite(p) for p in ps) or not .98 <= sum(ps) <= 1.02:
                return None, 'invalid_label_probabilities'
    native = {k: raw[k] for k in ('entity_id', 'label', 'label_probabilities', 'mechanism',
                                'uncertainty', 'support_ids', 'counter_ids') if k in raw}
    triple = dict(zip(('point', 'lo', 'hi'), values))
    if contract['kind'] == 'future_eps': native['forecast_eps'] = triple
    else: native['point_forecast'] = values[0]
    native['interval'] = {'level': contract['interval_level'], 'lo': values[1], 'hi': values[2]}
    row, error = normalize_forecast(native, task, entity, packet, corpus, prior, view=role)
    if row is not None:
        row['_typed_quantity'] = dict(quantity)
        row['_role'] = role
        row['_binding'] = binding(task, entity, packet)
    return row, error


ISSUE_KINDS = {'unit', 'period', 'denominator', 'accounting', 'scope', 'arithmetic',
               'source_conflict', 'uncertainty', 'classification', 'cross_entity_order'}


def validate_issues(raw, task, entity, packet, corpus):
    error = bound(raw, binding(task, entity, packet))
    if error: return [], error
    issues = raw.get('issues')
    if not isinstance(issues, list) or len(issues) > 3: return [], 'invalid_issues'
    result = []
    for issue in issues:
        if not isinstance(issue, dict) or not isinstance(issue.get('kind'), str) or issue.get('kind') not in ISSUE_KINDS:
            return [], 'invalid_issue_kind'
        if issue.get('candidate') not in ('fundamental', 'challenge'): return [], 'invalid_issue_candidate'
        if not isinstance(issue.get('detail'), str) or not 1 <= len(issue['detail']) <= 500:
            return [], 'invalid_issue_detail'
        try: issue['detail'].encode('utf-8')
        except UnicodeError: return [], 'invalid_issue_unicode'
        error = validate_ids(issue.get('source_ids'), packet, corpus, task, entity)
        if error: return [], error
        result.append(issue)
    return result, None
