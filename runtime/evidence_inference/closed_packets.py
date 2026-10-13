"""Referentially closed, budgeted original-source views; no inference or I/O.

Numbers, periods, units, table cells and measurement identities are retained.
Unavailable source pointers are moved to explicitly non-citable provenance,
never left in advertised source_id lists. This establishes a data contract,
not source relevance, forecast correctness or production faithfulness.
"""
from __future__ import annotations
from collections import Counter
import copy
import json


ROLE_ORDER = {
    'target_target_quarter_guidance': 0,
    'contract_flow_measurement': 1,
    'target_annual_financial_guidance': 3,
    'target_future_cost_or_tax': 4,
    'target_forward_operating_condition': 5,
    'target_current_operating_driver': 6,
    'going_concern_and_qualification': 6,
    'liquidity_capacity': 7,
    'debt_horizon': 7,
    'financial_measurement_original_row': 12,
    'historical_eps_row': 14,
    'earnings_comparability': 16,
    'target_historical_earnings_components': 18,
    'target_common_income_and_dilution': 19,
    'target_capital_plan_and_execution_condition': 20,
    'period_header': 25,
    'accounting_context': 25,
    'evidence': 60,
}
CITATION_KEYS = {'source_ids', 'citation_ids', 'support_ids', 'counter_ids'}


def encoded_chars(value):
    return len(json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(',', ':')))


def references(value, known, path='$', inherited_role=None):
    """Return exact original handle references plus their dependency role."""
    if isinstance(value, dict):
        role = value.get('dependency_role', value.get('role', inherited_role))
        for key, child in value.items():
            yield from references(child, known, path + '.' + str(key), role)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from references(child, known, path + '[' + str(index) + ']', inherited_role)
    elif isinstance(value, str) and value in known:
        yield path, value, inherited_role


def _pointer_lists(value, known, allowed, omitted, path='$'):
    if isinstance(value, dict):
        result = {}
        for key, child in value.items():
            child_path = path + '.' + str(key)
            if key == 'citation_links_by_measurement':
                if not isinstance(child, dict): raise ValueError('citation links must be a dictionary')
                links = {}
                for mid, ids in child.items():
                    if not isinstance(ids, list) or any(not isinstance(fid, str) for fid in ids):
                        raise ValueError('measurement links must be original source-handle lists')
                    kept = [fid for fid in ids if fid in allowed]
                    lost = [fid for fid in ids if fid not in allowed]
                    links[mid] = kept  # Keep every measurement key, including an empty link.
                    if lost: omitted.append({'path': child_path + '.' + str(mid),
                        'original_source_ids': ids, 'visible_citable_source_ids': kept,
                        'noncitable_source_ids': lost, 'citable': False})
                result[key] = links
            elif key in CITATION_KEYS:
                if not isinstance(child, list) or any(not isinstance(fid, str) for fid in child):
                    raise ValueError('citation pointers must be source-handle lists')
                kept = [fid for fid in child if fid in allowed]
                lost = [fid for fid in child if fid not in allowed]
                result[key] = kept
                if lost: omitted.append({'path': child_path, 'original_source_ids': child,
                    'visible_citable_source_ids': kept, 'noncitable_source_ids': lost, 'citable': False})
            else: result[key] = _pointer_lists(child, known, allowed, omitted, child_path)
        return result
    if isinstance(value, list):
        result = []
        for index, child in enumerate(value):
            child_path = path + '[' + str(index) + ']'
            if isinstance(child, str) and child in known and child not in allowed:
                omitted.append({'path': child_path, 'original_source_ids': [child],
                    'visible_citable_source_ids': [], 'noncitable_source_ids': [child], 'citable': False})
                result.append(None)
            else: result.append(_pointer_lists(child, known, allowed, omitted, child_path))
        return result
    if isinstance(value, str) and value in known and value not in allowed:
        omitted.append({'path': path, 'original_source_ids': [value],
                        'visible_citable_source_ids': [], 'noncitable_source_ids': [value], 'citable': False})
        return None
    return copy.deepcopy(value)


def _financial_priority(packet, registry):
    """Prefer observed EPS identity inputs and current-quarter driver rows."""
    financial = packet.get('observations', {}).get('financial_measurement', {})
    if not isinstance(financial, dict): return {}
    links = financial.get('citation_links_by_measurement', {})
    if not isinstance(links, dict): raise ValueError('invalid financial citation links')
    mid_priority = {}
    for identity in financial.get('historical_eps_identities', []):
        for key in ('numerator_measurement_id', 'shares_measurement_id'):
            if identity.get(key): mid_priority[identity[key]] = 2
    current = financial.get('latest_observed_quarter_end')
    for table in financial.get('tables', []):
        columns = table.get('columns', {})
        current_columns = {int(k) for k, col in columns.items()
                           if col.get('period_end') == current and col.get('period_months') == 3}
        for row in table.get('rows', []):
            for cell in row.get('cells', []):
                if len(cell) >= 2 and cell[1] in current_columns:
                    mid_priority[cell[0]] = min(mid_priority.get(cell[0], 11), 11)
    priorities = {}
    for mid, priority in mid_priority.items():
        for fid in links.get(mid, []):
            if fid in registry: priorities[fid] = min(priorities.get(fid, priority), priority)
    return priorities


def select_closed_packet(packet, *, card_chars=22000, selected_ids=()):
    """Return canonical selected cards, intact measurements and explicit omissions.

    The caller still verifies corpus/date/entity scope when resolving a chosen
    handle. This function never creates a span, alters source text, adds a fact,
    admits an unseen full-registry ID, or silently catches a malformed context.
    """
    if not isinstance(packet, dict) or not isinstance(packet.get('cards'), list):
        raise ValueError('packet/cards shape')
    if not isinstance(card_chars, int) or isinstance(card_chars, bool) or card_chars < 2:
        raise ValueError('card budget must include at least the two JSON list brackets')
    cards = packet['cards']; registry = {}
    for card in cards:
        if not isinstance(card, dict) or not isinstance(card.get('id'), str): raise ValueError('source card shape')
        if card['id'] in registry: raise ValueError('duplicate source handle')
        for key in ('doc_id', 'doc_date', 'role', 'text', 'span_start', 'span_end', 'citable'):
            if key not in card: raise ValueError('incomplete original source card')
        registry[card['id']] = card
    if any(fid not in registry for fid in selected_ids): raise ValueError('unknown explicit source selection')
    contract=packet.get('observations',{}).get('contract_flow_measurement')
    if isinstance(contract,dict):
        # Preserve the previous selected evidence before adding a complete
        # contract/offset bundle. A fixed22K substitution had removed useful
        # regional revenue facts in the actual source audit.
        added=set(contract.get('source_cards_added',[]))
        if not added<=set(registry):raise ValueError('unknown contract source additions')
        baseline=copy.deepcopy(packet)
        baseline['observations'].pop('contract_flow_measurement',None)
        baseline['cards']=[card for card in baseline['cards'] if card['id'] not in added]
        base=select_closed_packet(baseline,card_chars=card_chars,
            selected_ids=[fid for fid in selected_ids if fid not in added])
        required=set(selected_ids)|{card['id'] for card in base['cards']}
        for context in contract.get('observed_bundle',{}).get('source_contexts',[]):
            required.update(context.get('source_ids',[]))
        if not required<=set(registry):raise ValueError('unknown contract context reference')
        required_chars=encoded_chars([registry[fid] for fid in registry if fid in required])
        if required_chars>35000:
            base['optional_measurement_diagnostics']={'contract_flow_omitted':
                'complete bundle plus baseline exceeds35000 source-card characters',
                'required_card_chars':required_chars}
            return base
        card_chars=max(card_chars,required_chars)
        selected_ids=tuple(required)
    refs = list(references(packet.get('observations', {}), set(registry)))
    counts = Counter(fid for _, fid, _ in refs)
    priorities = _financial_priority(packet, registry)
    for path, fid, context_role in refs:
        priority = ROLE_ORDER.get(context_role, ROLE_ORDER.get(registry[fid]['role'], 45))
        if '.citation_links_by_measurement.' in path: priority = min(priority, 12)
        priorities[fid] = min(priorities.get(fid, priority), priority)
    for fid in selected_ids: priorities[fid] = -1
    original_order = {card['id']: i for i, card in enumerate(cards)}
    ordered = sorted(cards, key=lambda card: (
        priorities.get(card['id'], ROLE_ORDER.get(card['role'], 50)),
        -counts[card['id']], original_order[card['id']]))
    visible = []; used = 2  # Actual compact JSON list punctuation.
    for card in ordered:
        addition = encoded_chars(card) + int(bool(visible))
        if used + addition <= card_chars:
            visible.append(copy.deepcopy(card)); used += addition
    allowed = {card['id'] for card in visible if card['citable'] is True}
    omitted = []
    value = {key: copy.deepcopy(child) for key, child in packet.items() if key not in ('cards', 'observations')}
    value['cards'] = visible
    value['observations'] = _pointer_lists(packet.get('observations', {}), set(registry), allowed, omitted)
    unresolved = list(references(value['observations'], set(registry)))
    if any(fid not in allowed for _, fid, _ in unresolved): raise ValueError('internal citation closure violation')
    value['noncitable_source_provenance'] = {
        'status': 'Omitted or uncitable reference provenance; never usable as support_ids.',
        'references': omitted,
        'source_records': {fid: {key: registry[fid][key] for key in
                ('doc_id', 'doc_date', 'span_start', 'span_end', 'role')} | {'citable': False}
            for fid in sorted({fid for row in omitted for fid in row['noncitable_source_ids'] if fid in registry})},
    }
    value['context_omissions'] = {
        'cards_not_shown': len(cards) - len(visible), 'original_card_count': len(cards),
        'selected_card_chars': encoded_chars(visible), 'card_budget_chars': card_chars,
        'advertised_reference_count': len(unresolved), 'noncitable_reference_groups': len(omitted),
        'usable_citable_cards': len(allowed), 'observed_numeric_values_removed': False,
        'policy': 'Only selected citable cards are advertised as citation handles; omitted associations stay explicitly noncitable.',
    }
    if isinstance(contract,dict):
        retained={card['id'] for card in visible}
        if not required<=retained:raise ValueError('baseline or complete contract bundle lost')
        value['context_omissions']['baseline_source_inclusion_preserved']=True
        value['context_omissions']['complete_contract_source_inclusion_preserved']=True
    return value
