"""Budgeted role DAG. Only the coordinator mutates predictions/checkpoints."""
from __future__ import annotations
import concurrent.futures
import json
import math
import time
from ..packets import add_source, retrieve_queries
from .contracts import binding, bound, digest, index_items, normalize_typed, validate_ids, validate_issues
from .prompts import SYSTEM, make_payload, encode_payload, visible_packet

ROLE_TIMEOUT = {'source': 40, 'fundamental': 75, 'challenge': 75, 'recovery': 55,
                'critic': 45, 'referee': 60, 'reasons': 40}


def execute_collaboration(task, active, full_packets, corpus, priors, results, views, rejections,
                          client, checkpoint, plan, *, write_reasons=True):
    """No consensus voting or proxy-scoring of an unvalidated posterior.

    Both independent views see the same immutable source packet, not each other's
    forecast. Critics inspect hypotheses, referees select/correct, reasons follow
    a frozen decision. All HTTP attempts share one client's global cap/deadline.
    """
    n = len(active); count = min(4, max(1, math.ceil(n / 4)))
    candidates = {e['entity_id']: {} for e in active}
    issues = {e['entity_id']: [] for e in active}
    packets = {e['entity_id']: visible_packet(full_packets[e['entity_id']]) for e in active}
    # Measure real context before deciding batch size. Financial observations
    # can dominate card text; splitting preserves them better than deleting them.
    for proposed in range(count, 5):
        size = max(1, math.ceil(n / proposed))
        batches = [active[i:i + size] for i in range(0, n, size)]
        lengths = [len(json.dumps(make_payload('fundamental', task, b, packets, priors),
                   ensure_ascii=False, separators=(',', ':'), allow_nan=False)) for b in batches]
        if max(lengths, default=0) <= 135000: break
    plan.update(version=1, phase='source', batches=len(batches), workers=4,
                planned_calls=5 * len(batches) + int(write_reasons),
                max_recovery_calls=min(3, max(0, client.max_calls - 5 * len(batches) - int(write_reasons))),
                shared_house_client=True, evidence_kind_boundary=True,
                correlated_same_model_views=True, majority_vote=False,
                posterior_utility_selection=False, events=[], reason_diagnostics={},
                task_digest=digest(task), official_score_measured=False)
    plan['forecast_context_chars'] = lengths

    def reject(eid, role, error):
        rejections[eid].append({'stage': role, 'reason': error or 'missing_row'})

    def request(role, batch, *, final_answer=None):
        if client.disabled or not client.enabled or time.monotonic() + 8 >= client.deadline:
            return {}, 'stage_unavailable'
        try:
            payload = make_payload(role, task, batch, packets, priors,
                         candidates=candidates if role in ('critic', 'referee') else None,
                         issues=issues if role == 'referee' else None, final_answer=final_answer,
                         current_decisions=results if role in ('critic', 'referee') else None)
            prompt = encode_payload(payload)
            response = client.complete(SYSTEM, prompt, thinking=False, expected_key='items', stage=role,
                                      request_timeout_seconds=ROLE_TIMEOUT[role])
            if role == 'reasons': return response, None
            return index_items(response, [e['entity_id'] for e in batch])
        except Exception as exc:
            # Never echo a rejected response, token, header or free-form exception.
            return {}, getattr(exc, 'reason_code', type(exc).__name__)

    def run_stage(jobs):
        # Results returned in the scheduled order, never response-completion order.
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(request, role, batch) for role, batch in jobs]
            return [(role, batch, *future.result()) for (role, batch), future in zip(jobs, futures)]

    # Preserve a limited budget for at least one full-roster forecast pass.
    source_jobs = [('source', b) for b in batches] if client.max_calls >= 3 * len(batches) else []
    for role, batch, rows, error in run_stage(source_jobs):
        for e in batch:
            eid = e['entity_id']; raw = rows.get(eid); failure = error or bound(raw, binding(task, e, packets[eid]))
            selected = raw.get('selected_ids') if isinstance(raw, dict) else None
            if not failure: failure = validate_ids(selected, packets[eid], corpus, task, e, minimum=0, maximum=8)
            queries = raw.get('queries') if isinstance(raw, dict) else None
            if not failure and (not isinstance(queries, list) or len(queries) > 3 or
                         any(not isinstance(q, str) or not 1 <= len(q) <= 200 for q in queries)):
                failure = 'invalid_queries'
            if failure: reject(eid, role, failure); continue
            chunks = retrieve_queries(corpus, task, e, queries, limit=3, budget=6000)
            added = []
            for chunk in chunks:
                added.extend(add_source(full_packets[eid], chunk, corpus, task, e, role='followup'))
            packets[eid] = visible_packet(full_packets[eid], selected_ids=selected + added)
            plan['events'].append({'role': role, 'entity_id': eid, 'selected_handles': len(selected),
                                   'retrieved_handles': len(added), 'evidence_digest': digest(packets[eid])})
    checkpoint()
    plan['phase'] = 'independent_forecasts'
    jobs = [(role, b) for role in ('fundamental', 'challenge') for b in batches]
    for role, batch, rows, error in run_stage(jobs):
        for e in batch:
            eid = e['entity_id']
            row, failure = normalize_typed(rows.get(eid), task, e, packets[eid], corpus, priors[eid], role)
            if row is None: reject(eid, role, error or failure); continue
            candidates[eid][role] = row; views[eid][role] = row
    # Deterministic fallback hierarchy: primary -> alternative -> complete prior.
    for e in active:
        eid = e['entity_id']; c = candidates[eid]
        results[eid] = c.get('fundamental', c.get('challenge', priors[eid]))
    checkpoint()

    missing = [e for e in active if not candidates[e['entity_id']]]
    recovery_jobs = [('recovery', missing[i:i + size]) for i in range(0, len(missing), size)]
    recovery_jobs = recovery_jobs[:plan['max_recovery_calls']]
    plan['phase'] = 'recovery'
    for role, batch, rows, error in run_stage(recovery_jobs):
        for e in batch:
            eid = e['entity_id']
            row, failure = normalize_typed(rows.get(eid), task, e, packets[eid], corpus, priors[eid], role)
            if row is None: reject(eid, role, error or failure); continue
            candidates[eid]['fundamental'] = row; views[eid]['fundamental'] = row; results[eid] = row
    checkpoint()

    plan['phase'] = 'critique'
    review_batches = [[e for e in b if candidates[e['entity_id']]] for b in batches]
    review_batches = [b for b in review_batches if b]
    for role, batch, rows, error in run_stage([('critic', b) for b in review_batches]):
        for e in batch:
            eid = e['entity_id']
            found, failure = validate_issues(rows.get(eid), task, e, packets[eid], corpus)
            if not failure and any(issue['candidate'] not in candidates[eid] for issue in found):
                failure = 'critique_of_unavailable_candidate'
            if error or failure: reject(eid, role, error or failure)
            else: issues[eid] = found
    checkpoint()

    plan['phase'] = 'referee'
    for role, batch, rows, error in run_stage([('referee', b) for b in review_batches]):
        for e in batch:
            eid = e['entity_id']; raw = rows.get(eid)
            failure = error or bound(raw, binding(task, e, packets[eid]))
            adopted = None
            if not failure and raw.get('action') == 'select':
                candidate = raw.get('candidate')
                adopted = candidates[eid].get(candidate) if isinstance(candidate, str) else None
                if adopted is None: failure = 'unavailable_selected_candidate'
            elif not failure and raw.get('action') == 'revise':
                index = raw.get('issue_index')
                if not isinstance(index, int) or isinstance(index, bool) or not 0 <= index < len(issues[eid]):
                    failure = 'revision_requires_validated_issue'
                else:
                    adopted, failure = normalize_typed(raw.get('forecast'), task, e, packets[eid], corpus, priors[eid], 'referee')
                    forecast = raw.get('forecast')
                    refs = forecast.get('support_ids', []) if isinstance(forecast, dict) else []
                    if adopted is not None and not set(refs).intersection(issues[eid][index]['source_ids']):
                        adopted = None; failure = 'revision_does_not_cite_issue'
            elif not failure: failure = 'invalid_referee_action'
            if adopted is not None:
                results[eid] = adopted
                plan['events'].append({'role': role, 'entity_id': eid, 'action': raw['action'],
                                       'selected_role': adopted.get('_role')})
            else: reject(eid, role, failure)
    checkpoint()

    # Freeze the whole roster before writing any reason. Workers cannot alter it.
    final_answer = {'entity_predictions': [{k: v for k, v in results[e['entity_id']].items()
                           if not k.startswith('_')} for e in task['entities']]}
    final_answer['_forecast_assumptions'] = {e['entity_id']: {
        'status': 'model hypotheses, not source observations',
        'mechanism': results[e['entity_id']].get('_reasoning', ''),
        'uncertainty': results[e['entity_id']].get('_uncertainty', '')} for e in task['entities']}
    frozen = digest(final_answer['entity_predictions'])
    plan['final_predictions_digest'] = frozen
    plan['phase'] = 'reason_writer'
    reasons = []
    if write_reasons and task['entities'] and not client.disabled and client.attempts < client.max_calls:
        # One unit reason call; choose a compact original packet for every entity.
        reason_packets = {e['entity_id']: visible_packet(full_packets[e['entity_id']],
                  selected_ids=results[e['entity_id']].get('_source_ids', []), card_chars=6000) for e in task['entities']}
        for packet in reason_packets.values():
            packet['observations'] = {}
            packet['context_omissions']['observations_not_shown'] = True
            packet['context_omissions']['reason_role_inputs'] = 'Frozen answers, explicit forecast assumptions and original source cards.'
        original = packets; packets = reason_packets
        response, error = request('reasons', task['entities'], final_answer=final_answer)
        if error: plan['reason_diagnostics'] = {'status': error}
        else:
            from .reasons import compose_reasons
            try:
                reasons, diagnostics = compose_reasons(response, task, packets, corpus, final_answer, frozen)
                plan['reason_diagnostics'] = diagnostics
            except (ValueError, TypeError, KeyError, OverflowError, UnicodeError):
                # Optional explanation failure cannot erase complete predictions.
                reasons = []; plan['reason_diagnostics'] = {'status': 'reason_validation_error'}
        packets = original
    current = [{k: v for k, v in results[e['entity_id']].items() if not k.startswith('_')}
               for e in task['entities']]
    if digest(current) != frozen or digest(final_answer['entity_predictions']) != frozen:
        raise RuntimeError('prediction_freeze_mutated')
    plan.update(phase='complete', candidate_counts={eid: len(c) for eid, c in candidates.items()},
                critique_counts={eid: len(v) for eid, v in issues.items()},
                adopted_roles={e['entity_id']: results[e['entity_id']].get('_role', 'prior') for e in active},
                submitted_reasons=reasons, requests_attempted=client.attempts)
    return plan
