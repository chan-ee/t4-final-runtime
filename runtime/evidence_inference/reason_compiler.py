"""Compile actual decision records to the optional official reason contract.

Grouping never silently drops part of a roster. Reason, citation and evidence
caps remain strict. An answer-only advisory over-count does not remove otherwise
valid prose: the actual grader's sealed answer projection remains authoritative.
Local checks cannot certify that answer cap, hidden-reason coverage, causal
correctness, Final NLI or official R.
"""
from .harness import official_reason_rail as rail
from .harness.contracts import digest
from .harness.reasons import _rail_docs
import json
from .decision_witness import render_parts as witness_parts, validate_record as validate_witness_record


def number(value):
    return format(value, ".7g")


def implication(entry):
    answer = entry["answer"]
    result = f"{entry['entity_id']}: point {number(answer['point_forecast'])}"
    if "label" in answer:
        result += f", label {answer['label']}"
    return result


def pieces(entry):
    decision = entry["decision"]
    eid = entry["entity_id"]
    if decision['kind']=='statistical_practice_anchor':
        family=decision['family'];features=decision['features'];model=decision['model']
        inputs=json.dumps(decision['task_inputs'],ensure_ascii=False,sort_keys=True,separators=(',',':'))
        values=json.dumps(features,ensure_ascii=False,sort_keys=True,separators=(',',':'))
        premise=f"{eid}: supplied task quantities {inputs}; source-derived historical inputs and statistical transforms {values}."
        if family=='macro':
            mechanism=(f"Observed source publication cadence gives a heuristic number of additional release events. "
                f"At threshold {model['release_threshold']}, the forecast uses one directional regime and otherwise the other; "
                "historical revision magnitudes set the level adjustment and uncertainty. Cadence and direction are statistical hypotheses, "
                "not guaranteed future publication dates or known revised values.")
        else:
            pairs=', '.join(n+'*'+number(c)for n,c in zip(model['feature_names'],model['coefficients']))
            mechanism="Statistical forecasting hypothesis: sum("+pairs+"). "
            if model.get('offset_feature'):mechanism+='Add '+model['offset_feature']+'. '
            if family=='eps_growth':mechanism+='This predicts EPS; native growth is100*(forecast EPS/task prior EPS-1). '
            elif family=='eps_direction':mechanism+='Compare predicted EPS with the supplied prior EPS for the directional label. '
            elif family=='reaction':mechanism+='Forecast is company-minus-benchmark return percent, with the supplied neutral threshold for labels. '
            elif family=='cot':mechanism+='The score is forecast net-position change as percent of base open interest, not an ordinal rank. '
            elif family=='rates':mechanism+='Output is future Treasury yield change in bps, not the policy-rate action. '
            elif family=='cpi':mechanism+='Lagged source changes forecast the requested first-print monthly component percent; it is not an observed future print. '
            mechanism+='Fitted associations are not causal proof. Fit-residual intervals have unverified future coverage.'
        return premise,mechanism,implication(entry)
    if decision["kind"] == "statistical_eps_anchor":
        inputs = decision["inputs"]
        latest, old = decision["observations"]
        premise = (f"{eid}: task prior-year target-quarter EPS {number(inputs['prior'])} USD/share; "
                   f"source diluted EPS {number(inputs['last'])} for three months ended {latest['period_end']}, "
                   f"and {number(inputs['prior_last'])} for corresponding three months ended {old['period_end']}. ")
        method = decision["method"]
        if method == "lad3":
            coeff = ','.join(number(x) for x in decision['coefficients'])
            mechanism = (f"Statistical assumption: coefficients ({coeff}) combine prior target-quarter EPS, "
                         "latest EPS and prior target-quarter minus prior corresponding latest-quarter EPS. "
                         "This uses recent earnings state and a seasonal contrast, not an observed future result or a causal estimate. "
                         "Native growth is 100*(predicted EPS/task prior EPS-1). Residual quantiles use a separate earlier period; "
                         "future coverage and stability are unverified.")
        else:
            mechanism = (f"Statistical {method} forecast from the stated historical EPS observations; "
                         "growth is 100*(predicted EPS/task prior EPS-1). The future is uncertain; "
                         "the historical relation need not persist and residual quantiles are not a coverage guarantee.")
        consequence = (f"{eid}: predicted EPS {number(decision['physical_forecast']['point'])} USD/share "
                       f"implies growth {number(entry['answer']['point_forecast'])} percent.")
        return premise, mechanism, consequence
    if decision["kind"] == "house_eps_adjustments":
        anchor=decision["anchor"]
        anchor_entry=dict(entry,decision=anchor)
        premise,_,_=pieces(anchor_entry)
        # The original source passages qualify the adjustment premises; the
        # numerical deltas below remain estimates and never become source quotes.
        premise += " Cited driver facts: " + " ".join(a['premise']for a in decision['adjustments'])
        links=[f"Assumed delta {number(a['eps_delta'])} USD/share: {a['mechanism']} Assumes: {'; '.join(a['assumptions'])}."
               for a in decision['adjustments']]
        mechanism=(f"{eid}: anchor EPS {number(anchor['physical_forecast']['point'])} USD/share. "
                   +(" ".join(links) if links else "No source-supported adjustment is made. ")
                   )
        consequence=f"{eid}: forecast EPS {number(decision['physical_forecast']['point'])} USD/share implies growth {number(entry['answer']['point_forecast'])} percent."
        return premise,mechanism,consequence
    if decision["kind"] == "statistical_cpi_anchor":
        premise = (f"{eid}: task and source latest seasonally adjusted monthly CPI change "
                   f"{number(decision['latest_mom_pct'])} percent for {decision['latest_ref_month']}. "
                   f"Arithmetic means of cited regular-gasoline weekly prices are "
                   f"{number(decision['pump_mean_previous'])} and {number(decision['pump_mean_current'])} USD/gallon "
                   f"for the previous and target reference months; their derived change is {number(decision['features'][3])} percent.")
        mechanism = ("Regular-gasoline pump prices provide a contemporaneous price proxy for gasoline and energy CPI. "
                     "The forecast uses the fitted price-change coefficient, latest component change and seasonal terms. "
                     "Regular-grade unadjusted prices are not observed all-types seasonally adjusted CPI; "
                     "price transmission, grade composition and seasonal adjustment remain statistical assumptions. "
                     "Separate earlier residuals supply the interval without a proven future coverage guarantee.")
        return premise, mechanism, f"{eid}: this model implies {number(entry['answer']['point_forecast'])} percent monthly CPI change for {decision['target_ref_month']}."
    if decision["kind"] == "house_same_response":
        mechanism = decision.get("mechanism", "")
        if not mechanism.strip():
            raise ValueError("decision_missing_inferential_link")
        assumptions = decision.get("assumptions", [])
        if assumptions:
            mechanism += " Assumptions: " + "; ".join(assumptions)
        premise = f"{eid}: " + decision.get('premise_fact',{}).get('quote',entry["source_quote"])
        if decision.get("task_inputs"):
            premise += " Task inputs: " + decision["task_inputs"]
        consequence = implication(entry)
        if decision.get('arithmetic_witness') is not None:
            observed, conditional, arithmetic = witness_parts(decision['arithmetic_witness'])
            premise += ' ' + observed
            mechanism += ' ' + conditional
            consequence += '; declared native-point bridge ' + arithmetic + '. Label, ordering and interval derivation are unverified.'
        return premise, f"{eid}: {mechanism}", consequence
    raise ValueError("unsupported_decision_kind")


def make_reason(entries, reason_id):
    parts = [pieces(e) for e in entries]
    mechanisms = list(dict.fromkeys(p[1] for p in parts))
    premises=[p[0]for p in parts]
    if entries[0]['decision']['kind']=='house_same_response':
        # Shared original passages and identical inferential links are stated
        # once with every affected entity named. No source span/text is edited.
        source_groups={};link_groups={};task_input_groups={};witness_premises=[]
        for entry in entries:
            eid=entry['entity_id'];decision=entry['decision']
            source_groups.setdefault(decision.get('premise_fact',{}).get('quote',entry['source_quote']),[]).append(eid)
            text=decision['mechanism']
            if decision.get('assumptions'):
                text+=' Assumptions: '+'; '.join(decision['assumptions'])
            if decision.get('arithmetic_witness') is not None:
                observed, conditional, _ = witness_parts(decision['arithmetic_witness'])
                witness_premises.append(eid+': '+observed)
                text+=' '+conditional
            link_groups.setdefault(text,[]).append(eid)
            if decision.get('task_inputs'):
                task_input_groups.setdefault(decision['task_inputs'],[]).append(eid)
        task_inputs=[','.join(ids)+': task inputs '+text for text,ids in task_input_groups.items()]
        premises=[','.join(ids)+': '+text for text,ids in source_groups.items()]+task_inputs+witness_premises
        mechanisms=[','.join(ids)+': '+text for text,ids in link_groups.items()]
    if entries[0]['decision']['kind']=='house_eps_adjustments':
        mechanisms.append("Code sums conditional EPS deltas with the eligible statistical anchor, then converts to percent growth against task prior EPS. These deltas are uncertain estimates, not observed source amounts; future coverage is unverified.")
    citations, seen = [], set()
    for entry in entries:
        for citation in entry["citations"]:
            identity = citation["doc_id"], citation["span_start"], citation["span_end"]
            if identity not in seen:
                citations.append(citation)
                seen.add(identity)
    return {"reason_id": reason_id, "premise": "\n".join(premises),
            "mechanism": "\n".join(mechanisms), "answer_implication": "\n".join(p[2] for p in parts),
            "scope": {"entities": [e["entity_id"] for e in entries]}, "citations": citations}


def _reason_preflight(answer, docs, cutoff, actual_ids):
    """Check actual scopes, and isolate only the advisory answer-size finding.

    The empty-row view is internal and never becomes an answer. It checks only
    the reason/citation/deny-list resource rules, without guessing a sealed key
    or implying that the organizer can judge the actual answer.
    """
    reasons = answer.get("submitted_reasons")
    if not isinstance(reasons, list):
        return ["invalid_reason_list"], {"status": "invalid_reason_list"}
    for reason in reasons:
        if not isinstance(reason, dict):
            return ["invalid_reason_object"], {"status": "invalid_reason_object"}
        scope = reason.get("scope")
        if not isinstance(scope, dict):
            return ["invalid_reason_scope"], {"status": "invalid_reason_scope"}
        ids = scope.get("entities")
        if (not isinstance(ids, list) or not ids
                or any(not isinstance(e, str) or not e for e in ids)
                or len(set(ids)) != len(ids) or not set(ids) <= actual_ids):
            return ["invalid_actual_reason_scope"], {"status": "invalid_actual_reason_scope"}
    findings = rail.check_submitted_reasons(answer, docs, cutoff)
    codes = [f.code for f in findings]
    diagnostic = {"actual_grader_answer_cap_certified": False,
                  "actual_answer_cap_unknown": True,
                  "auxiliary_view_never_submitted": True}
    if findings and any(code != "cap_answer_bytes" for code in codes):
        diagnostic.update(status="known_reason_rail_failure", codes=codes)
        return codes, diagnostic
    checking = answer
    if findings:
        checking = dict(answer, entity_predictions=[])
        reason_findings = rail.check_submitted_reasons(checking, docs, cutoff)
        if reason_findings:
            codes = [f.code for f in reason_findings]
            diagnostic.update(status="known_reason_only_failure", codes=codes)
            return codes, diagnostic
        diagnostic.update(status="actual_answer_cap_unknown", advisory_codes=codes)
    else:
        diagnostic["status"] = "local_advisory_passed"
    plan = rail.reasons_judged(checking, docs, cutoff)
    if any(not entry["judged"] for entry in plan):
        diagnostic.update(status="local_reason_only_plan_refused", reason_only_plan=plan)
        return ["local_reason_only_plan_refused"], diagnostic
    diagnostic["reason_only_plan"] = plan
    return [], diagnostic


def compile_reasons(task, final_answer, ledger, corpus):
    diagnostics = {"accepted": 0, "groups": [], "covered_entities": [],
                   "official_reasoning_grade_measured": False,
                   "actual_grader_answer_cap_certified": False,
                   "actual_answer_cap_unknown": True,
                   "advisory_answer_cap_overflow": False,
                   "scope": "decision consistency, source provenance and published local shape/caps only"}
    predictions = final_answer.get("entity_predictions") if isinstance(final_answer, dict) else None
    task_rows = task.get("entities") if isinstance(task, dict) else None
    if (not isinstance(predictions, list) or not isinstance(task_rows, list) or not task_rows
            or any(not isinstance(row, dict) or not isinstance(row.get("entity_id"), str)
                   or not row["entity_id"] for row in predictions + task_rows)):
        diagnostics["context_error"] = "invalid_final_roster"
        return [], diagnostics
    actual_ids = [row["entity_id"] for row in predictions]
    task_ids = [row["entity_id"] for row in task_rows]
    if (len(set(actual_ids)) != len(actual_ids) or len(set(task_ids)) != len(task_ids)
            or set(actual_ids) != set(task_ids)):
        diagnostics["context_error"] = "invalid_final_roster"
        return [], diagnostics
    if not isinstance(ledger, dict) or not isinstance(ledger.get("entries"), list):
        diagnostics["context_error"] = "invalid_decision_ledger"
        return [], diagnostics
    try:
        before = digest(predictions)
        task_digest = digest(task)
    except (ValueError, TypeError, OverflowError):
        diagnostics["context_error"] = "invalid_reason_context"
        return [], diagnostics
    if ledger.get("task_digest") != task_digest:
        diagnostics["context_error"] = "task_changed"
        return [], diagnostics
    rows = {row["entity_id"]: row for row in predictions}
    groups, seen_entries = {}, set()
    for entry in ledger["entries"]:
        if (not isinstance(entry, dict) or not isinstance(entry.get("entity_id"), str)
                or entry["entity_id"] not in rows or entry["entity_id"] in seen_entries
                or not isinstance(entry.get("decision"), dict)
                or not isinstance(entry["decision"].get("kind"), str)
                or not isinstance(entry["decision"].get("method", "joint"), str)):
            diagnostics["context_error"] = "invalid_decision_ledger"
            return [], diagnostics
        eid = entry["entity_id"]
        seen_entries.add(eid)
        try:
            if not isinstance(entry.get("answer"), dict):
                raise ValueError("invalid_cached_decision_answer")
            cached_answer_digest = digest(entry["answer"])
        except (ValueError, TypeError, OverflowError):
            diagnostics["context_error"] = "invalid_cached_decision_answer"
            return [], diagnostics
        if cached_answer_digest != entry.get("answer_digest"):
            diagnostics["context_error"] = "cached_decision_answer_changed"
            return [], diagnostics
        witness = entry['decision'].get('arithmetic_witness')
        if witness is not None:
            try:
                entity = next(e for e in task_rows if e['entity_id'] == eid)
                validate_witness_record(witness, task, entity, rows[eid]['point_forecast'])
            except (ValueError, TypeError, KeyError, OverflowError):
                diagnostics['context_error'] = 'arithmetic_witness_record_changed'
                return [], diagnostics
        projected = {k: rows[eid][k] for k in ("entity_id", "label", "point_forecast", "interval") if k in rows[eid]}
        if digest(projected) != entry.get("answer_digest"):
            diagnostics["context_error"] = "answer_changed_after_decision"
            return [], diagnostics
        key = entry["decision"].get("kind"), entry["decision"].get("method", "joint")
        groups.setdefault(key, []).append(entry)
    reasons = []
    docs = _rail_docs(corpus.docs)

    def checked(entries):
        try:
            reason = make_reason(entries, "r" + str(len(reasons) + 1))
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            return None, [str(exc)], {}
        proposed = dict(final_answer, submitted_reasons=reasons + [reason])
        errors, preflight = _reason_preflight(proposed, docs, task["cutoff_date"], set(rows))
        if errors:
            return None, errors, preflight
        return reason, [], preflight

    def commit(entries):
        ids = [e["entity_id"] for e in entries]
        if len(reasons) >= rail.MAX_REASONS:
            diagnostics["groups"].append({"entities": ids, "status": "omitted_unit_reason_count_cap"})
            return
        reason, errors, preflight = checked(entries)
        if errors:
            diagnostics["groups"].append({"entities": ids, "status": "omitted_local_rail_findings", "codes": errors})
            return
        reasons.append(reason)
        diagnostics["covered_entities"].extend(ids)
        diagnostics["groups"].append({"entities": ids, "status": "accepted", "local_reason_rail": preflight})
        if preflight.get("status") == "actual_answer_cap_unknown":
            diagnostics["advisory_answer_cap_overflow"] = True

    for entries in groups.values():
        # The total 6500-byte unit cap cannot be avoided by dividing the prose.
        # Reuse a common statistical mechanism once instead of per-entity padding.
        reason, errors, _ = checked(entries)
        if not errors:
            commit(entries)
            continue
        # Isolate an individually invalid link without discarding other grounded
        # decisions. Nothing is clipped, and every omitted entity is reported.
        pending = []
        for entry in entries:
            _, single_errors, _ = checked([entry])
            if single_errors:
                commit([entry])
                continue
            _, proposed_errors, _ = checked(pending + [entry])
            if proposed_errors and pending:
                commit(pending)
                pending = []
            pending.append(entry)
        if pending:
            commit(pending)
    if digest(predictions) != before:
        raise ValueError("reason_compilation_changed_predictions")
    diagnostics.update(accepted=len(reasons), predictions_digest=before,
                       complete_reason_roster=set(diagnostics["covered_entities"]) == set(rows))
    return reasons, diagnostics
