"""Compile actual decision records to the optional official reason contract.

Grouping never silently drops part of a roster. A group exceeding published
caps is omitted with diagnostics. Local checks cannot certify hidden-reason
coverage, causal correctness, Final NLI or official R.
"""
from .harness import official_reason_rail as rail
from .harness.contracts import digest
from .harness.reasons import _rail_docs
import json


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
        return premise, f"{eid}: {mechanism}", implication(entry)
    raise ValueError("unsupported_decision_kind")


def make_reason(entries, reason_id):
    parts = [pieces(e) for e in entries]
    mechanisms = list(dict.fromkeys(p[1] for p in parts))
    premises=[p[0]for p in parts]
    if entries[0]['decision']['kind']=='house_same_response':
        # Shared original passages and identical inferential links are stated
        # once with every affected entity named. No source span/text is edited.
        source_groups={};link_groups={};task_input_groups={}
        for entry in entries:
            eid=entry['entity_id'];decision=entry['decision']
            source_groups.setdefault(decision.get('premise_fact',{}).get('quote',entry['source_quote']),[]).append(eid)
            text=decision['mechanism']
            if decision.get('assumptions'):
                text+=' Assumptions: '+'; '.join(decision['assumptions'])
            link_groups.setdefault(text,[]).append(eid)
            if decision.get('task_inputs'):
                task_input_groups.setdefault(decision['task_inputs'],[]).append(eid)
        task_inputs=[','.join(ids)+': task inputs '+text for text,ids in task_input_groups.items()]
        premises=[','.join(ids)+': '+text for text,ids in source_groups.items()]+task_inputs
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


def compile_reasons(task, final_answer, ledger, corpus):
    predictions = final_answer["entity_predictions"]
    before = digest(predictions)
    diagnostics = {"accepted": 0, "groups": [], "covered_entities": [],
                   "official_reasoning_grade_measured": False,
                   "scope": "decision consistency, source provenance and published local shape/caps only"}
    if ledger["task_digest"] != digest(task):
        diagnostics["context_error"] = "task_changed"
        return [], diagnostics
    rows = {row["entity_id"]: row for row in predictions}
    if len(rows) != len(predictions) or set(rows) != {e["entity_id"] for e in task["entities"]}:
        diagnostics["context_error"] = "invalid_final_roster"
        return [], diagnostics
    groups = {}
    for entry in ledger["entries"]:
        eid = entry["entity_id"]
        projected = {k: rows[eid][k] for k in ("entity_id", "label", "point_forecast", "interval") if k in rows[eid]}
        if digest(projected) != entry["answer_digest"]:
            diagnostics["context_error"] = "answer_changed_after_decision"
            return [], diagnostics
        key = entry["decision"]["kind"], entry["decision"].get("method", "joint")
        groups.setdefault(key, []).append(entry)
    reasons = []
    docs = _rail_docs(corpus.docs)

    def checked(entries):
        try:
            reason = make_reason(entries, "r" + str(len(reasons) + 1))
        except ValueError as exc:
            return None, [str(exc)]
        proposed = dict(final_answer, submitted_reasons=reasons + [reason])
        findings = rail.check_submitted_reasons(proposed, docs, task["cutoff_date"])
        plan = rail.reasons_judged(proposed, docs, task["cutoff_date"])
        if findings or any(not p["judged"] for p in plan):
            return None, [f.code for f in findings] or ["local_judged_plan_refused"]
        return reason, []

    def commit(entries):
        ids = [e["entity_id"] for e in entries]
        if len(reasons) >= rail.MAX_REASONS:
            diagnostics["groups"].append({"entities": ids, "status": "omitted_unit_reason_count_cap"})
            return
        reason, errors = checked(entries)
        if errors:
            diagnostics["groups"].append({"entities": ids, "status": "omitted_local_rail_findings", "codes": errors})
            return
        reasons.append(reason)
        diagnostics["covered_entities"].extend(ids)
        diagnostics["groups"].append({"entities": ids, "status": "accepted"})

    for entries in groups.values():
        # The total 6500-byte unit cap cannot be avoided by dividing the prose.
        # Reuse a common statistical mechanism once instead of per-entity padding.
        reason, errors = checked(entries)
        if not errors:
            commit(entries)
            continue
        # Isolate an individually invalid link without discarding other grounded
        # decisions. Nothing is clipped, and every omitted entity is reported.
        pending = []
        for entry in entries:
            _, single_errors = checked([entry])
            if single_errors:
                commit([entry])
                continue
            _, proposed_errors = checked(pending + [entry])
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
