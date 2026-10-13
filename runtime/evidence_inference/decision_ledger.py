"""Bind reasons to the actual decision, exact numeric answer and owned sources."""
import copy
import math

from .harness.contracts import digest
from .harness.reasons import _resolve_refs
from .final_policy import decision_exception
from .reason_premise import validate_record
from .decision_witness import validate_record as validate_witness_record


def projection(row):
    return {k: copy.deepcopy(row[k]) for k in ("entity_id", "label", "point_forecast", "interval") if k in row}


def build_ledger(task, results, packets, corpus):
    entries, omitted = [], []
    for entity in task["entities"]:
        eid = entity["entity_id"]
        row = results[eid]
        decision = row.get("_forecast_decision")
        if not isinstance(decision, dict):
            omitted.append({"entity_id": eid, "reason": "no_decision_record"})
            continue
        native = decision.get("native_output")
        expected = {k: row[k] for k in ("point_forecast", "interval", "label") if k in row}
        if native != expected:
            omitted.append({"entity_id": eid, "reason": "decision_answer_mismatch"})
            continue
        source_ids = decision.get("source_ids")
        refs = [{"entity_id": eid, "source_ids": source_ids}]
        try:
            if decision.get('kind') == 'house_same_response':
                decision = copy.deepcopy(decision)
                witness = row.get('_decision_witness')
                if witness is not None:
                    decision['arithmetic_witness'] = validate_witness_record(witness, task, entity, row['point_forecast'])
                    decision['arithmetic_witness_status'] = 'declared_arithmetic_verified_only'
                else:
                    decision['arithmetic_witness_status'] = 'absent_arithmetic_unverified'
            quote, citations, _ = _resolve_refs(refs, task, packets, corpus.docs)
            if decision.get('premise_fact') is not None and not validate_record(
                    decision['premise_fact'],source_ids,packets[eid],corpus):
                raise ValueError('decision_premise_source_mismatch')
            if not all(type(x) in (int, float) and math.isfinite(x) for x in
                       (row["point_forecast"], row["interval"]["lo"], row["interval"]["hi"])):
                raise ValueError("nonfinite_decision")
            if decision.get("artifact_available_by", task["cutoff_date"]) > task["cutoff_date"] and not decision_exception(decision):
                raise ValueError("decision_artifact_unavailable")
            # A statistical record may not be edited independently of its arithmetic.
            if decision["kind"] == "statistical_eps_anchor":
                a, b, c = (decision["inputs"][key] for key in ("prior", "last", "prior_last"))
                observations = decision["observations"]
                if len(observations) != 2 or [o["value"] for o in observations] != [b, c]:
                    raise ValueError("decision_observation_value_mismatch")
                for observation in observations:
                    doc = corpus.docs.get(observation["doc_id"])
                    if doc is None or doc.date != observation["doc_date"] or observation["period_end"] > doc.date:
                        raise ValueError("decision_observation_period_or_date_mismatch")
                method = decision["method"]
                if method == "lad3":
                    eps = sum(x * y for x, y in zip(decision["coefficients"], (a, b, a - c)))
                elif method == "prior":
                    eps = a
                elif method == "last":
                    eps = b
                elif method == "seasonal":
                    eps = a + b - c
                elif method.startswith("shrink_"):
                    eps = a + float(method.removeprefix("shrink_")) * (b - c)
                else:
                    raise ValueError("unknown_decision_formula")
                if eps != decision["physical_forecast"]["point"]:
                    raise ValueError("decision_arithmetic_mismatch")
                if not math.isclose(100 * (eps / a - 1), row["point_forecast"], abs_tol=1e-9):
                    raise ValueError("decision_native_transform_mismatch")
                if a != entity.get("prior_year_q_eps"):
                    raise ValueError("decision_task_input_changed")
            elif decision["kind"] == "statistical_cpi_anchor":
                features = decision["features"]
                coefficients = decision["coefficients"]
                if len(features) != 4 or len(coefficients) != 4:
                    raise ValueError("decision_cpi_feature_shape")
                point = decision["intercept"] + sum(a*b for a,b in zip(coefficients, features))
                if point != row["point_forecast"]:
                    raise ValueError("decision_cpi_arithmetic_mismatch")
                change = 100*(decision["pump_mean_current"]/decision["pump_mean_previous"]-1)
                if change != features[3] or features[0] != entity.get("latest_published_mom_pct"):
                    raise ValueError("decision_cpi_observation_mismatch")
            elif decision["kind"] == "house_eps_adjustments":
                anchor=decision["anchor"]
                point=anchor["physical_forecast"]["point"]+sum(a["eps_delta"]for a in decision["adjustments"])
                if point != decision["physical_forecast"]["point"]:
                    raise ValueError("decision_adjustment_sum_mismatch")
                base=anchor["inputs"]["prior"]
                if base != entity.get("prior_year_q_eps") or not math.isclose(100*(point/base-1),row["point_forecast"],abs_tol=1e-9):
                    raise ValueError("decision_adjustment_native_mismatch")
            elif decision['kind']=='statistical_practice_anchor':
                features=decision['features'];model=decision['model'];family=decision['family']
                if family=='macro':
                    sign=model['late_sign']if features['source_release_events']>=model['release_threshold']else model['early_sign']
                    expected_point=decision['task_inputs']['latest_precutoff_estimate']+sign*features['historical_revision_scale']
                    if row.get('label')!=('up'if sign>0 else'down'):raise ValueError('practice_macro_label_mismatch')
                else:
                    names=model['feature_names'];coeff=model['coefficients']
                    if len(names)!=len(coeff):raise ValueError('practice_equation_shape')
                    expected_point=sum(features[n]*c for n,c in zip(names,coeff))
                    if model.get('offset_feature'):expected_point+=features[model['offset_feature']]
                    if family=='eps_growth':
                        expected_point=100*(expected_point/decision['task_inputs']['prior_year_q_eps']-1)
                if expected_point!=row['point_forecast']:raise ValueError('practice_arithmetic_mismatch')
        except (ValueError, TypeError, KeyError, ZeroDivisionError, OverflowError) as exc:
            omitted.append({"entity_id": eid, "reason": str(exc)})
            continue
        entries.append({"entity_id": eid, "decision": copy.deepcopy(decision),
                        "answer": projection(row), "answer_digest": digest(projection(row)),
                        "source_refs": refs, "source_quote": quote, "citations": citations,
                        "classification_labels": copy.deepcopy(task["target"].get("labels", []))})
    return {"purpose": "internal actual-decision record, not competition output schema",
            "task_digest": digest(task), "entries": entries, "omitted": omitted,
            "complete_decision_roster": len(entries) == len(task["entities"]),
            "official_reasoning_grade_measured": False}
