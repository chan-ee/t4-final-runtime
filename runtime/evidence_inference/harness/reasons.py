"""Optional unit reasons, bound to frozen predictions and original source cards.

House chooses source handles and writes mechanisms; it never supplies trusted
quotes or offsets. This adapter does not mutate the prediction answer. Its
checks establish shape, provenance and published local caps, not inferential
quality, hidden-reason coverage or an official reasoning score.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import re
from collections.abc import Mapping
from types import SimpleNamespace

from ..packets import observation_exclusions
from . import official_reason_rail as rail


def digest(value):
    """Canonical digest shared by task and frozen entity_predictions bindings."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"),
                         ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _field(value, name, default=None):
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _date(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("invalid_calendar_date")
    return dt.date.fromisoformat(value)


def _docs(corpus):
    value = corpus if isinstance(corpus, Mapping) else getattr(corpus, "docs", None)
    return value if isinstance(value, Mapping) else {}


def _rail_docs(documents):
    return {
        doc_id: rail.CorpusDoc(doc_id, _field(doc, "text", ""),
                               _field(doc, "date", _field(doc, "doc_date")))
        for doc_id, doc in documents.items()
        if isinstance(doc_id, str) and isinstance(_field(doc, "text"), str)
    }


def _named(text, entity):
    names = [entity.get("entity_id"), entity.get("name")]
    return any(isinstance(name, str) and name and
               re.search(r"(?<!\w)" + re.escape(name) + r"(?!\w)", text, re.I)
               for name in names)


def _resolve_refs(refs, task, packets, documents):
    """Resolve only entity-owned handles to eligible exact original slices."""
    roster = {row["entity_id"]: row for row in task["entities"]}
    if not isinstance(refs, list) or not refs or len(refs) > len(roster):
        raise ValueError("invalid_source_refs")
    cutoff = _date(task["cutoff_date"])
    citations, quotes, entities, seen = [], [], [], set()
    seen_handles = set()
    for ref in refs:
        if not isinstance(ref, dict) or set(ref) != {"entity_id", "source_ids"}:
            raise ValueError("invalid_source_ref_shape")
        entity_id, identifiers = ref["entity_id"], ref["source_ids"]
        if not isinstance(entity_id, str) or entity_id not in roster:
            raise ValueError("unknown_source_entity")
        if (not isinstance(identifiers, list) or not identifiers or len(identifiers) > 6
                or any(not isinstance(identifier, str) for identifier in identifiers)):
            raise ValueError("invalid_source_ids")
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("duplicate_source_ids")
        packet = packets.get(entity_id)
        if not isinstance(packet, dict) or packet.get("entity_id") != entity_id:
            raise ValueError("source_packet_entity_mismatch")
        cards = packet.get("cards")
        if not isinstance(cards, list) or any(not isinstance(card, dict) for card in cards):
            raise ValueError("invalid_source_packet")
        lookup = {card.get("id"): card for card in cards if isinstance(card.get("id"), str)}
        if len(lookup) != len(cards):
            raise ValueError("ambiguous_source_handles")
        if entity_id not in entities:
            entities.append(entity_id)
        for identifier in identifiers:
            handle = (entity_id, identifier)
            if handle in seen_handles:
                raise ValueError("duplicate_source_ids")
            seen_handles.add(handle)
            card = lookup.get(identifier)
            if card is None or card.get("citable") is not True:
                raise ValueError("unknown_or_uncitable_source_handle")
            doc_id = card.get("doc_id")
            if not isinstance(doc_id, str) or doc_id == "task" or doc_id not in documents:
                raise ValueError("source_not_in_frozen_corpus")
            doc = documents[doc_id]
            date = _field(doc, "date", _field(doc, "doc_date"))
            if _date(date) > cutoff or card.get("doc_date") != date:
                raise ValueError("source_date_mismatch_or_future")
            bound_entities = _field(doc, "entity_ids", ())
            if not (_field(doc, "shared", False) is True or
                    (isinstance(bound_entities, (tuple, list)) and entity_id in bound_entities)):
                raise ValueError("source_wrong_entity")
            start, end = card.get("span_start"), card.get("span_end")
            text = _field(doc, "text")
            if (not isinstance(start, int) or isinstance(start, bool)
                    or not isinstance(end, int) or isinstance(end, bool)
                    or not isinstance(text, str) or not 0 <= start < end <= len(text)):
                raise ValueError("invalid_original_source_span")
            quote = text[start:end]
            if quote != card.get("text") or not quote.strip():
                raise ValueError("source_quote_changed")
            observation_doc = SimpleNamespace(text=text, date=date)
            exclusions = observation_exclusions(observation_doc, task["cutoff_date"],
                                                roster[entity_id].get("latest_published_ref_month"))
            if any(max(start, left) < min(end, right) for left, right in exclusions):
                raise ValueError("source_future_observation_span")
            key = (doc_id, start, end)
            if key not in seen:
                quotes.append(quote)
                citations.append({"doc_id": doc_id, "span_start": start, "span_end": end})
                seen.add(key)
    return "\n".join(quotes), citations, entities


def _render_reason(item, task, packets, documents, frozen_digest, contract_digest):
    if not isinstance(item, dict):
        raise ValueError("invalid_reason_item")
    if item.get("final_predictions_digest") != frozen_digest:
        raise ValueError("reason_prediction_binding_mismatch")
    if item.get("contract_digest") != contract_digest:
        raise ValueError("reason_contract_binding_mismatch")
    if "task_digest" in item and item["task_digest"] != contract_digest:
        raise ValueError("reason_task_binding_mismatch")
    if any(key in item for key in ("premise", "citations", "doc_id", "span_start", "span_end")):
        raise ValueError("model_authored_quote_or_offsets")
    mechanism, implication = item.get("mechanism"), item.get("answer_implication")
    if (not isinstance(mechanism, str) or not mechanism.strip()
            or not isinstance(implication, str) or not implication.strip()):
        raise ValueError("missing_reason_text")
    assumptions = item.get("assumptions", [])
    if (not isinstance(assumptions, list) or len(assumptions) > 8
            or any(not isinstance(value, str) or not value.strip() for value in assumptions)):
        raise ValueError("invalid_reason_assumptions")
    premise, citations, entities = _resolve_refs(item.get("source_refs"), task, packets, documents)
    roster = {row["entity_id"]: row for row in task["entities"]}
    if any(not _named(implication, roster[entity_id]) for entity_id in entities):
        raise ValueError("reason_implication_missing_entity")
    if assumptions:
        mechanism += "\nAssumptions: " + "; ".join(assumptions)
    return {"reason_id": "r1", "premise": premise, "mechanism": mechanism,
            "answer_implication": implication, "scope": {"entities": entities},
            "citations": citations}


def compose_reasons(raw_items, task, packets, corpus, final_answer, frozen_digest):
    """Return (valid optional reasons, diagnostics), preserving all input rows.

    Each House item binds to digest(task) and the supplied frozen
    digest(final_answer['entity_predictions']). Only source_refs with the
    entity's original packet handles can become the premise/citations.
    Dropped reasons never cause an answer mutation or an empty reason block.
    """
    diagnostics = {"status": "not_composed", "accepted": 0, "rejected": [],
                   "reasoning_score_measured": False,
                   "semantic_consistency_verified": False,
                   "validation": "local_published_shape_provenance_and_caps"}
    try:
        if not isinstance(final_answer, dict) or not isinstance(task, dict):
            raise ValueError("invalid_reason_context")
        predictions = final_answer.get("entity_predictions")
        if not isinstance(predictions, list) or digest(predictions) != frozen_digest:
            raise ValueError("frozen_predictions_changed")
        rows = task.get("entities")
        if (not isinstance(rows, list) or not rows or any(not isinstance(row, dict)
                or not isinstance(row.get("entity_id"), str) for row in rows)):
            raise ValueError("invalid_task_roster")
        roster = [row["entity_id"] for row in rows]
        named = [row.get("entity_id") for row in predictions if isinstance(row, dict)]
        if (len(set(roster)) != len(roster) or len(named) != len(predictions)
                or len(named) != len(roster) or len(set(named)) != len(named)
                or set(named) != set(roster)):
            raise ValueError("invalid_final_roster")
        contract_digest = digest(task)
        _date(task.get("cutoff_date"))
        if not isinstance(packets, Mapping):
            raise ValueError("invalid_source_packets")
        documents = _docs(corpus)
        checking_corpus = _rail_docs(documents)
        items = raw_items.get("items") if isinstance(raw_items, dict) else raw_items
        if not isinstance(items, list):
            raise ValueError("invalid_reason_envelope")
    except (ValueError, TypeError, KeyError, OverflowError) as exc:
        diagnostics.update(status="omitted_invalid_context", context_error=str(exc))
        return [], diagnostics
    reasons = []
    if len(items) > 12:
        diagnostics["raw_items_truncated"] = len(items) - 12
    for index, item in enumerate(items[:12]):
        if len(reasons) == rail.MAX_REASONS:
            diagnostics["unprocessed_after_unit_cap"] = len(items) - index
            break
        try:
            reason = _render_reason(item, task, packets, documents, frozen_digest, contract_digest)
            reason["reason_id"] = "r" + str(len(reasons) + 1)
            prospective = {**final_answer, "submitted_reasons": reasons + [reason]}
            findings = rail.check_submitted_reasons(prospective, checking_corpus, task["cutoff_date"])
            plan = rail.reasons_judged(prospective, checking_corpus, task["cutoff_date"])
            if findings or any(not entry["judged"] for entry in plan):
                diagnostics["rejected"].append({"index": index,
                    "code": "official_local_reason_rail",
                    "findings": [{"code": finding.code, "reason_index": finding.claim_index}
                                 for finding in findings], "judged_plan": plan})
                continue
            reasons.append(reason)
        except (ValueError, TypeError, KeyError, OverflowError) as exc:
            diagnostics["rejected"].append({"index": index, "code": str(exc)})
    if digest(final_answer["entity_predictions"]) != frozen_digest:
        diagnostics.update(status="omitted_predictions_changed_during_compose", accepted=0)
        return [], diagnostics
    diagnostics.update(status="composed" if reasons else "omitted_no_valid_reasons",
                       accepted=len(reasons), final_predictions_digest=frozen_digest,
                       contract_digest=contract_digest)
    if reasons:
        diagnostics["judged_plan"] = rail.reasons_judged(
            {**final_answer, "submitted_reasons": reasons}, checking_corpus, task["cutoff_date"])
    return reasons, diagnostics
