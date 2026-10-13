"""Conditional bounded rounding for oversized House answer projections.

The published reasoning grader caps its numeric-answer projection at3000bytes.
Round only when it is oversized, keep local statistical rows unchanged, and
commit only if twelve significant digits fix the cap. This is a disclosed
delivery tradeoff, not an improvement in predictive skill.
"""
import copy
import math
from .harness import official_reason_rail as rail


def answer_bytes(rows):
    return rail._compact_bytes([{k:r[k]for k in rail._ANSWER_FIELDS if k in r}for r in rows])


def compact_house_answers(results, ids):
    ordered=[results[eid]for eid in ids];before=answer_bytes(ordered)
    report={'applied':False,'answer_bytes_before':before,'significant_digits':12,
            'changed_entities':[],'maximum_native_delta':0.,'scored_quality_measured':False}
    if before<=rail.MAX_ANSWER_BYTES:return report
    candidates=copy.deepcopy(results);deltas=[];changed=[]
    for eid in ids:
        row=candidates[eid]
        if row.get('_fallback',True)or row.get('_local_model')or row.get('_forecast_decision'):
            continue
        if not row.get('_reasoning'):continue
        entity_changed=False
        for obj,key in ((row,'point_forecast'),(row['interval'],'lo'),(row['interval'],'hi')):
            original=obj[key]
            if type(original)not in(int,float)or not math.isfinite(original):return report
            value=float(format(original,'.12g'));delta=abs(value-original)
            if delta>max(abs(original)*5e-12,math.ulp(original)):return report
            if value!=original:obj[key]=value;entity_changed=True;deltas.append(delta)
        if entity_changed:changed.append(eid)
    after=answer_bytes([candidates[eid]for eid in ids]);report['answer_bytes_if_rounded']=after
    if after>rail.MAX_ANSWER_BYTES:return report
    for eid in changed:
        results[eid]=candidates[eid]
    report.update(applied=bool(changed),answer_bytes_after=after,changed_entities=changed,
                  maximum_native_delta=max(deltas,default=0.))
    return report
