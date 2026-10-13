"""Optional model-delivery trial; absence of delivery is not a forecast score.

This only checks the participant's adoption path. Source truth, semantic NLI,
calibration and model identity are not certified here. Platform proxy/model
identity and the submitted immutable image remain external assumptions.
"""

POINT_SOURCES={'house_final','converted_house_eps','arithmetic_house',
               'code_from_house_forecast_diluted_common_income_and_shares'}
INTERVAL_SOURCES={'house_final','converted_house_eps','code_from_house_joint_forecast_eps_scenarios'}

class ModelDeliveryUnverified(RuntimeError):pass

def delivery_findings(task,results,views,client,protocol):
    """Reject synthetic/fallback/carried or pooled contributions by provenance.

    A model that independently returns a point equal to a prior is allowed.
    Equality is not evidence of copying or failed delivery.
    """
    errors={}
    if protocol not in ('deliberative','direct_control'):errors['unit']='single_decision_protocol_required'
    if client is None or not client.enabled or client.attempts<=0:errors['transport']='no_completed_route_attempt'
    elif client.custom_transport:errors['transport']='synthetic_transport_is_not_official_house'
    for entity in task['entities']:
        eid=entity['entity_id'];row=results.get(eid,{});why=[];sources=row.get('_field_sources',{})
        if row.get('_fallback',True):why.append('fallback_row')
        if len(views.get(eid,{}))!=1 or not set(views.get(eid,{})).issubset({'deliberative','final_recovery'}):why.append('no_single_verified_final')
        if sources.get('point_forecast')not in POINT_SOURCES:why.append('point_not_wholly_from_verified_final')
        if sources.get('interval')not in INTERVAL_SOURCES:why.append('interval_not_wholly_from_verified_final')
        if task['target']['type']=='classification'and sources.get('label')!='house_final':why.append('label_not_from_verified_final')
        if why:errors[eid]=why
    return errors
