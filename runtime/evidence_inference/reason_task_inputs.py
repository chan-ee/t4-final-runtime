"""State trusted quantity references the reason grader cannot read from rows."""
import json


def reason_task_inputs(task,entity,contract,*,conversion_applied=True,used_fields=None):
    result={}
    if isinstance(used_fields,list)and used_fields:
        if len(used_fields)>12 or len(set(used_fields))!=len(used_fields)or any(
                not isinstance(key,str)or key not in entity or key in('entity_id','name','corpus_ref')for key in used_fields):
            raise ValueError('invalid used task fields')
        result['used_task_features']={key:entity[key]for key in used_fields}
    reference=contract.get('reference_observation')
    if isinstance(reference,dict):
        result['task_reference']={k:v for k,v in reference.items()if k in
            ('value','unit','period','as_of','maturity_years','vintage','base_report','base_net_contracts','base_open_interest')and v is not None}
    transform=contract.get('response_to_output',{'scale':1.,'offset':0.})
    if conversion_applied and transform!={'scale':1.,'offset':0.}:
        result['code_conversion_to_native_output']=transform
    name=task['target']['name']
    if name=='eps_outcome':
        result['declared_label_comparison']={k:entity[k]for k in
            ('consensus_eps','threshold_pct','beat_condition','miss_condition','inline_condition')if k in entity}
    elif name=='eps_yoy_direction'and'prior_year_q_eps'in entity:
        result['declared_label_comparison']={'prior_year_q_eps':entity['prior_year_q_eps'],
                                             'unit':entity.get('currency','USD/share')}
    if result:result['output_unit']=contract.get('output_unit',contract.get('unit'))
    return json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'))if result else None
