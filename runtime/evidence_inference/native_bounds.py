"""Respect explicit native scales; mentioning probability is not a range rule."""
import math
import re
from .native_units import declared_native_unit, _normal


def native_bounds(task,entity):
    target=task.get('target',{})
    explicit=target.get('bounds')
    if explicit is not None:
        if not isinstance(explicit,(list,tuple))or len(explicit)!=2 or any(
                type(v)not in(int,float)or not math.isfinite(v)for v in explicit)or explicit[0]>explicit[1]:
            raise ValueError('Malformed trusted target bounds')
        return tuple(map(float,explicit))
    # The same target/prompt/row precedence defines both response units and
    # range checks. Input-row units must not override an explicit output scale.
    unit,status=declared_native_unit(task,entity)
    if status.startswith('conflicting_') or status=='negated_output_unit_clause_unresolved':
        return None
    unit=_normal(unit)if unit else None
    name=str(target.get('name','')).casefold()
    probability=bool(re.search(r'probabilit|(?:^|[_-])(?:prob|chance|likelihood)(?:[_-]|$)',name))
    if re.search(r'probability_(?:change|delta|difference|growth)(?:_|$)|(?:change|delta|difference|growth)_(?:in_|of_)?probability',name):
        return None  # A change is not a probability level; signed values can be valid.
    if unit in('probability percent','probability pct','percent probability'):
        return(0.,100.)
    if unit in('probability','probability 0 to 1','probability fraction','decimal probability'):
        return(0.,1.)
    if probability and unit in('%','percent','percentage','pct'):
        return(0.,100.)
    # Explicit unrelated units override incidental mentions of probability.
    if unit is not None:return None
    if name=='credit_event_12m':return(0.,1.)
    if probability and re.search(r'\b(?:between\s+0\s+and\s+1|0\s*(?:to|[-–])\s*1)\b',task.get('prompt',''),re.I):
        return(0.,1.)
    return None
