"""Explicit target-output units; input currency alone does not define a target.

No scale is inferred from a number, and this helper performs no conversions.
An unresolved prompt stays unresolved rather than declaring an input currency
to be the forecast's unit. Published family conversions remain elsewhere.
"""
import re

_UNIT = (r'probability\s+percent|probability\s+(?:fraction|0\s+to\s+1)|'
         r'percentage[- ]points?|basis[- ]points?|percent(?:age)?|pct|bps|'
         r'USD\s*/\s*share|USD\s+per\s+share|USD|dollars?|ratio|unitless')
_OUTPUT = re.compile(
    r'\b(?:outputs?|point_forecast|native_forecast|target(?:\s+(?:quantity|value))?|'
    r'point\s+(?:forecast|estimate))\b[^.;\n]{0,80}?'
    r'\b(?:in|as)\s+(?:a\s+|an\s+)?(?P<unit>' + _UNIT + r')\b', re.I)


def _normal(unit):
    value = ' '.join(unit.casefold().replace('_', ' ').replace('-', ' ').split())
    value = re.sub(r'\s*/\s*', '/', value)
    aliases = {'pct': 'percent', 'percentage': 'percent', '%': 'percent',
               'bps': 'basis points', 'bp': 'basis points', 'basis point': 'basis points',
               'percentage point': 'percentage points', 'dollar': 'usd', 'dollars': 'usd',
               'usd per share': 'usd/share'}
    return aliases.get(value, value)


def declared_native_unit(task, entity):
    target = task.get('target', {})
    target_units = [value for value in (target.get('unit'), target.get('units'),
                    task.get('target_unit'), task.get('output_unit'))
                    if isinstance(value, str) and value.strip()]
    prompt_matches = list(_OUTPUT.finditer(task.get('prompt', '')))
    prompt_units = [match['unit'] for match in prompt_matches
                    if not re.search(r'\b(?:not|never|no)\b', match.group(), re.I)]
    # The explicit output clause may describe several different quantities.
    # Ambiguity is not permission to choose an arbitrary one or transform it.
    declarations = target_units or prompt_units
    if declarations:
        if len({_normal(value) for value in declarations}) != 1:
            return None, 'conflicting_output_unit_declarations'
        if target_units and prompt_units and any(_normal(value) != _normal(target_units[0]) for value in prompt_units):
            return None, 'conflicting_target_and_prompt_units'
        return declarations[0], 'explicit_target_unit' if target_units else 'explicit_prompt_output_unit'
    row_units = [value for value in (entity.get('unit'), entity.get('units'))
                 if isinstance(value, str) and value.strip()]
    if row_units:
        if len({_normal(value) for value in row_units}) != 1:
            return None, 'conflicting_row_units'
        return row_units[0], 'declared_row_unit'
    if prompt_matches:return None, 'negated_output_unit_clause_unresolved'
    return None, 'native_unit_unresolved_in_task_prompt'
