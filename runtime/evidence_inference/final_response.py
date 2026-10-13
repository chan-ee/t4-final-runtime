"""Research-only final-answer adapter; no transport or model is invoked.

Separates an unfinished reasoning block from submitted forecast JSON and retains
safe response diagnostics. This corrects a reproduced parser counterexample,
not a measured prediction or official-runtime failure.
"""
from dataclasses import dataclass
import json
import re


@dataclass
class FinalResponse:
    value: dict | None
    diagnostic: dict


def _decoder():
    def refuse_nonfinite(value):
        raise ValueError('nonfinite JSON value')
    return json.JSONDecoder(parse_constant=refuse_nonfinite)


def _unquoted_closing_marker(text):
    """Find an explicit marker outside quoted strings, or return -1."""
    quoted = escaped = False
    for index, char in enumerate(text):
        if quoted:
            if escaped: escaped = False
            elif char == '\\': escaped = True
            elif char == '"': quoted = False
        elif char == '"': quoted = True
        elif text.startswith('</think>', index): return index
    return -1


def _unfence(text):
    if not text.startswith('```'): return text
    newline = text.find('\n')
    if newline < 0: return ''
    text = text[newline + 1:]
    if text.rstrip().endswith('```'): text = text.rstrip()[:-3].rstrip()
    return text


def _canonical_final(value, expected_key):
    """Match the two envelopes supported by legacy._rows, without guessing.

    Aliases apply only to the forecast stage. Equal duplicate envelopes are
    harmless; conflicting or malformed declared envelopes are refused.
    Row quantities and sources remain subject to downstream validation.
    """
    keys = (expected_key, 'entity_predictions') if expected_key == 'predictions' else (expected_key,)
    if not isinstance(value, dict):
        return None, 'wrong_final_shape', None
    present = [key for key in keys if key in value]
    if not present or any(not isinstance(value[key], list) for key in present):
        return None, 'wrong_final_shape', None
    if len(present) > 1 and value[present[0]] != value[present[1]]:
        return None, 'conflicting_final_envelopes', None
    source_key = present[0]
    canonical = dict(value)
    canonical[expected_key] = value[source_key]
    if expected_key == 'predictions':
        canonical.pop('entity_predictions', None)
    return canonical, 'final_json', source_key


def read_final_response(response, *, expected_key='predictions'):
    diagnostic={'finish_reason':None,'content_type':None,'complete_rows':0,
        'partial':False,'reasoning_field_present':False}
    if not isinstance(response,dict):
        return FinalResponse(None,{**diagnostic,'status':'malformed_response'})
    choices=response.get('choices')
    if not isinstance(choices,list) or len(choices)!=1 or not isinstance(choices[0],dict):
        return FinalResponse(None,{**diagnostic,'status':'malformed_choices'})
    choice=choices[0];message=choice.get('message')
    if not isinstance(message,dict):
        return FinalResponse(None,{**diagnostic,'status':'malformed_message'})
    content=message.get('content');reason=choice.get('finish_reason')
    diagnostic['finish_reason']=reason if reason in ('stop','length','content_filter','tool_calls',None) else 'other'
    diagnostic['content_type']=type(content).__name__
    diagnostic['reasoning_field_present']=bool(message.get('reasoning_content'))
    if not isinstance(content,str) or not content.strip():
        status='reasoning_only' if diagnostic['reasoning_field_present'] else 'no_final_content'
        return FinalResponse(None,{**diagnostic,'status':status})
    if reason in ('content_filter','tool_calls'):
        return FinalResponse(None,{**diagnostic,'status':'not_a_text_final'})
    text=content.strip()
    decoder=_decoder()
    # A whole JSON answer wins before interpreting any reasoning markers.
    # Thus </think> inside a quoted JSON value always remains ordinary data.
    try:
        value=decoder.decode(_unfence(text))
        value,status,source_key=_canonical_final(value,expected_key)
        if value is None:return FinalResponse(None,{**diagnostic,'status':status})
        diagnostic['complete_rows']=len(value[expected_key])
        return FinalResponse(value,{**diagnostic,'status':status,'source_envelope':source_key})
    except (ValueError,json.JSONDecodeError):
        pass
    # The public NVIDIA template can prefill <think> before generation. A
    # draft may itself start with JSON; only the suffix after an explicit
    # unquoted closing marker is a final candidate. Never recover draft rows.
    close=_unquoted_closing_marker(text)
    if close>=0:
        leading=text.startswith('<think>')
        text=text[close+len('</think>'):].strip()
        diagnostic['reasoning_protocol']='leading_think_block' if leading else 'prefilled_opening_think'
        if not text:return FinalResponse(None,{**diagnostic,'status':'no_final_content'})
        if _unquoted_closing_marker(text)>=0:
            return FinalResponse(None,{**diagnostic,'status':'invalid_reasoning_boundary'})
    elif text.startswith('<think>'):
        return FinalResponse(None,{**diagnostic,'status':'unfinished_reasoning'})
    text=_unfence(text)
    if not text:return FinalResponse(None,{**diagnostic,'status':'no_final_content'})
    try:
        value=decoder.decode(text)
        value,status,source_key=_canonical_final(value,expected_key)
        if value is None:
            return FinalResponse(None,{**diagnostic,'status':status})
        diagnostic['complete_rows']=len(value[expected_key])
        return FinalResponse(value,{**diagnostic,'status':status,'source_envelope':source_key})
    except (ValueError,json.JSONDecodeError):
        pass
    # Only a declared final array at the beginning of an output object can
    # recover complete rows after a length stop. Provisional JSON buried in
    # prose/reasoning is never promoted to a final prediction.
    if reason!='length':return FinalResponse(None,{**diagnostic,'status':'invalid_final_json'})
    # A malformed reasoning prefix can leave a quote open and hide its raw
    # closing marker from the quote-aware scan. It must not be recovered as
    # a partial final answer. Whole valid JSON already won above; ordinary
    # partial finals without a marker keep their existing recovery behavior.
    if '</think>' in content and not diagnostic.get('reasoning_protocol'):
        return FinalResponse(None,{**diagnostic,'status':'ambiguous_reasoning_boundary',
            'boundary_policy':'unclassified_raw_marker_disables_prefix_recovery'})
    keys=(expected_key,'entity_predictions') if expected_key=='predictions' else (expected_key,)
    match=re.match(r'\s*\{\s*"('+ '|'.join(re.escape(key) for key in keys)+r')"\s*:\s*\[',text)
    if not match:return FinalResponse(None,{**diagnostic,'status':'invalid_final_json'})
    at=match.end();rows=[]
    while at<len(text):
        while at<len(text) and text[at] in ' \t\r\n,':at+=1
        # Once the array closes, an incomplete outer object may already carry
        # a second, conflicting envelope. Do not certify its first array by
        # ignoring that suffix. Only truncation inside the declared row array
        # is eligible for complete-row recovery.
        if at<len(text) and text[at]==']':
            return FinalResponse(None,{**diagnostic,'status':'incomplete_final_envelope'})
        if at>=len(text) or text[at]!='{':break
        try:row,end=decoder.raw_decode(text[at:])
        except (ValueError,json.JSONDecodeError):break
        if not isinstance(row,dict) or not isinstance(row.get('entity_id'),str):break
        rows.append(row);at+=end
    if not rows:return FinalResponse(None,{**diagnostic,'status':'invalid_final_json'})
    return FinalResponse({expected_key:rows,'_partial_response':True},
        {**diagnostic,'status':'partial_final_rows','partial':True,'complete_rows':len(rows),'source_envelope':match.group(1)})
