"""Bounded House calls with explicit final-channel and checkpoint diagnostics."""
from __future__ import annotations
import hashlib
import json
import math
import os
import threading
import time
import urllib.error
import urllib.request
from .final_response import read_final_response

class BudgetExhausted(RuntimeError):pass
class Unavailable(RuntimeError):
    def __init__(self,message,*,reason_code='unavailable'):
        super().__init__(message)
        self.reason_code=reason_code

def parse_json(raw, *, expected_key='predictions', finish_reason='stop'):
    if isinstance(raw,dict) and 'choices' in raw:response=raw
    else:
        if isinstance(raw,dict):raw=json.dumps(raw,allow_nan=False)
        response={'choices':[{'finish_reason':finish_reason,'message':{'content':raw}}]}
    result=read_final_response(response,expected_key=expected_key)
    if result.value is None:raise ValueError(result.diagnostic['status'])
    return result.value

class HouseClient:
    def __init__(self,transport=None,max_calls=25,deadline_seconds=480,model=None,max_retries=0):
        self.endpoint=os.environ.get('MODEL_ENDPOINT','').rstrip('/')
        self.model=model or os.environ.get('MODEL_NAME','house')
        self.token=os.environ.get('MODEL_TOKEN','')
        self.custom_transport=transport is not None
        self.transport=transport or self._http
        self.enabled=self.custom_transport or bool(self.endpoint and self.token)
        self.max_calls=min(25,max(0,max_calls));self.deadline=time.monotonic()+deadline_seconds
        self.max_retries=min(1,max(0,int(max_retries)))
        self.attempts=0;self.errors=[];self.records=[];self.lock=threading.Lock();self.disabled=False
        self._request_local=threading.local()
    def _http(self,payload,timeout):
        url=self.endpoint+('/chat/completions' if self.endpoint.endswith('/v1') else '/v1/chat/completions')
        req=urllib.request.Request(url,data=json.dumps(payload,allow_nan=False).encode(),headers={'Content-Type':'application/json','Authorization':'Bearer '+self.token},method='POST')
        until=time.monotonic()+timeout;chunks=[];size=0
        with urllib.request.urlopen(req,timeout=timeout) as response:
            record=getattr(self._request_local,'record',None)
            if record is not None:
                with self.lock:record['http_status']=int(response.status)
            while True:
                left=until-time.monotonic()
                if left<=0:raise TimeoutError('request_body_deadline')
                try:response.fp.raw._sock.settimeout(left)
                except (AttributeError,OSError):pass
                reader=getattr(response,'read1',response.read)
                chunk=reader(min(65536,2*1024*1024+1-size))
                if not chunk:break
                size+=len(chunk);chunks.append(chunk)
                if size>2*1024*1024:raise ValueError('response_size')
        if record is not None:
            with self.lock:record['response_body_bytes']=size
        return json.loads(b''.join(chunks))
    def snapshot(self):
        with self.lock:
            return {'requests_attempted':self.attempts,'request_cap':self.max_calls,'admitted_requests_known':False,
                'house_environment_available':self.enabled,'transport_kind':'synthetic' if self.custom_transport else 'house_http',
                'request_errors':list(self.errors[:25]),'requests':[dict(r) for r in self.records[:25]],
                'remaining_seconds':round(max(0,self.deadline-time.monotonic()),3)}
    def complete(self,system,user,max_tokens=4000,thinking=False,*,stage='forecast',expected_key='predictions',request_timeout_seconds=80,temperature=0,top_p=None,low_effort=False):
        if not self.enabled:raise Unavailable('House environment unavailable')
        if type(temperature) not in (int,float) or not math.isfinite(temperature) or not 0<=temperature<=2:
            raise ValueError('invalid_temperature')
        if top_p is not None and (type(top_p) not in (int,float) or not math.isfinite(top_p) or not 0<top_p<=1):
            raise ValueError('invalid_top_p')
        payload={'model':self.model,'messages':[{'role':'system','content':system},{'role':'user','content':user}],
            'temperature':temperature,'seed':int(os.environ.get('QFBENCH_SEED','0')),'max_tokens':min(4000,max_tokens),
            'chat_template_kwargs':{'enable_thinking':bool(thinking)}}
        if top_p is not None:payload['top_p']=top_p
        if thinking and low_effort:payload['chat_template_kwargs']['low_effort']=True
        digest=hashlib.sha256(json.dumps(payload,allow_nan=False,sort_keys=True,separators=(',',':')).encode()).hexdigest()
        for retry in range(self.max_retries+1):
            with self.lock:
                remaining=self.deadline-time.monotonic()
                if self.disabled:raise Unavailable('House route unavailable')
                if self.attempts>=self.max_calls or remaining<5:raise BudgetExhausted('request/time cap')
                self.attempts+=1
                record={'attempt':self.attempts,'stage':stage,'retry':retry,'payload_sha256':digest,
                    'prompt_chars':len(user),'request_timeout_seconds':round(min(max(1,float(request_timeout_seconds)),max(1,remaining-3)),3),'thinking':bool(thinking),'max_output_tokens':payload['max_tokens'],
                    'low_effort':bool(thinking and low_effort),'temperature':temperature,'top_p':top_p,
                    'remaining_before_seconds':round(remaining,3),'status':'in_progress'}
                self.records.append(record)
            start=time.monotonic()
            try:
                self._request_local.record=record
                response=self.transport(payload,min(max(1,float(request_timeout_seconds)),max(1,remaining-3)))
                with self.lock:record['transport_returned']=True
                if self.custom_transport and not (isinstance(response,dict) and 'choices' in response):
                    if isinstance(response,dict):response=json.dumps(response,allow_nan=False)
                    response={'choices':[{'finish_reason':'stop','message':{'content':response}}]}
                parsed=read_final_response(response,expected_key=expected_key)
                usage=response.get('usage',{}) if isinstance(response,dict) else {}
                safe_usage={k:v for k,v in usage.items() if isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v)} if isinstance(usage,dict) else {}
                details=usage.get('completion_tokens_details',{}) if isinstance(usage,dict) else {}
                if isinstance(details,dict):
                    value=details.get('reasoning_tokens')
                    if isinstance(value,(int,float)) and not isinstance(value,bool) and math.isfinite(value):safe_usage['reasoning_tokens']=value
                with self.lock:
                    record.update(parsed.diagnostic);record['elapsed_seconds']=round(time.monotonic()-start,3);record['usage']=safe_usage
                if parsed.value is None:
                    with self.lock:self.errors.append(parsed.diagnostic['status'])
                    raise Unavailable('No verified final JSON',reason_code=parsed.diagnostic['status'])
                return parsed.value
            except urllib.error.HTTPError as exc:
                with self.lock:
                    record.update(status='HTTP_'+str(exc.code),http_status=exc.code,elapsed_seconds=round(time.monotonic()-start,3));self.errors.append('HTTP_'+str(exc.code))
                    if exc.code in (401,403,404,429):self.disabled=True
                if self.disabled or exc.code<500:raise Unavailable('House refused',reason_code='HTTP_'+str(exc.code)) from None
            except Unavailable:raise
            except (TimeoutError,OSError,ValueError,KeyError,TypeError,RuntimeError) as exc:
                with self.lock:
                    record.update(status=type(exc).__name__,elapsed_seconds=round(time.monotonic()-start,3));self.errors.append(type(exc).__name__)
                raise Unavailable('House request failed',reason_code=type(exc).__name__) from None
            if retry<self.max_retries and time.monotonic()+10<self.deadline:time.sleep(.15)
            else:break
        raise Unavailable('House transient retry exhausted')
