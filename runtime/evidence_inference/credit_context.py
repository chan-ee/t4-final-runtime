"""Retrieve financial capacity, constraints and qualifications together.

This module produces exact current-corpus windows, never credit labels or fitted
probabilities. The lexical rules contain no issuers or financial target values.
"""
from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class ContextWindow:
    doc_id: str
    start: int
    end: int
    text: str
    role: str
    source_date: str


_RULES = (
    ('operating_cash_flow', r'operating\s+activities|cash\s+flows?\s+from\s+operations'),
    ('liquidity_capacity', r'cash\s+and\s+cash\s+equivalents|(?:available|total|unrestricted)\s+(?:cash|liquidity)|(?:availability|managed\s+accessibility)\s+under|available\s+(?:borrowings|liquidity)'),
    ('debt_horizon', r'springing\s+maturity|(?:scheduled\s+to|will)\s+mature|(?:current|scheduled)\s+maturities|(?:debt|notes).{0,100}maturit'),
    ('covenant_status', r'in\s+compliance\s+with.{0,100}covenant|covenant.{0,100}(?:waiv|amend|compliance)|(?:have|has)\s+(?:been\s+)?(?:waived|amended)'),
    ('going_concern_and_qualification', r'substantial\s+doubt|(?:management|the\s+company)\s+believes.{0,150}(?:sufficient|adequate)|alleviat.{0,100}(?:doubt|uncertainty)'),
)


def is_credit_task(task):
    target=task.get('target',{})
    return 'credit_event' in target.get('name','').lower() or 'credit_event' in target.get('labels',[])


def credit_context_windows(corpus,task,entity,*,max_chars=7000):
    if not is_credit_task(task):return []
    candidates=[]
    for role,pattern in _RULES:
        ranked=[]
        for doc in corpus.docs.values():
            if doc.date>task['cutoff_date']:continue
            if doc.shared or entity['entity_id'] not in doc.entity_ids:continue
            normalized=doc.text.translate(str.maketrans({'\u200b':' ','\u200c':' ','\u200d':' ','\ufeff':' '}))
            for match in re.finditer(pattern,normalized,re.I):
                before=1200 if role=='operating_cash_flow' else (450 if role=='going_concern_and_qualification' else 350)
                after=1000 if role=='going_concern_and_qualification' else 650
                start=max(0,match.start()-before);end=min(len(doc.text),match.end()+after)
                if start and not doc.text[start-1].isspace():
                    boundary=doc.text.find(' ',start,min(match.start(),start+100))
                    if boundary>=0:start=boundary+1
                if end<len(doc.text) and not doc.text[end-1].isspace() and not doc.text[end].isspace():
                    boundary=re.search(r'\s',doc.text[end:min(len(doc.text),end+100)])
                    if boundary:end+=boundary.start()
                    else:
                        boundary=max(doc.text.rfind(' ',match.end(),end),doc.text.rfind('\n',match.end(),end))
                        if boundary>match.end():end=boundary
                local=normalized[start:end];near=normalized[max(0,match.start()-100):match.end()+250]
                if role=='going_concern_and_qualification' and not re.search(r'substantial\s+doubt|going\s+concern|liquidity|cash\s+flows?|meet.{0,35}obligations|working\s+capital',near,re.I):
                    # Adequate tax accruals or environmental reserves are not a
                    # management statement about surviving the forecast horizon.
                    continue
                # Actual financial figures and reporting periods outrank generic
                # risk-factor vocabulary. Scores have no mapping to event labels.
                score=3*min(4,len(re.findall(r'\$\s*\(?\s*[\d,.]+',near)))
                score+=2*min(4,len(re.findall(r'\b20\d{2}\b',local)))
                score+=4*bool(re.search(r'(?:thousands|millions|billions)',local,re.I))
                score+=3*bool(re.search(r'\b(?:was|were|as\s+of|at\s+December|year\s+ended|months?\s+ended|weeks?\s+ended)\b',local,re.I))
                if role=='operating_cash_flow':
                    tail=normalized[match.end():match.end()+250]
                    score+=16*bool(re.match(r'\s*(?:\(\s*\d\s*\))?\s*\$?\s*\(?\s*[-\d]',tail))
                    score+=12*bool(re.search(r'\bwas\s+\$\s*[\d,.]+',tail))
                    score+=3*bool(re.search(r'net\s+cash.{0,70}$',normalized[max(0,match.start()-100):match.start()],re.I))
                    score-=14*bool(re.search(r'adjustments\s+to\s+reconcile.{0,100}$',normalized[max(0,match.start()-140):match.start()],re.I))
                    score-=10*bool(re.search(r'(?:defined|define|definition).{0,180}$',normalized[max(0,match.start()-200):match.start()],re.I))
                if role=='going_concern_and_qualification':
                    score+=4*len(set(re.findall(r'initially|alleviat|assumes|believes|sufficient|substantial',local,re.I)))
                if role=='covenant_status':score+=8*bool(re.search(r'(?:we|company).{0,40}(?:was|were|is|are)\s+in\s+compliance',local,re.I))
                # Select deterministic source windows; current-year references are
                # not evidence of a future target, and are not used as labels.
                ranked.append((score,doc.date,-match.start(),doc.doc_id,ContextWindow(doc.doc_id,start,end,doc.text[start:end],role,doc.date)))
        if ranked:candidates.append(max(ranked,key=lambda x:x[:-1])[-1])
    selected=[];used=0
    for window in candidates:
        if used+len(window.text)>max_chars:continue
        if any(window.doc_id==x.doc_id and max(0,min(window.end,x.end)-max(window.start,x.start))>.8*min(len(window.text),len(x.text)) for x in selected):continue
        selected.append(window);used+=len(window.text)
    return selected


def supplement_credit_evidence(corpus,task,entity,selected,budget_chars=14000):
    """Adapter returning the caller's Chunk type with unchanged original spans."""
    if not selected or not is_credit_task(task):return selected
    candidates=credit_context_windows(corpus,task,entity,max_chars=min(7000,budget_chars))
    constructor=type(selected[0]);result=[];used=0
    for c in list(candidates)+list(selected):
        if used+len(c.text)>budget_chars:continue
        if any(c.doc_id==x.doc_id and max(0,min(c.end,x.end)-max(c.start,x.start))>.7*min(len(c.text),len(x.text)) for x in result):continue
        result.append(constructor(c.doc_id,c.start,c.end,c.text));used+=len(c.text)
        if len(result)>=10:break
    return result or selected
