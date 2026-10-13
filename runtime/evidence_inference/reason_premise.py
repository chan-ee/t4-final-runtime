"""Bind a short complete factual quotation to an already selected source.

Only the reason premise becomes shorter. Original prediction claim/citation
spans remain unchanged. Exact text and boundaries do not certify semantic
relevance, future support, or an official reasoning grade.
"""
import re


def complete_boundary(text, start, end):
    if start and not (text[start-1] in '\n\r' or text[:start].rstrip().endswith(('.', '!', '?'))):
        return False
    if end<len(text) and not (text[end] in '\n\r' or
                             text[end-1] in '.!?' and text[end].isspace()):
        return False
    return True


def bind_premise(quote, source_ids, packet, corpus):
    if not isinstance(quote,str) or not 12<=len(quote)<=240 or quote!=quote.strip():
        return None
    if not re.search(r'[^\W\d_]',quote):return None
    selected=set(source_ids);found={}
    for card in packet['cards']:
        if card['id']not in selected:continue
        doc=corpus.docs.get(card['doc_id'])
        if doc is None:continue
        text=card['text'];offset=0
        while True:
            index=text.find(quote,offset)
            if index<0:break
            start=card['span_start']+index;end=start+len(quote)
            if doc.text[start:end]==quote and complete_boundary(doc.text,start,end):
                found[(card['doc_id'],start,end)]={'quote':quote,'source_id':card['id'],
                    'doc_id':card['doc_id'],'span_start':start,'span_end':end}
            offset=index+1
    return next(iter(found.values()))if len(found)==1 else None


def validate_record(record, source_ids, packet, corpus):
    if not isinstance(record,dict):return False
    return bind_premise(record.get('quote'),source_ids,packet,corpus)==record
