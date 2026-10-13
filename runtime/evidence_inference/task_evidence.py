"""Canonical own-row task citations, matching official corpus.py at eef782bf.

The table is organizer input, not an additional corpus document. Reasons must
never cite `task`: the reason grader does not resolve that reserved claim id.
"""
import json


def own_row_claim(task, entity):
    offset = 0
    for row in task['entities']:
        text = json.dumps(row, ensure_ascii=False, separators=(', ', ': '))
        if row['entity_id'] == entity['entity_id']:
            # An actual canonical key/value fragment keeps long descriptive
            # rows out of the claim. Offsets count original characters, never
            # UTF-8 bytes. The byte bound is conservative, not a tokenizer test.
            preferred = ('consensus_eps','latest_precutoff_estimate','latest_published_mom_pct',
                         'prior_year_q_eps','prior_margin_pct')
            numeric = [key for key,value in row.items() if type(value)in(int,float,bool)
                       and key not in('entity_id','cik','sic','naics')]
            statuses = [key for key in('rating','status','sector','industry')
                        if isinstance(row.get(key),str)and row[key]]
            keys = list(dict.fromkeys([key for key in preferred if key in row] + numeric + statuses + ['entity_id']))
            fragment = None
            for key in keys:
                candidate = json.dumps(key,ensure_ascii=False) + ': ' + json.dumps(row[key],ensure_ascii=False)
                if len(candidate.encode('utf-8'))<=160:
                    fragment=candidate;break
            if fragment is None:fragment='"entity_id": '+json.dumps(row['entity_id'],ensure_ascii=False)
            start = text.index(fragment)
            quote = fragment
            while len(quote.encode('utf-8')) > 160:
                quote = quote[:-1]
            return {'doc_id': 'task', 'span_start': offset + start,
                    'span_end': offset + start + len(quote), 'claim': quote}
        offset += len(text) + 1
    raise ValueError('task evidence entity absent from trusted roster')
