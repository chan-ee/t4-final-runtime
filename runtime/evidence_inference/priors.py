"""Current-task numerical priors from published vintage columns; no external artifacts."""
from __future__ import annotations

import datetime as dt
import math
import re
import statistics


def quantile(values, probability):
    values=sorted(values)
    location=(len(values)-1)*probability
    left=int(location);right=min(left+1,len(values)-1)
    return values[left]+(location-left)*(values[right]-values[left])


def vintage_prior(task,entity,chunks,corpus=None):
    current=entity.get('latest_precutoff_estimate')
    if isinstance(current,bool) or not isinstance(current,(int,float)) or not math.isfinite(current):return None
    target_name=task.get('target',{}).get('name','').lower()
    prompt=task.get('prompt','').lower()
    if 'revision' not in target_name:return None
    if target_name!='next_estimate_revision_direction' and not any(x in prompt for x in ('revised level','forecast of the revised value','forecast of the revised estimate')):
        return None
    cutoff=task.get('cutoff_date','')
    texts=[]
    if corpus is not None:
        for doc in corpus.docs.values():
            if not doc.shared and entity['entity_id'] in doc.entity_ids:
                texts.append((doc.text,min(cutoff,doc.date)))
    else:
        texts=[(c.text,cutoff) for c in chunks]
    for text,source_cutoff in texts:
        grid=[[part.strip() for part in line.split('|')] for line in text.splitlines() if '|' in line]
        header=next((row for row in grid if row and row[0]=='reference_month' and any(x.startswith('as_of_') for x in row[1:])),None)
        if header is None:continue
        columns=[(i,x.removeprefix('as_of_')) for i,x in enumerate(header) if i>0 and x.startswith('as_of_') and x.removeprefix('as_of_')<=source_cutoff]
        if len(columns)<2:continue
        matrix={}
        for row in grid:
            if not row or not re.fullmatch(r'\d{4}-\d{2}',row[0]) or row[0]>source_cutoff[:7]:continue
            values=[]
            for index,_ in columns:
                try:value=float(row[index].replace(',',''))
                except (ValueError,IndexError):value=None
                values.append(value)
            matrix[row[0]]=values
        if not matrix:continue
        dates=[dt.date.fromisoformat(date) for _,date in columns]
        steps=None;horizon_basis=None
        try:
            # Reuse the same inferred early/late publication cadence as the
            # primary measurement ledger. Median day gaps treated DGORDER's
            # late-to-late month as one release instead of two.
            from .numerical_priors import revision_release_horizon
            inferred=revision_release_horizon([date.isoformat() for date in dates],
                entity.get('latest_precutoff_vintage',columns[-1][1]),entity['resolving_release_date'])
            if inferred is not None:steps,horizon_basis=inferred
        except (KeyError,ValueError,TypeError,statistics.StatisticsError):pass
        if steps is None or steps>=len(columns):return None
        deltas=[]
        for j in range(len(columns)-steps):
            # An annual/system-wide vintage can revise many old months at once.
            # Keep such transitions in the observed data, but do not let them define
            # an ordinary monthly revision prior when the task does not request one.
            old=[]
            for month,values in matrix.items():
                age=(dates[j].year-int(month[:4]))*12+dates[j].month-int(month[5:])
                a,b=values[j],values[j+steps]
                if a is not None and b is not None and age>=4:old.append(b-a)
            systematic=len(old)>=3 and sum(abs(x)>1e-10 for x in old)>=.6*len(old)
            if systematic and not any(x in task.get('prompt','').lower() for x in ('annual update target','benchmark revision target')):
                continue
            available=sorted(month for month,values in matrix.items() if values[j] is not None)
            for month in available[-3:]:
                a,b=matrix[month][j],matrix[month][j+steps]
                if a is not None and b is not None:deltas.append(b-a)
        if len(deltas)<3:return None
        center=statistics.median(deltas)
        low=min(quantile(deltas,.05),center,0.)
        high=max(quantile(deltas,.95),center,0.)
        if low==high:
            spread=max(abs(current)*1e-5,1e-6);low-=spread;high+=spread
        up=sum(x>1e-10 for x in deltas);down=sum(x<-1e-10 for x in deltas)
        label='up' if up>down else 'down'
        return {'point':current+center,'lo':current+low,'hi':current+high,
                'label':label,'observed_revisions':len(deltas),'horizon_releases':steps,
                'horizon_basis':horizon_basis,'future_release_schedule_verified':False}
    return None
