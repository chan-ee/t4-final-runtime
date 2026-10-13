"""Link arithmetic measurement IDs to exact original citation handles.

MIDs identify quantities and cannot be submitted as source FIDs. A current
amount is not a future forecast, and semantic accounting scope stays open.
"""
from .corpus import Chunk
from .packets import add_source

def attach_measurement_citations(task,entity,corpus,packet,ledger,compact):
    retained={cell[0]for table in compact.get('tables',[])for row in table.get('rows',[])for cell in row.get('cells',[])}
    required={item[k]for item in ledger.get('historical_eps_identities',[])for k in ('numerator_measurement_id','shares_measurement_id')if item.get(k)}
    latest=ledger.get('latest_observed_quarter_end');selected=[]
    for row in ledger.get('measurements',[]):
        mid=row['measurement_id']
        if mid in retained and(mid in required or(row['period_end']==latest and row['period_months']==3)):
            selected.append(row)
    links={};spans={};omitted=[]
    for row in selected:
        source=row['source'];key=(source['doc_id'],source['span_start'],source['span_end'])
        if key not in spans:
            if len(spans)>=32:omitted.append(row['measurement_id']);continue
            doc=corpus.docs.get(source['doc_id'])
            if not doc:continue
            a,b=source['span_start'],source['span_end']
            if doc.text[a:b]!=source['quote']:raise ValueError('measurement source row mismatch')
            spans[key]=add_source(packet,Chunk(doc.doc_id,a,b,doc.text[a:b]),corpus,task,entity,role='financial_measurement_original_row')
        if spans[key]:links[row['measurement_id']]=spans[key]
    compact['citation_links_by_measurement']=links
    compact['citation_handle_note']='E*M* measurement IDs are arithmetic/provenance references, never submit them as support_ids. Use linked E*F* IDs or other existing source_facts IDs for claims. Derived aggregates and future scenarios are not original quotations.'
    compact['citation_link_diagnostics']={'new_or_reused_original_rows':len(spans),'linked_measurements':len(links),'omitted_measurement_ids_by_row_cap':omitted}
    return compact
