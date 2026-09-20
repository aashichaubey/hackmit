import copy

from tokenese.evo_benchmark import gate, workflow_comparison


def corpus_and_rows():
    cases, rows = [], []
    for i in range(12):
        qs = [{'id':f'{i}-{j}'} for j in range(10)]
        cases.append({'id':str(i),'questions':qs,'provenance':{'annotation_status':'independently_reviewed'}})
        for q in qs:
            rows.append({'case_id':str(i), 'question_id':q['id'], 'category':'owner', 'correct':True,
                         'expected_found':True, 'answer':{'found':True,'answer':'Ana'},
                         'usage':{'input_tokens':50,'output_tokens':5,'cached_input_tokens':0}})
    return cases, rows


def test_119_is_not_enough_if_new_critical_error():
    cases, rows = corpus_and_rows()
    base = copy.deepcopy(rows)
    for row in base:
        row['usage']['input_tokens'] = 100
    assert gate(rows,[base],cases)['status'] == 'qualified'
    rows[0]['correct'] = False
    assert gate(rows,[base],cases)['status'] == 'no_qualified_candidate'
    base[0]['correct'] = False
    assert gate(rows,[base],cases)['status'] == 'qualified'
    cases[0]['provenance']['annotation_status'] = 'model_drafted_unreviewed'
    assert gate(rows,[base],cases)['status'] == 'incomplete_evaluation'


def test_workflow_charges_extraction_once_at_each_workload():
    cases, product = corpus_and_rows()
    raw = copy.deepcopy(product)
    for row in raw:
        row['usage']['input_tokens'] = 100
    extraction = {c['id']:{'result':{'usage':{'input_tokens':100,'output_tokens':0,'cached_input_tokens':0}}} for c in cases}
    result = workflow_comparison(cases,raw,product,extraction)
    assert result['1']['product_input_tokens'] == 12*150
    assert result['4']['product_input_tokens'] == 12*300
    assert result['10']['product_input_tokens'] == 12*600
    extraction['0']['result']['usage'] = None
    assert workflow_comparison(cases,raw,product,extraction)['1']['product_usd'] is None
