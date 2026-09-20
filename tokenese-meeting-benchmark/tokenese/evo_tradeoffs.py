"""Display measured efficiency/quality tradeoffs without rewriting old gates."""
from pathlib import Path
import json
import streamlit as st
from .evo_public_data import REPORTS


def load_tradeoffs(directory: Path = REPORTS) -> list[dict]:
    rows=[]
    heldout_name='heldout-product-policy.json' if (directory/'heldout-product-policy.json').exists() else 'heldout.json'
    for name,label,split in [('pilot-minimal.json','Minimal separator','Training'),('blocks.json','Speaker blocks','Training'),(heldout_name,'Minimal separator','Fresh test · workspace policy' if heldout_name=='heldout-product-policy.json' else 'Fresh test')]:
        path=directory/name
        if not path.exists():
            continue
        try:
            report=json.loads(path.read_text());raw=report['methods']['raw'];encoded=report['methods']['encoded']
            if raw['actual_input_tokens'] is None or encoded['actual_input_tokens'] is None:
                continue
            rows.append({'format':label,'evaluation':split,'questions':raw['questions'],
                         'input saving %':100*(1-encoded['actual_input_tokens']/raw['actual_input_tokens']),
                         'F1 loss (pp)':100*(raw['macro_f1']-encoded['macro_f1']),
                         'answerability loss (pp)':100*(raw['balanced_answerability']-encoded['balanced_answerability']),
                         'raw F1 %':100*raw['macro_f1'],'encoded F1 %':100*encoded['macro_f1'],
                         'rejected outputs':encoded['invalid_answers'],
                         'answerable F1 loss (pp)':100*(raw['answerable_f1']-encoded['answerable_f1']) if 'answerable_f1' in raw and 'answerable_f1' in encoded else None,
                         'cost saving %':100*(1-encoded['actual_usd']/raw['actual_usd']) if raw['actual_usd'] and encoded['actual_usd'] is not None else None})
        except (OSError,ValueError,KeyError,TypeError,ZeroDivisionError):
            continue
    return rows


def render():
    rows=load_tradeoffs()
    if not rows:
        return
    st.subheader('How much quality would you trade for fewer tokens?')
    tolerance=st.slider('Exploratory F1 loss tolerance · percentage points',0.0,20.0,5.0,.5,key='quality_tradeoff_tolerance')
    st.caption('A small quality decrease can be worth exploring. This control filters measured tradeoffs; it does not change answers, labels, historical gates or the selected workspace format.')
    st.vega_lite_chart(rows,{'mark':{'type':'circle','size':150},'height':240,
        'encoding':{'x':{'field':'input saving %','type':'quantitative','title':'Actual input tokens saved (%)'},
                    'y':{'field':'F1 loss (pp)','type':'quantitative','title':'Answer F1 lost (percentage points)'},
                    'color':{'field':'format','type':'nominal'},'shape':{'field':'evaluation','type':'nominal'},
                    'tooltip':[{'field':key} for key in rows[0]]}},width='stretch')
    displayed=[]
    for row in rows:
        displayed.append({**{key:round(value,2) if isinstance(value,float) else value for key,value in row.items()},
                          'within selected F1 tolerance':row['F1 loss (pp)']<=tolerance})
    st.dataframe(displayed,hide_index=True)
    st.caption('Rejected outputs include quotations that could not be grounded in the source. They are failures, not successful abstentions. Test workspace-policy scoring is a separately recorded correction; frozen native scores remain in the report.')
    st.caption('Each point compares the same questions, model and output limit against its own raw baseline. Negative F1 loss means improvement. F1 is reference-span overlap, not a percentage of facts preserved. Fresh-test and training points have different source meetings.')
