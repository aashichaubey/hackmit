"""Artifact replay and a session-only meeting workspace. No calls on render."""
from __future__ import annotations

import json
import os
from pathlib import Path

import streamlit as st

from .evo_data import DATA, REPORTS
from .evo_grammar import GrammarSpec, compile_facts
from .evo_memory import compile_meeting, answer_from_memory, memory_key, workflow_usage
from .facts import MeetingFacts
from .evo_usage import Runner
from .evo_eval import context_for
from .facts import Answer
from .prompts import answer_prompt


def read_artifact(name: str, directory: Path = REPORTS) -> dict:
    path = directory / name
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {"artifact_error": f"Cannot read {name}"}


def render_public_benchmarks():
    public_directory = REPORTS.parent / 'publicqa'
    catalog = read_artifact('catalog.json', public_directory)
    if not catalog:
        return
    if 'counts' not in catalog:
        catalog = {'artifact_error': catalog.get('artifact_error', 'Incomplete public QA catalog')}
    if catalog.get('artifact_error'):
        st.warning(catalog['artifact_error'])
        return
    with st.expander('Published meeting QA · broader evaluation', expanded=True):
        st.write('Human-annotated questions test answer coverage, abstention and source grounding. Structural compression keeps every utterance; no extraction call is needed.')
        results = []
        for filename, label in [('pilot.json', '@ speaker notation'), ('pilot-readable.json', 'Readable speaker notation'), ('pilot-minimal.json', 'Minimal speaker notation')]:
            report = read_artifact(filename, public_directory)
            for method, metrics in report.get('methods', {}).items():
                results.append({'experiment': label, 'method': method, 'questions': metrics['questions'],
                                'answer F1': f"{metrics['macro_f1']:.2%}", 'balanced answerability': f"{metrics['balanced_answerability']:.2%}",
                                'ungrounded': metrics['ungrounded_answers'], 'API input tokens': metrics['actual_input_tokens'],
                                'API cost USD': metrics['actual_usd'], 'advances': report['advance']})
        if results:
            st.dataframe(results, hide_index=True)
        st.caption('Training pilots, not qualification. F1 is Unicode token overlap with published human spans. Each experiment has a matched raw baseline; output limits differ between iterations. Missing usage stays unknown.')
        local_search = read_artifact('local-search.json', public_directory)
        if local_search and 'datasets' not in local_search:
            local_search = {'artifact_error':local_search.get('artifact_error', 'Incomplete local compression report')}
        if local_search.get('artifact_error'):
            st.warning(local_search['artifact_error'])
            local_search = {}
        if local_search:
            st.markdown('**Consecutive-speaker inheritance · local screen**')
            st.dataframe([{'dataset':name, 'contexts':row['unique_contexts'],
                           'context token saving':f"{row['saved_fraction']:.2%}", 'reconstruction failures':row['round_trip_failures']}
                          for name,row in local_search['datasets'].items()], hide_index=True)
            st.caption('Local token counts, not actual API savings. Exact reconstruction does not establish model comprehension.')
        inheritance = read_artifact('inheritance.json', public_directory)
        if inheritance.get('methods'):
            raw_i, encoded_i = inheritance['methods']['raw'], inheritance['methods']['encoded']
            st.warning(f"Four-question inheritance diagnostic: input tokens {raw_i['actual_input_tokens']:,} → {encoded_i['actual_input_tokens']:,}, but answer F1 {raw_i['macro_f1']:.0%} → {encoded_i['macro_f1']:.0%}. Research only; quality did not hold.")
        dialogue = read_artifact('dialogue.json', public_directory)
        if dialogue and not all(k in dialogue for k in ('questions','response_token_overlap_f1','attribution_f1','actual_input_tokens')):
            dialogue = {'artifact_error':dialogue.get('artifact_error', 'Incomplete dialogue report')}
        if dialogue.get('artifact_error'):
            st.warning(dialogue['artifact_error'])
            dialogue = {}
        if dialogue:
            st.markdown('**MISeD dialogue baseline**')
            response_col, attribution_col, usage_col = st.columns(3)
            response_col.metric('Response overlap F1', f"{dialogue['response_token_overlap_f1']:.2%}")
            attribution_col.metric('Attribution F1', f"{dialogue['attribution_f1']:.2%}")
            usage_col.metric('Input tokens · eight questions', f"{dialogue['actual_input_tokens']:,}" if dialogue['actual_input_tokens'] is not None else 'Unknown')
            st.caption('Raw transcript with published preceding dialogue. This measures reference overlap and citation matching, not factual entailment or a compressed dialogue system.')
        st.dataframe([{'dataset':name, **counts} for name,counts in catalog['counts'].items()], hide_index=True)
        st.caption('MISeD dialogue and attribution data are loaded separately; extractive pilot scores do not measure dialogue quality. Shared meeting families are excluded from future held-out evaluation.')
        st.markdown('[MeetingQA](https://github.com/adobe-research/meetingqa) · [MeeQA](https://github.com/reutapel/MeeQA) · [MISeD](https://github.com/google-research-datasets/MISeD)')


def render_evolution(directory: Path = REPORTS):
    from .evo_tradeoffs import render as render_tradeoffs
    render_tradeoffs()
    render_public_benchmarks()
    search = read_artifact("search.json", directory)
    local = read_artifact("local.json", directory)
    frozen = read_artifact("frozen.json", directory)
    test = read_artifact("test.json", directory)
    st.subheader("A smaller language. Meaning put to the test.")
    st.caption("Discover → question → expose a failure → repair one rule. The model never receives a grammar legend.")
    if not search and not local:
        st.info("No evolution run yet. The local compiler can be screened without API calls: python -m tokenese.evo_search --local")
        return
    a, b, c = st.columns(3)
    labels = {'qualified':'Qualified','incomplete_evaluation':'Incomplete','no_qualified_candidate':'No passing candidate'}
    with a:
        st.metric("Language qualification", labels.get(frozen.get('language_status'),'Not evaluated'))
    with b:
        st.metric("Product qualification", labels.get(frozen.get('product_status'),'Not evaluated'))
    with c:
        ledger = test.get("ledger") or search.get("ledger") or {}
        st.metric("Frozen experiment spend", f"${ledger['known_usd']:.4f}" if 'known_usd' in ledger else "Not measured")
    if frozen and frozen.get("language_status") != "qualified":
        st.warning("The original frozen experiment did not qualify. Published human-annotated datasets now support a separate successor study above; its training pilots do not authorize automatic encoded answers.")
    generations = search.get("generations", [])
    if generations:
        generation = st.slider("Generation", 1, len(generations), len(generations))-1 if len(generations) > 1 else 0
        current = generations[generation]
        candidates = current['candidates']
        chart = [{"grammar":r['grammar_id'], "input_tokens":r['summary']['visible_prompt_tokens'],
                  "accuracy":r['summary']['accuracy'], "origin":r.get('origin','seed')} for r in candidates]
        st.vega_lite_chart(chart, {'mark':{'type':'circle','size':110}, 'height':260,
            'encoding':{'x':{'field':'input_tokens','type':'quantitative','title':'Answer input tokens','scale':{'zero':False}},
                        'y':{'field':'accuracy','type':'quantitative','title':'Exact answer accuracy','scale':{'domain':[0,1]}},
                        'color':{'field':'origin','type':'nominal','title':'Candidate origin'},
                        'tooltip':[{'field':'grammar'},{'field':'input_tokens'},{'field':'accuracy'},{'field':'origin'}]}}, width='stretch')
        st.caption("Development probes in this generation only. X: local full-prompt token estimates. Y: exact model QA accuracy. A round trip alone is not a comprehension result.")
        candidate = st.selectbox("Inspect language", candidates, format_func=lambda r: f"{r['grammar_id']} · {r['summary']['correct']}/{r['summary']['total']} correct · {r.get('origin','seed')}")
        grammar = GrammarSpec(**candidate['grammar'])
        left, right = st.columns(2)
        with left:
            st.markdown("**Language rules**")
            st.json(grammar.to_dict())
            st.caption(f"Parent: {candidate.get('parent_id') or 'initial seed'} · Grammar ID: {grammar.id}")
        with right:
            path = DATA / "dev.json"
            if path.exists():
                cases = json.loads(path.read_text())['cases']
                example = st.selectbox("Public development excerpt", cases, format_func=lambda c:c['id'])
                st.code(compile_facts(MeetingFacts(facts=example['facts']), grammar).text, language=None)
                st.markdown(f"[Source]({example['provenance']['url']}) · {example['provenance']['annotation_status']}")
    elif local:
        st.dataframe([{k:r[k] for k in ('grammar_id','context_tokens','visible_input_tokens')} for r in local.get('local_frontier',[])], hide_index=True)
        st.caption("Local token estimates only. Model comprehension has not been measured.")
    st.markdown("**What a failure taught the language**")
    repairs = search.get('repairs', [])
    meaningful = sorted([r for r in repairs if r['result']['fixed']],key=lambda r:(not r['result']['confirmed'],-len(r['result']['fixed']),r['result']['additional_tokens']))
    if meaningful:
        repair = st.selectbox('Inspect a repair', meaningful, format_func=lambda r:f"{r['category']} · {r['result']['status']} · {r['result']['additional_tokens']:+} probe tokens")
        qid = (repair['result']['paired_fixes'] or repair['result']['fixed'])[0]
        before = next(r for r in repair['before'] if r['question_id'] == qid)
        after = next(r for r in repair['after'] if r['question_id'] == qid)
        l, r = st.columns(2)
        with l:
            st.write(before['question'])
            st.error(f"Before: {(before.get('answer') or {}).get('answer','invalid')}")
            st.write('Expected:', before['answers'])
        with r:
            st.success(f"After: {(after.get('answer') or {}).get('answer','invalid')}")
            st.write('Rule:', repair['category'])
            st.write('Regressions:', repair['result']['regressions'])
        contrast = before.get('contrast_id')
        if contrast:
            comparison = [r for r in repair['after'] if r['question_id'] == contrast]
            if comparison:
                st.write('Contrast:', comparison[0]['question'], comparison[0]['answers'], comparison[0]['answer'])
        st.caption('Improvement is provisional unless its contrast passes without regression and it generalizes across two source meetings.')
    else:
        st.info("No recorded repair has improved an answer yet. The application does not invent a successful repair story.")
    if test:
        st.markdown("**Unseen sources · frozen test**")
        rows = []
        for source, methods in test['summaries'].items():
            for method, metrics in methods.items():
                rows.append({'facts':source,'method':method,'correct':metrics['correct'],'questions':metrics['total'],
                             'actual_input_tokens':metrics['actual_input_tokens'],'answer_usd':metrics['workflow_answer_usd'],
                             'worst_category':metrics['worst_category_accuracy']})
        st.dataframe(rows, hide_index=True)
        st.caption('Actual provider usage; replayed rows retain original usage. Answer cost excludes extraction; complete workflow below includes it once.')
        st.dataframe([{'questions_per_meeting':n, **v} for n,v in test['workflow'].items()], hide_index=True)
        with st.expander('Category results, evidence limits, and qualification gates'):
            st.json({'language':test['language_gate'],'product':test['product_gate']})
    successor = read_artifact('classifier-next-dev.json', directory)
    if successor.get('paired_diagnostics'):
        with st.expander('Classifier development · unqualified follow-up'):
            st.caption('Four development excerpts, unchanged broader questions. This follow-up does not alter the frozen language or qualify a new product route.')
            summary = successor['summaries']
            st.dataframe([{'method':label,'correct':summary[key]['correct'],'questions':summary[key]['total'],
                           'answer_input_tokens':summary[key]['actual_input_tokens']}
                          for label,key in [('Raw notes','raw_qa'),('Original extraction','baseline_diagnostic_qa'),
                                            ('Source-span selection','candidate_diagnostic_qa')]],hide_index=True)
            paired = successor['paired_diagnostics']['candidate']
            if paired['new_errors_vs_raw']:
                st.warning(f"Candidate introduced {len(paired['new_errors_vs_raw'])} errors on questions raw notes answered correctly. Savings do not pass the quality requirement.")
            else:
                st.info('Development results cannot qualify this candidate. A fresh independent evaluation is required.')
            st.write(f"Encoded: {paired['encoded_correct']}/{paired['encoded_questions']} correct. Raw fallback: {paired['fallback_correct']}/{paired['fallback_questions']} correct.")
            st.dataframe([{'questions_per_excerpt':n,**v} for n,v in successor['diagnostic_workflows']['candidate'].items()],hide_index=True)
            st.caption('Diagnostic costs include extraction once. Unsupported-content flags were bypassed only for this QA diagnosis; this is not a deployable workflow.')
            st.json({'candidate_extraction':summary['candidate_extraction'],'shared_ledger_at_followup':successor.get('ledger')})
    source_study = read_artifact('source-grammar-dev.json', directory)
    if source_study.get('workflows'):
        with st.expander('Complete-source grammar · development pilot'):
            st.caption('A separate reversible source grammar. It retains unsupported/background content and makes no extraction call. No grammar legend is sent to the answerer.')
            left, right = source_study['raw_summary'],source_study['encoded_summary']
            st.write(f"Raw: {left['correct']}/{left['total']}. Source grammar: {right['correct']}/{right['total']}.")
            st.write(f"Actual answer input: {left['actual_input_tokens']:,} → {right['actual_input_tokens']:,} tokens.")
            if not source_study['pilot_pass']:
                st.warning('This pilot did not pass its quality-and-savings requirement. It is not active in the meeting workspace.')
            st.dataframe([{'questions_per_excerpt':n,**v} for n,v in source_study['workflows'].items()],hide_index=True)
            st.caption('Original development labels and scores remain unchanged. Exact reversibility does not establish model comprehension.')
            stability = read_artifact('source-grammar-stability.json', directory)
            if stability.get('summary'):
                st.dataframe([{k:v for k,v in row.items() if k != 'output_counts'} for row in stability['summary']],hide_index=True)
                st.caption('Five real repeats per method on selected failures only; not a representative benchmark or best-of-five score.')
    label_review = read_artifact('label-review.json', directory)
    if label_review.get('counts'):
        with st.expander('Annotation audit · independent review pending'):
            st.caption('Offline checks on development annotations only. These are review candidates, not confirmed error counts; original labels and scores remain unchanged.')
            st.dataframe([{'review_category':key,'candidates':value} for key,value in label_review['counts'].items()],hide_index=True)
            st.write('A concise source-only review packet is saved as reports/evolution/label-review.md. It contains three selected issues and their complete public source excerpts, without model answers.')
            st.caption('This selected development review cannot substitute for a fresh, independently checked qualification corpus.')
    with st.expander('Experiment identity and accounting'):
        st.json({'identity':search.get('identity'), 'frozen':frozen, 'ledger':ledger if generations or frozen else {}})


def render_workspace(directory: Path = REPORTS):
    mode = st.radio('Workspace', ['Transcript exploration', 'Original fact-memory experiment'], horizontal=True, key='workspace_mode')
    if mode == 'Transcript exploration':
        from .evo_transcript_workspace import render
        render()
        return
    st.subheader("One meeting. Follow-up questions.")
    st.caption("Extraction runs once per notes/model/grammar configuration. Each answer still pays for its input context. Pasted notes remain in this session.")
    frozen = read_artifact('frozen.json', directory)
    research = st.checkbox('Research preview: allow the frozen unqualified grammar', value=False)
    if research:
        st.warning('This is an explicit research preview. Quality is not qualified; unsupported or ambiguous content still uses raw notes.')
    notes = st.text_area('Your meeting notes', height=220, key='evo_notes', placeholder='Paste meeting notes, then ask a question…')
    if notes.strip():
        from .evo_transcript import compile_minimal_transcript
        from .evo_transcript_search import compile_inheritance
        with st.expander('Lossless transcript notation · local preview'):
            notation = st.selectbox('Structural notation', ['Consecutive-speaker inheritance', 'Minimal separator'], key='evo_structural_preview')
            structural = (compile_inheritance if notation == 'Consecutive-speaker inheritance' else compile_minimal_transcript)(notes)
            st.caption('Every utterance is preserved. This preview makes no model calls; comprehension is evaluated separately.')
            left, right = st.columns(2)
            left.metric('Original context tokens', structural.source_tokens)
            right.metric('Notation context tokens', structural.encoded_tokens,
                         delta=structural.encoded_tokens-structural.source_tokens, delta_color='inverse')
            st.code(structural.text, language=None)
            st.caption(f'Route: {structural.profile} · {structural.reason}')
    question = st.text_input('Ask a follow-up', key='evo_question')
    if st.button('Answer', type='primary', key='evo_answer'):
        if not notes.strip() or not question.strip():
            st.error('Add meeting notes and a question.')
        elif not os.getenv('OPENAI_API_KEY'):
            st.error('Set OPENAI_API_KEY to answer a question.')
        else:
            from openai import OpenAI
            try:
                key = memory_key(notes, frozen, research)
                if st.session_state.get('evo_memory_key') != key:
                    # A new in-memory runner also discards any old private replay entries.
                    st.session_state['evo_runner'] = Runner(OpenAI(), public=False)
                    st.session_state['evo_memory'] = compile_meeting(notes, frozen, st.session_state['evo_runner'], research=research)
                    st.session_state['evo_memory_key'] = key
                memory = st.session_state['evo_memory']
                with st.spinner('Reading the selected context…'):
                    answer_from_memory(memory, question, st.session_state['evo_runner'])
            except Exception as exc:
                st.error(str(exc))
    memory = st.session_state.get('evo_memory')
    if memory and st.session_state.get('evo_memory_key') == memory_key(notes, frozen, research):
        st.caption(f"Route: {memory.route} · {memory.reason}")
        for result in memory.answers:
            with st.chat_message('user'):
                st.write(result['question'])
            with st.chat_message('assistant'):
                if result.get('error'):
                    st.error(result['error'])
                else:
                    st.write((result.get('answer') or {}).get('answer', 'No structured answer'))
                st.caption(f"Actual API usage: {result.get('usage')} · {'local replay' if result.get('replayed') else 'measured request'}")
        st.write('Workflow:', workflow_usage(memory))
        with st.expander('Inspect compiled facts and source evidence'):
            if memory.compiled:
                st.code(memory.compiled.text, language=None)
            if memory.facts:
                st.dataframe([f.model_dump() for f in memory.facts.facts], hide_index=True)
            st.write('Ambiguities:', memory.ambiguity_notes)
        if st.button('Compare representations · extra model calls', key='evo_compare'):
            if not question.strip():
                st.error('Add a question for the comparison.')
            else:
                try:
                    runner = st.session_state['evo_runner']
                    comparison_memory = memory if memory.compiled else compile_meeting(notes,frozen,runner,research=True)
                    if comparison_memory.compiled is None or comparison_memory.facts is None:
                        st.info('No valid compiled facts are available to compare; the selected answer remains raw.')
                    else:
                        case = {'notes':notes,'facts':comparison_memory.facts.model_dump()['facts']}
                        contexts = {'raw':notes, 'english':context_for(case,None,'english'),
                                    'prose':context_for(case,None,'prose'), 'encoded':comparison_memory.compiled.text}
                        results = {method:runner.call(answer_prompt(context,question),Answer,purpose='demonstration') for method,context in contexts.items()}
                        st.session_state['evo_comparison'] = {'memory_key':memory.key,'question':question,'results':results}
                except Exception as exc:
                    st.error(str(exc))
        comparison = st.session_state.get('evo_comparison')
        if comparison and comparison['memory_key'] == memory.key and comparison['question'] == question:
            st.caption('Explicit demonstration overhead, separate from the selected product workflow. Encoded quality may be unqualified; no gold answer is assumed for your notes.')
            st.dataframe([{'method':method,'answer':(r.get('answer') or {}).get('answer'),
                           'input_tokens':(r.get('usage') or {}).get('input_tokens'),
                           'output_tokens':(r.get('usage') or {}).get('output_tokens'),
                           'local_replay':r.get('replayed'), 'error':r.get('error')} for method,r in comparison['results'].items()],hide_index=True)
    else:
        st.info('New notes or settings start a new session memory when you answer.')
