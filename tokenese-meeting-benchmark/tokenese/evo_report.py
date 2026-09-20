"""Readable report from saved public measurements; never runs the answer model."""
from __future__ import annotations

import json
import csv
from collections import Counter

from .evo_data import REPORTS, DATA
from .evo_usage import Ledger, dollars


def percent(value):
    return 'unavailable' if value is None else f'{value:.1%}'


def write_report(directory=REPORTS):
    frozen = json.loads((directory/'frozen.json').read_text())
    validation = json.loads((directory/'validation.json').read_text())
    test = json.loads((directory/'test.json').read_text())
    search = json.loads((directory/'search.json').read_text())
    ablations = json.loads((directory/'ablations.json').read_text())
    ledger = Ledger(directory/'ledger.sqlite').summary()
    evolved = test['summaries']['gold']['evolved']
    raw = test['summaries']['gold']['raw']
    english = test['summaries']['gold']['english']
    prose = test['summaries']['gold']['prose']
    savings = {name:1-evolved['actual_input_tokens']/value['actual_input_tokens'] if value['actual_input_tokens'] and evolved['actual_input_tokens'] is not None else None
               for name,value in [('raw',raw),('english',english),('prose',prose)]}
    context_savings = {name:1-evolved['context_tokens']/value['context_tokens'] if value['context_tokens'] else None
                       for name,value in [('raw',raw),('english',english),('prose',prose)]}
    lines = ['# Tokenese evolution — measured results', '',
             f"Language status: **{frozen['language_status']}**. Product status: **{frozen['product_status']}**. Accounting: **{ledger['accounting_status']}**.", '',
             f"On the frozen provisional test corpus, the selected grammar used **{percent(savings['raw'])} fewer answer-input tokens than raw excerpts**, "
             f"with **{evolved['correct']}/{evolved['total']}** exact field-preservation answers. "
             f"Same-fact input savings were **{percent(savings['english'])} versus compact English** and **{percent(savings['prose'])} versus explicit prose**.", '',
             f"The test language metric gate **{'passed' if test['language_gate']['metric_gate_pass'] else 'failed'}**: "
             f"{evolved['correct']}/120 against the 119/120 threshold, with {len(test['language_gate']['new_critical_errors'])} new critical errors relative to at least one English baseline. "
             'Independent annotation review is an additional requirement, not the only missing gate.', '',
             '**This is not a qualified general-purpose compression claim.** Public annotations were model-drafted and have not been independently reviewed. '
             'Raw-excerpt savings include fact extraction; only the same-fact comparison isolates encoding. '
             'The product retains raw notes for unsupported content, uncertainty, or a missing qualified grammar.', '',
             f"Context-only local token reductions (excluding question/instruction/schema overhead): raw **{percent(context_savings['raw'])}**, compact English **{percent(context_savings['english'])}**, prose **{percent(context_savings['prose'])}**.", '',
             '## Frozen language comparison', '',
             '| Split | Representation | Exact answers | Actual input tokens | Actual output tokens | Answer cost |',
             '|---|---|---:|---:|---:|---:|']
    for split,report in [('validation',validation),('test',test)]:
        for method,m in report['summaries']['gold'].items():
            cost = 'unknown' if m['workflow_answer_usd'] is None else f"${m['workflow_answer_usd']:.6f}"
            lines.append(f"| {split} | {method} | {m['correct']}/{m['total']} | {m['actual_input_tokens']} | {m['actual_output_tokens']} | {cost} |")
    lines += ['', 'These are provider input/output measurements including schema overhead. Replayed rows retain original provider usage. '
              'Logical workflow cost is separate from actual research spending; replay is not a provider cache hit.', '',
              '## Broader source-question / classifier results', '',
              '| Test route | Exact source-question answers | Unsupported answers on unknowns | Wrong/missing abstentions | Input tokens |',
              '|---|---:|---:|---:|---:|']
    for method,m in test['summaries']['parsed'].items():
        lines.append(f"| {method} | {m['correct']}/{m['total']} | {m['unsupported_answers']} | {m['wrong_or_missing_abstentions']} | {m['actual_input_tokens']} |")
    metrics = [x['metrics'] for x in test['extraction'].values() if x['metrics'] is not None]
    tp = sum(m['matched'] for m in metrics)
    test_cases = json.loads((DATA/'test.json').read_text())['cases']
    ng = sum(len(c['facts']) for c in test_cases)
    np = sum(m['predicted'] for m in metrics)
    precision,recall = tp/np if np else 0,tp/ng if ng else 0
    f1 = 2*precision*recall/(precision+recall) if precision+recall else 0
    routes = dict(Counter(x['route'] for x in test['extraction'].values()))
    lines += ['', f"Classifier exact semantic-tuple precision **{percent(precision)}**, recall **{percent(recall)}**, F1 **{percent(f1)}**. "
              'Duplicates count. This is a literal-preservation diagnostic against provisional gold; harmless paraphrases can fail it. '
              f"Source-question QA is the complementary usefulness measure. Extraction metrics were available for {len(metrics)}/12 excerpts; failed extractions still contribute all their gold facts as recall misses.", '', f"Frozen-route diagnostics: `{json.dumps(routes)}`.", '',
              '## Recurring workflow cost', '',
              'Extraction is charged once per meeting, answer context once per independent question. These are the first 1/4/10 recorded source questions per meeting, not scaled averages.', '',
              '| Questions / meeting | Raw input | Product input incl. extraction | Raw cost | Product cost | Cost savings |',
              '|---:|---:|---:|---:|---:|---:|']
    for n,w in test['workflow'].items():
        raw_cost = 'unknown' if w['raw_usd'] is None else f"${w['raw_usd']:.6f}"
        product_cost = 'unknown' if w['product_usd'] is None else f"${w['product_usd']:.6f}"
        lines.append(f"| {n} | {w['raw_input_tokens']} | {w['product_input_tokens']} | {raw_cost} | {product_cost} | {percent(w['cost_savings_fraction'])} |")
    lines += ['', 'A raw fallback after extraction can preserve answer behavior while costing more than starting with raw notes. '
              'The shipped unqualified default skips extraction and answers raw. Research previews are explicitly labeled.', '',
              '## Discovery and repair', '',
              f"Seed 42; **{len(search['generations'])} generations**, at most six candidate slots and sixteen questions per round. "
              'Earlier development runs before the suite/probe audit are retained separately and all their paid calls remain in the ledger.', '',
              f"Selected grammar: `{frozen['grammar_id']}`.", '', '```json',json.dumps(frozen['grammar'],indent=2,ensure_ascii=False),'```','']
    seed_summary = test['summaries']['gold'].get('initial_seed')
    if seed_summary and seed_summary['actual_input_tokens'] == evolved['actual_input_tokens'] and seed_summary['correct'] == evolved['correct']:
        lines.append('**Evolution did not beat the initial English seed on the selected held-out result.** Shorter development candidates existed, but traded away accuracy. A seed winning is an observed failure of the stronger search-improvement hypothesis, not evidence of discovery.')
    repaired = [r for r in search['repairs'] if r['result']['confirmed']]
    for repair in repaired:
        lines += ['', f"Controlled repair `{repair['parent_id']}` → `{repair['child_id']}`: `{repair['category']}`; "
                  f"fixed {len(repair['result']['fixed'])} probe answers across {repair['result']['source_meetings_improved']} source IDs, "
                  f"{repair['result']['additional_tokens']:+} visible probe-input tokens, no new shared-probe regressions. "
                  'Paired fixes include synthetic development contrasts; this does not qualify the child on the full suite.']
    matched = ablations['matched_search_comparison']
    lines += ['',f"Matched repair/no-repair ablation: **{matched.get('repair_probe_evaluations')} vs {matched.get('no_repair_probe_evaluations')} logical probes** over the first two rounds.",
              '```json',json.dumps(matched.get('paired_rounds',[]),indent=2),'```','',
              'The ablation compares the same questions and logical budgets; cache reuse changes actual spending. Later adaptive rounds and full finalists are not a matched causal comparison.', '',
              '### Four-question batching (development only)', '',
              '| Method | Correct / 16 | Batch input tokens | Batch answer cost |', '|---|---:|---:|---:|']
    for method,batches in ablations['four_question_batches'].items():
        usage = [b['usage'] for b in batches]
        known = all(u is not None for u in usage)
        cost = sum(dollars(u) for u in usage) if known else None
        lines.append(f"| {method} | {sum(sum(b['correct']) for b in batches)}/16 | {sum(u['input_tokens'] for u in usage) if known else 'unknown'} | {f'${cost:.6f}' if known else 'unknown'} |")
    lines += ['', 'Batching is a supporting comparison, not the central language-discovery claim. See `ablations.json` for exact requests and single-question measurements.', '',
              '## Spending and reproducibility', '',
              f"**{ledger['actual_calls']} actual API calls; ${ledger['known_usd']:.6f} known cost** across annotation, failed drafts, all development searches, repairs, ablations, extraction, validation, and test. "
              f"Unknown usage records: **{ledger['unknown_usage_calls']}**. Global ceiling: 3,500 calls / $10. "
              'Rates: $0.40/M input, $0.10/M provider-cached input, $1.60/M output for the pinned reference model.', '',
              '[Pinned-model pricing](https://developers.openai.com/api/docs/models/gpt-4.1-mini). Usage is recorded from the API, not inferred from local character counts.', '',
              '```sh', '.venv/bin/python -m tokenese.evo_benchmark --audit', '.venv/bin/python -m pytest -q', '.venv/bin/streamlit run app.py', '```', '',
              'Artifacts: `frozen.json` contains model/tokenizer/prompt/schema/compiler/data hashes; `search.json` has generations and controlled repairs; '
              '`validation.json` and `test.json` retain every public question, response, and usage record; `verification.json` reproduces gates and arithmetic. '
              '`ledger.sqlite` is the shared persistent research budget/replay store. Private notes never enter it.', '',
              '## Evaluation changes and limits', '',
              '- Kept V1 and earlier results intact. Preserved original public source questions as a separate holistic track.',
              '- Added 36 attributed public source excerpts, 360 bounded-language questions, 360 broader source questions, and 40 deterministic development contrast questions.',
              '- Changed extraction grading from loose quote overlap to multiset exact semantic fields; added precision/recall and field diagnostics.',
              '- Added strict Unicode-aware answer grading: names, signs, C++/C#, negation, and this/next date modifiers cannot collapse through ASCII punctuation deletion.',
              '- Added source/category breakdowns, hallucination/abstention counts, paired critical regressions, and descriptive Wilson intervals. Correlated questions are not independent statistical evidence.',
              '- Model-authored facts still contain annotation weaknesses (for example, meeting dates mistaken for deadlines and discussion mistaken for commitment in development). No independent reviewer is claimed.',
              '- Cross-organization splits prevent series overlap but confound organization with distribution shift. Meeting updates can summarize multiple sessions; the unit is a source document/excerpt.',
              '- Literal field-preservation questions are intentionally narrow. The broader track can reveal schema coverage loss and also has provisional labels.',
              '- Missing independent review, failed 119/120 quality gates, new critical errors, or insufficient same-fact savings prevent qualification. Gates were not relaxed to obtain a win.',
              '- Custom model weights, arbitrary literal rewriting, aliases, external compression services, and an optional model grammar proposer were not needed for this bounded local-mutation implementation.', '']
    (directory/'README.md').write_text('\n'.join(lines))
    summary = {'language_status':frozen['language_status'],'product_status':frozen['product_status'],'grammar_id':frozen['grammar_id'],
               'test_language':evolved,'input_savings_fraction':savings,'context_savings_fraction':context_savings,'classifier':{'precision':precision,'recall':recall,'f1':f1,'measured_cases':len(metrics),'total_cases':len(test_cases)},
               'workflow':test['workflow'],'ledger':ledger,'confirmed_development_repairs':len(repaired)}
    (directory/'metrics.json').write_text(json.dumps(summary,indent=2)+'\n')
    review_rows = []
    for split,report in [('validation',validation),('test',test)]:
        cases = {c['id']:c for c in json.loads((DATA/f'{split}.json').read_text())['cases']}
        for track in ('gold','parsed'):
            for method,rows in report[track].items():
                for row in rows:
                    if row['correct']:
                        continue
                    predicted = row.get('answer') or {}
                    category = 'provider_failure' if row.get('error') else 'wrong_abstention' if row['expected_found'] and not predicted.get('found') else 'unsupported_answer' if not row['expected_found'] and predicted.get('found') else 'value_mismatch'
                    case = cases[row['case_id']]
                    question = next(q for q in case['questions']+case.get('holistic_questions',[]) if q['id']==row['question_id'])
                    review_rows.append({'split':split,'track':track,'method':method,'case_id':row['case_id'],'question_id':row['question_id'],
                        'question':row['question'],'expected':json.dumps(row['answers'],ensure_ascii=False),'expected_found':row['expected_found'],
                        'predicted':predicted.get('answer'),'predicted_found':predicted.get('found'), 'failure_type':category,
                        'source_url':case['provenance']['url'],'evidence':json.dumps(question['evidence'],ensure_ascii=False),
                        'annotation_status':case['provenance']['annotation_status'],'reviewer':'','review_decision':''})
    if review_rows:
        with (directory/'annotation-review.csv').open('w',newline='') as handle:
            writer = csv.DictWriter(handle,fieldnames=list(review_rows[0]))
            writer.writeheader()
            writer.writerows(review_rows)
    print(json.dumps({k:v for k,v in summary.items() if k in ('language_status','product_status','input_savings_fraction','classifier','confirmed_development_repairs')},indent=2))
    return summary


if __name__ == '__main__':
    write_report()
