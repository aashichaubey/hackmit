from tokenese.evo_eval import score_answer, summarize, paired_regressions, classifier_metrics, evaluate
from tokenese.facts import Fact, MeetingFacts
from tokenese.evo_grammar import GrammarSpec


def row(qid='q1', correct=True, category='owner'):
    return {"question_id": qid, "case_id": 'c', "category": category, "correct": correct,
            "expected_found": True, "answer": {"found": True, "answer": 'Ana'},
            "usage": {"input_tokens": 20, "output_tokens": 4, "cached_input_tokens": 0}}


def test_found_flag_and_modifier_grade():
    question = {"answers": ['next Friday'], "expected_found": True}
    assert score_answer({"found": True, "answer": 'next Friday'}, question)
    assert not score_answer({"found": True, "answer": 'Friday'}, question)
    assert not score_answer({"found": False, "answer": 'next Friday'}, question)
    assert not score_answer(None, question)


def test_unicode_and_significant_symbols_are_not_erased():
    for predicted, gold in [('北京','上海'), ('C++','C#'), ('5','-5'), ('Łukasz','Lukasz')]:
        assert not score_answer({'found':True,'answer':predicted},{'expected_found':True,'answers':[gold]})
    assert score_answer({'found':True,'answer':'  ŁUKASZ. '},{'expected_found':True,'answers':['Łukasz']})


def test_invalid_and_unknown_usage_do_not_become_zero():
    r = row(correct=False)
    r.update(answer=None, usage=None, error='refused')
    result = summarize([r])
    assert result['accuracy'] == 0
    assert result['actual_input_tokens'] is None
    assert result['invalid_or_failed'] == 1


def test_paired_regression_union_and_multiset_extraction():
    assert paired_regressions([row(correct=False)], [[row()], [row(correct=False)]])['new_critical_errors'] == ['q1']
    f = Fact(kind='action', person='Ana', text='review', deadline='next Friday', source_quote='Ana reviews')
    gold = MeetingFacts(facts=[f, f])
    assert classifier_metrics(gold, MeetingFacts(facts=[f]))['recall'] == .5
    wrong = f.model_copy(update={'person': 'Bo'})
    assert classifier_metrics(MeetingFacts(facts=[f]), MeetingFacts(facts=[wrong]))['f1'] == 0


def test_failed_extraction_never_leaks_gold_to_answerer():
    case = {'id':'c', 'split':'dev', 'notes':'Original notes', 'facts':[{'kind':'action','text':'GOLD SECRET','source_quote':'Original'}],
            'provenance':{'source_id':'source'}, 'questions':[{'id':'q','question':'Who?', 'answers':['not found'], 'expected_found':False,'category':'unknown'}]}
    class Fake:
        def call(self, prompt, shape, **kwargs):
            assert 'GOLD SECRET' not in prompt and 'Original notes' in prompt
            assert 'Grammar:' not in prompt
            return {'answer':{'found':False,'answer':'not found'}, 'usage':None,'error':None}
    assert evaluate(GrammarSpec(), [case], Fake(), parsed_facts={})[0]['correct']
