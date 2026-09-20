from tokenese.evo_memory import compile_meeting, answer_from_memory, product_identity, code_hashes, verify_frozen, memory_key, freeze_digest
from tokenese.evo_eval import experiment_identity
from tokenese.evo_grammar import GrammarSpec
from tokenese.evo_usage import Runner


class SessionRunner(Runner):
    def __init__(self, unsupported=False, empty=False):
        super().__init__(None, public=False)
        self.calls = []
        self.unsupported, self.empty = unsupported, empty
    def call(self, prompt, shape, purpose='probe', **kwargs):
        self.calls.append(purpose)
        if purpose == 'extraction':
            answer = {'facts':[] if self.empty else [{'kind':'action','person':'Ana','text':'review','deadline':'Friday','source_quote':'Ana will review by Friday.'}],
                      'unsupported_content':self.unsupported,'ambiguity_notes':[]}
        else:
            answer = {'found':True,'answer':'Ana'}
        return {'answer':answer,'error':None,'usage':{'input_tokens':100,'output_tokens':20,'cached_input_tokens':0},'replayed':False}


def manifest():
    grammar = GrammarSpec()
    result = {'identity':experiment_identity(), 'product_identity':product_identity(), 'code_hashes':code_hashes(),
            'grammar':grammar.to_dict(), 'grammar_id':grammar.id, 'language_status':'qualified','product_status':'qualified',
            'dataset_manifest':{},'candidate_ids':[grammar.id],'search_hash':'test-fixture'}
    result['freeze_hash'] = freeze_digest(result)
    return result


def test_four_followups_extract_once_and_private_state_only():
    runner = SessionRunner()
    notes = 'Meeting opening remarks. '*20 + 'Ana will review by Friday.'
    memory = compile_meeting(notes,manifest(),runner)
    assert memory.route == 'encoded'
    for question in ['Who?', 'Who owns review?', 'Assignee?', 'Who reviews?']:
        answer_from_memory(memory,question,runner)
    assert runner.calls.count('extraction') == 1
    assert runner.calls.count('product_answer') == 4
    assert runner.ledger is None


def test_unqualified_empty_unsupported_and_stale_manifest_fall_back():
    notes = 'Ana will review by Friday.'
    m = manifest()
    m['product_status'] = 'incomplete_evaluation'
    runner = SessionRunner()
    assert compile_meeting(notes,m,runner).route == 'raw'
    assert runner.calls == []
    for runner in (SessionRunner(empty=True), SessionRunner(unsupported=True)):
        memory = compile_meeting(notes,manifest(),runner)
        assert memory.route == 'raw'
        assert answer_from_memory(memory,'Who?',runner)['answer']['found']
    m = manifest()
    m['code_hashes'] = {}
    assert compile_meeting(notes,m,SessionRunner()).route == 'raw'
    assert memory_key(notes,m) != memory_key(notes+'changed',m)
