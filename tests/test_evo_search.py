import json
import random

from tokenese.evo_data import DATA
from tokenese.evo_grammar import GrammarSpec
from tokenese.evo_mutate import seed_population, mutate
from tokenese.evo_search import run_search, probe_ids, pareto


def test_local_run_deterministic_and_never_calls_model(tmp_path):
    class NoNetwork:
        def call(self,*args,**kwargs):
            raise AssertionError('local screen made a network call')
    left = run_search(output=tmp_path/'a.json', local=True, seed=42, runner=NoNetwork())
    right = run_search(output=tmp_path/'b.json', local=True, seed=42, runner=NoNetwork())
    assert left == right
    assert all(not s['integrity_errors'] for s in left['seed_population'])


def test_mutation_is_repeatable_and_heldout_guard(tmp_path):
    a, b = random.Random(42), random.Random(42)
    assert [mutate(GrammarSpec(),a).id for _ in range(20)] == [mutate(GrammarSpec(),b).id for _ in range(20)]
    (tmp_path/'selection-lock.json').write_text('{}')
    import pytest
    with pytest.raises(ValueError, match='frozen'):
        run_search(output=tmp_path/'search.json',local=False)
    assert run_search(output=tmp_path/'local.json',local=True)['status'] == 'local_only'


def test_probe_keeps_contrasts_paired():
    from tokenese.evo_data import load_development
    cases = load_development()
    ids = probe_ids(cases,[],0)
    assert len(ids) == 16
    qs = {q['id']:q for c in cases for q in c['questions']}
    assert sum(bool(qs[i].get('contrast_id')) for i in ids) == 8
    assert all(qs[i].get('contrast_id',i) in ids for i in ids)
