from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from tokenese.evo_usage import Ledger, Runner, BudgetExceeded, dollars
from tokenese.facts import Answer


class FakeClient:
    def __init__(self, fail=False, missing=False):
        self.responses = self
        self.calls = 0
        self.fail, self.missing = fail, missing
    def with_options(self, **kwargs):
        assert kwargs['max_retries'] == 0
        return self
    def parse(self, **kwargs):
        self.calls += 1
        if self.fail:
            raise TimeoutError('unknown provider usage')
        return SimpleNamespace(id='r', output_parsed=Answer(found=True, answer='Ana'), usage=None if self.missing else
            SimpleNamespace(input_tokens=100, output_tokens=10, input_tokens_details=SimpleNamespace(cached_tokens=20)))


def test_persisted_budget_and_replay_attribution(tmp_path):
    ledger = Ledger(tmp_path/'ledger.db', max_calls=1, max_usd=1)
    client = FakeClient()
    runner = Runner(client, ledger)
    first = runner.call('public', Answer)
    replay = runner.call('public', Answer)
    assert client.calls == 1 and not first['replayed'] and replay['replayed']
    assert replay['usage']['cached_input_tokens'] == 20
    assert Ledger(ledger.path, max_calls=100).summary()['limits']['max_calls'] == 1
    with pytest.raises(BudgetExceeded):
        runner.call('new', Answer)


def test_failures_keep_reservation_and_private_has_no_disk(tmp_path):
    ledger = Ledger(tmp_path/'ledger.db')
    result = Runner(FakeClient(fail=True), ledger).call('public', Answer)
    assert result['usage'] is None
    assert ledger.summary()['unknown_usage_calls'] == 1
    assert ledger.summary()['reserved_or_spent_usd'] > 0
    with pytest.raises(ValueError):
        Runner(FakeClient(), ledger, public=False)


def test_concurrent_reservations_cannot_overspend(tmp_path):
    ledger = Ledger(tmp_path/'ledger.db', max_calls=3)
    def reserve(i):
        try:
            ledger.reserve(str(i), 'probe', .01)
            return True
        except BudgetExceeded:
            return False
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert sum(pool.map(reserve, range(20))) == 3
    assert ledger.summary()['actual_calls'] == 3


def test_dollars_cached_subset_not_added_twice():
    assert dollars({'input_tokens':1000000, 'cached_input_tokens':1000000, 'output_tokens':0}) == .1


def test_pacing_is_shared_between_runners(tmp_path,monkeypatch):
    from tokenese import evo_usage
    waits = []
    monkeypatch.setattr(evo_usage.time,'time',lambda:100.0)
    monkeypatch.setattr(evo_usage.time,'sleep',waits.append)
    first = Ledger(tmp_path/'ledger.db')
    second = Ledger(tmp_path/'ledger.db')
    first.pace(85000)
    second.pace(85000)
    assert waits == [30.0]


def test_successor_pilot_cannot_bypass_discovery_ceiling(tmp_path):
    ledger = Ledger(tmp_path/'ledger.db')
    with ledger.connect() as db:
        db.executemany("INSERT INTO calls(key,purpose,status,reserved,created) VALUES(?,'classifier_development','pending',0,0)",
                       [(str(i),) for i in range(1500)])
    for purpose in ('probe','finalist','validation','classifier_development'):
        with pytest.raises(BudgetExceeded):
            ledger.reserve('overflow',purpose,0)


def test_replicates_are_budgeted_distinct_calls_with_identical_provider_inputs(tmp_path):
    ledger = Ledger(tmp_path/'ledger.db',max_calls=3)
    ledger.pace = lambda *_: None
    client = FakeClient()
    requests = []
    original = client.parse
    def capture(**kwargs):
        requests.append(kwargs)
        return original(**kwargs)
    client.parse = capture
    runner = Runner(client,ledger)
    first = runner.call('public',Answer)
    repeated = runner.call('public',Answer,replicate=1)
    replay = runner.call('public',Answer,replicate=1)
    assert requests[0] == requests[1]
    assert first['request_hash'] != repeated['request_hash']
    assert replay['replayed'] and ledger.summary()['actual_calls'] == 2
    assert runner.call('public',Answer)['request_hash'] == first['request_hash']
    with pytest.raises(ValueError):
        runner.call('public',Answer,replicate=-1)


def test_successor_allocation_is_explicit_immutable_and_preserves_global_reserve(tmp_path):
    ledger=Ledger(tmp_path/'ledger.db',max_calls=6,max_usd=1)
    with pytest.raises(BudgetExceeded):
        ledger.reserve('a','successor_discovery:new',0)
    purpose=ledger.allocate_successor_discovery('new',2,4,'User-directed successor study')
    assert ledger.allocate_successor_discovery('new',2,4,'User-directed successor study')==purpose
    with pytest.raises(ValueError):
        ledger.allocate_successor_discovery('new',3,4,'User-directed successor study')
    with pytest.raises(BudgetExceeded):
        ledger.allocate_successor_discovery('another',1,4,'Would overallocate')
    ledger.reserve('a',purpose,0)
    Ledger(ledger.path,max_calls=600)  # Restart cannot expand either limit.
    ledger.reserve('b',purpose,0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve('c',purpose,0)
    for i in range(4):
        ledger.reserve(str(i),'qualification',0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve('overflow','qualification',0)
    assert ledger.summary()['actual_calls']==6


def test_successor_allocation_still_enforces_dollars(tmp_path):
    ledger=Ledger(tmp_path/'ledger.db',max_calls=10,max_usd=.01)
    purpose=ledger.allocate_successor_discovery('new',2,4,'Bounded follow-up')
    with pytest.raises(BudgetExceeded):
        ledger.reserve('a',purpose,.02)


def test_successor_dispatch_preserves_highest_recorded_reserve(tmp_path):
    ledger=Ledger(tmp_path/'ledger.db',max_calls=10)
    ledger.allocate_successor_discovery('a',1,8,'Reserve eight evaluation calls')
    b=ledger.allocate_successor_discovery('b',1,0,'Second allocation cannot weaken reserve')
    ledger.reserve('raw1','baseline',0)
    ledger.reserve('raw2','baseline',0)
    with pytest.raises(BudgetExceeded):
        ledger.reserve('candidate',b,0)
