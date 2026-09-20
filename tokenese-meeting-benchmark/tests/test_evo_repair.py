import pytest

from tokenese.evo_grammar import GrammarSpec
from tokenese.evo_repair import propose_repairs, accept_repair


def rows(fixed=False, regress=False):
    return [{'question_id':'q1','correct':fixed,'source_id':'s1','contrast_id':'q2','category':'proposal'},
            {'question_id':'q2','correct':not regress,'source_id':'s2','contrast_id':'q1','category':'proposal'}]


def test_repair_is_structural_and_regression_rejected():
    grammar = GrammarSpec(kind_forms=('!', '?', '=', '+', '−'))
    edits = propose_repairs(grammar, rows())
    assert len(edits) == 1 and edits[0][0].kind_forms[1] == 'proposal'
    assert accept_repair(rows(), rows(fixed=True), {'parent':10,'child':12})['accepted']
    assert not accept_repair(rows(), rows(fixed=True, regress=True), {'parent':10,'child':12})['accepted']
    assert not accept_repair(rows(), rows(fixed=True), {'parent':10,'child':12})['confirmed']
    with pytest.raises(ValueError):
        accept_repair(rows(), rows()[:1], {'parent':10,'child':12})
