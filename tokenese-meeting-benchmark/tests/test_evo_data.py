import copy
import json

import pytest

from tokenese.evo_data import load_cases, validate_contrast_pair, DATA, load_development
from tokenese.evo_contrasts import make_contrasts


def test_bundled_splits_and_deterministic_contrasts():
    all_sources, all_series = set(), set()
    for split in ('dev','validation','test'):
        cases = load_cases(DATA/f'{split}.json', split)
        assert len(cases) == 12 and sum(len(c['questions']) for c in cases) == 120
        sources = {c['provenance']['source_id'] for c in cases}
        series = {c['provenance']['series_id'] for c in cases}
        assert not all_sources & sources and not all_series & series
        all_sources |= sources
        all_series |= series
        assert all(c['provenance']['annotation_status'] != 'independently_reviewed' for c in cases)
    for pair in make_contrasts():
        validate_contrast_pair(pair)
    assert sum(len(c['questions']) for c in load_development()) == 160


def test_split_quote_and_two_field_contrast_rejected(tmp_path):
    payload = json.loads((DATA/'dev.json').read_text())
    path = tmp_path/'dev.json'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(path,'test')
    payload['cases'][0]['questions'][0]['evidence'][0]['quote'] = 'fabricated source'
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        load_cases(path,'dev')
    pair = copy.deepcopy(make_contrasts()[0])
    pair['after']['facts'][0]['deadline'] = 'tomorrow'
    with pytest.raises(ValueError):
        validate_contrast_pair(pair)
