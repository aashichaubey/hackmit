import pytest

from tokenese.evo_public_data import adapt_extractive,meeting_family,require_disjoint,validate_spans
from tokenese.evo_transcript import compile_transcript,decode_transcript


def row(answer='yes',start=11,impossible=False):
    return {'title':'ES2002a','id':'q1','context':'Speaker 0: yes','question':'Approved?',
            'answers':{'text':[answer],'answer_start':[start]},'is_impossible':impossible}


def test_adapter_quarantines_bad_offsets_without_relabeling():
    good=row()
    bad=row(start=0);bad['id']='bad'
    cases,excluded=adapt_extractive([good,bad],'meetingqa','train')
    assert len(cases)==1 and excluded==[{'id':'bad','reason':'invalid_source_offset'}]
    assert good['answers']['answer_start']==[11]


def test_human_references_are_joint_spans_and_annotators_are_alternatives():
    a=row();b=row('Speaker 0: yes',0);b['id']='q2'
    cases,_=adapt_extractive([a,b],'meetingqa','train')
    assert len(cases)==1 and len(cases[0]['references'])==2
    unknown=row('',-1,True)
    assert validate_spans(unknown)==[]
    cases,excluded=adapt_extractive([a,unknown],'meetingqa','train')
    assert not cases and excluded[0]['reason']=='annotator_answerability_disagreement'


def test_shared_ami_sessions_cannot_cross_dataset_split_boundary():
    assert meeting_family('ES2002a')==meeting_family('es2002d')
    with pytest.raises(ValueError):
        require_disjoint([{'family_id':meeting_family('ES2002a')}],[{'family_id':meeting_family('ES2002c')}])


def test_transcript_codec_keeps_speakers_dates_negation_and_every_turn():
    for source in ('Speaker 0: Not Friday.\n Speaker 1: Next Friday.\n Speaker 1: Next Friday.\n',
                   '& SPEAKER_00: Never. & SPEAKER_12: C++ costs $50.'):
        encoded=compile_transcript(source)
        assert encoded.profile!='raw'
        assert decode_transcript(encoded.text,encoded.profile)==source
        assert encoded.encoded_tokens < encoded.source_tokens
    for source in ('Literal @0: is data','Speaker 0: A & SPEAKER_1: B','plain notes'):
        assert compile_transcript(source).text==source


def test_native_scoring_requires_all_joint_spans_but_allows_annotator_alternatives():
    from tokenese.evo_public_eval import score
    case={'context':'yes and Friday','expected_found':True,
          'references':[[{'text':'yes'},{'text':'Friday'}],[{'text':'yes and Friday'}]]}
    def result(spans,found=True):
        return {'answer':{'found':found,'spans':spans},'error':None}
    assert score(case,result(['yes','Friday']))['f1']==1
    assert score(case,result(['yes']))['f1'] < 1
    assert score(case,result(['invented']))['ungrounded']
    assert not score(case,result([],True))['valid']
    assert score(case,result([],False))['f1']==0
    assert score(case,{'answer':None,'error':'timeout'})['f1']==0


def test_native_scoring_decodes_headers_and_preserves_negation_unicode_symbols():
    from tokenese.evo_public_eval import score,overlap
    case={'context':'Speaker 0: Łukasz said no.','expected_found':True,
          'references':[[{'text':'Speaker 0: Łukasz said no.'}]]}
    result={'answer':{'found':True,'spans':['@0: Łukasz said no.']},'error':None}
    assert score(case,result,'meetingqa')['f1']==1
    assert not score(case,result,'meetingqa')['ungrounded']
    assert overlap('C++','C')['f1']<1
    assert overlap('not approved','approved')['f1']<1


def test_unknown_api_usage_is_not_reported_as_free():
    from tokenese.evo_public_eval import aggregate,score
    case={'context':'nothing','expected_found':False,'references':[[]]}
    result={'answer':{'found':False,'spans':[]},'error':None,'usage':None}
    row={'dataset':'x','meeting_id':'m','result':result,'score':score(case,result)}
    metrics=aggregate([row])
    assert metrics['actual_usd'] is None and metrics['actual_input_tokens'] is None
    assert metrics['unanswerable_accuracy']==1 and metrics['unknown_usage_calls']==1


def test_readable_candidate_is_lossless_and_leaves_other_profiles_unchanged():
    from tokenese.evo_transcript import compile_readable_transcript
    source='& SPEAKER_00: No. & SPEAKER_1: Friday, not Monday.'
    encoded=compile_readable_transcript(source)
    assert encoded.text=='Speaker 00: No. Speaker 1: Friday, not Monday.'
    assert decode_transcript(encoded.text,encoded.profile)==source
    assert encoded.encoded_tokens < encoded.source_tokens
    for source in ('Speaker 0: No.', '& SPEAKER_0: Literal Speaker 9: is data.'):
        encoded=compile_readable_transcript(source)
        assert encoded.profile=='raw' and encoded.text==source


def test_minimal_candidate_preserves_literal_markers_and_numeric_ids():
    from tokenese.evo_transcript import compile_minimal_transcript
    source='& SPEAKER_00: No. & SPEAKER_2: yes.'
    encoded=compile_minimal_transcript(source)
    assert encoded.text=='SPEAKER_00: No. SPEAKER_2: yes.'
    assert decode_transcript(encoded.text,encoded.profile)==source
    for source in ('& SPEAKER_0: Literal SPEAKER_9: is data.', 'ordinary text'):
        assert compile_minimal_transcript(source).profile=='raw'


def test_reference_interval_union_does_not_require_duplicate_or_nested_evidence():
    from tokenese.evo_public_eval import score,reference_text
    case={'context':'approved Friday','expected_found':True,
          'references':[[{'start':0,'end':8,'text':'approved'},
                         {'start':0,'end':8,'text':'approved'},
                         {'start':0,'end':3,'text':'app'}]]}
    result={'answer':{'found':True,'spans':['approved']},'error':None}
    assert reference_text(case,case['references'][0])=='approved'
    assert score(case,result)['f1']==1
    assert len(case['references'][0])==3
