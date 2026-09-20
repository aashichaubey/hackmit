from types import SimpleNamespace
import pytest
from tokenese.evo_public_eval import SpanAnswer
from tokenese.evo_transcript_workspace import prepare,answer
from tokenese.evo_usage import Runner


class Client:
    def __init__(self,spans):
        self.responses=self;self.spans=spans;self.calls=0
    def with_options(self,**kwargs):
        return self
    def parse(self,**kwargs):
        self.calls+=1
        return SimpleNamespace(id='test',output_parsed=SpanAnswer(found=True,spans=self.spans),usage=None)


def test_workspace_retains_general_notes_and_makes_one_private_call_with_replay():
    notes='& SPEAKER_0: Not Friday. & SPEAKER_1: Monday.'
    encoding=prepare(notes,'minimal');client=Client(['Monday.']);runner=Runner(client,public=False)
    first=answer(notes,'When?','minimal',encoding,runner)
    second=answer(notes,'When?','minimal',encoding,runner)
    assert first['result']['answer']['spans']==['Monday.']
    assert second['result']['replayed'] and client.calls==1
    assert prepare('Freeform notes remain intact.','blocks').text=='Freeform notes remain intact.'


def test_workspace_rejects_ungrounded_answers_and_public_storage():
    notes='Nothing decided.';encoding=prepare(notes,'raw')
    result=answer(notes,'When?','raw',encoding,Runner(Client(['Friday']),public=False))
    assert result['result']['error']
    with pytest.raises(ValueError):
        answer(notes,'When?','raw',encoding,Runner(Client(['Friday'])))
    with pytest.raises(ValueError):
        answer('Different notes','When?','raw',encoding,Runner(Client([]),public=False))


def test_workspace_preview_updates_without_api_calls(monkeypatch):
    from streamlit.testing.v1 import AppTest
    from openai import OpenAI
    def forbidden(*args,**kwargs):
        raise AssertionError('Preview must not create a model client')
    monkeypatch.setattr(OpenAI,'__init__',forbidden)
    app=AppTest.from_string('from tokenese.evo_transcript_workspace import render\nrender()').run()
    app.text_area(key='transcript_notes').set_value('& SPEAKER_0: Friday. & SPEAKER_1: Next Monday.').run()
    assert not app.exception and len(app.metric)==3
    assert app.session_state['transcript_prepared'].encoded_tokens < app.session_state['transcript_prepared'].source_tokens
    before=app.session_state['transcript_preparation_key']
    app.text_input(key='transcript_question').set_value('When?').run()
    assert app.session_state['transcript_preparation_key']==before


def test_auto_named_transcript_preserves_names_and_restores_inherited_quote():
    from tokenese.evo_transcript_blocks import decode_blocks
    notes='Natalie Tran: Hello.\nD Tran: First.\nD Tran: Second.\nD Tran: Third.'
    encoding=prepare(notes,'auto')
    assert encoding.profile=='named'
    assert encoding.encoded_tokens < encoding.source_tokens
    assert decode_blocks(encoding.text,encoding.profile)==notes
    result=answer(notes,'What next?','auto',encoding,Runner(Client(['  - Second.']),public=False))
    assert result['method']=='blocks'
    assert result['result']['answer']['spans']==['D Tran: Second.']


def test_auto_handles_minimal_freeform_and_already_compact_notes():
    from tokenese.evo_transcript_workspace import fallback_message
    notes='& SPEAKER_0: First. & SPEAKER_1: Second.'
    encoding=prepare(notes,'auto')
    result=answer(notes,'What next?','auto',encoding,Runner(Client(['Second.']),public=False))
    assert not result['result'].get('error')
    for notes in ['Meeting notes: decide next week.', 'Natalie Tran: Hello.\nD Tran: First.\n  - Second.']:
        encoding=prepare(notes,'auto')
        assert encoding.text==notes
        assert encoding.encoded_tokens==encoding.source_tokens
    assert 'already contain compact' in fallback_message(notes,'auto')


def test_auto_is_ui_default_and_named_notes_compress():
    from streamlit.testing.v1 import AppTest
    app=AppTest.from_string('from tokenese.evo_transcript_workspace import render\nrender()').run()
    assert app.selectbox[0].value=='Automatic (recommended)'
    app.text_area(key='transcript_notes').set_value('Natalie Tran: Hello.\nD Tran: First.\nD Tran: Second.\nD Tran: Third.').run()
    assert not app.exception
    assert app.session_state['transcript_prepared'].profile=='named'
    assert not app.info


def test_malformed_answer_is_recoverable_and_not_replayed():
    from tokenese.evo_transcript_workspace import answer_error_message,workspace_prompt
    class TruncatedClient(Client):
        def parse(self,**kwargs):
            self.calls+=1
            if self.calls==1:
                raise ValueError('Invalid JSON: EOF while parsing a string [type=json_invalid]')
            return SimpleNamespace(id='recovered',output_parsed=SpanAnswer(found=True,spans=['Monday.']),usage=None)
    notes='Monday.'
    encoding=prepare(notes,'auto')
    client=TruncatedClient([])
    runner=Runner(client,public=False)
    first=answer(notes,'When?','auto',encoding,runner)
    assert first['result']['error'] and first['result']['answer'] is None
    assert 'Try again' in answer_error_message(first['result']['error'])
    second=answer(notes,'When?','auto',encoding,runner)
    assert second['result']['answer']['spans']==['Monday.']
    assert not second['result']['replayed'] and client.calls==2
    assert '400 words' in workspace_prompt(notes,'When?')


def test_workspace_keeps_verified_spans_with_explicit_partial_warning():
    notes='Monday. Budget undecided.'
    result=answer(notes,'What happened?','auto',prepare(notes,'auto'),Runner(Client(['Monday.','Budget is approved.']),public=False))['result']
    assert not result.get('error')
    assert result['answer']['spans']==['Monday.']
    assert '1 quotation(s) omitted' in result['warning']
    assert 'incomplete' in result['warning']
    failed=answer(notes,'What happened?','auto',prepare(notes,'auto'),Runner(Client(['Budget is approved.']),public=False))['result']
    assert failed['error'] and 'warning' not in failed
