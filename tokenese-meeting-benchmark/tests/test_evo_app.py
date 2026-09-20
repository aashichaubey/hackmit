from streamlit.testing.v1 import AppTest
from pathlib import Path


def test_view_renders_without_artifacts(tmp_path):
    code = f"from pathlib import Path\nfrom tokenese.evo_view import render_evolution, render_workspace\nrender_evolution(Path({str(tmp_path)!r}))\nrender_workspace(Path({str(tmp_path)!r}))"
    app = AppTest.from_string(code).run(timeout=20)
    assert not app.exception
    assert app.info


def test_whole_application_renders_without_api_calls(monkeypatch):
    monkeypatch.delenv('OPENAI_API_KEY', raising=False)
    from openai import OpenAI
    def forbid_client(*args,**kwargs):
        raise AssertionError('Rendering the app must not create an API client')
    monkeypatch.setattr(OpenAI,'__init__',forbid_client)
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / 'app.py').run(timeout=20)
    assert not app.exception
    assert [tab.label for tab in app.tabs] == [
        'V1 demo',
        'V1 benchmark',
        'Research archive',
        'V2 transcript study',
    ]


def test_view_renders_tiny_saved_trace(tmp_path):
    import json
    from tokenese.evo_grammar import GrammarSpec
    g = GrammarSpec()
    trace = {'generations':[{'candidates':[{'grammar':g.to_dict(),'grammar_id':g.id,'origin':'seed',
        'summary':{'correct':1,'total':1,'accuracy':1,'visible_prompt_tokens':100}}]}], 'repairs':[]}
    (tmp_path/'search.json').write_text(json.dumps(trace))
    code = f"from pathlib import Path\nfrom tokenese.evo_view import render_evolution\nrender_evolution(Path({str(tmp_path)!r}))"
    assert not AppTest.from_string(code).run(timeout=20).exception
