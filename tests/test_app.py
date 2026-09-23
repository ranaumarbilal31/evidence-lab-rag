from pathlib import Path
from streamlit.testing.v1 import AppTest


def test_research_pause_opens_existing_byok_without_live_calls():
    app = AppTest.from_file(str(Path(__file__).parents[1] / 'app.py'), default_timeout=20)
    app.secrets['GEMINI_API_KEY'] = ''
    app.secrets['FREE_TIER_CONFIRMED'] = False
    app.secrets['RESEARCH_PAUSE_SHARED'] = True
    app.run()
    assert not app.exception
    action = next(b for b in app.button if b.label == 'Use my API key instead')
    action.click().run()
    assert not app.exception
    assert app.session_state['use_personal'] is True
    assert any(t.label == 'API key' for t in app.text_input)


def test_no_key_preview_is_honest_and_all_scenarios_render(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20)
    # Never inherit a developer's real credential while testing the no-key state.
    app.secrets["GEMINI_API_KEY"] = ""
    app.secrets["FREE_TIER_CONFIRMED"] = False
    app.run()
    assert not app.exception
    assert "Better answers" in app.title[0].value
    for scenario in ["clean", "malicious", "irrelevant", "conflicting", "insufficient"]:
        app.selectbox[0].select(scenario).run()
        assert not app.exception
        assert any("not an API result" in warning.value or "Saved demonstration — not a live response" in warning.value
                   for warning in app.warning)
        assert all(button.disabled for button in app.button if button.label == "Check the evidence")
    app.radio[0].set_value("Use my documents").run()
    assert not app.exception
    assert all(button.disabled for button in app.button if button.label in {"Index my documents", "Check the evidence"})
