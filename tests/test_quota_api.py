import json
from types import SimpleNamespace

import pytest

from rag.api import Gemini
from rag.config import Settings, Limits, GEN_MODEL, EMBED_MODEL
from rag.models import Draft, RagError, QuotaError, SchemaError
from rag.quota import Governor
from rag.store import Store


def test_live_requires_key_tier_and_explicit_limits():
    assert not Settings.from_mapping({}).ready
    assert not Settings.from_mapping({"GEMINI_API_KEY": "fixture"}).ready


def test_daily_reservation_refunds_unused_but_counts_attempts(tmp_path):
    gov = Governor({"model": Limits(100, 100000, 3)}, tmp_path / "quota.json")
    with gov.job({"model": 3}) as ticket:
        gov.take("model", 10, ticket)
    assert gov.snapshot()["attempts"]["model"] == 1
    with gov.job({"model": 2}) as ticket:
        gov.take("model", 10, ticket)
        gov.take("model", 10, ticket)
        with pytest.raises(QuotaError):
            gov.take("model", 10, ticket)
    restored = Governor({"model": Limits(100, 100000, 3)}, tmp_path / "quota.json")
    with pytest.raises(QuotaError):
        with restored.job({"model": 1}):
            pass


def test_minute_token_limit_and_daily_reset():
    date = ["2026-09-13"]
    gov = Governor({"model": Limits(1, 100, 1)}, clock=lambda: 100.0, day=lambda: date[0])
    with gov.job({"model": 1}) as ticket:
        with pytest.raises(QuotaError):
            gov.take("model", 101, ticket)
        gov.take("model", 10, ticket)
    date[0] = "2026-09-14"
    assert gov.snapshot()["attempts"]["model"] == 0


def fake_client(monkeypatch, operation):
    from google import genai
    sdk = SimpleNamespace(models=SimpleNamespace(generate_content=operation), close=lambda: None)
    monkeypatch.setattr(genai, "Client", lambda **kw: sdk)
    settings = Settings("fake-test-key", True, Limits(100, 100000, 100), Limits(100, 100000, 100))
    gov = Governor({GEN_MODEL: settings.generation, EMBED_MODEL: settings.embedding})
    return Gemini(settings, gov, Store())


def test_sdk_secret_errors_are_redacted(monkeypatch):
    def operation(**kwargs):
        raise RuntimeError("fake-test-key plus visitor document text")
    client = fake_client(monkeypatch, operation)
    with pytest.raises(RagError) as err:
        client.ask("answer", "Answer", {"question": "test"}, Draft)
    assert "fake-test-key" not in str(err.value)
    assert "visitor document" not in str(err.value)


def test_valid_responses_are_cached_but_invalid_responses_are_not(monkeypatch):
    calls = []
    def operation(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(candidates=[SimpleNamespace(finish_reason="STOP")],
            text=json.dumps({"answer": "Supported answer", "sources": []}), usage_metadata=None)
    client = fake_client(monkeypatch, operation)
    client.ask("answer", "Answer", {"question": "test"}, Draft)
    client.ask("answer", "Answer", {"question": "test"}, Draft)
    assert len(calls) == 1 and client.usage["cache_hits"] == 1
    assert calls[0]["model"] == GEN_MODEL
    assert calls[0]["config"].tools is None
    assert calls[0]["config"].response_schema is None
    assert calls[0]["config"].response_json_schema == Draft.model_json_schema()
    assert calls[0]["config"].automatic_function_calling.disable is True


def test_truncated_output_is_rejected(monkeypatch):
    client = fake_client(monkeypatch, lambda **kw: SimpleNamespace(
        candidates=[SimpleNamespace(finish_reason="MAX_TOKENS")], text='{"answer":"bad"}', usage_metadata=None))
    with pytest.raises(SchemaError):
        client.ask("answer", "Answer", {}, Draft)


def test_rate_limit_pauses_without_paid_or_model_fallback(monkeypatch):
    from google.genai.errors import ClientError
    calls = []
    def operation(**kwargs):
        calls.append(kwargs["model"])
        raise ClientError(429, {"error": {"message": "Quota exhausted"}})
    client = fake_client(monkeypatch, operation)
    with pytest.raises(QuotaError):
        client.ask("answer", "Answer", {}, Draft)
    assert calls == [GEN_MODEL]
    with pytest.raises(QuotaError):
        client.ask("answer", "Answer", {}, Draft)
    assert calls == [GEN_MODEL]


def test_service_errors_have_bounded_retries(monkeypatch):
    from google.genai.errors import ServerError
    calls = []
    monkeypatch.setattr("rag.api.time.sleep", lambda _: None)
    def operation(**kwargs):
        calls.append(kwargs["model"])
        raise ServerError(503, {"error": {"message": "Unavailable"}})
    client = fake_client(monkeypatch, operation)
    with pytest.raises(RagError):
        client.ask("answer", "Answer", {}, Draft)
    assert calls == [GEN_MODEL] * 3
