import json
import socket
from dataclasses import replace
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from rag import providers
from rag.ingest import ingest
from rag.models import Draft, RagError, QuotaError, SchemaError
from rag.providers import Connection, PersonalClient, SessionGate, check_connection, public_addresses
from rag.store import Store, Index, build_index


def connection(key="visitor-key", **changes):
    return replace(Connection.create("OpenAI", key), dimensions=3, **changes)


def transport(calls):
    def send(url, headers, body):
        calls.append((url, headers, body))
        if url.endswith("/embeddings") or url.endswith(":embedContent"):
            return 200, {}, {"data": [{"embedding": [1., 0., 0.]}], "embedding": {"values": [1., 0., 0.]}}
        gemini = url.endswith(":generateContent")
        payload = json.loads(body["contents"][0]["parts"][0]["text"] if gemini else body["messages"][1]["content"])
        schema = (body["generationConfig"]["responseJsonSchema"] if gemini else body["response_format"]["json_schema"]["schema"])["title"]
        documents = payload.get("documents", [])
        sources = [{"chunk_id": d["chunk_id"], "quote": d["text"]} for d in documents]
        if schema == "Safety":
            value = {"items": [{"chunk_id": d["chunk_id"], "decision": "safe", "reason": "Policy"} for d in documents]}
        elif schema == "Relevance":
            value = {"items": [{"chunk_id": d["chunk_id"], "relevant": True, "reason": "Policy", "claims": [{"text": "Policy", "source": s}]} for d, s in zip(documents, sources)]}
        elif schema == "Conflict":
            value = {"detected": False, "explanation": "", "sources": []}
        elif schema == "Sufficiency":
            value = {"sufficient": True, "missing": [], "sources": [c["source"] for c in payload["claims"]]}
        elif schema == "Validation":
            value = {"supported": True, "unsupported_claims": []}
        else:
            value = {"answer": "Connected" if not sources else "Employees receive 20 days annual leave.", "sources": sources[:1]}
        return 200, {}, {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(value)}}],
                         "candidates": [{"finishReason": "STOP", "content": {"parts": [{"text": json.dumps(value)}]}}]}
    return send


@pytest.mark.parametrize("provider", ["Gemini", "OpenAI", "Custom"])
def test_connection_checks_both_capabilities_and_never_owner_key(monkeypatch, provider):
    calls = []
    monkeypatch.setattr(providers, "post_json", transport(calls))
    config = Connection.create(provider, "my-key", "https://example.com/v1", "answer-model", "embed-model")
    checked = check_connection(config)
    assert checked.dimensions == 3 and len(calls) == 2
    assert all("my-key" in str(headers) for _, headers, _ in calls)
    assert "my-key" not in repr(checked) and "my-key" not in str(checked.public_config)


@pytest.mark.parametrize("code,error,match", [(401, RagError, "rejected this API key"), (403, RagError, "permission"),
    (402, QuotaError, "credit"), (404, RagError, "unavailable"), (400, RagError, "request format"),
    (429, QuotaError, "rate limit"), (302, RagError, "Redirects")])
def test_safe_errors_identify_embedding_stage(monkeypatch, code, error, match):
    monkeypatch.setattr(providers, "post_json", lambda *args: (code, {}, {"error": "secret-key-private-text"}))
    with pytest.raises(error, match=match) as exc:
        check_connection(connection())
    assert "Embedding check" in str(exc.value)
    assert "secret-key-private-text" not in str(exc.value)


def test_generation_capability_failure_is_specific(monkeypatch):
    calls = []
    base = transport(calls)
    monkeypatch.setattr(providers, "post_json", lambda url, *args: base(url, *args) if url.endswith("embeddings") else (400, {}, None))
    with pytest.raises(RagError, match="Structured generation check"):
        check_connection(connection())


def test_unsupported_anthropic_is_explicit():
    with pytest.raises(RagError, match="both embeddings"):
        Connection.create("Anthropic", "test")


@pytest.mark.parametrize("url", ["http://example.com", "https://user:key@example.com", "https://example.com?key=secret",
    "https://example.com/#fragment", "https://example.com:8080", "https://example.com\\@localhost", "https://example.com\n"])
def test_unsafe_endpoint_syntax_rejected(url):
    with pytest.raises(RagError):
        Connection.create("Custom", "key", url, "g", "e")


@pytest.mark.parametrize("ip", ["127.0.0.1", "10.0.0.1", "169.254.169.254", "::1", "::ffff:127.0.0.1", "192.168.1.1"])
def test_private_resolution_rejected(monkeypatch, ip):
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: [(None, None, None, None, (ip, 443))])
    with pytest.raises(RagError, match="public internet"):
        public_addresses("example.com")


def test_transport_pins_checked_address_and_never_follows_redirect(monkeypatch):
    seen = []
    monkeypatch.setattr(providers, "public_addresses", lambda host: ["8.8.8.8"])
    class Response:
        status = 302
        def read(self, size): return b""
        def getheaders(self): return [("Location", "https://127.0.0.1")]
    class Wire:
        def __init__(self, host, address): seen.append((host, address))
        def request(self, *args): seen.append(args)
        def getresponse(self): return Response()
        def close(self): pass
    monkeypatch.setattr(providers, "PinnedHTTPSConnection", Wire)
    assert providers.post_json("https://example.com/v1/embeddings", {}, {})[0] == 302
    assert seen[0] == ("example.com", "8.8.8.8") and len(seen) == 2


def test_retries_bounded_and_cache_resumes(monkeypatch):
    calls, sleeps = [], []
    monkeypatch.setattr(providers.time, "sleep", sleeps.append)
    def fail(*args):
        calls.append(args)
        return 429, {"Retry-After": "1"}, None
    monkeypatch.setattr(providers, "post_json", fail)
    client = PersonalClient(connection(), Store())
    with pytest.raises(QuotaError): client.embed("Policy")
    assert len(calls) == 3 and sleeps == [1, 1]
    calls.clear()
    monkeypatch.setattr(providers, "post_json", transport(calls))
    client.embed("Policy")
    client.embed("Policy")
    assert len(calls) == 1 and client.usage["cache_hits"] == 1


def test_malformed_output_not_cached(monkeypatch):
    monkeypatch.setattr(providers, "post_json", lambda *a: (200, {}, {"choices": [{"finish_reason": "stop", "message": {"content": "not JSON"}}]}))
    client = PersonalClient(connection(), Store())
    for _ in range(2):
        with pytest.raises(SchemaError): client.ask("answer", "Answer", {}, Draft)
    assert client.usage["requests"] == 2 and client.usage["cache_hits"] == 0


def test_connection_namespace_and_index_compatibility(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "post_json", transport(calls))
    cache = Store()
    a = PersonalClient(connection(), cache)
    b = PersonalClient(connection(embedding_model="different"), cache)
    a.embed("same text"); b.embed("same text")
    assert len(calls) == 2
    chunks, _ = ingest([("x.json", b'{"text":"Policy", "source_doc":"Handbook"}')])
    index = build_index(chunks, a)
    restored = Index.from_dict(index.to_dict(), a.index_config)
    assert restored.chunks[0].location == chunks[0].location
    with pytest.raises(RagError): Index.from_dict(index.to_dict(), b.index_config)
    with pytest.raises(RagError): Index.from_dict(index.to_dict())
    with pytest.raises(RagError): index.retrieve([1, 0])


def test_session_gate_allows_only_one_active_operation():
    a, b = SessionGate(), SessionGate()
    with a.job():
        with pytest.raises(RagError):
            with a.job(): pass
        with b.job(): pass
    with a.job(): pass


def app_without_owner():
    app = AppTest.from_file(str(Path(__file__).parents[1] / "app.py"), default_timeout=20)
    app.secrets["GEMINI_API_KEY"] = ""
    app.secrets["FREE_TIER_CONFIRMED"] = False
    app.run()
    return app


def button(app, label):
    return next(b for b in app.button if b.label == label)


def connect_app(app, key):
    app.toggle[0].set_value(True).run()
    app.selectbox(key="provider").select("OpenAI").run()
    app.text_input(key="personal_key").set_value(key)
    button(app, "Check and connect").click().run()
    assert not app.exception
    assert app.session_state.connection.api_key == key


@pytest.mark.parametrize("mode", ["Standard RAG", "Detect & Block", "Full Safety Wall"])
def test_app_byok_works_with_exhausted_shared_quota(monkeypatch, mode):
    from rag.quota import Governor
    from contextlib import contextmanager
    @contextmanager
    def exhausted(*a, **k):
        raise QuotaError("Shared quota exhausted")
        yield
    monkeypatch.setattr(Governor, "job", exhausted)
    calls = []
    monkeypatch.setattr(providers, "post_json", transport(calls))
    app = app_without_owner()
    connect_app(app, "visitor-A")
    app.radio[1].set_value(mode).run()
    button(app, "Check the evidence").click().run()
    assert not app.exception
    assert app.session_state.results[1].status == "answered"
    assert all("visitor-A" in str(headers) for _, headers, _ in calls)
    # The shared 20-second cooldown and five-question cap do not apply.
    app.session_state.question_times = [__import__('time').time()] * 5
    button(app, "Check the evidence").click().run()
    assert not app.exception and app.session_state.results[1].status == "answered"


def test_two_sessions_disconnect_and_reconnect(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "post_json", transport(calls))
    a, b = app_without_owner(), app_without_owner()
    connect_app(a, "visitor-A"); connect_app(b, "visitor-B")
    a.session_state.private_store.put("private", "Only A")
    assert b.session_state.private_store.get("private") is None
    button(a, "Check the evidence").click().run()
    assert a.session_state.sample_indexes and not b.session_state.sample_indexes
    button(a, "Disconnect").click().run()
    assert not a.exception and a.session_state.connection is None
    assert not a.session_state.sample_indexes and a.session_state.results is None
    assert a.session_state.private_store.get("private") is None
    assert b.session_state.connection.api_key == "visitor-B"
    a.text_input(key="personal_key").set_value("visitor-C")
    button(a, "Check and connect").click().run()
    assert not a.exception and a.session_state.connection.api_key == "visitor-C"


@pytest.mark.parametrize("mode", ["Standard RAG", "Detect & Block", "Full Safety Wall"])
def test_json_upload_index_and_citations_in_app(monkeypatch, mode):
    import streamlit
    from types import SimpleNamespace
    calls = []
    monkeypatch.setattr(providers, "post_json", transport(calls))
    upload = SimpleNamespace(name="policy.json", getvalue=lambda: b'[{"text":"Employees receive 20 days annual leave.","source_doc":"Handbook","chunk_id":"leave"}]')
    monkeypatch.setattr(streamlit, "file_uploader", lambda *a, **k: [upload])
    app = app_without_owner()
    connect_app(app, "json-user")
    app.radio[0].set_value("Use my documents").run()
    app.checkbox[0].set_value(True).run()
    button(app, "Index my documents").click().run()
    assert not app.exception and app.session_state.private_index is not None
    app.text_input(key="question_private").set_value("How much leave?")
    app.radio[1].set_value(mode).run()
    button(app, "Check the evidence").click().run()
    assert not app.exception
    result = app.session_state.results[1]
    assert result.status == "answered" and "record 1" in result.citations[0]["location"]
    assert all("json-user" in str(headers) for _, headers, _ in calls)


def test_failed_reconnect_does_not_keep_old_credentials(monkeypatch):
    calls = []
    monkeypatch.setattr(providers, "post_json", transport(calls))
    app = app_without_owner(); connect_app(app, "old-key")
    monkeypatch.setattr(providers, "post_json", lambda *a: (401, {}, None))
    app.text_input(key="personal_key").set_value("bad-key")
    button(app, "Check and connect").click().run()
    assert not app.exception and app.session_state.connection is None
    assert button(app, "Check the evidence").disabled


def test_provider_change_disconnects_and_clears_index(monkeypatch):
    monkeypatch.setattr(providers, "post_json", transport([]))
    app = app_without_owner(); connect_app(app, "old-key")
    button(app, "Check the evidence").click().run()
    app.selectbox(key="provider").select("Custom").run()
    assert not app.exception and app.session_state.connection is None
    assert not app.session_state.sample_indexes and app.session_state.results is None
