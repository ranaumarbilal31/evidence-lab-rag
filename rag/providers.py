"""Session-only BYOK clients. Never fall back to owner credentials."""
from __future__ import annotations

import http.client
import ipaddress
import json
import socket
import ssl
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from typing import Protocol
from urllib.parse import quote, urlsplit

import numpy as np
from pydantic import ValidationError

from .api import fingerprint, input_bound, serialized
from .config import GEN_MODEL, EMBED_MODEL, MAX_INPUT, MAX_OUTPUT, PROMPT_VERSION
from .models import Draft, RagError, QuotaError, SchemaError


class AIClient(Protocol):
    usage: dict
    def ask(self, stage, system, payload, schema): ...
    def embed(self, text): ...
    def close(self): ...


@dataclass(frozen=True)
class Connection:
    provider: str
    api_key: str = field(repr=False)
    endpoint: str = ""
    generation_model: str = ""
    embedding_model: str = ""
    dimensions: int = 0

    @classmethod
    def create(cls, provider, key, endpoint="", generation="", embedding=""):
        if provider == "Anthropic":
            raise RagError("Direct Anthropic keys are not supported: this app requires one key with both embeddings and structured generation. Use Gemini, OpenAI, or a compatible provider offering both.")
        if provider not in {"Gemini", "OpenAI", "Custom"}:
            raise RagError("Choose a supported provider.")
        if not key.strip() or any(ord(c) < 32 or ord(c) > 126 for c in key.strip()):
            raise RagError("Enter a valid API key.")
        if provider == "Gemini":
            endpoint, generation, embedding = "https://generativelanguage.googleapis.com/v1beta", GEN_MODEL, EMBED_MODEL
        elif provider == "OpenAI":
            endpoint, generation, embedding = "https://api.openai.com/v1", "gpt-4.1-mini", "text-embedding-3-small"
        endpoint = validate_endpoint(endpoint)
        if not generation.strip() or not embedding.strip():
            raise RagError("Enter both a generation model and an embedding model.")
        if any(len(m) > 200 or any(ord(c) < 32 for c in m) for m in (generation, embedding)):
            raise RagError("The model name is invalid.")
        return cls(provider, key.strip(), endpoint, generation.strip(), embedding.strip())

    @property
    def index_config(self):
        return {"provider": self.provider, "endpoint": self.endpoint,
                "model": self.embedding_model, "dimensions": self.dimensions,
                "version": "byok-text-v1"}

    @property
    def public_config(self):
        return {**self.index_config, "generation_model": self.generation_model,
                "embedding_model": self.embedding_model}


def validate_endpoint(endpoint):
    try:
        parts = urlsplit(endpoint.strip())
        port = parts.port
        if (parts.scheme != "https" or not parts.hostname or parts.username or parts.password
                or parts.query or parts.fragment or port not in (None, 443)
                or "\\" in endpoint or any(ord(c) < 33 or ord(c) > 126 for c in endpoint)):
            raise ValueError
        return endpoint.rstrip("/")
    except ValueError:
        raise RagError("Use a public HTTPS API base URL on port 443, without credentials, query parameters, or fragments.") from None


def public_addresses(host):
    try:
        addresses = list(dict.fromkeys(item[4][0] for item in socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)))
        if not addresses or any(not ipaddress.ip_address(ip).is_global for ip in addresses):
            raise RagError("The API endpoint must resolve only to public internet addresses. Local and private endpoints are not supported.")
        return addresses
    except (OSError, ValueError):
        raise RagError("The API endpoint could not be resolved. Check its public hostname.") from None


class PinnedHTTPSConnection(http.client.HTTPSConnection):
    """Connect to the checked IP, preserving hostname verification/SNI.

    No proxy environment, second DNS lookup, or redirects can change the destination.
    """
    def __init__(self, host, address):
        super().__init__(host, timeout=45, context=ssl.create_default_context())
        self.address = address

    def connect(self):
        raw = socket.create_connection((self.address, 443), timeout=self.timeout)
        try:
            self.sock = self._context.wrap_socket(raw, server_hostname=self.host)
        except BaseException:
            raw.close()
            raise


def post_json(url, headers, payload):
    parts = urlsplit(url)
    address = public_addresses(parts.hostname)[0]
    connection = PinnedHTTPSConnection(parts.hostname, address)
    try:
        connection.request("POST", parts.path or "/", serialized(payload).encode("utf-8"),
                           {"Content-Type": "application/json", **headers})
        response = connection.getresponse()
        body = response.read(2 * 1024 * 1024 + 1)
        if len(body) > 2 * 1024 * 1024:
            raise RagError("The provider response exceeds the allowed size.")
        # Error bodies may contain credentials or document text. Never expose them.
        data = json.loads(body) if 200 <= response.status < 300 else None
        return response.status, dict(response.getheaders()), data
    except RagError:
        raise
    except Exception:
        raise RagError("The provider could not be reached or returned an unreadable response. Please retry later.") from None
    finally:
        connection.close()


class SessionGate:
    def __init__(self):
        self.lock = threading.Lock()

    @contextmanager
    def job(self, expected=None):
        if not self.lock.acquire(blocking=False):
            raise RagError("Another operation is already running in your session.")
        try:
            yield {}
        finally:
            self.lock.release()


class PersonalClient:
    def __init__(self, connection, cache):
        self.connection, self.cache = connection, cache
        self.usage = {"requests": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}

    @property
    def index_config(self):
        return self.connection.index_config

    @property
    def configuration(self):
        return self.connection.public_config

    def close(self):
        pass  # Each transport call closes its socket; credentials remain session-owned.

    def _request(self, operation, payload):
        config = self.connection
        headers = ({"x-goog-api-key": config.api_key} if config.provider == "Gemini"
                   else {"Authorization": "Bearer " + config.api_key})
        for attempt in range(3):
            self.usage["requests"] += 1
            code, headers_out, data = post_json(config.endpoint + operation, headers, payload)
            if 200 <= code < 300:
                if not isinstance(data, dict):
                    raise SchemaError("The provider returned an invalid response object.")
                return data
            if 300 <= code < 400:
                raise RagError("The endpoint redirected the request. Redirects are blocked; enter the final API base URL.")
            if code == 401:
                raise RagError("The provider rejected this API key. Check or replace it.")
            if code == 403:
                raise RagError("This key lacks permission for the selected model. Check model access with your provider.")
            if code == 402:
                raise QuotaError("Your provider account has insufficient credit. Check its billing or use another compatible key.")
            if code in {404, 405}:
                raise RagError("The selected model or API operation is unavailable. This app requires embeddings and structured generation through the same key.")
            if code in {400, 422}:
                raise RagError("The provider rejected the request format. Choose models supporting embeddings and JSON-schema structured generation.")
            if code == 429:
                try:
                    delay = float(next((v for k, v in headers_out.items() if k.lower() == "retry-after"), "0"))
                except (ValueError, TypeError):
                    delay = 0
                if attempt < 2 and 0 < delay <= 10:
                    time.sleep(delay)
                    continue
                raise QuotaError("Your key's rate limit or quota was reached. Progress is cached; check your provider's limits or credits and retry later.")
            if code in {500, 502, 503, 504} and attempt < 2:
                time.sleep(attempt + 1)
                continue
            raise RagError("The provider is temporarily unavailable. Please retry later.")

    def ask(self, stage, system, payload, schema):
        config = self.connection
        if input_bound(system, payload, schema) > MAX_INPUT:
            raise RagError("The evidence exceeds this demo's input budget. Use a shorter question or smaller documents.")
        key = fingerprint(["generate", config.public_config, PROMPT_VERSION, stage, system, payload, schema.model_json_schema()])
        cached = self.cache.get(key)
        if cached is not None:
            self.usage["cache_hits"] += 1
            return schema.model_validate(cached)
        if config.provider == "Gemini":
            data = self._request("/models/" + quote(config.generation_model, safe="") + ":generateContent", {
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [{"text": serialized(payload)}]}],
                "generationConfig": {"temperature": 0, "maxOutputTokens": MAX_OUTPUT,
                    "responseMimeType": "application/json", "responseJsonSchema": schema.model_json_schema()}})
            try:
                candidate = data["candidates"][0]
                if candidate.get("finishReason") != "STOP":
                    raise ValueError
                text = "".join(p.get("text", "") for p in candidate["content"]["parts"] if not p.get("thought"))
                usage = data.get("usageMetadata", {})
                self.usage["input_tokens"] += usage.get("promptTokenCount", 0)
                self.usage["output_tokens"] += usage.get("candidatesTokenCount", 0) + usage.get("thoughtsTokenCount", 0)
            except (KeyError, IndexError, TypeError, ValueError):
                raise SchemaError("The provider blocked, truncated, or malformed its structured response.") from None
        else:
            data = self._request("/chat/completions", {"model": config.generation_model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": serialized(payload)}],
                "temperature": 0, "max_tokens": MAX_OUTPUT,
                "response_format": {"type": "json_schema", "json_schema": {
                    "name": schema.__name__, "strict": True, "schema": schema.model_json_schema()}}})
            try:
                choice = data["choices"][0]
                if choice["finish_reason"] != "stop" or choice["message"].get("refusal"):
                    raise ValueError
                text = choice["message"]["content"]
                usage = data.get("usage") or {}
                self.usage["input_tokens"] += usage.get("prompt_tokens", 0)
                self.usage["output_tokens"] += usage.get("completion_tokens", 0)
            except (KeyError, IndexError, TypeError, ValueError):
                raise SchemaError("The provider blocked, truncated, or malformed its structured response.") from None
        try:
            parsed = schema.model_validate_json(text)
        except (ValidationError, ValueError, TypeError):
            raise SchemaError("The provider did not return the required structured JSON. No unchecked answer was returned.") from None
        self.cache.put(key, parsed.model_dump())
        return parsed

    def embed(self, text):
        config = self.connection
        if not text or len(text.encode()) > 4096:
            raise RagError("Embedding input is empty or too long.")
        key = fingerprint(["embedding", config.index_config, text])
        cached = self.cache.get(key)
        if cached is not None:
            self.usage["cache_hits"] += 1
            return cached
        if config.provider == "Gemini":
            data = self._request("/models/" + quote(config.embedding_model, safe="") + ":embedContent", {
                "model": "models/" + config.embedding_model, "content": {"parts": [{"text": text}]},
                "outputDimensionality": 768})
            embedding = data.get("embedding")
            values = embedding.get("values") if isinstance(embedding, dict) else None
        else:
            data = self._request("/embeddings", {"model": config.embedding_model, "input": text, "encoding_format": "float"})
            records = data.get("data")
            values = records[0].get("embedding") if isinstance(records, list) and len(records) == 1 and isinstance(records[0], dict) else None
        try:
            vector = np.asarray(values, dtype=np.float32)
            if (vector.ndim != 1 or not 1 <= vector.size <= 8192 or not np.isfinite(vector).all()
                    or not np.linalg.norm(vector) or (config.dimensions and vector.size != config.dimensions)):
                raise ValueError
        except (TypeError, ValueError):
            raise SchemaError("The provider returned invalid or incompatible embeddings.") from None
        value = vector.tolist()
        self.cache.put(key, value)
        return value


def check_connection(connection):
    """No visitor documents or owner credentials are used in capability checks."""
    from .store import Store
    cache = Store()
    client = PersonalClient(connection, cache)
    try:
        try:
            vector = client.embed("Connection test.")
        except RagError as exc:
            raise type(exc)("Embedding check: " + str(exc)) from None
        try:
            client.ask("connection", "Return answer 'Connected' and an empty sources array.", {"test": "connection"}, Draft)
        except RagError as exc:
            raise type(exc)("Structured generation check: " + str(exc)) from None
        return replace(connection, dimensions=len(vector))
    finally:
        client.close()
        cache.close()
