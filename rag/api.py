from __future__ import annotations

import hashlib
import json
import time

from pydantic import ValidationError

from .config import (GEN_MODEL, EMBED_MODEL, DIMENSIONS, EMBED_VERSION,
                     MAX_INPUT, MAX_OUTPUT, PROMPT_VERSION)
from .models import RagError, QuotaError, SchemaError, AccessError


def serialized(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def fingerprint(value):
    return hashlib.sha256(serialized(value).encode()).hexdigest()


def input_bound(system, payload, schema):
    # Conservative UTF-8 byte bound, including schema and framing allowance.
    # This deliberately admits less text than a typical 4096-token context.
    return len((system + serialized(payload) + serialized(schema.model_json_schema())).encode()) + 128


class Gemini:
    def __init__(self, settings, governor, cache, ticket=None):
        if not settings.ready:
            raise RagError("Live AI is not configured. The owner must set a Free-tier key and active quota limits.")
        from google import genai
        from google.genai import types
        # Force Developer API with fixed origin; no Vertex, proxy endpoint, or tool fallback.
        self.sdk = genai.Client(api_key=settings.api_key, vertexai=False,
            http_options=types.HttpOptions(base_url="https://generativelanguage.googleapis.com",
                timeout=45000, retry_options=types.HttpRetryOptions(attempts=1)))
        self.governor, self.cache = governor, cache
        self.ticket = ticket if ticket is not None else {}
        self.usage = {"requests": 0, "cache_hits": 0, "input_tokens": 0, "output_tokens": 0}

    def close(self):
        self.sdk.close()

    def _call(self, model, bound, operation):
        from google.genai import errors
        for attempt in range(3):
            self.governor.take(model, bound, self.ticket)
            self.usage["requests"] += 1
            try:
                return operation()
            except errors.APIError as exc:
                code = exc.code
                if code == 429:
                    retry = None
                    body = getattr(exc, "response_json", None) or {}
                    if isinstance(body, dict):
                        for detail in body.get("error", body).get("details", []):
                            if isinstance(detail, dict) and "retryDelay" in detail:
                                try:
                                    retry = float(str(detail["retryDelay"]).rstrip("s"))
                                except (TypeError, ValueError):
                                    pass
                    if retry is not None and 0 < retry <= 10 and attempt < 2:
                        time.sleep(retry)
                        continue
                    delay = max(60, retry or 60)
                    # Provider daily quota violations must not be retried every minute.
                    if 'perday' in json.dumps(body).lower().replace('_', ''):
                        from datetime import datetime, timedelta
                        from zoneinfo import ZoneInfo
                        now = datetime.now(ZoneInfo('America/Los_Angeles'))
                        reset = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
                        delay = max(delay, reset.timestamp() - now.timestamp())
                    self.governor.pause(delay)
                    raise QuotaError("The API quota is unavailable or exhausted. Progress is cached; retry after the provider reset.") from None
                if code in {500, 502, 503, 504} and attempt < 2:
                    time.sleep(attempt + 1)
                    continue
                if code in {401, 403}:
                    raise AccessError("The API rejected access. The owner must check key permissions, account eligibility, and model access.") from None
                if code == 404:
                    raise RagError("The configured API model is unavailable. No substitute or paid fallback was used.") from None
                raise RagError("The API request failed. No unchecked answer was returned.") from None
            except (QuotaError, RagError):
                raise
            except Exception:
                # Never echo SDK exceptions: they can contain request text or credentials.
                raise RagError("The API could not be reached or timed out. Please retry later.") from None

    def ask(self, stage, system, payload, schema):
        from google.genai import types
        bound = input_bound(system, payload, schema)
        if bound > MAX_INPUT:
            raise RagError("The evidence exceeds this demo's input budget. Please use a shorter question or smaller documents.")
        key = fingerprint(["generate", GEN_MODEL, PROMPT_VERSION, stage, system, payload,
                           schema.model_json_schema(), MAX_OUTPUT, 0])
        cached = self.cache.get(key)
        if cached is not None:
            self.usage["cache_hits"] += 1
            return schema.model_validate(cached)
        response = self._call(GEN_MODEL, bound + MAX_OUTPUT,
            lambda: self.sdk.models.generate_content(model=GEN_MODEL,
                contents=serialized(payload), config=types.GenerateContentConfig(
                    system_instruction=system, temperature=0,
                    max_output_tokens=MAX_OUTPUT,
                    thinking_config=types.ThinkingConfig(thinking_budget=0),
                    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
                    response_mime_type="application/json",
                    response_json_schema=schema.model_json_schema())))
        candidates = response.candidates or []
        if not candidates or str(candidates[0].finish_reason).split(".")[-1] != "STOP":
            raise SchemaError("The API blocked or truncated its response. No unchecked answer was returned.")
        try:
            parsed = schema.model_validate_json(response.text or "")
        except (ValidationError, ValueError):
            raise SchemaError("A verification response was malformed. No unchecked answer was returned.") from None
        usage = response.usage_metadata
        if usage:
            self.usage["input_tokens"] += usage.prompt_token_count or 0
            self.usage["output_tokens"] += (usage.candidates_token_count or 0) + (usage.thoughts_token_count or 0)
        self.cache.put(key, parsed.model_dump())
        return parsed

    def embed(self, text):
        from google.genai import types
        if not text or len(text.encode()) > 4096:
            raise RagError("Embedding input is empty or too long.")
        key = fingerprint(["embedding", EMBED_MODEL, DIMENSIONS, EMBED_VERSION, text])
        cached = self.cache.get(key)
        if cached is not None:
            self.usage["cache_hits"] += 1
            return cached
        response = self._call(EMBED_MODEL, len(text.encode()) + 128,
            lambda: self.sdk.models.embed_content(model=EMBED_MODEL, contents=text,
                config=types.EmbedContentConfig(output_dimensionality=DIMENSIONS)))
        # Embedding 2 does not support task_type. Each call contains one independent text.
        import numpy as np
        vectors = response.embeddings or []
        values = vectors[0].values if len(vectors) == 1 else None
        vector = np.asarray(values, dtype=np.float32)
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all() or not np.linalg.norm(vector):
            raise RagError("The API returned an invalid embedding.")
        value = vector.tolist()
        self.cache.put(key, value)
        return value
