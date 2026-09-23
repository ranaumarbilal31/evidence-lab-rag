"""Explicit same-provider pool. Capacity belongs to a project, never to a key."""
from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict
from pathlib import Path

from .api import Gemini, fingerprint, input_bound
from .config import GEN_MODEL, EMBED_MODEL, MAX_OUTPUT, Limits
from .models import AccessError, RagError, QuotaError
from .quota import Governor


class SharedPool:
    def __init__(self, settings, ledger_root=None, clock=time.time, client_factory=Gemini):
        self.keys = settings.credentials
        self.client_factory = client_factory
        self.clock = clock
        self.disabled = set()
        self.lock = threading.RLock()
        self.serial = threading.Lock()
        self.waiting = 0
        self.governors = {}
        for project in dict.fromkeys(k.project_id for k in self.keys):
            members = [k for k in self.keys if k.project_id == project]
            # Conflicting declarations never multiply capacity: use the minimum.
            limits = {model: Limits(*(min(getattr(getattr(k, attr), unit) for k in members)
                                      for unit in ('rpm', 'tpm', 'rpd')))
                      for model, attr in ((GEN_MODEL, 'generation'), (EMBED_MODEL, 'embedding'))}
            ledger = None
            if ledger_root:
                ledger_root = Path(ledger_root)
                ledger_root.mkdir(parents=True, exist_ok=True)
                ledger = ledger_root / ('quota.json' if project == 'legacy' else f'quota-{fingerprint(project)[:20]}.json')
            self.governors[project] = Governor(limits, ledger, clock=clock)

    @contextmanager
    def job(self, expected=None):
        # One operation at a time per host; three waiting visitors. Individual
        # requests are charged atomically to their actual project before sending.
        with self.lock:
            if self.waiting >= 4:
                raise QuotaError('The shared demo is busy. Try again or connect your own key.')
            self.waiting += 1
        acquired = False
        try:
            acquired = self.serial.acquire(timeout=45)
            if not acquired:
                raise QuotaError('The shared demo is busy. Your request was not started.')
            yield {}
        finally:
            if acquired:
                self.serial.release()
            with self.lock:
                self.waiting -= 1

    def eligible(self, key, model, bound):
        gov = self.governors[key.project_id]
        with gov.lock:
            gov._rollover()
            limit = gov.limits[model]
            recent = [(t, n) for t, n in gov.minute[model] if self.clock() - t < 60]
            return (key.api_key not in self.disabled and self.clock() >= gov.paused_until
                    and gov.daily['calls'].get(model, 0) + gov.reserved[model] < limit.rpd
                    and bound <= limit.tpm and len(recent) < limit.rpm
                    and sum(n for _, n in recent) + bound <= limit.tpm)

    def snapshot(self):
        return {'approximate': True, 'projects': len(self.governors),
                'configured_keys': len(self.keys), 'disabled_keys': len(self.disabled),
                'projects_usage': [g.snapshot() for g in self.governors.values()]}

    def minute_wait(self, model, bound):
        """Wait for the shortest local minute window only, never a daily reset."""
        waits = []
        for key in self.keys:
            gov = self.governors[key.project_id]
            with gov.lock:
                gov._rollover()
                now, limit = self.clock(), gov.limits[model]
                if (key.api_key in self.disabled or now < gov.paused_until or bound > limit.tpm
                        or gov.daily['calls'].get(model, 0) >= limit.rpd):
                    continue
                recent = sorted((t, n) for t, n in gov.minute[model] if now - t < 60)
                if recent:
                    waits.append(max(0, recent[0][0] + 60.05 - now))
        return min(waits) if waits else None


class SharedClient:
    def __init__(self, pool, cache):
        self.pool, self.cache = pool, cache
        self.usage = {'requests': 0, 'cache_hits': 0, 'input_tokens': 0, 'output_tokens': 0, 'pool_rotations': 0}
        self.last_project = None
        self.configuration = {'credential_mode': 'shared_gemini_pool', 'quota_estimates': True}

    def close(self):
        pass

    def _run(self, method, model, bound, *args):
        failed_projects = set()
        if not any(self.pool.eligible(k, model, bound) for k in self.pool.keys):
            wait = self.pool.minute_wait(model, bound)
            if wait is not None and wait <= 60.05:
                time.sleep(min(wait, 60))
                if wait > 60:
                    time.sleep(wait - 60)
        # At most one attempt per configured credential here; the Gemini adapter
        # itself has bounded retries. Generic errors never cause provider fallback.
        for key in self.pool.keys:
            if key.project_id in failed_projects or not self.pool.eligible(key, model, bound):
                continue
            client = self.pool.client_factory(key.settings(), self.pool.governors[key.project_id], self.cache)
            try:
                value = getattr(client, method)(*args)
                if self.last_project is not None and self.last_project != key.project_id:
                    self.usage['pool_rotations'] += 1
                self.last_project = key.project_id
                return value
            except QuotaError:
                failed_projects.add(key.project_id)
                # The adapter usually installs RetryInfo's cooldown. Preserve it.
                self.pool.governors[key.project_id].pause(60)
            except AccessError:
                with self.pool.lock:
                    self.pool.disabled.add(key.api_key)
            finally:
                for name, count in client.usage.items():
                    self.usage[name] = self.usage.get(name, 0) + count
                client.close()
        raise QuotaError('Shared Gemini capacity is unavailable or exhausted. Saved examples still work. Connect your own API key or retry after the quota reset.')

    def ask(self, stage, system, payload, schema):
        # Let valid completed work remain available even when every project is paused.
        from .config import PROMPT_VERSION, MAX_INPUT
        bound = input_bound(system, payload, schema)
        if bound > MAX_INPUT:
            raise RagError('The evidence exceeds the input budget. Use shorter evidence.')
        key = fingerprint(['generate', GEN_MODEL, PROMPT_VERSION, stage, system, payload,
                           schema.model_json_schema(), MAX_OUTPUT, 0])
        cached = self.cache.get(key)
        if cached is not None:
            self.usage['cache_hits'] += 1
            return schema.model_validate(cached)
        return self._run('ask', GEN_MODEL, bound + MAX_OUTPUT, stage, system, payload, schema)

    def embed(self, text):
        from .config import DIMENSIONS, EMBED_VERSION
        if not text or len(text.encode()) > 4096:
            raise RagError('Embedding input is empty or too long.')
        key = fingerprint(['embedding', EMBED_MODEL, DIMENSIONS, EMBED_VERSION, text])
        cached = self.cache.get(key)
        if cached is not None:
            self.usage['cache_hits'] += 1
            return cached
        return self._run('embed', EMBED_MODEL, len(text.encode()) + 128, text)
