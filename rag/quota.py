"""Process-wide throttling; local CLI ledgers can persist across restarts."""
from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from zoneinfo import ZoneInfo

from .models import QuotaError


class Governor:
    def __init__(self, limits, ledger=None, clock=time.time, day=None):
        self.limits = limits
        self.ledger = ledger
        self.clock = clock
        # Gemini daily quota resets at midnight Pacific time (including DST).
        self.day = day or (lambda: datetime.now(ZoneInfo("America/Los_Angeles")).date().isoformat())
        self.lock = threading.RLock()
        self.serial = threading.Lock()
        self.waiting = 0
        self.reserved = {k: 0 for k in limits}
        self.minute = {k: [] for k in limits}
        self.daily = {"day": self.day(), "calls": {k: 0 for k in limits}}
        if ledger and ledger.exists():
            self.daily = json.loads(ledger.read_text())
        self.paused_until = 0.0

    def _save(self):
        if self.ledger:
            self.ledger.parent.mkdir(parents=True, exist_ok=True)
            temp = self.ledger.with_suffix(".tmp")
            temp.write_text(json.dumps(self.daily))
            temp.replace(self.ledger)

    def _rollover(self):
        if self.daily["day"] != self.day():
            self.daily = {"day": self.day(), "calls": {k: 0 for k in self.limits}}

    @contextmanager
    def job(self, expected):
        """Reserve expected daily calls; retries require additional capacity."""
        with self.lock:
            self._rollover()
            if self.clock() < self.paused_until:
                raise QuotaError("Live requests are temporarily paused by the API. Please try later.")
            if self.waiting >= 4:
                raise QuotaError("The demo is busy (one active job and three waiting). Please try again later.")
            for model, count in expected.items():
                if self.daily["calls"].get(model, 0) + self.reserved[model] + count > self.limits[model].rpd:
                    raise QuotaError("The remaining daily allowance cannot cover this job. Try after the quota reset.")
            for model, count in expected.items():
                self.reserved[model] += count
            self.waiting += 1
        ticket = dict(expected)
        try:
            if not self.serial.acquire(timeout=45):
                raise QuotaError("The demo is busy. Your request was not started; please try again later.")
            try:
                yield ticket
            finally:
                self.serial.release()
        finally:
            with self.lock:
                for model, remaining in ticket.items():
                    self.reserved[model] -= remaining
                self.waiting -= 1

    def take(self, model, token_bound, ticket):
        """Record BEFORE sending. Provider failures still use an attempt."""
        while True:
            with self.lock:
                self._rollover()
                now = self.clock()
                if now < self.paused_until:
                    raise QuotaError("Live requests are paused by the API. Please try later.")
                limit = self.limits[model]
                if token_bound > limit.tpm:
                    raise QuotaError("This request exceeds the configured token allowance. Use shorter documents or a shorter question.")
                recent = [(t, n) for t, n in self.minute[model] if now - t < 60]
                self.minute[model] = recent
                if len(recent) >= limit.rpm or sum(n for _, n in recent) + token_bound > limit.tpm:
                    wait = 60 - (now - recent[0][0]) + 0.05
                else:
                    wait = 0
                if not wait:
                    reserved_here = ticket.get(model, 0)
                    if not reserved_here and self.daily["calls"].get(model, 0) + self.reserved[model] >= limit.rpd:
                        raise QuotaError("The configured daily API allowance is exhausted. Progress has been kept.")
                    if reserved_here:
                        ticket[model] -= 1
                        self.reserved[model] -= 1
                    self.daily["calls"][model] = self.daily["calls"].get(model, 0) + 1
                    self.minute[model].append((now, token_bound))
                    self._save()
                    return
            # One minute is an expected rate-limit window, not a failed answer.
            # Keep each sleep bounded while the Streamlit status remains visible.
            if wait > 61:
                raise QuotaError("The per-minute API allowance is used. Please retry in about a minute; cached progress is retained.")
            time.sleep(min(wait, 60))

    def pause(self, seconds=60):
        with self.lock:
            self.paused_until = max(self.paused_until, self.clock() + seconds)

    def snapshot(self):
        with self.lock:
            self._rollover()
            return {"day_pacific": self.daily["day"], "attempts": dict(self.daily["calls"]),
                    "jobs_active_or_waiting": self.waiting,
                    "limits": {k: vars(v) for k, v in self.limits.items()}}
