from dataclasses import replace

import pytest

from rag.config import Settings, SharedKey, Limits, GEN_MODEL, EMBED_MODEL
from rag.models import AccessError, QuotaError, RagError, Draft
from rag.shared import SharedPool, SharedClient
from rag.store import Store


def settings(*projects):
    return Settings('', False, Limits(0, 0, 0), Limits(0, 0, 0), tuple(
        SharedKey(f'private-{i}', project, True, Limits(100, 100000, 10), Limits(100, 100000, 10))
        for i, project in enumerate(projects)))


def factory(calls, failures=None):
    class Client:
        def __init__(self, settings, governor, cache):
            self.key, self.governor = settings.api_key, governor
            self.usage = {'requests': 0, 'cache_hits': 0}
        def embed(self, text):
            self.governor.take(EMBED_MODEL, 10, {})
            self.usage['requests'] += 1
            calls.append(self.key)
            if failures and self.key in failures:
                raise failures[self.key]('Redacted error')
            return [1., 0.]
        def close(self):
            pass
    return Client


def test_same_project_shares_quota_and_never_adds_capacity():
    pool = SharedPool(settings('one', 'one'), client_factory=factory([]))
    assert len(pool.governors) == 1
    pool.governors['one'].daily['calls'][EMBED_MODEL] = 10
    with pytest.raises(QuotaError):
        SharedClient(pool, Store()).embed('Policy')


def test_daily_exhaustion_rotates_to_independent_project():
    calls = []
    pool = SharedPool(settings('one', 'two'), client_factory=factory(calls))
    pool.governors['one'].daily['calls'][EMBED_MODEL] = 10
    with pool.job():
        client = SharedClient(pool, Store())
        assert client.embed('Policy') == [1., 0.]
    assert calls == ['private-1']
    assert pool.governors['two'].daily['calls'][EMBED_MODEL] == 1


def test_quota_failure_skips_all_keys_of_project_and_cooldown_persists():
    calls = []
    pool = SharedPool(settings('one', 'one', 'two'), clock=lambda: 100,
                      client_factory=factory(calls, {'private-0': QuotaError}))
    client = SharedClient(pool, Store())
    assert client.embed('Policy') == [1., 0.]
    assert calls == ['private-0', 'private-2']
    assert pool.governors['one'].paused_until == 160
    client.embed('Next')
    assert calls[-1] == 'private-2'


def test_invalid_key_disabled_but_other_key_can_work():
    calls = []
    pool = SharedPool(settings('one', 'one'), client_factory=factory(calls, {'private-0': AccessError}))
    SharedClient(pool, Store()).embed('Policy')
    assert calls == ['private-0', 'private-1']
    assert len(pool.disabled) == 1
    assert 'private' not in str(pool.snapshot())
    assert 'private' not in repr(settings('one'))


def test_nonquota_errors_do_not_rotate():
    calls = []
    pool = SharedPool(settings('one', 'two'), client_factory=factory(calls, {'private-0': RagError}))
    with pytest.raises(RagError):
        SharedClient(pool, Store()).embed('Policy')
    assert calls == ['private-0']


def test_minute_tokens_and_cooldown_are_project_scoped():
    now = [100.]
    pool = SharedPool(settings('one', 'two'), clock=lambda: now[0])
    a, b = pool.keys
    pool.governors['one'].minute[GEN_MODEL] = [(100, 100000)]
    assert not pool.eligible(a, GEN_MODEL, 10)
    assert pool.eligible(b, GEN_MODEL, 10)
    now[0] = 161
    assert pool.eligible(a, GEN_MODEL, 10)


def test_cached_result_works_with_exhausted_pool():
    from rag.api import fingerprint
    from rag.config import PROMPT_VERSION, MAX_OUTPUT
    cache = Store()
    key = fingerprint(['generate', GEN_MODEL, PROMPT_VERSION, 'answer', 'system', {}, Draft.model_json_schema(), MAX_OUTPUT, 0])
    cache.put(key, {'answer': 'Previously completed', 'sources': []})
    pool = SharedPool(settings())
    assert SharedClient(pool, cache).ask('answer', 'system', {}, Draft).answer == 'Previously completed'


def test_pool_config_legacy_and_ten_key_limit():
    legacy = dict(GEMINI_API_KEY='legacy-key', FREE_TIER_CONFIRMED=True, GEN_RPM=5, GEN_TPM=60000,
                  GEN_RPD=20, EMBED_RPM=10, EMBED_TPM=60000, EMBED_RPD=100)
    assert Settings.from_mapping(legacy).credentials[0].project_id == 'legacy'
    rows = [dict(api_key=f'key-{i}', project_id=f'p-{i}', enabled=True, free_tier_confirmed=True,
                 gen_rpm=1, gen_tpm=10000, gen_rpd=20, embed_rpm=1, embed_tpm=10000, embed_rpd=20) for i in range(10)]
    parsed = Settings.from_mapping({**legacy, 'shared_keys': rows})
    assert len(parsed.credentials) == 10
    assert all(k.api_key != 'legacy-key' for k in parsed.credentials)
    with pytest.raises(ValueError):
        Settings.from_mapping({'shared_keys': rows + [rows[0]]})
    with pytest.raises(ValueError, match='Duplicate'):
        Settings.from_mapping({'shared_keys': [rows[0], rows[0]]})


def test_persistent_ledger_does_not_reset_when_key_changes(tmp_path):
    pool = SharedPool(settings('one'), tmp_path, client_factory=factory([]))
    SharedClient(pool, Store()).embed('Policy')
    original = settings('one')
    changed = replace(original, shared_keys=(replace(original.shared_keys[0], api_key='rotated'),))
    restored = SharedPool(changed, tmp_path)
    assert restored.governors['one'].daily['calls'][EMBED_MODEL] == 1
