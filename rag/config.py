from dataclasses import dataclass, field
from collections.abc import Mapping

GEN_MODEL = "gemini-3.6-flash"
EMBED_MODEL = "gemini-embedding-2"
DIMENSIONS = 768
PROMPT_VERSION = "1.1"
EMBED_VERSION = "text-no-task-type-v1"
MAX_INPUT = 4096
MAX_OUTPUT = 1536
# Experimental detection settings. Keep unchanged until separate development
# probes show improvement; never tune these against held-out outcomes.
DETECTION_POLICY = 'heuristic_first'
VECTOR_CHECK_ENABLED = False
VECTOR_THRESHOLD = 0.9  # Uncalibrated candidate; not a probability.


@dataclass(frozen=True)
class Limits:
    rpm: int
    tpm: int
    rpd: int


@dataclass(frozen=True)
class SharedKey:
    api_key: str = field(repr=False)
    project_id: str
    confirmed: bool
    generation: Limits
    embedding: Limits

    def settings(self):
        return Settings(self.api_key, self.confirmed, self.generation, self.embedding)


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    confirmed: bool
    generation: Limits
    embedding: Limits
    shared_keys: tuple[SharedKey, ...] = ()
    research_pause_shared: bool = False

    @classmethod
    def from_mapping(cls, values: Mapping):
        def limit(prefix):
            return Limits(*(max(0, int(values.get(f"{prefix}_{unit}", 0))) for unit in ("RPM", "TPM", "RPD")))
        entries = values.get('shared_keys', [])
        if not isinstance(entries, (list, tuple)) or len(entries) > 10:
            raise ValueError('shared_keys must contain at most 10 entries.')
        keys = []
        seen = set()
        for entry in entries:
            if str(entry.get('enabled', False)).lower() != 'true':
                continue
            project = str(entry.get('project_id', '')).strip()
            key = str(entry.get('api_key', '')).strip()
            if not project or not key:
                raise ValueError('Enabled shared keys require a project_id and api_key.')
            if key in seen:
                raise ValueError('Duplicate shared API key.')
            seen.add(key)
            limits = lambda prefix: Limits(*(max(0, int(entry.get(f'{prefix}_{unit}', 0))) for unit in ('rpm', 'tpm', 'rpd')))
            keys.append(SharedKey(key, project, str(entry.get('free_tier_confirmed', False)).lower() == 'true', limits('gen'), limits('embed')))
        return cls(str(values.get("GEMINI_API_KEY", "")),
                   str(values.get("FREE_TIER_CONFIRMED", "false")).lower() == "true",
                   limit("GEN"), limit("EMBED"), tuple(keys),
                   str(values.get('RESEARCH_PAUSE_SHARED', False)).lower() == 'true')

    @property
    def ready(self):
        if self.shared_keys:
            return any(key.settings().ready for key in self.shared_keys)
        return bool(self.api_key and self.confirmed and all(
            value > 0 for limits in (self.generation, self.embedding)
            for value in (limits.rpm, limits.tpm, limits.rpd)))

    @property
    def credentials(self):
        if self.shared_keys:
            return tuple(k for k in self.shared_keys if k.settings().ready)
        return (SharedKey(self.api_key, 'legacy', self.confirmed, self.generation, self.embedding),) if self.ready else ()
