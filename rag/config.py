from dataclasses import dataclass
from collections.abc import Mapping

GEN_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "gemini-embedding-2"
DIMENSIONS = 768
PROMPT_VERSION = "1.1"
EMBED_VERSION = "text-no-task-type-v1"
MAX_INPUT = 4096
MAX_OUTPUT = 1536


@dataclass(frozen=True)
class Limits:
    rpm: int
    tpm: int
    rpd: int


@dataclass(frozen=True)
class Settings:
    api_key: str
    confirmed: bool
    generation: Limits
    embedding: Limits

    @classmethod
    def from_mapping(cls, values: Mapping):
        def limit(prefix):
            return Limits(*(max(0, int(values.get(f"{prefix}_{unit}", 0))) for unit in ("RPM", "TPM", "RPD")))
        return cls(str(values.get("GEMINI_API_KEY", "")),
                   str(values.get("FREE_TIER_CONFIRMED", "false")).lower() == "true",
                   limit("GEN"), limit("EMBED"))

    @property
    def ready(self):
        return bool(self.api_key and self.confirmed and all(
            value > 0 for limits in (self.generation, self.embedding)
            for value in (limits.rpm, limits.tpm, limits.rpd)))
