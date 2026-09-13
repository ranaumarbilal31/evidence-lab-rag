from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Evidence(StrictModel):
    chunk_id: str
    quote: str = Field(min_length=1, max_length=1200)


class SafetyItem(StrictModel):
    chunk_id: str
    decision: Literal["safe", "suspicious", "uncertain"]
    reason: str = Field(max_length=300)


class Safety(StrictModel):
    items: list[SafetyItem]


class Claim(StrictModel):
    text: str = Field(min_length=1, max_length=400)
    source: Evidence


class RelevantItem(StrictModel):
    chunk_id: str
    relevant: bool
    reason: str = Field(max_length=300)
    claims: list[Claim]


class Relevance(StrictModel):
    items: list[RelevantItem]


class Conflict(StrictModel):
    detected: bool
    explanation: str = Field(max_length=500)
    sources: list[Evidence]


class Sufficiency(StrictModel):
    sufficient: bool
    missing: list[str]
    sources: list[Evidence]


class Draft(StrictModel):
    answer: str = Field(min_length=1, max_length=1800)
    sources: list[Evidence]


class Validation(StrictModel):
    supported: bool
    unsupported_claims: list[str]


@dataclass(frozen=True)
class Chunk:
    id: str
    document_hash: str
    filename: str
    page: int | None
    text: str


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass
class Result:
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    timings: dict[str, float] = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    cache_hit: bool = False


class RagError(Exception):
    """Only fixed, safe messages from this exception are shown to visitors."""


class QuotaError(RagError):
    pass


class SchemaError(RagError):
    pass

