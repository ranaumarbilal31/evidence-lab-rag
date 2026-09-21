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


class MissingFacts(StrictModel):
    """Output of the recovery module's missing-fact analysis: short, search-friendly
    descriptions of what a question needs that a just-quarantined chunk would have
    addressed -- never the quarantined chunk's own (untrusted) wording repeated back."""
    facts: list[str]


@dataclass(frozen=True)
class Chunk:
    id: str
    document_hash: str
    filename: str
    page: int | None
    text: str
    location: str | None = None


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


@dataclass(frozen=True)
class DetectionResult:
    """One chunk's Safety Wall verdict.

    `confidence` is an ordinal signal from whichever layer produced the flag
    (heuristic pattern match or semantic classifier judgment); it is NOT a
    calibrated probability and must never be interpreted as one.
    """
    chunk_id: str
    flagged: bool
    flag_reason: str
    confidence: float
    source: str  # "heuristic" | "classifier"


@dataclass(frozen=True)
class RecoveredChunk:
    """A single piece of replacement evidence that survived exclusion, deduplication,
    and support verification. `document_hash`/`page` are carried along (beyond the
    minimal spec shape) purely so the pipeline can reconstruct a faithful `Chunk` and
    keep citation provenance correct -- they are not part of any trust decision."""
    chunk_id: str
    text: str
    source_doc: str
    score: float
    document_hash: str = ""
    page: int | None = None
    location: str | None = None


@dataclass
class RecoveryResult:
    excluded_chunks: list[str]
    missing_facts: list[str]
    recovered_chunks: list[RecoveredChunk] = field(default_factory=list)
    recovery_status: str = "failed"  # "full" | "partial" | "failed"
    rejected: list[dict] = field(default_factory=list)  # audit trail: candidates considered and turned away


@dataclass(frozen=True)
class Decision:
    """The Safety Wall's final verdict, computed deterministically in code (see
    rag.decision.decide) from VERIFIED evidence only -- never from raw retrieval or
    from a quarantined/unverified chunk."""
    decision: str  # "answer" | "partial_answer" | "abstain"
    verified_chunks: list[str]
    unsupported_facts: list[str]
    reason: str


@dataclass
class Result:
    status: str
    answer: str
    citations: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    excluded: list[dict] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)
    recovery: dict | None = None
    decision: dict | None = None
    # The raw 'validate' stage verdict (supported / unsupported_claims), recorded for
    # audit purposes whether or not it passed -- None whenever that stage did not run
    # (baseline, the conflict/abstain-before-generation paths, or disable="validation").
    validation: dict | None = None
    timings: dict[str, float] = field(default_factory=dict)
    configuration: dict = field(default_factory=dict)
    usage: dict = field(default_factory=dict)
    cache_hit: bool = False


@dataclass
class SafetyWallReport:
    """Single structured record of one full Safety Wall run (Pipeline.run_safety_wall):
    query -> existing retrieval -> detection/quarantine -> missing-fact analysis ->
    bounded recovery -> existing verification -> decision -> existing generation ->
    existing citation validation. Assembled entirely from an existing Result (plus the
    initial retrieval hits); it adds no retrieval, generation, or citation logic of its
    own -- see Pipeline.run_safety_wall in rag/pipeline.py."""
    query: str
    status: str
    initial_evidence: list[dict]
    flagged_chunks: list[dict]
    quarantined_chunks: list[dict]
    missing_facts: list[str]
    recovery_attempts: list[dict]
    recovered_chunks: list[dict]
    recovery_status: str | None
    final_verified_evidence: list[dict]
    decision: dict | None
    answer: str
    citations: list[dict]
    validation: dict | None


class RagError(Exception):
    """Only fixed, safe messages from this exception are shown to visitors."""


class QuotaError(RagError):
    pass


class SchemaError(RagError):
    pass

