"""Safety Wall -- the decision/control layer.

Sits between evidence gathering (retrieval -> detection -> quarantine -> recovery ->
relevance/conflict/sufficiency, all unchanged by this module) and generation. Turns
the already-computed Sufficiency signal into exactly one of three decision states
using plain, deterministic Python -- this is a code-level control, not a fresh LLM
judgment call. The underlying evidence checks (relevance, sufficiency) remain the
existing semantic LLM stages, reused as-is; this module only decides what to do with
their already-validated output.

By the time `decide()` is called, `chunks` already reflects every upstream trust
decision:
  - quarantined chunks were deleted (rag.detection) and can never come back
  - only recovery-verified, deduplicated replacements were merged back in (rag.recovery)
  - only chunks judged relevant, with an exact-quote-grounded claim, remain (the
    existing relevance stage in rag.pipeline)
So `chunks` here is already "verified evidence" in the sense this layer requires --
`decide()` never re-derives that, it only decides what to do with what is left, and
exactly how much of it the generator is allowed to see.
"""
from __future__ import annotations

from typing import Mapping

from .models import Chunk, Claim, Decision, Sufficiency

ANSWER = "answer"
PARTIAL_ANSWER = "partial_answer"
ABSTAIN = "abstain"


def decide(chunks: Mapping[str, Chunk], claims: list[Claim], sufficiency: Sufficiency | None) -> Decision:
    """Deterministic decision over VERIFIED evidence only. No model call here.

    `sufficiency` is the existing 'sufficiency' LLM stage's already mechanically-
    validated output (its `sources` were already run through `check_sources` by the
    caller before this is invoked), or None when that stage was skipped by the
    `disable="sufficiency"` ablation -- preserved exactly as before: proceed on
    whatever verified evidence survived relevance filtering.
    """
    if not chunks or not claims:
        return Decision(ABSTAIN, [], [],
                         "No verified evidence remains after quarantine, recovery, and relevance filtering.")

    if sufficiency is None:
        return Decision(ANSWER, sorted(chunks), [],
                         "Sufficiency check disabled for this run; proceeding on verified relevant evidence.")

    if sufficiency.sufficient:
        return Decision(ANSWER, sorted(chunks), [],
                         "All facts required to answer the question are supported by verified evidence.")

    supported_ids = sorted({source.chunk_id for source in sufficiency.sources})
    if supported_ids:
        return Decision(PARTIAL_ANSWER, supported_ids, list(sufficiency.missing),
                         "Some requested facts are supported by verified evidence; the rest could not be "
                         "recovered and must not be guessed.")

    return Decision(ABSTAIN, [], list(sufficiency.missing),
                     "No part of the question is supported by trustworthy, verified evidence.")


def context_for(decision: Decision, chunks: Mapping[str, Chunk]) -> dict[str, Chunk]:
    """The exact evidence the generator is permitted to see for this decision.

    For `answer`, every currently-trusted chunk already passed relevance filtering,
    so the full set is used -- unchanged from this project's existing generation
    behavior. For `partial_answer`, generation is restricted to only the chunks the
    sufficiency check actually cited as supporting: a chunk that survived relevance
    filtering but was never confirmed as answering anything is withheld from the
    generator, so it structurally cannot be guessed from or cited.
    """
    if decision.decision == PARTIAL_ANSWER:
        return {chunk_id: chunks[chunk_id] for chunk_id in decision.verified_chunks if chunk_id in chunks}
    return dict(chunks)
