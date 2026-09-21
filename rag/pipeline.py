from __future__ import annotations

import time
from dataclasses import asdict

from .api import input_bound
from .config import GEN_MODEL, EMBED_MODEL, PROMPT_VERSION, MAX_INPUT
from .decision import ABSTAIN, PARTIAL_ANSWER, decide, context_for
from .detection import BASE, SAFETY_PROMPT, PATTERNS, suspicious, detect
from .models import (Chunk, Decision, Safety, Relevance, Conflict, Sufficiency, Draft, Validation,
                     MissingFacts, Evidence, Result, SafetyWallReport, RagError, SchemaError, QuotaError)
from .recovery import MISSING_FACT_PROMPT, RECOVERY_VERIFY_PROMPT, recover as recover_evidence

# `suspicious` and `PATTERNS` now live in rag.detection (the Safety Wall module) and are
# re-exported here unchanged so existing imports of `rag.pipeline.suspicious` still work.
PROMPTS = {
    "safety": SAFETY_PROMPT,
    "relevance": BASE + "For EVERY chunk judge relevance to the question and extract at most 2 answer-bearing claims. Use exact, contiguous quotes and the original chunk ID. Relevant context without an answer-bearing claim has no claims. Give each chunk ID once.",
    "conflict": BASE + "Do these claims disagree about the SAME subject, scope and conditions? Different groups or circumstances are not conflicts. Never choose a newer source automatically. If conflicting, cite at least two incompatible exact source quotations. Otherwise sources must be empty.",
    "sufficiency": BASE + "Does the evidence explicitly answer EVERY part of the question? Similar subject matter alone is insufficient. List missing parts, and exact source quotes for supported parts. sufficient=true requires sources and no missing parts.",
    "answer": BASE + "Answer the question concisely using only evidence. Cite exact supporting quotations with original chunk IDs. If evidence is missing, say so; sources may be empty only for an abstention. Do not invent citations.",
    "validate": BASE + "Check EVERY factual assertion in the answer against the evidence, including quantities and scope. Also check that it answers the question. supported=true only if every assertion is supported, all requested parts are answered, and unsupported_claims is empty.",
    "missing_fact": MISSING_FACT_PROMPT,
    "recovery_verify": RECOVERY_VERIFY_PROMPT,
}
SCHEMAS = {"safety": Safety, "relevance": Relevance, "conflict": Conflict,
           "sufficiency": Sufficiency, "answer": Draft, "validate": Validation,
           "missing_fact": MissingFacts, "recovery_verify": Relevance}


def check_sources(sources, chunks, required=False):
    if required and not sources:
        raise SchemaError("The response did not provide supporting evidence.")
    result, seen = [], set()
    for source in sources:
        chunk = chunks.get(source.chunk_id)
        if chunk is None or not source.quote.strip() or source.quote not in chunk.text:
            raise SchemaError("The response cited an unknown source or invented quotation.")
        key = (source.chunk_id, source.quote)
        if key not in seen:
            seen.add(key)
            result.append({"chunk_id": chunk.id, "filename": chunk.filename,
                           "page": chunk.page, "quote": source.quote, **({"location": chunk.location} if chunk.location else {})})
    return result


def exact_coverage(items, chunks):
    ids = [i.chunk_id for i in items]
    if len(ids) != len(set(ids)) or set(ids) != set(chunks):
        raise SchemaError("The checker omitted, duplicated, or invented a chunk ID.")


def evidence_payload(question, chunks):
    return {"question": question, "documents": [{"chunk_id": c.id, "text": c.text} for c in chunks]}


def common_context(question, hits):
    if not question.strip() or len(question) > 1000:
        raise RagError("Enter a question between 1 and 1,000 characters.")
    selected = []
    for hit in hits:
        candidate = selected + [hit]
        payload = evidence_payload(question, [h.chunk for h in candidate])
        if all(input_bound(PROMPTS[stage], payload, SCHEMAS[stage]) <= MAX_INPUT
               for stage in ("safety", "relevance", "answer")):
            selected = candidate
    return selected


class Pipeline:
    def __init__(self, client):
        self.client = client

    def run(self, question, hits, protected=True, disable=None, recover=False, retrieve_fn=None):
        if disable not in {None, "injection", "relevance", "conflict", "sufficiency", "validation"}:
            raise ValueError("Unknown ablation")
        start = time.perf_counter()
        before = dict(self.client.usage)
        result = Result("error", "No answer was produced.", configuration={
            "generation_model": GEN_MODEL, "embedding_model": EMBED_MODEL,
            "prompt_version": PROMPT_VERSION, "protected": protected, "disabled": disable,
            **getattr(self.client, "configuration", {})})

        def ask(stage, payload):
            tick = time.perf_counter()
            try:
                return self.client.ask(stage, PROMPTS[stage], payload, SCHEMAS[stage])
            finally:
                result.timings[stage] = time.perf_counter() - tick

        def abstain(message, unsupported_facts=()):
            # Always (re-)sets result.decision to the true final verdict -- this can
            # override an earlier tentative "answer"/"partial_answer" recorded before
            # citation/validation checks ran, since those checks are what actually
            # decide whether generated text is trustworthy enough to release.
            result.status, result.answer = "insufficient_evidence", message
            result.decision = asdict(Decision(ABSTAIN, [], list(unsupported_facts), message))
            return result

        try:
            chosen = common_context(question, hits)
            result.configuration["retrieved_ids"] = [h.chunk.id for h in hits]
            result.configuration["context_ids"] = [h.chunk.id for h in chosen]
            for hit in hits:
                if hit not in chosen:
                    result.excluded.append({"chunk_id": hit.chunk.id, "reason": "Application input budget", "stage": "context"})
            if len(chosen) < len(hits):
                result.warnings.append("Some retrieved chunks did not fit the input budget; this can hide relevant or conflicting evidence.")
            chunks = {h.chunk.id: h.chunk for h in chosen}
            if not chunks:
                return abstain("No retrieved evidence fits this question and the input budget.")
            if protected and disable != "injection":
                # Safety Wall: every chunk gets a detection verdict (audit trail, kept on
                # the result regardless of outcome); flagged chunks are quarantined here
                # -- deleted from `chunks` before anything downstream, including
                # generation, can see their text.
                detections = detect(chunks, question, ask)
                result.detections = [asdict(d) for d in detections]
                quarantined = {}
                for d in detections:
                    if d.flagged:
                        result.excluded.append({"chunk_id": d.chunk_id, "reason": d.flag_reason, "stage": "injection"})
                        quarantined[d.chunk_id] = chunks.pop(d.chunk_id)
                if any(e["stage"] == "injection" for e in result.excluded):
                    result.warnings.append("Potential prompt injection detected; suspicious or uncertain chunks were excluded.")
                # Evidence Recovery: a separate, opt-in capability (recover=False by default,
                # so Standard RAG and the existing detect-and-block protected pipeline are
                # byte-for-byte unchanged). Only ever fills in the same `chunks` dict that the
                # rest of this method already consumes -- nothing downstream needs to know a
                # chunk came from recovery instead of the original retrieval. A quarantined
                # chunk id is excluded from the recovery search and can never re-enter `chunks`.
                if quarantined and recover and retrieve_fn is not None:
                    recovery = recover_evidence(question, quarantined, set(chunks), ask, self.client.embed, retrieve_fn)
                    result.recovery = asdict(recovery)
                    for rc in recovery.recovered_chunks:
                        chunks[rc.chunk_id] = Chunk(rc.chunk_id, rc.document_hash, rc.source_doc, rc.page, rc.text, rc.location)
            if not chunks:
                return abstain("All retrieved evidence was excluded. There is not enough accepted evidence to answer.")
            claims = []
            decision = None
            generation_context = chunks
            if protected:
                relevance = ask("relevance", evidence_payload(question, chunks.values()))
                exact_coverage(relevance.items, chunks)
                for item in relevance.items:
                    for claim in item.claims:
                        if claim.source.chunk_id != item.chunk_id:
                            raise SchemaError("A claim was attached to a different source.")
                        check_sources([claim.source], chunks, required=True)
                    if not item.relevant and disable != "relevance":
                        result.excluded.append({"chunk_id": item.chunk_id, "reason": item.reason, "stage": "relevance"})
                        del chunks[item.chunk_id]
                    else:
                        claims.extend(item.claims)
                if not chunks:
                    return abstain("The retrieved information is not relevant enough to answer this question.")
                # Use quotes, not unchecked paraphrases, as the evidence for subsequent stages.
                accepted_sources = [c.source for c in claims]
                check_sources(accepted_sources, chunks)
                claim_payload = {"question": question, "claims": [c.model_dump() for c in claims]}
                if disable != "conflict" and len(claims) >= 2:
                    conflict = ask("conflict", claim_payload)
                    if conflict.detected:
                        citations = check_sources(conflict.sources, chunks, required=True)
                        if len({(c["chunk_id"], c["quote"]) for c in citations}) < 2:
                            raise SchemaError("A conflict was reported without two distinct supporting passages.")
                        result.status = "conflict"
                        result.answer = "The accepted evidence contains conflicting information. Compare the cited passages before deciding which policy applies."
                        result.citations = citations
                        return result
                    if conflict.sources:
                        raise SchemaError("The conflict checker returned inconsistent fields.")
                enough = None
                if disable != "sufficiency":
                    enough = ask("sufficiency", claim_payload)
                    check_sources(enough.sources, chunks)
                    if enough.sufficient and (enough.missing or not enough.sources):
                        raise SchemaError("The evidence checker returned inconsistent fields.")
                # Decision layer: deterministic, code-level -- turns the (already
                # mechanically-validated) sufficiency signal into exactly one of
                # answer/partial_answer/abstain. Quarantined chunks are already gone
                # from `chunks`; recovered chunks only ever arrived here after passing
                # recovery's own verification and deduplication checks.
                decision = decide(chunks, claims, enough)
                if decision.decision == ABSTAIN:
                    return abstain(decision.reason, unsupported_facts=decision.unsupported_facts)
                result.decision = asdict(decision)
                # The generator is handed ONLY the evidence this decision permits: the
                # full trusted set for `answer`, but just the sufficiency-cited chunks
                # for `partial_answer` -- a relevant-but-unconfirmed chunk is withheld
                # entirely, so its content cannot be guessed from or cited.
                generation_context = context_for(decision, chunks)
            payload = evidence_payload(question, generation_context.values())
            if decision is not None and decision.decision == PARTIAL_ANSWER:
                payload["unsupported_facts"] = decision.unsupported_facts
            draft = ask("answer", payload)
            if not protected:
                # Preserve model behavior for research, but never render fabricated source links.
                result.status, result.answer = "answered", draft.answer
                try:
                    result.citations = check_sources(draft.sources, chunks)
                except SchemaError:
                    result.warnings.append("The baseline produced invalid citations; they are not shown as valid sources.")
                if not draft.sources:
                    result.warnings.append("The baseline provided no supporting citations.")
                return result
            try:
                result.citations = check_sources(draft.sources, generation_context, required=True)
            except SchemaError:
                result.citations = []
                return abstain("The generated answer did not provide valid supporting citations.")
            if disable != "validation":
                validation = ask("validate", {"question": question, "answer": draft.answer,
                    "evidence": [s.model_dump() for s in draft.sources]})
                result.validation = validation.model_dump()
                if not validation.supported or validation.unsupported_claims:
                    result.citations = []
                    return abstain("The generated answer could not be fully supported by its citations.")
            result.status = "partial_answer" if decision is not None and decision.decision == PARTIAL_ANSWER else "answered"
            result.answer = draft.answer
            return result
        except QuotaError as exc:
            # A citation set from an earlier, already-validated stage (e.g. citations were
            # checked, then the *next* call -- validation -- hit a live quota pause) must
            # never survive onto a non-"answered"/"partial_answer" result: the status here
            # says the run did not complete, so nothing citation-shaped should imply it did.
            result.citations = []
            result.status, result.answer = "quota_exceeded", str(exc)
            return result
        except RagError as exc:
            result.citations = []
            result.status, result.answer = "error", str(exc)
            return result
        except Exception:
            result.citations = []
            result.status, result.answer = "error", "A verification step failed. No unchecked answer was returned."
            return result
        finally:
            result.timings["total"] = time.perf_counter() - start
            result.usage = {key: self.client.usage.get(key, 0) - before.get(key, 0) for key in self.client.usage}
            result.cache_hit = result.usage.get("requests", 0) == 0 and result.usage.get("cache_hits", 0) > 0

    def run_safety_wall(self, question, index):
        """The full Safety Wall as a single orchestration entry point:

            query -> index.retrieve (existing) -> Pipeline.run(..., protected=True,
            recover=True, retrieve_fn=index.retrieve) -- i.e. this class's existing
            detection/quarantine, missing-fact analysis, bounded recovery, relevance/
            conflict/sufficiency verification, and decision layer, followed by this
            class's existing generation and citation validation -- -> SafetyWallReport.

        Reuses run() and Index.retrieve entirely; adds no new retrieval, generation, or
        citation logic. It only assembles what run() already computed -- detections,
        quarantine/exclusion, recovery, decision, citations, and the raw validation
        verdict -- into one structured, auditable report shaped for the "Full Safety
        Wall" evaluation mode, as distinct from Standard RAG (protected=False) and the
        existing detect-and-block-only mode (protected=True, recover=False).
        """
        hits = index.retrieve(self.client.embed(question))
        result = self.run(question, hits, protected=True, recover=True, retrieve_fn=index.retrieve)

        # Resolve chunk_ids back to their content for the report: the original retrieval
        # hits cover everything except recovered chunks, whose content only exists on
        # result.recovery (recover_evidence() never mutates the caller's `hits`).
        by_id = {hit.chunk.id: hit.chunk for hit in hits}
        for recovered in (result.recovery or {}).get("recovered_chunks", []):
            by_id[recovered["chunk_id"]] = Chunk(recovered["chunk_id"], recovered.get("document_hash", ""),
                recovered["source_doc"], recovered.get("page"), recovered["text"], recovered.get("location"))

        def describe(chunk_id, reason=None):
            chunk = by_id.get(chunk_id)
            entry = {"chunk_id": chunk_id, "filename": chunk.filename if chunk else None,
                     "text": chunk.text if chunk else None}
            if reason is not None:
                entry["reason"] = reason
            return entry

        recovery = result.recovery or {}
        verified_ids = (result.decision or {}).get("verified_chunks", [])

        return SafetyWallReport(
            query=question,
            status=result.status,
            initial_evidence=[describe(hit.chunk.id) for hit in hits],
            flagged_chunks=[d for d in result.detections if d["flagged"]],
            quarantined_chunks=[describe(e["chunk_id"], e["reason"])
                                 for e in result.excluded if e["stage"] == "injection"],
            missing_facts=recovery.get("missing_facts", []),
            # "Recovery attempts": every recovery candidate this run actually considered
            # and turned away (recovery.py's own audit trail) -- accepted candidates are
            # reported separately below, as recovered_chunks.
            recovery_attempts=recovery.get("rejected", []),
            recovered_chunks=recovery.get("recovered_chunks", []),
            recovery_status=recovery.get("recovery_status"),
            final_verified_evidence=[describe(chunk_id) for chunk_id in verified_ids],
            decision=result.decision,
            answer=result.answer,
            citations=result.citations,
            validation=result.validation,
        )
