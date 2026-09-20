from __future__ import annotations

import json
import hashlib
import os
import time
from dataclasses import asdict
from pathlib import Path

import streamlit as st

from rag.api import Gemini, fingerprint
from rag.config import Settings, GEN_MODEL, EMBED_MODEL
from rag.ingest import ingest
from rag.models import Result, RagError, QuotaError
from rag.pipeline import Pipeline
from rag.quota import Governor
from rag.recovery import MAX_RECOVERY_ATTEMPTS, RECOVERY_TOP_K
from rag.samples import SCENARIOS
from rag.store import Index, Store, build_index

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Evidence Lab · Secure RAG", page_icon="◈", layout="wide")


def read_settings():
    values = dict(os.environ)
    try:
        values.update(dict(st.secrets))
    except (FileNotFoundError, st.errors.StreamlitSecretNotFoundError):
        pass
    try:
        return Settings.from_mapping(values)
    except (ValueError, TypeError):
        return Settings.from_mapping({})


@st.cache_resource
def governor_for(generation, embedding):
    return Governor({GEN_MODEL: generation, EMBED_MODEL: embedding})


@st.cache_resource
def public_cache():
    # ONLY immutable public sample requests may use this shared cache.
    return Store()


def init_session():
    defaults = {"private_store": None, "private_index": None, "results": None,
                "upload_fingerprint": None, "question_times": [], "last_submit": 0.0,
                "result_key": None, "private_warnings": []}
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value
    if st.session_state.private_store is None:
        st.session_state.private_store = Store()


def clear_session():
    st.session_state.private_store.close()
    for key in list(st.session_state):
        del st.session_state[key]
    st.rerun()


def show_result(result, heading):
    st.subheader(heading)
    labels = {"answered": "Answer with evidence", "conflict": "Conflicting evidence",
              "insufficient_evidence": "Insufficient evidence", "quota_exceeded": "Live quota paused", "error": "Unable to verify"}
    st.caption(labels.get(result.status, result.status).upper())
    # Plain text avoids model-authored Markdown links, remote images, or HTML.
    st.text(result.answer)
    for warning in result.warnings:
        st.warning(warning)
    if result.citations:
        st.markdown("**Supporting passages**")
        for citation in result.citations:
            page = f" · page {citation['page']}" if citation["page"] else ""
            st.caption(f"{citation['filename']}{page}")
            st.text(citation["quote"])
            st.caption(f"Source: {citation['chunk_id']}")
    with st.expander("How this result was produced"):
        st.json({"excluded_evidence": result.excluded, "timings_seconds": result.timings,
                 "usage": result.usage, "configuration": result.configuration,
                 "cache_hit": result.cache_hit})


def show_safety_wall_result(report, heading):
    """Render a SafetyWallReport (Pipeline.run_safety_wall) as 9 compact, auditable
    stages. Pure display: every value shown here already exists on the report --
    nothing is computed, scored, or inferred by the UI. Quarantined chunks are shown
    only under stage 3, clearly marked UNTRUSTED; they are never part of
    `final_verified_evidence` or the generation context, because run_safety_wall never
    puts them there (see rag/pipeline.py)."""
    st.subheader(heading)
    labels = {"answered": "Answer with evidence", "partial_answer": "Partial answer with evidence",
              "conflict": "Conflicting evidence", "insufficient_evidence": "Insufficient evidence",
              "quota_exceeded": "Live quota paused", "error": "Unable to verify"}
    st.caption(labels.get(report.status, report.status).upper())

    flagged_ids = {c["chunk_id"] for c in report.flagged_chunks}

    with st.expander("1 · Retrieved evidence", expanded=False):
        if report.initial_evidence:
            for chunk in report.initial_evidence:
                st.caption(f"`{chunk['chunk_id']}` · {chunk['filename']}")
        else:
            st.caption("No evidence was retrieved.")

    with st.expander("2 · Safety screening", expanded=False):
        clean = [c for c in report.initial_evidence if c["chunk_id"] not in flagged_ids]
        st.markdown(f"**Clean chunks ({len(clean)})**")
        for chunk in clean:
            st.caption(f"`{chunk['chunk_id']}` · {chunk['filename']}")
        st.markdown(f"**Flagged chunks ({len(report.flagged_chunks)})**")
        if report.flagged_chunks:
            for detection in report.flagged_chunks:
                st.caption(f"`{detection['chunk_id']}` · {detection['source']} · confidence {detection['confidence']:.2f}")
                st.text(detection["flag_reason"])
        else:
            st.caption("No chunk was flagged.")

    with st.expander("3 · Quarantine", expanded=False):
        if report.quarantined_chunks:
            st.warning("Quarantined chunks are UNTRUSTED. They are never included in the generation context.")
            for chunk in report.quarantined_chunks:
                st.caption(f"`{chunk['chunk_id']}` · {chunk['filename']} · UNTRUSTED")
                st.text(chunk.get("reason", ""))
        else:
            st.caption("Nothing was quarantined.")

    with st.expander("4 · Missing fact", expanded=False):
        if report.missing_facts:
            for fact in report.missing_facts:
                st.write(f"- {fact}")
        else:
            st.caption("No missing fact was attributed (nothing quarantined, or no fact could be tied to a question).")

    with st.expander("5 · Evidence recovery", expanded=False):
        recovery_needed = bool(report.quarantined_chunks)
        recovery_attempted = bool(report.missing_facts)
        st.write(f"Recovery needed: {'yes' if recovery_needed else 'no'}")
        st.write(f"Recovery attempted: {'yes' if recovery_attempted else 'no'}")
        if recovery_attempted:
            st.caption(f"Recovery budget: {MAX_RECOVERY_ATTEMPTS} retrieval attempt(s), top {RECOVERY_TOP_K} candidate(s) per attempt.")
            st.caption(f"Recovery query: {'; '.join(report.missing_facts)}")
            excluded_ids = sorted(c["chunk_id"] for c in report.quarantined_chunks)
            st.caption(f"Excluded chunk IDs: {', '.join(excluded_ids) if excluded_ids else 'none'}")
            recovered_ids = [c["chunk_id"] for c in report.recovered_chunks]
            st.caption(f"Recovered chunk IDs: {', '.join(recovered_ids) if recovered_ids else 'none'}")
            st.caption(f"Recovery status: {report.recovery_status or 'n/a'}")
            if report.recovery_attempts:
                st.markdown("**Rejected recovery candidates**")
                for rejected in report.recovery_attempts:
                    st.caption(f"`{rejected.get('chunk_id', 'n/a')}`: {rejected['reason']}")
        else:
            st.caption("No recovery attempt was made.")

    with st.expander("6 · Verified evidence", expanded=False):
        st.caption("Only this evidence was allowed to reach generation.")
        if report.final_verified_evidence:
            for chunk in report.final_verified_evidence:
                st.caption(f"`{chunk['chunk_id']}` · {chunk['filename']}")
        else:
            st.caption("No evidence was verified as sufficient.")

    with st.expander("7 · Decision", expanded=False):
        if report.decision:
            st.write(f"Decision: **{report.decision['decision']}**")
            st.caption(f"Reason: {report.decision['reason']}")
            if report.decision.get("unsupported_facts"):
                st.markdown("**Unsupported facts**")
                for fact in report.decision["unsupported_facts"]:
                    st.write(f"- {fact}")
        else:
            st.caption("No decision was recorded for this run.")

    with st.expander("8 · Final answer", expanded=True):
        st.text(report.answer)
        if report.citations:
            st.markdown("**Supporting passages**")
            for citation in report.citations:
                page = f" · page {citation['page']}" if citation["page"] else ""
                st.caption(f"{citation['filename']}{page}")
                st.text(citation["quote"])
                st.caption(f"Source: {citation['chunk_id']}")

    with st.expander("9 · Citation validation", expanded=False):
        if report.validation is not None:
            st.write(f"Mechanically validated as supported: {report.validation.get('supported')}")
            if report.validation.get("unsupported_claims"):
                st.markdown("**Unsupported claims**")
                for claim in report.validation["unsupported_claims"]:
                    st.write(f"- {claim}")
        validation_failed = report.validation is not None and not report.validation.get("supported", True)
        if report.status == "insufficient_evidence" and (validation_failed or not report.citations):
            st.error("Citation validation did not pass. The answer above is the safe fallback (abstention), "
                      "not an unverified answer presented as trustworthy.")
        elif report.citations:
            st.success("Citations passed the existing mechanical validation.")


init_session()
settings = read_settings()
governor = governor_for(settings.generation, settings.embedding)

with st.sidebar:
    st.markdown("### ◈ Evidence Lab")
    st.caption("SECURE & RELIABLE RAG")
    st.divider()
    st.markdown("**A small research demo**")
    st.write("Explore what happens when retrieved evidence is malicious, irrelevant, conflicting, or incomplete.")
    st.caption("Free APIs · No local AI models")
    if settings.ready:
        st.success("Live API configured")
    else:
        st.info("Preview mode · live API setup pending")
    with st.expander("Service & usage"):
        st.write("Free hosting can sleep. API limits are shared by all visitors, and requests may pause.")
        st.json(governor.snapshot())
        st.caption("App counters are estimates and reset on host restart. Provider Free-tier quotas remain authoritative.")
    st.button("Clear my session", on_click=clear_session, use_container_width=True)
    st.caption("Research prototype: evidence checks reduce risk; they do not guarantee truth or complete attack protection.")

st.caption("RETRIEVE  /  CHECK  /  EXPLAIN")
st.title("Better answers start with better evidence.")
st.write("Compare ordinary RAG with a pipeline that checks the information before it answers.")
st.info("Use only public or synthetic, non-sensitive documents. Document text and questions are processed by Google’s API. Uploads also pass through this hosting service.")

source = st.radio("Choose your evidence", ["Explore sample scenarios", "Use my documents"], horizontal=True)
index, cache, warnings = None, public_cache(), []
scenario_id = None
if source == "Explore sample scenarios":
    scenario_id = st.selectbox("Scenario", list(SCENARIOS), format_func=lambda k: f"{SCENARIOS[k]['icon']} · {SCENARIOS[k]['title']}")
    sample = SCENARIOS[scenario_id]
    st.write(sample["description"])
    with st.expander("Read the source documents"):
        for filename, text in sample["documents"].items():
            st.markdown(f"**{filename}**")
            st.text(text)
    index_path = ROOT / "demo" / f"{scenario_id}.index.json"
    if index_path.exists():
        try:
            index = Index.from_dict(json.loads(index_path.read_text(encoding="utf-8")))
        except (RagError, ValueError, KeyError, TypeError):
            st.error("The bundled index is incompatible. The owner must rebuild it before live use.")
    else:
        st.caption("The owner still needs to generate this scenario’s API embedding index. No live results are claimed.")
    capture_path = ROOT / "demo" / f"{scenario_id}.capture.json"
    with st.expander("View a demonstration example", expanded=not settings.ready):
        if capture_path.exists():
            capture = json.loads(capture_path.read_text(encoding="utf-8"))
            st.warning(f"Saved demonstration — not a live response. Captured {capture['captured_at']} using {capture['model']}.")
            st.caption(f"Saved question: {capture['question']}")
            show_result(Result(**capture["protected"]), "Saved protected result")
            if "baseline" in capture:
                show_result(Result(**capture["baseline"]), "Saved ordinary RAG result")
            if not capture.get("human_reviewed", False):
                st.caption("This capture has not received independent human review.")
        else:
            st.warning("Illustrative expected behavior — manually authored, not an API result or research measurement.")
            st.text(sample["expected_answer"])
            st.caption(f"Expected status: {sample['expected_status']}")
    default_question = sample["question"]
else:
    cache = st.session_state.private_store
    st.caption("Up to 3 UTF-8 text/Markdown files or text PDFs · 2 MB each · 30 pages each · 50 chunks total")
    uploads = st.file_uploader("Choose non-sensitive documents", type=["pdf", "txt", "md"], accept_multiple_files=True)
    default_question = ""
    consent = st.checkbox("These documents are public or synthetic and contain no sensitive, confidential, or personal information.")
    prepared = None
    current_hash = None
    if uploads:
        files = [(f.name, f.getvalue()) for f in uploads]
        current_hash = fingerprint([(name, hashlib.sha256(data).hexdigest()) for name, data in files])
        if current_hash != st.session_state.upload_fingerprint:
            st.session_state.private_index = None
            st.session_state.results = None
        try:
            prepared, warnings = ingest(files)
            st.caption(f"{len(prepared)} chunks. Indexing needs at most {len(prepared)} embedding requests before cache hits and retries.")
        except RagError as exc:
            st.error(str(exc))
    if st.button("Index my documents", disabled=not (prepared and consent and settings.ready)):
        client = None
        try:
            with governor.job({EMBED_MODEL: len(prepared)}) as ticket:
                client = Gemini(settings, governor, cache, ticket)
                progress = st.progress(0.0, text="Embedding through the free API…")
                with st.spinner("Building a private session index…"):
                    built = build_index(prepared, client, lambda value: progress.progress(value))
                st.session_state.private_index = built
                st.session_state.private_warnings = warnings
                st.session_state.upload_fingerprint = current_hash
                st.success("Your session index is ready.")
        except RagError as exc:
            st.warning(str(exc))
        finally:
            if client:
                client.close()
    if current_hash and current_hash == st.session_state.upload_fingerprint:
        index = st.session_state.private_index
        for warning in st.session_state.private_warnings:
            st.warning(warning)

question = st.text_input("Ask about this evidence", value=default_question, max_chars=1000,
                         key=f"question_{scenario_id or 'private'}", placeholder="What does the policy actually say?")
# A custom question is visitor data even when the corpus is public.
if scenario_id and question != SCENARIOS[scenario_id]["question"]:
    cache = st.session_state.private_store
pipeline_choice = st.radio("Pipeline", ["Standard RAG", "Detect & Block", "Full Safety Wall"], index=1, horizontal=True,
    help="Standard RAG: no safety checks, for comparison. Detect & Block: flags and quarantines suspicious "
         "evidence before answering. Full Safety Wall: also analyzes what a quarantine cost the question and "
         "attempts one bounded, independently-verified recovery of the missing fact.")
compare = False
if pipeline_choice != "Standard RAG":
    compare = st.checkbox("Compare with Standard RAG", value=True)
identity = fingerprint([source, scenario_id, st.session_state.upload_fingerprint, question, pipeline_choice, compare])
if identity != st.session_state.result_key:
    st.session_state.results = None
if st.button("Check the evidence", type="primary", disabled=not (settings.ready and index and question.strip())):
    now = time.time()
    history = [t for t in st.session_state.question_times if now - t < 3600]
    if now - st.session_state.last_submit < 20 or len(history) >= 5:
        st.warning("Please wait between questions. This free demo allows five new questions per session per hour.")
    else:
        client = None
        if pipeline_choice == "Full Safety Wall":
            reservation = {GEN_MODEL: 11 if compare else 10, EMBED_MODEL: 3 if compare else 2}
        elif pipeline_choice == "Detect & Block":
            reservation = {GEN_MODEL: 7 if compare else 6, EMBED_MODEL: 1}
        else:
            reservation = {GEN_MODEL: 1, EMBED_MODEL: 1}
        try:
            with governor.job(reservation) as ticket:
                st.session_state.last_submit = now
                st.session_state.question_times = history + [now]
                client = Gemini(settings, governor, cache, ticket)
                pipeline = Pipeline(client)
                with st.status("Retrieving and checking evidence…", expanded=True) as status:
                    if pipeline_choice == "Full Safety Wall":
                        st.write("Retrieving evidence, screening it, and attempting bounded recovery where needed.")
                        primary_kind = "safety_wall"
                        primary = pipeline.run_safety_wall(question, index)
                        secondary = None
                        if compare and primary.status not in {"quota_exceeded", "error"}:
                            st.write("Running Standard RAG on freshly retrieved evidence.")
                            hits = index.retrieve(client.embed(question))
                            secondary = pipeline.run(question, hits, protected=False)
                    else:
                        tick = time.perf_counter()
                        hits = index.retrieve(client.embed(question))
                        retrieval_seconds = time.perf_counter() - tick
                        protected_mode = pipeline_choice == "Detect & Block"
                        primary_kind = "protected" if protected_mode else "baseline"
                        st.write("Checking instructions, relevance, conflicts, and answer support." if protected_mode
                                 else "Running ordinary RAG with no safety checks.")
                        primary = pipeline.run(question, hits, protected=protected_mode)
                        primary.timings["retrieval"] = retrieval_seconds
                        secondary = None
                        if compare and primary.status not in {"quota_exceeded", "error"}:
                            st.write("Running Standard RAG on the same retrieved evidence.")
                            secondary = pipeline.run(question, hits, protected=False)
                    status.update(label="Evidence check complete" if primary.status not in {"error", "quota_exceeded"} else "Live request paused or failed", state="complete")
                st.session_state.results = (primary_kind, primary, secondary)
                st.session_state.result_key = identity
        except RagError as exc:
            st.warning(str(exc))
        finally:
            if client:
                client.close()

if st.session_state.results:
    primary_kind, primary, secondary = st.session_state.results
    primary_heading = {"safety_wall": "Full Safety Wall", "protected": "Detect & Block", "baseline": "Standard RAG"}[primary_kind]
    render_primary = show_safety_wall_result if primary_kind == "safety_wall" else show_result
    if secondary:
        left, right = st.columns(2, gap="large")
        with left:
            render_primary(primary, primary_heading)
        with right:
            show_result(secondary, "Standard RAG")
    else:
        render_primary(primary, primary_heading)
    st.download_button("Download this comparison", json.dumps({"pipeline": primary_kind, "primary": asdict(primary),
        "compare": asdict(secondary) if secondary else None}, indent=2), "rag-comparison.json", "application/json")

st.divider()
st.caption("Evidence Lab · Research demonstration · $0 service budget · No automatic paid fallback")
