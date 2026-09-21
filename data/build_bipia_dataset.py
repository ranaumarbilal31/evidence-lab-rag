"""Extract and adapt BIPIA benchmark data into corpus.json and eval_questions.json.

Reads clean contexts from ../BIPIA/benchmark/{email,table,code}/test.jsonl and
adversarial prompts from ../BIPIA/benchmark/text_attack_test.json.  Produces two
files in --out-dir that the rag-safety-wall retriever and evaluation harness can
consume directly.

Usage:
    python data/build_bipia_dataset.py
    python data/build_bipia_dataset.py --bipia-dir ../BIPIA/benchmark --out-dir data/
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _word_count(text: str) -> int:
    return len(text.split())


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    """Truncate at the last sentence boundary before *max_tokens* words."""
    words = text.split()
    if len(words) <= max_tokens:
        return text
    truncated = " ".join(words[:max_tokens])
    # Try to cut at a sentence boundary
    for sep in (". ", "? ", "! ", ".\n", "\n\n"):
        pos = truncated.rfind(sep)
        if pos > len(truncated) // 2:
            return truncated[: pos + 1].strip()
    return truncated.strip()


def _pad_to_tokens(text: str, supplement: str, min_tokens: int) -> str:
    """If *text* is too short, append *supplement* to reach ~min_tokens."""
    if _word_count(text) >= min_tokens:
        return text
    combined = text.rstrip() + "\n\n" + supplement.strip()
    return combined


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# Source loaders
# ---------------------------------------------------------------------------

def _load_jsonl(path: Path) -> list[dict]:
    items = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def _load_email(bipia_dir: Path) -> list[dict]:
    """Each line: {context: str, question: str, ideal: str}."""
    path = bipia_dir / "email" / "test.jsonl"
    raw = _load_jsonl(path)
    out = []
    for item in raw:
        out.append({
            "source": "email",
            "context": _normalize_whitespace(item["context"]),
            "question": item.get("question", ""),
            "ideal": item.get("ideal", "unknown"),
        })
    return out


def _load_table(bipia_dir: Path) -> list[dict]:
    """Each line: {context: str (markdown table), question: str, ideal: str}."""
    path = bipia_dir / "table" / "test.jsonl"
    raw = _load_jsonl(path)
    out = []
    for item in raw:
        # Tables are already string; keep formatting for structure
        ctx = item["context"].strip()
        out.append({
            "source": "table",
            "context": ctx,
            "question": item.get("question", ""),
            "ideal": item.get("ideal", "unknown"),
        })
    return out


def _load_code(bipia_dir: Path) -> list[dict]:
    """Each line: {error: list[str], code: list[str], context: list[str],
    ideal: list[str], context_url: str, ...}.  No 'question' field -- we
    synthesize one from the error traceback."""
    path = bipia_dir / "code" / "test.jsonl"
    raw = _load_jsonl(path)
    out = []
    for item in raw:
        ctx_parts = item.get("context", [])
        context_text = "\n".join(ctx_parts) if isinstance(ctx_parts, list) else str(ctx_parts)
        context_text = context_text.strip()

        # Build context: code snippet + context (answer) text
        code_text = "\n".join(item.get("code", []))
        full_context = f"Code:\n{code_text}\n\nExplanation:\n{context_text}" if code_text else context_text

        # Synthesize question from error
        errors = item.get("error", [])
        # Find the last meaningful error line (usually the exception message)
        error_summary = ""
        for line in reversed(errors):
            line = line.strip()
            if line and not line.startswith("Traceback") and not line.startswith("File "):
                error_summary = line
                break
        if not error_summary and errors:
            error_summary = errors[-1].strip() or "this error"
        question = f"How do you fix this error: {error_summary}" if error_summary else "How do you fix this code error?"

        # required_facts: the ideal fix + key diagnostic term
        ideal_lines = item.get("ideal", [])
        ideal_text = "\n".join(ideal_lines) if isinstance(ideal_lines, list) else str(ideal_lines)

        out.append({
            "source": "code",
            "context": _normalize_whitespace(full_context),
            "question": question,
            "ideal": ideal_text.strip() if ideal_text.strip() else "unknown",
        })
    return out


def _load_stub(bipia_dir: Path, name: str) -> list[dict]:
    """Stub loader for qa/abstract -- returns empty if data isn't available."""
    path = bipia_dir / name / "test.jsonl"
    if not path.exists():
        return []
    try:
        raw = _load_jsonl(path)
        out = []
        for item in raw:
            ctx = item.get("context", "")
            if isinstance(ctx, list):
                ctx = "\n".join(ctx)
            out.append({
                "source": name,
                "context": _normalize_whitespace(ctx),
                "question": item.get("question", ""),
                "ideal": item.get("ideal", "unknown"),
            })
        return out
    except Exception as e:
        print(f"[stub] Could not load {name}: {e}; skipping.", file=sys.stderr)
        return []


def _load_attacks(bipia_dir: Path) -> list[str]:
    """Flatten text_attack_test.json {category: [prompt, ...]} into a flat list."""
    path = bipia_dir / "text_attack_test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    payloads = []
    for prompts in data.values():
        payloads.extend(prompts)
    return payloads


# ---------------------------------------------------------------------------
# Chunk + eval generation
# ---------------------------------------------------------------------------

MIN_TOKENS = 15   # floor — some BIPIA emails are legitimately ~20 words
MAX_TOKENS = 500  # ceiling — truncate if longer
ABSTAIN_FRACTION = 0.18  # ~18% unrecoverable


def _make_recovery_text(context: str, ideal: str) -> str:
    """Build a recovery chunk: last ~60% of the context, or the context with ideal appended."""
    words = context.split()
    start = max(0, len(words) * 2 // 5)  # take the last 60%
    recovery = " ".join(words[start:])
    if ideal and ideal.lower() != "unknown":
        recovery = recovery.rstrip() + " " + ideal.strip()
    return recovery.strip()


def build(bipia_dir: Path, out_dir: Path, seed: int = 42):
    rng = random.Random(seed)

    # Load sources
    samples: list[dict] = []
    samples.extend(_load_email(bipia_dir))
    samples.extend(_load_table(bipia_dir))
    samples.extend(_load_code(bipia_dir))
    # Stubs for future sources
    samples.extend(_load_stub(bipia_dir, "qa"))
    samples.extend(_load_stub(bipia_dir, "abstract"))

    attacks = _load_attacks(bipia_dir)
    if not attacks:
        print("ERROR: No attack payloads found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(samples)} context samples, {len(attacks)} attack payloads.")

    # Decide which indices are unrecoverable (~18%) — only from answerable
    # samples (non-"unknown" ideal) so we test recovery failure, not just
    # unanswerable questions.
    answerable_indices = [i for i, s in enumerate(samples) if s["ideal"].lower() != "unknown"]
    n_abstain = max(1, round(len(samples) * ABSTAIN_FRACTION))
    n_abstain = min(n_abstain, len(answerable_indices))  # can't exceed pool
    abstain_indices = set(rng.sample(answerable_indices, n_abstain))
    print(f"Unrecoverable (abstain) cases: {n_abstain}/{len(samples)} ({n_abstain/len(samples)*100:.1f}%)")

    corpus: list[dict] = []
    evals: list[dict] = []
    source_counters: dict[str, int] = {}

    for idx, sample in enumerate(samples):
        source = sample["source"]
        source_counters[source] = source_counters.get(source, 0) + 1
        local_idx = source_counters[source] - 1
        prefix = f"bipia_{source}_{local_idx:03d}"

        context = sample["context"]
        question = sample["question"]
        ideal = sample["ideal"]
        is_unrecoverable = idx in abstain_indices

        # --- Chunk length control ---
        if _word_count(context) > MAX_TOKENS:
            context = _truncate_to_tokens(context, MAX_TOKENS)
        if _word_count(context) < MIN_TOKENS and ideal and ideal.lower() != "unknown":
            context = _pad_to_tokens(context, ideal, MIN_TOKENS)

        # --- 1. Primary clean chunk ---
        clean_id = f"{prefix}_clean"
        source_doc = f"BIPIA/{source}/test.jsonl:{local_idx}"
        corpus.append({
            "chunk_id": clean_id,
            "source_doc": source_doc,
            "text": context,
            "is_injected": False,
            "attacked": False,
        })

        # --- 2. Injected variant ---
        attack_payload = attacks[idx % len(attacks)]
        injected_id = f"{prefix}_injected"
        injected_text = context + " " + attack_payload
        # Truncate if the combined text is too long
        if _word_count(injected_text) > MAX_TOKENS + 100:
            injected_text = _truncate_to_tokens(injected_text, MAX_TOKENS + 100)
        corpus.append({
            "chunk_id": injected_id,
            "source_doc": source_doc,
            "text": injected_text,
            "is_injected": True,
            "attacked": True,
        })

        # --- 3. Recovery clean chunk (skip for unrecoverable cases) ---
        recovery_id: str | None = None
        if not is_unrecoverable:
            recovery_id = f"{prefix}_recovery"
            recovery_text = _make_recovery_text(context, ideal)
            if _word_count(recovery_text) > MAX_TOKENS:
                recovery_text = _truncate_to_tokens(recovery_text, MAX_TOKENS)
            # Pad short recovery chunks — use full context as fallback
            if _word_count(recovery_text) < MIN_TOKENS:
                supplement = ideal if (ideal and ideal.lower() != "unknown") else context
                recovery_text = _pad_to_tokens(recovery_text, f"Summary: {supplement}", MIN_TOKENS)
            corpus.append({
                "chunk_id": recovery_id,
                "source_doc": source_doc,
                "text": recovery_text,
                "is_injected": False,
                "attacked": False,
            })

        # --- Eval case ---
        # required_facts: ideal answer if available, else key context excerpt
        if ideal and ideal.lower() != "unknown":
            required_facts = [ideal]
        else:
            # Use first 100 words of context as the fact reference
            required_facts = [" ".join(context.split()[:100])]

        # Unrecoverable cases → "abstain"; everything else (including
        # ideal="unknown" with recovery) → "answer" to test the pipeline's
        # ability to answer from recovered evidence even when ideal is unknown.
        expected_behavior = "abstain" if is_unrecoverable else "answer"

        evals.append({
            "test_id": prefix,
            "query": question,
            "required_facts": required_facts,
            "target_clean_chunk": clean_id,
            "recovery_clean_chunk": recovery_id,  # None for abstain cases
            "target_injected_chunk": injected_id,
            "expected_behavior": expected_behavior,
        })

    # Write outputs
    out_dir.mkdir(parents=True, exist_ok=True)

    corpus_path = out_dir / "corpus.json"
    corpus_path.write_text(json.dumps(corpus, indent=2, ensure_ascii=False), encoding="utf-8")

    evals_path = out_dir / "eval_questions.json"
    evals_path.write_text(json.dumps(evals, indent=2, ensure_ascii=False), encoding="utf-8")

    # Summary
    n_injected = sum(1 for c in corpus if c["is_injected"])
    n_clean = len(corpus) - n_injected
    n_abstain_actual = sum(1 for e in evals if e["expected_behavior"] == "abstain")
    n_answer = len(evals) - n_abstain_actual

    print(f"\n--- Output Summary ---")
    print(f"corpus.json : {len(corpus)} chunks ({n_clean} clean, {n_injected} injected)")
    print(f"eval_questions.json: {len(evals)} cases ({n_answer} answer, {n_abstain_actual} abstain)")
    print(f"Written to: {out_dir.resolve()}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Build BIPIA-derived benchmark dataset for Evidence Recovery RAG Safety Wall.")
    parser.add_argument("--bipia-dir", type=Path, default=Path(__file__).resolve().parent.parent / "BIPIA" / "benchmark",
                        help="Path to BIPIA/benchmark/ directory (default: ../BIPIA/benchmark relative to this script)")
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent,
                        help="Output directory for corpus.json and eval_questions.json (default: same dir as this script)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic attack/abstain assignment")
    args = parser.parse_args()

    if not args.bipia_dir.is_dir():
        print(f"ERROR: BIPIA benchmark directory not found: {args.bipia_dir}", file=sys.stderr)
        sys.exit(1)

    build(args.bipia_dir, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
