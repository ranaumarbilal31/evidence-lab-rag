"""Owner-only setup and research commands; never exposed in the public app."""
from __future__ import annotations

import argparse
import csv
import json
import os
import threading
import time
import tomllib
import zipfile
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .api import Gemini, fingerprint
from .config import Settings, GEN_MODEL, EMBED_MODEL, PROMPT_VERSION
from .dataset import generate_cases, write_cases
from .ingest import ingest, LOCAL_LIMITS
from .models import RagError
from .pipeline import Pipeline
from .quota import Governor
from .samples import SCENARIOS
from .store import Store, Index, build_index

ROOT = Path(__file__).resolve().parents[1]
MODES = ["baseline", "protected", "without_injection", "without_relevance", "without_conflict", "without_sufficiency", "without_validation"]


def dump(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temp.replace(path)


def settings_and_governor():
    values = dict(os.environ)
    secrets = ROOT / ".streamlit" / "secrets.toml"
    if secrets.exists():
        values.update(tomllib.loads(secrets.read_text(encoding="utf-8")))
    settings = Settings.from_mapping(values)
    if not settings.ready:
        raise RagError("Configure .streamlit/secrets.toml with a Free-tier Gemini key, billing confirmation, and active quotas. Do not put the key in chat.")
    return settings, Governor({GEN_MODEL: settings.generation, EMBED_MODEL: settings.embedding},
                             ROOT / "research" / "quota.json")


class MemorySample:
    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self.peak = self.process.memory_info().rss
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.05):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()


def prepare_demo(capture=False):
    settings, governor = settings_and_governor()
    cache = Store(ROOT / "research" / "demo-cache.sqlite")
    try:
        for key, sample in SCENARIOS.items():
            path = ROOT / "demo" / f"{key}.index.json"
            if path.exists():
                index = Index.from_dict(json.loads(path.read_text(encoding="utf-8")))
            else:
                chunks, _ = ingest([(n, t.encode()) for n, t in sample["documents"].items()])
                with governor.job({EMBED_MODEL: len(chunks)}) as ticket:
                    client = Gemini(settings, governor, cache, ticket)
                    try:
                        index = build_index(chunks, client)
                    finally:
                        client.close()
                dump(path, index.to_dict())
                print(f"Prepared API embedding index: {key}")
            if capture:
                capture_path = ROOT / "demo" / f"{key}.capture.json"
                if capture_path.exists():
                    continue
                with governor.job({GEN_MODEL: 7, EMBED_MODEL: 1}) as ticket:
                    client = Gemini(settings, governor, cache, ticket)
                    try:
                        hits = index.retrieve(client.embed(sample["question"]))
                        pipeline = Pipeline(client)
                        protected = pipeline.run(sample["question"], hits)
                        if protected.status in {"error", "quota_exceeded"}:
                            raise RagError(protected.answer)
                        baseline = pipeline.run(sample["question"], hits, protected=False)
                        if baseline.status in {"error", "quota_exceeded"}:
                            raise RagError(baseline.answer)
                        dump(capture_path, {"captured_at": datetime.now(timezone.utc).isoformat(),
                            "model": GEN_MODEL, "question": sample["question"],
                            "protected": asdict(protected), "baseline": asdict(baseline),
                            "human_reviewed": False})
                    finally:
                        client.close()
                print(f"Captured API outputs: {key}; human review still required")
    finally:
        cache.close()


def evaluate(args):
    settings, governor = settings_and_governor()
    source = ROOT / "research" / "cases.jsonl"
    if not source.exists():
        raise RagError("Generate and human-review the dataset first: python -m rag.cli dataset")
    cases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    selected = [c for c in cases if c["split"] == args.split][:args.limit]
    if not all(c["human_label_reviewed"] is True for c in selected):
        raise RagError("Review expected labels in cases.jsonl and set human_label_reviewed=true per reviewed case before evaluation.")
    mode_list = MODES if args.mode == "all" else [args.mode]
    print(f"Selected {len(selected)} cases × {len(mode_list)} modes. Up to 6 checking/generation calls per protected mode, 1 baseline call, plus embeddings and retries. Runs checkpoint and may require multiple quota resets.")
    outdir = ROOT / "research" / "runs"
    cache = Store(ROOT / "research" / "eval-cache.sqlite")
    try:
        for case in selected:
            identity = fingerprint([case, GEN_MODEL, EMBED_MODEL, PROMPT_VERSION])
            retrieval_path = outdir / f"{case['id']}-{identity[:12]}.retrieval.json"
            # Shared initial retrieval is reused exactly across modes and resumptions.
            if retrieval_path.exists():
                from .models import Chunk, Hit
                saved = json.loads(retrieval_path.read_text())
                hits = [Hit(Chunk(**h["chunk"]), h["score"]) for h in saved["hits"]]
            else:
                chunks, _ = ingest([(n, t.encode()) for n, t in case["documents"].items()], LOCAL_LIMITS)
                with governor.job({EMBED_MODEL: len(chunks) + 1}) as ticket:
                    client = Gemini(settings, governor, cache, ticket)
                    try:
                        index = build_index(chunks, client)
                        hits = index.retrieve(client.embed(case["question"]))
                        dump(retrieval_path, {"case_id": case["id"], "fingerprint": identity,
                             "hits": [{"chunk": asdict(h.chunk), "score": h.score} for h in hits]})
                    finally:
                        client.close()
            for mode in mode_list:
                result_path = outdir / f"{case['id']}-{identity[:12]}-{mode}.json"
                if result_path.exists():
                    previous = json.loads(result_path.read_text())
                    if previous["result"]["status"] not in {"error", "quota_exceeded"}:
                        continue
                # Generation caches are isolated by mode so ablations are independently run.
                mode_cache = Store(ROOT / "research" / f"{mode}-cache.sqlite")
                try:
                    with governor.job({GEN_MODEL: 1 if mode == "baseline" else 6}) as ticket:
                        client = Gemini(settings, governor, mode_cache, ticket)
                        try:
                            tick = time.perf_counter()
                            with MemorySample() as memory:
                                result = Pipeline(client).run(case["question"], hits,
                                    protected=mode != "baseline", disable=mode.removeprefix("without_") if mode.startswith("without_") else None)
                            row = {"case_id": case["id"], "fingerprint": identity, "split": case["split"],
                                "category": case["category"], "mode": mode, "result": asdict(result),
                                "wall_seconds": time.perf_counter() - tick, "peak_rss_mb": memory.peak / 1024**2,
                                "captured_at": datetime.now(timezone.utc).isoformat()}
                            dump(result_path, row)
                            print(f"{case['id']} / {mode}: {result.status}")
                            if result.status in {"error", "quota_exceeded"}:
                                raise RagError(result.answer)
                        finally:
                            client.close()
                finally:
                    mode_cache.close()
    finally:
        cache.close()


HUMAN_FIELDS = ["reviewed", "answer_correct", "citations_supported", "attack_succeeded",
                "observed_conflict", "observed_abstention", "benign_false_positive", "notes"]


def export_scores():
    out = ROOT / "research" / "human-scores.csv"
    prior = {}
    if out.exists():
        with out.open(newline="", encoding="utf-8") as handle:
            prior = {row["run_file"]: row for row in csv.DictReader(handle)}
    rows = []
    for path in sorted((ROOT / "research" / "runs").glob("*.json")):
        run = json.loads(path.read_text())
        if "result" not in run:
            continue
        row = {"run_file": path.name, "case_id": run["case_id"], "mode": run["mode"], "category": run["category"],
               "status": run["result"]["status"], "answer": run["result"]["answer"],
               **{f: prior.get(path.name, {}).get(f, "") for f in HUMAN_FIELDS}}
        rows.append(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["run_file", "case_id", "mode", "category", "status", "answer"] + HUMAN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Exported {len(rows)} rows for human scoring. Fill binary fields with 0 or 1; use reviewed=1 only after review.")


def summarize():
    from .evaluation import summarize_scores
    result = summarize_scores(ROOT / "research")
    dump(ROOT / "research" / "summary.json", result)
    print(json.dumps(result, indent=2))


def package():
    """Allowlist-only source archive; private research and secrets cannot enter it."""
    target = ROOT / "dist" / "evidence-lab-deploy.zip"
    target.parent.mkdir(exist_ok=True)
    files = [ROOT / name for name in ["app.py", "requirements.txt", "requirements-dev.txt", ".gitignore",
        "README.md", "DEPLOYMENT.md", "LIMITATIONS.md", "RELEASE_STATUS.md", "secrets.example.toml", ".streamlit/config.toml"]]
    files += list((ROOT / "rag").glob("*.py")) + list((ROOT / "tests").glob("*.py"))
    files += list((ROOT / "demo").glob("*.json"))
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            if path.exists():
                archive.write(path, path.relative_to(ROOT))
    print(f"Created {target}")


def preflight():
    settings, _ = settings_and_governor()
    missing = []
    for key in SCENARIOS:
        index_file = ROOT / "demo" / f"{key}.index.json"
        capture_file = ROOT / "demo" / f"{key}.capture.json"
        if not index_file.exists():
            missing.append(f"{key}: API embedding index")
        else:
            Index.from_dict(json.loads(index_file.read_text()))
        if not capture_file.exists():
            missing.append(f"{key}: live capture")
        elif not json.loads(capture_file.read_text()).get("human_reviewed"):
            missing.append(f"{key}: human review of capture")
    if missing:
        raise RagError("Not ready for live-demo delivery: " + "; ".join(missing))
    print("Local deployment prerequisites passed. Hosted signed-out, session-isolation, and live tests are still required.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("dataset")
    prepare = sub.add_parser("prepare-demo")
    prepare.add_argument("--capture", action="store_true")
    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--split", choices=["development", "held_out"], default="development")
    evaluate_parser.add_argument("--mode", choices=MODES + ["all"], default="all")
    evaluate_parser.add_argument("--limit", type=int, default=150)
    for name in ["export-scores", "summarize", "package", "preflight"]:
        sub.add_parser(name)
    args = parser.parse_args()
    try:
        if args.command == "dataset":
            write_cases(ROOT / "research" / "cases.jsonl")
            print("Created 150 synthetic cases (50 development, 100 held-out). Expected labels require human review.")
        elif args.command == "prepare-demo":
            prepare_demo(args.capture)
        elif args.command == "evaluate":
            evaluate(args)
        elif args.command == "export-scores":
            export_scores()
        elif args.command == "summarize":
            summarize()
        elif args.command == "package":
            package()
        else:
            preflight()
    except (RagError, ValueError) as exc:
        parser.exit(2, f"Paused: {exc}\n")


if __name__ == "__main__":
    main()
