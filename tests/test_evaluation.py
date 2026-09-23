import csv
import json
from collections import Counter

from rag.dataset import generate_cases, write_cases
from rag.evaluation import summarize_scores


def test_dataset_balance_and_disjoint_families(tmp_path):
    cases = generate_cases()
    assert len(cases) == 150
    assert Counter(c["category"] for c in cases) == {k: 30 for k in ["clean", "malicious", "irrelevant", "conflicting", "insufficient"]}
    dev = [c for c in cases if c["split"] == "development"]
    held = [c for c in cases if c["split"] == "held_out"]
    assert len(dev) == 50 and len(held) == 100
    assert not {c["family"] for c in dev} & {c["family"] for c in held}
    assert all(c["human_label_reviewed"] is False for c in cases)
    path = tmp_path / "cases.jsonl"
    write_cases(path)
    import pytest
    with pytest.raises(ValueError):
        write_cases(path)


def test_missing_live_results_are_not_passing_metrics(tmp_path):
    result = summarize_scores(tmp_path)
    assert result["status"] == "pending_human_scoring"
    assert result["live_runs"] == 0 and not result["targets"]


def test_human_paired_metrics_and_operational_failures(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    rows = []
    for category in ["clean", "malicious", "conflicting", "insufficient"]:
        for mode in ["baseline", "protected"]:
            correct = mode == "protected" or category == "clean"
            identity = category + "0123456789012345"
            name = f"{category}-{identity[:12]}-{mode}.json"
            record = {"case_id": category, "fingerprint": identity, "category": category,
                      "split": "held_out", "mode": mode, "wall_seconds": 3, "peak_rss_mb": 100,
                      "result": {"status": "answered", "usage": {"cache_hits": 0}}}
            (run_dir / name).write_text(json.dumps(record))
            rows.append({"run_file": name, "case_id": category, "mode": mode, "category": category,
                "status": "answered", "reviewed": 1, "answer_correct": int(correct),
                "citations_supported": int(correct), "attack_succeeded": int(category == "malicious" and mode == "baseline"),
                "observed_conflict": int(category == "conflicting" and correct),
                "observed_abstention": int(category == "insufficient" and correct), "benign_false_positive": 0})
    (run_dir / "failed.json").write_text(json.dumps({"case_id": "failed", "fingerprint": "failed0123456789",
        "mode": "baseline", "result": {"status": "quota_exceeded"}}))
    with (tmp_path / "human-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize_scores(tmp_path)
    assert report["operational_failures"] == 1
    assert report["targets"]["held_out"]["relative_attack_reduction"] == 1.0
    assert report["modes"]["held_out/protected"]["conflicting_recall"] == 1.0


def test_recovery_metrics_do_not_require_human_review(tmp_path):
    """Recovery attempted/succeeded/failed are structural facts on the run itself, so
    they must be visible before any human-scores.csv exists at all."""
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    (run_dir / "a-000000000000-safety_wall.json").write_text(json.dumps({
        "case_id": "a", "fingerprint": "000000000000", "category": "malicious", "split": "development",
        "mode": "safety_wall", "wall_seconds": 4, "peak_rss_mb": 100,
        "result": {"status": "answered", "usage": {"cache_hits": 0},
                   "recovery": {"recovery_status": "full", "missing_facts": ["x"], "recovered_chunks": [], "rejected": []}}}))
    (run_dir / "b-000000000000-safety_wall.json").write_text(json.dumps({
        "case_id": "b", "fingerprint": "000000000000", "category": "malicious", "split": "development",
        "mode": "safety_wall", "wall_seconds": 4, "peak_rss_mb": 100,
        "result": {"status": "insufficient_evidence", "usage": {"cache_hits": 0},
                   "recovery": {"recovery_status": "failed", "missing_facts": ["x"], "recovered_chunks": [], "rejected": []}}}))
    (run_dir / "c-000000000000-protected.json").write_text(json.dumps({
        "case_id": "c", "fingerprint": "000000000000", "category": "clean", "split": "development",
        "mode": "protected", "wall_seconds": 2, "peak_rss_mb": 100,
        "result": {"status": "answered", "usage": {"cache_hits": 0}, "recovery": None}}))
    report = summarize_scores(tmp_path)
    assert report["status"] == "pending_human_scoring"  # no human-scores.csv yet
    assert report["recovery"]["development/safety_wall"] == {
        "recovery_attempted": 2, "recovery_succeeded": 1, "recovery_failed": 1, "recovery_success_rate": 0.5}
    assert "development/protected" not in report["recovery"]  # recovery=None is never counted


def test_protected_vs_safety_wall_target_answers_q2_q3_q4(tmp_path):
    run_dir = tmp_path / "runs"
    run_dir.mkdir()
    rows = []
    # "insufficient": detect-and-block abstains (answer_correct=0); the full Safety Wall
    # recovers the missing fact and gets it right (answer_correct=1) -- answers Q2.
    # "clean": the full Safety Wall introduces one additional benign false positive that
    # detect-and-block did not have -- answers Q3.
    # Latencies (wall_seconds) differ by 3s between the two modes -- answers Q4.
    plan = {
        ("insufficient", "protected"): dict(answer_correct=0, benign_false_positive=0, wall_seconds=2),
        ("insufficient", "safety_wall"): dict(answer_correct=1, benign_false_positive=0, wall_seconds=5),
        ("clean", "protected"): dict(answer_correct=1, benign_false_positive=0, wall_seconds=2),
        ("clean", "safety_wall"): dict(answer_correct=1, benign_false_positive=1, wall_seconds=5),
        ("malicious", "protected"): dict(answer_correct=1, benign_false_positive=0, wall_seconds=2),
        ("malicious", "safety_wall"): dict(answer_correct=1, benign_false_positive=0, wall_seconds=5),
    }
    for (category, mode), values in plan.items():
        identity = category + "0123456789012345"
        name = f"{category}-{identity[:12]}-{mode}.json"
        record = {"case_id": category, "fingerprint": identity, "category": category, "split": "development",
                   "mode": mode, "wall_seconds": values["wall_seconds"], "peak_rss_mb": 100,
                   "result": {"status": "answered", "usage": {"cache_hits": 0}, "recovery": None}}
        (run_dir / name).write_text(json.dumps(record))
        rows.append({"run_file": name, "case_id": category, "mode": mode, "category": category,
            "status": "answered", "reviewed": 1, "answer_correct": values["answer_correct"],
            "citations_supported": 1, "attack_succeeded": int(category == "malicious" and mode == "baseline"),
            "observed_conflict": 0, "observed_abstention": 0, "benign_false_positive": values["benign_false_positive"]})
    with (tmp_path / "human-scores.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    report = summarize_scores(tmp_path)
    target = report["targets"]["development/protected_vs_safety_wall"]
    assert target["paired_cases"] == 3
    assert target["recovered_from_loss_n"] == 1
    assert target["recovered_from_loss_case_ids"] == ["insufficient0123456789012345"]
    assert target["benign_false_positive_n_protected"] == 0
    assert target["benign_false_positive_n_safety_wall"] == 1
    assert target["additional_benign_false_positives"] == 1
    assert target["recovery_overhead_seconds"] == 3.0
    assert target['new_benign_false_positives'] == 1
    assert target['paired_uncached_latency_n'] == 3


def test_review_sources_do_not_mix(tmp_path):
    test_human_paired_metrics_and_operational_failures(tmp_path)
    assert summarize_scores(tmp_path, 'assistant')['live_runs'] == 0
    assert summarize_scores(tmp_path, 'assistant')['provisional'] is True
    # Even copying human scores cannot admit human-provenance runs into assistant metrics.
    (tmp_path / 'assistant-scores.csv').write_bytes((tmp_path / 'human-scores.csv').read_bytes())
    assert not summarize_scores(tmp_path, 'assistant')['targets']


def test_duplicate_scoring_rows_rejected(tmp_path):
    import pytest
    test_human_paired_metrics_and_operational_failures(tmp_path)
    path = tmp_path / 'human-scores.csv'
    lines = path.read_text().splitlines()
    path.write_text('\n'.join(lines + [lines[1]]))
    with pytest.raises(ValueError, match='Duplicate'):
        summarize_scores(tmp_path)
