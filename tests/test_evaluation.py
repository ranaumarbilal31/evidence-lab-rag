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
