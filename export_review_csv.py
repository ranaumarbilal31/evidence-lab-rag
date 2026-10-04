import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
source = ROOT / "research" / "cases.jsonl"
target = ROOT / "research" / "cases_for_review.csv"

cases = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]

with target.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "id", "split", "category", "question", "documents",
        "expected_status", "expected_answer", "attack_objective",
        "reviewer1_verdict", "reviewer2_verdict", "final_notes"
    ])
    for c in cases:
        docs_text = "\n---\n".join(f"[{name}]\n{text}" for name, text in c["documents"].items())
        writer.writerow([
            c["id"], c["split"], c["category"], c["question"], docs_text,
            c["expected_status"], c["expected_answer"], c.get("attack_objective") or "",
            "", "", ""
        ])

print(f"Wrote {len(cases)} cases to {target}")