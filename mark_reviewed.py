"""Mark specific cases as human-reviewed. Requires an explicit case-id list.

`human_label_reviewed` is the gate that lets a `--review-source human` evaluation
proceed, so it must never be applied in bulk by accident. This deliberately has no
"mark everything" mode: pass the exact ids you have actually checked against the
case's question, documents, expected answer/status and attack objective.
"""
import json
import sys
from pathlib import Path

path = Path("research/cases.jsonl")
if len(sys.argv) < 2:
    raise SystemExit(
        "Usage: python mark_reviewed.py CASE_ID [CASE_ID ...]\n"
        "Refusing to mark cases that were not named. Review them first."
    )
if not path.exists():
    raise SystemExit(f"{path} not found. Generate the dataset first: python -m rag.cli dataset")

requested = set(sys.argv[1:])
cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
unknown = requested - {c["id"] for c in cases}
if unknown:
    raise SystemExit(f"Unknown case ids: {', '.join(sorted(unknown))}")

marked = 0
for case in cases:
    if case["id"] in requested and case.get("human_label_reviewed") is not True:
        case["human_label_reviewed"] = True
        marked += 1

path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
remaining = sum(c.get("human_label_reviewed") is not True for c in cases)
print(f"Marked {marked} case(s) as human_label_reviewed = true; {remaining} still unreviewed.")
