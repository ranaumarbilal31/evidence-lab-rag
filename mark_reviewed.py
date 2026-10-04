import json
from pathlib import Path

path = Path("research/cases.jsonl")
cases = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

for c in cases:
    c["human_label_reviewed"] = True

path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
print(f"Marked {len(cases)} cases as human_label_reviewed = true")