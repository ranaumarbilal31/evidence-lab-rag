# Reproducible evaluation

The default review source remains `human`. Assistant review is explicitly
provisional and never changes `human_label_reviewed`. Label reviews live in
`research/assistant-label-reviews.jsonl`, bound to each complete case fingerprint.
Disputed, missing or stale reviews are excluded and listed with counts.

Run one command at a time:

```powershell
python -m rag.cli evaluate --split development --mode baseline --review-source assistant
python -m rag.cli evaluate --split development --mode protected --review-source assistant
python -m rag.cli evaluate --split development --mode safety_wall --review-source assistant
```

Each prints a manifest ID. Unchanged code/data/settings reuse the same experiment;
changed inputs create a different directory under `research/experiments/`.
The manifest contains no credentials. Source and dataset snapshots, retrieval,
mode caches, results and scores remain private and excluded from deployment.
The owner commands share an OS process lock; it releases even on process exit.

```powershell
python -m rag.cli export-scores --review-source assistant --manifest MANIFEST_ID
python -m rag.cli summarize --review-source assistant --manifest MANIFEST_ID
python -m rag.cli freeze --manifest MANIFEST_ID
python -m rag.cli evaluate --split held_out --mode baseline --review-source assistant --manifest MANIFEST_ID
python -m rag.cli evaluate --split held_out --mode protected --review-source assistant --manifest MANIFEST_ID
python -m rag.cli evaluate --split held_out --mode safety_wall --review-source assistant --manifest MANIFEST_ID
```

Fill the exported `assistant-scores.csv` only after reading each complete output,
its original documents and the README's scoring definitions. Leave disputed rows
unreviewed. Human reviewers instead use `human-scores.csv` and `--review-source human`.
Exports preserve existing scores. No automated label matching is a quality score.

Freeze requires all 150 development runs (50 cases × three modes) and reviewed
development score rows. Held-out runs reject changes to the frozen code, dataset,
models, budgets or review source. Do not tune against held-out outcomes.

Summaries report paired attack reduction and clean accuracy; recovery gains and
regressions; new false positives separately from net change; and median paired
uncached latency differences. Zero baseline attack success makes relative
reduction undefined, not a pass. Timings include rate-limit waits. Missing data
and operational failures do not count as success or model abstention.

Set `RESEARCH_PAUSE_SHARED=true` on the hosted site before spending its project's
quota locally. The flag is local to each deployment; changing a local file cannot
pause a remote host. Provider quotas are authoritative across machines. A full
run may need multiple daily resets. Saved demos and personal connections remain
available while shared live processing is paused.
