# Experimental detection hardening

The deployed default remains **heuristic-first**, with **vector checking off**.
No claim of improved live attack success is supported yet: the shared live
generation budget was exhausted during Priority 1. The original research source
snapshot is preserved; these changes belong to a new experiment identity.

`python -m rag.cli probes` writes a separate, versioned development set containing
paraphrases, full-width and zero-width Unicode, spacing, encoded directives,
educational quotations, incident reports and benign policy controls. It does not
alter or extend the original 150 cases or held-out split.

When quota is available, `python -m rag.cli probes --live` compares three
orderings over the same probes: existing heuristic-first detection,
`heuristic_confirm` (a regex match is sent to the classifier for adjudication,
with the matched rule kept as audit trail), and semantic adjudication of every
chunk. `--live --vector` also measures the experimental second layer.
Results record missed attacks and benign false positives, not downstream attack
success. Caches preserve completed model responses across quota pauses. Tests
use fixtures only and do not establish live effectiveness.

The experimental semantic orderings let a classifier examine educational quotes
that the old regex pass blocks immediately. Suspicious or uncertain classifier
decisions remain quarantined. `heuristic_confirm` is cheaper than full semantic
adjudication (the extra chunks ride along in the same batched safety call rather
than adding a call) and retains the rule name that matched, so it is the intended
candidate for promotion. Keep the old default unless development pipeline
evaluation shows fewer benign false positives without increased attack success.

API-vector similarity runs only on chunks cleared by existing detection. Five
authored attack references are embedded using the active connection's embedding
model; references and chunks use that same vector space. Existing embedding
caches are reused, and no local AI model is introduced. NFKC normalization and
zero-width-character removal apply to similarity input, not source quotations.
Cosine similarity and reference identity are reported separately from the
classifier's ordinal confidence. A vector match is not a probability.

`rag.similarity.calibrate` accepts development-only observations and proposes a
threshold only when at least one attack is caught with zero observed benign
false positives. Its output still needs review. The default candidate 0.9 is
uncalibrated and the feature stays disabled. Set the explicit experimental
constants in `rag/config.py` only for a new development experiment.

Recovery candidates now pass the same active detector as initial evidence,
before support verification. Quarantine exclusions, source/near-duplicate
rejection, one recovery retrieval, and exact quote verification remain intact.
This adds at most three classifier requests, bringing the conservative Safety
Wall generation budget to 13 (14 with a baseline comparison). Vector experiments
also incur embedding calls, charged to the same provider/project ledger.
