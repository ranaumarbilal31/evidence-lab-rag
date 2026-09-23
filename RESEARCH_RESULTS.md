# Research checkpoint — 23 September 2026

**Partial real development results; assistant review only. No security target is claimed.**

All 150 original synthetic case labels were inspected against their questions,
documents, expected behavior, supporting files and attack objectives. Separate
assistant review records contain per-case rationale and content fingerprints.
All human-label flags remain false. There were no disputed expected labels.

The real baseline run completed 18 development cases before the application cap
of 20 generation attempts was reached (two attempts were retries). Each result
was inspected and scored separately from human scores. Answers were correct in
18/18; citation support was 17/18. One conflict response correctly identified
both deadlines but gave no supporting citations. All four observed attacks
failed in this small baseline subset. These are not general robustness estimates.

Protected and Safety Wall development commands both paused at the same exhausted
generation budget. Held-out runs were deliberately not started: development is
incomplete and the freeze prerequisite is not met. The original 18-run source
snapshot and scores are retained in the private experiment directory; subsequent
tooling fixes have a separate identity and cannot be paired with that pilot.

| README question | Current answer |
|---|---|
| Attack-success reduction and clean-accuracy cost | Not measurable: zero paired baseline/protected results. |
| Recovery value | Not measurable: no protected/Safety Wall pairs. |
| Recovery's new benign false positives | Not measurable: no paired results. |
| Recovery latency cost | Not measurable: no paired uncached results. |

The suite verifies the reporting mechanics with fixtures; those fixtures are not
research observations. Full evaluation and independent human validation remain
pending quota and reviewer availability. The approved adaptation allows UI and
experimental hardening work to proceed with this limitation disclosed.
