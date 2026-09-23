# Model and quota review — 23 September 2026

The shared default remains `gemini-3.6-flash`. A real, small development probe
returned the correct 12-day domestic refund deadline. The same probe through
the existing SDK configuration failed for `gemini-3.5-flash-lite`; neither
acceptable compatibility nor higher project-specific daily capacity was established.
No pipeline change or model substitution was made for this review.

Google's [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits)
states that RPM, TPM and RPD are project/model limits, not per-key allowances.
The public tables do not confirm this project's active limits. Existing local
budgets (generation 5 RPM / 60,000 TPM / 20 RPD; embedding 10 RPM / 60,000 TPM /
100 RPD) are **unverified application budgets**, not promises of provider capacity.
Their historical basis is the previous owner's configuration and a reported
20-generation-request limit, not fresh AI Studio verification.

`FREE_TIER_CONFIRMED` and the private key were preserved. Billing was not enabled
or independently reverified. Live tests use the existing owner's attestation.
Historical captures retain their original model, timestamp and review status.
Re-running `prepare-demo --capture` skips existing captures; it does not validate
the new model by regeneration. A fresh full capture set remains pending and must
be staged, inspected and archived before replacing historical assets.
