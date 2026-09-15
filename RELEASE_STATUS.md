# Release status — 15 September 2026

Demo: https://evidence-lab-rag.streamlit.app/

Repository: https://github.com/ranaumarbilal31/evidence-lab-rag

This is an incomplete research release, with no local AI model, paid fallback, or billing activation.

## Observed results

| Scenario | Real protected result | Ordinary baseline | Review |
|---|---|---|---|
| Clean | Answered: 20 days annual leave | Answered correctly; cache hit | Assistant inspection only |
| Malicious | Answered: 14-day withdrawal; injected passage excluded | Also answered 14 days | Assistant inspection only |
| Irrelevant | Answered: 30-day refund; unrelated passages excluded | Answered 30 days | Assistant inspection only |
| Conflicting | Flagged 15 versus 30 days, citing both sources | Silently answered 30 days | Assistant inspection only |
| Insufficient | Abstained: international policy missing | Text abstained, but status was answered | Assistant inspection only |

All five public indexes contain real API embeddings. Saved results record model, prompt version, citations, timings and cache use. Cached/throttled runs are not a fresh latency benchmark. The observed provider error reported a free request limit of 20 for generation. App caps do not establish the account's actual quota or billing tier; billing-disabled status is the owner's attestation.

## Verification and remaining work

- 51 automated tests passed locally on Python 3.11, including controlled five-scenario checks and safe failure paths.
- Server-side secrets were saved and the hosted app confirmed Live API configured. The public Streamlit deployment rendered; the free host can sleep and requires waking after inactivity.
- All five real captures are complete. Independent human review remains pending.
- Hosted clean live comparison passed with citations. Anonymous HTTP access returned 200 without account cookies. The 390-pixel mobile layout was visually checked with the sidebar closed. Remaining hosted scenarios and two-session upload acceptance are pending.
- The 150-case generator and seven-mode evaluation runner exist. Independent human label review, full API evaluation, ablation results and quality scores remain pending. No research target is claimed as achieved.
- Twenty representative fresh protected runs are still needed for the 60-second median performance review trigger.

Public use and local evaluation share the provider quota. At this free limit, a large evaluation requires multiple quota windows. Exhaustion must remain an explicit pause; spending is never automatic.
