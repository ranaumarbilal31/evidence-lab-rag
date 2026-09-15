# Research limitations and current verification status

This system checks evidence; it does not certify truth or guarantee prompt-injection protection.

## Important limits

- The same hosted model performs semantic screening and answering. It can make correlated mistakes and can be influenced by adversarial content. Typed JSON and role separation are useful boundaries, not security proofs.
- Rules deliberately flag certain explicit instruction patterns, including some legitimate educational quotations. The test suite documents that false positive rather than hiding it.
- Contradictions depend on matching subject, scope, and conditions. The app does not independently authenticate document authors, check dates against an authority, or establish which conflicting policy is current.
- Exact citation checks establish that quoted text exists in the accepted source. They do not prove that the quote is true or entails every answer claim. The semantic verifier remains fallible.
- A conservative input budget can exclude retrieved chunks, including a conflicting source. The UI discloses these omissions. Retrieval quality and answer quality need separate interpretation.
- The public upload parser has size/page/chunk and PDF-stream checks but is not an operating-system sandbox. Host resource limits and malformed documents can still cause temporary unavailability. Do not use this prototype as a production document service.
- Anonymous session quotas deter casual overuse but can be bypassed by opening new sessions. Process-wide limits reduce load; provider quotas and billing-disabled projects enforce the $0 service boundary.
- Hosted storage and counters reset on restart. Visitor documents are temporary. Data retention by Streamlit/Google is governed by those services; the app cannot promise end-to-end deletion.
- The free API may throttle, change availability, or decline a model request. The app never substitutes a paid service, rotates accounts, or runs a local model.
- Synthetic cases use repeated templates. Family-disjoint splits reduce near-duplicate leakage but are not a comprehensive adversarial benchmark. Generator code is public and the split is not a secret benchmark.
- Cached/resumed runs are suitable for checkpointing functional outcomes, but not for unqualified latency comparisons. Independent fresh-run benchmarking would require fresh caches and sufficient quota.
- Local quota ledgers are single-process. Hosted process quotas and local evaluation do not share durable counters; the API provider enforces the common account limits.

## Results that must not be claimed yet

Fallback expected-behavior illustrations are manually authored. Unit tests inject controlled API responses. Neither establishes live model quality. Three real API captures and all five embedding indexes now exist; captures have not received independent human review.

The online deployment exists at https://evidence-lab-rag.streamlit.app/. Two remaining live captures are blocked by the current free quota. Human label review, full baseline/ablation outcomes, and complete hosted-browser acceptance are pending. The baseline also resisted the captured injection example; that example establishes no relative attack-success reduction. Do not claim research targets were met, 100% security, or independent human validation.

Targets remain: at least 50% relative attack-success reduction when baseline attacks succeed, at least 80% conflict and insufficiency recall, and no more than a 10-percentage-point clean-answer accuracy loss. Report sample counts, operational failures, false positives, and confidence limitations alongside results.
