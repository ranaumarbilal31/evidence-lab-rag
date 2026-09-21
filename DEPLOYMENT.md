# Required free online demo

Target: **Streamlit Community Cloud**, an HTTPS `streamlit.app` URL. The app must run independently of the developer's computer. No paid hosting or custom domain is needed.

## Owner prerequisites

Owner Gemini secrets are required only for the shared demo. Personal-key mode works without them. Install the updated `requirements.txt`, including the bounded DOCX/XLSX extraction dependencies. No visitor keys belong in deployment secrets, environment variables, source control, or persistent storage.

- A GitHub account and a repository you control.
- Streamlit Community Cloud sign-in with access to that repository. Any account/terms acceptance or access grants must be completed by the owner.
- For shared mode only: Gemini Free-tier credentials and active per-model limits. Owner billing remains disabled.

The app is deployed at **https://evidence-lab-rag.streamlit.app/** from `ranaumarbilal31/evidence-lab-rag`, branch `main`, entrypoint `app.py`. See [RELEASE_STATUS.md](RELEASE_STATUS.md) for completed checks and pending acceptance work.

## Prepare and verify

1. Follow README setup and run the tests.
2. Set local secrets privately and run `python -m rag.cli prepare-demo --capture`. Review all five real captures, setting `human_reviewed=true` only for captures that have been inspected.
3. Run `python -m rag.cli preflight`. It rejects absent/incompatible indexes and missing/unreviewed live captures. It checks local prerequisites, not hosted availability or the truth of an owner attestation.
4. Run `python -m rag.cli package` to create `dist/evidence-lab-deploy.zip`. It contains only allowlisted source/config/sample assets, never `.streamlit/secrets.toml`, `.env`, local research data, virtual environments, or caches. Inspect the file list before uploading. The example secrets file has no key.
5. Place the extracted source in your GitHub repository. Do not upload the parent workspace wholesale. Deploy from a stable commit; record its ID for rollback.

## Publish

In [Streamlit Community Cloud](https://share.streamlit.io/), create an app from the repository, branch, and entrypoint `app.py`. Python 3.11 is the locally tested version. The deployed host used the default Python 3.14 selection and rendered successfully; the complete test suite has not been run on that host runtime. Dependencies come from `requirements.txt`; development tools are unnecessary on the host.

Paste the configured secret values in Community Cloud **Secrets**, never into GitHub. Check that the Gemini key still belongs to a billing-disabled Free-tier project. Deploy and copy the actual assigned HTTPS URL. Do not invent a URL or present a local preview as the hosted result.

Community Cloud's filesystem is not durable. Demo embeddings and captures are versioned read-only assets; visitor uploads/indexes are private session memory. No startup embedding calls or hidden keep-alive traffic are used. App counters may reset on restart; provider quotas are authoritative. [Hosting](https://docs.streamlit.io/deploy/streamlit-community-cloud), [storage limits](https://docs.streamlit.io/develop/concepts/connections/connecting-to-data).

## Required hosted acceptance checks

- With shared credentials absent or shared quota exhausted, connect a compatible personal key and run Standard RAG, Detect & Block, and Full Safety Wall, including a JSON upload and source citations.
- Check Gemini and OpenAI with actual account/model access. For a custom endpoint, verify both required API operations. Check that unsupported Anthropic keys receive an explanation rather than a generation-only connection.
- Verify two personal sessions remain isolated and disconnect/reconnect removes old indexes/results. Changing the provider must require reconnecting and rebuilding indexes.
- Verify invalid credentials, missing model access, insufficient credit, and rate limits produce clear messages without exposing response bodies or credentials. Connection probes use synthetic text and may incur charges.
- Inspect upload previews for JSON/JSONL, CSV/TSV, DOCX, and XLSX. Verify provenance in citations and readable rejection of complex/unsupported content.

- Open the public link in a signed-out browser; no visitor account, key, or local software is needed.
- Run all five sample questions using the live API. Check baseline comparison, citations, exclusions, conflict behavior, and missing-evidence behavior. Save observed outcomes; failures remain failures.
- Open two independent browser sessions. Upload a distinct small public text file to each and confirm that documents, answers, and download contents do not cross sessions. Clear one session and verify the other remains intact.
- Test file/type/page/chunk limits and new-question throttling. Confirm excess concurrent work gets a clear retry response.
- Verify no secret appears in page text, rendered HTML, downloadable results, logs under app control, or the repository.
- Check a narrow/mobile viewport. Make sure questions, source passages, warnings, and comparisons remain usable.
- Restart the app and verify the bundled examples load without API embedding calls. Loss of temporary uploads is expected and disclosed.
- On API failure/quota exhaustion, custom live requests stop; saved results are explicitly marked with capture time/model and never substituted for a different question.
- Confirm the public link works while the developer's computer is off.

Free hosts may sleep and APIs may have outages; this is not an always-awake availability promise. Record the public URL and acceptance results in a release note only after they exist. If a release fails, redeploy the previous known-working commit with its compatible index/capture bundle. Check Community Cloud logs and Gemini quota dashboards for actual failures; no paid monitoring is required.
