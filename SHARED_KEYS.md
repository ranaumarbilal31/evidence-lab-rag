# Add your own shared API keys

The owner pool supports **up to 10 Gemini keys**. Add them privately through
`.streamlit/secrets.toml` locally, or the deployed app's Streamlit Secrets settings.
Do not put real keys in `secrets.example.toml`, GitHub, screenshots, or chat.

## Local setup

1. Open `.streamlit/secrets.toml`. This workspace already has ten disabled, blank
   `[[shared_keys]]` slots. Fill those slots instead of appending more. On a new
   checkout, copy `secrets.example.toml` to that path first.
2. For each key, enter the actual Google Cloud project ID and the generation and
   embedding budgets. Check the active limits for both configured models in your
   AI Studio project. Application budgets can be lower than those limits.
3. Set `free_tier_confirmed = true` only after verifying that project's billing
   is disabled. Set `enabled = true` when the slot is ready.
4. Save the file and restart Streamlit to load the updated configuration.

Example slot (blank and disabled deliberately):

```toml
[[shared_keys]]
api_key = ""
project_id = "your-actual-google-project-id"
enabled = false
free_tier_confirmed = false
gen_rpm = 0
gen_tpm = 0
gen_rpd = 0
embed_rpm = 0
embed_tpm = 0
embed_rpd = 0
```

Keep global settings such as `RESEARCH_PAUSE_SHARED` **above the first
`[[shared_keys]]` block**. TOML fields below a block belong to that slot.
Zero budgets keep a slot unavailable. Duplicate active keys and more than ten
slots are rejected. To remove a key from service, set its slot's `enabled` to
`false` and restart; revoke compromised keys in the provider console too.

## How quota and fallback work

Keys from the same Google project must use the same `project_id`: they share
RPM, TPM and daily counters. Ten keys in one project do not give ten times the
quota. If budgets disagree within a project, the smaller budgets apply.
Eligible configured projects can take over when another project is exhausted;
all use the same generation model, embedding model and dimensions.

Any enabled pool entries replace the legacy single-key configuration. When
all slots are disabled, the existing `GEMINI_API_KEY` configuration is used.
When all eligible pool capacity is exhausted, the interface directs visitors
to the existing personal-key connection. Personal requests never rotate into
the owner pool or another provider.

The owner pool is Gemini-only. OpenAI and custom OpenAI-compatible connections
remain available through the session-only **Use my API key** interface.

## Hosted setup and privacy

Paste the private configuration into your deployed app's Streamlit Secrets
settings and restart the app. Editing the local file does not update hosting.
Local and hosted counters are approximate and separate; provider quotas are
authoritative. Set hosted `RESEARCH_PAUSE_SHARED = true` while reserving shared
capacity for owner evaluation. Saved demos and personal connections remain usable.

`.streamlit/secrets.toml` is excluded from Git and deployment archives. Never
force-add it. Use the committed example and this guide to share configuration
instructions; keep the filled file private on a trusted machine.
