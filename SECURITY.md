# Security

This is a local, offline, single-user application — not a hosted service, so most conventional web-app security concerns don't apply. A few things worth knowing anyway:

- **Don't share save files or extracted game data that you consider sensitive.** They're never committed to this repository by design, but if you're troubleshooting in an issue, keep that in mind before pasting file contents.
- **Retrieved content (from the curated knowledge base) is always treated as untrusted data, never as instructions** — this is an architectural principle of the retrieval/synthesis design, not just a note here.
- **Never commit API keys, tokens, or credentials.** None should be required for local operation.
- If you find a genuine security issue, please open an issue or contact the maintainer directly rather than assuming urgency — this is not a production service with an incident response process.
