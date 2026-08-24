# Interactive Claim Extraction & Paraphrase Prompt Template

This is a standardized, copy-pasteable prompt template for operators performing interactive unstructured curation for X4 Advisor.

---

## Instructions for Operator

1. Open your LLM interface of choice (e.g. Claude 3.5 Sonnet, ChatGPT 4o, Gemini 1.5 Pro).
2. Copy the prompt below between the `--- PROMPT START ---` and `--- PROMPT END ---` markers.
3. Paste the prompt into the chat, replacing `[PASTE RAW SOURCE COMMUNITY GUIDE / WIKI TEXT HERE]` with the unedited text of the target X4 guide.
4. Save the LLM's outputs into local files under `data/sources/`:
   - **Pass 1 Output**: Save the extracted JSON claims array to `data/sources/<source_id>_c1.json`.
   - **Pass 2 Output**: Save the structured paraphrased Markdown text to `data/sources/<source_id>_p.md`.
   - **Pass 3 Output**: Save the re-extracted JSON claims array to `data/sources/<source_id>_c2.json`.

---

```markdown
--- PROMPT START ---
You are an expert X4: Foundations game analyst helping curate community knowledge for a local grounded advisor.

Your goal is to process the raw guide text provided below in three distinct passes to guarantee absolute factual fidelity and eliminate hallucination or epistemic drift.

### Raw Source Guide Input
"""
[PASTE RAW SOURCE COMMUNITY GUIDE / WIKI TEXT HERE]
"""

---

### Pass 1: Initial Typed Claim Extraction (C1)
Decompose the raw guide text into a list of discrete, typed factual claims.
Represent each claim as a JSON object with the following fields:
- `subject`: The specific entity or topic (e.g. "Cerberus Vanguard", "Claytronics Production", "Argon Prime").
- `predicate`: The attribute, metric, or relationship (e.g. "cargo_capacity", "min_price", "recommended_sector", "requires_ware").
- `object`: The claimed value, entity, or outcome (e.g. "1760", "2000", "True").
- `unit`: Measurement unit if applicable (e.g. "m3", "m/s", "credits"), or null if unitless.
- `qualifier`: Epistemic context, conditions, or attribution (e.g. "base game", "often recommended for miners", "the guide claims").

Output Pass 1 as a single valid JSON array block under a `### Pass 1: Claims JSON (C1)` header.

---

### Pass 2: Claim-First Paraphrase Generation (P)
Generate a clean, structured Markdown article **strictly from the Pass 1 claim set (C1)**.
Rules for Pass 2:
- Do NOT generate content directly from the raw source prose; rely exclusively on the facts captured in C1.
- Write clear, technical, non-conversational text.
- Organize the text under logical Markdown headers (`#`, `##`, `###`).
- Preserve all numbers, units, and entity names exactly as stated in C1.
- Preserve epistemic uncertainty (e.g., if C1 states "often recommended", do NOT state "is objectively best").

Output Pass 2 under a `### Pass 2: Paraphrased Article (P)` header.

---

### Pass 3: Re-Extracted Claim Set (C2)
Now, read **only the Pass 2 paraphrased article (P)** that you just generated, and re-extract the typed claim set.
Follow the exact same JSON schema as Pass 1 (`subject`, `predicate`, `object`, `unit`, `qualifier`).

Output Pass 3 as a single valid JSON array block under a `### Pass 3: Re-Extracted Claims JSON (C2)` header.
--- PROMPT END ---
```
