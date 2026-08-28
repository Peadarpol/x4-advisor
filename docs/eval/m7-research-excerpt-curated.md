# Curated Research Excerpt — Milestone M7 In-Scope Interventions Only

*Extracted from `docs/eval/deep_research_response_model_tuning.md`. This excerpt covers only the three interventions scoped into M7 (router JSON-schema constraints, BASE/DLC/UNKNOWN entity resolution, bounded-claims synthesis). Context distillation (Pillar 1), MiniCheck/verification cascades (§4.8–4.13), and speculative decoding are deliberately omitted — those are explicitly deferred, per the M7 plan, to a later milestone evaluated against real demonstrated need. See the full response document for that material if it becomes relevant later.*

---

## Intervention 1 — Router JSON-Schema Constraints (targets Granite's 77.8% routing accuracy)

### 4.1 What grammar guarantees

Suppose you define:

```json
{
  "type": "object",
  "properties": {
    "route": {
      "enum": ["STRUCTURED", "VECTOR", "BOTH", "ABSTAIN"]
    }
  },
  "required": ["route"],
  "additionalProperties": false
}
```

The constrained decoder can guarantee: valid JSON + route is one of the four permitted strings + required field exists + no additional fields.

It cannot guarantee: "Grand Exchange I is the correct entity." That's semantic validation. This distinction is fundamental.

### 4.3 Ollama implementation

Use Ollama's JSON Schema interface rather than manually managing GBNF unless you specifically need grammar-level control.

```python
from ollama import chat

response = chat(
    model="granite4.1:8b",
    messages=[
        {"role": "system", "content": ROUTER_PROMPT},
        {"role": "user", "content": user_query},
    ],
    format=ROUTER_SCHEMA,
    options={"temperature": 0},
)

route = Router.model_validate_json(response.message.content)
```

### 4.4 Don't put the whitelist entirely into the grammar

This is a common architectural mistake. Imagine 30,000 possible X4 ware names. Don't create a grammar enumerating every ware name — that creates an enormous grammar and makes maintenance unpleasant.

Instead: LLM → JSON schema → `"lookup_ware"`, `"requested_name": "Hull Parts"` → Python → SQLite canonical lookup → canonical ID.

**The application owns the whitelist. The grammar owns the syntax.**

### 4.5 Strongest architecture for tool parameters

Instead of `{"tool": "lookup_ware", "ware": "Hull Parts"}`, prefer `{"tool": "lookup_ware", "query": "Hull Parts"}`, then:

```python
candidate = normalize(tool_call["query"])
entity = db.resolve_exact_or_alias(entity_type="ware", query=candidate)
if entity is None:
    reject_tool_call()
```

The model cannot directly dictate SQL. This eliminates an entire class of hallucinated parameters.

### 4.6 Granite-specific recommendation

`temperature = 0`, structured output ON, `additionalProperties = false`, enum ON, max generation tokens very low. Then: schema validation → domain validation → parameter validation → SQL parameterization.

IBM specifically positions Granite 4.1 as improved for tool calling and instruction following. The current 77.8% routing accuracy should not be interpreted as "Granite isn't capable of routing" — interpret it as "Granite is being given too much freedom in a task that has a finite formal language." Constrain the output and validate the arguments.

### 4.7 Latency of constrained decoding

llama.cpp's LLGuidance integration reports approximately 50 μs average CPU time per token for JSON-schema token masking, with 0.5 ms p99. That's effectively irrelevant beside a 10–30 second LLM generation. **Use constrained decoding aggressively for routing — there is very little reason not to.**

---

## Intervention 2 — BASE/DLC/UNKNOWN Entity Resolution (targets Qwen's 50% abstention accuracy)

This is a knowledge representation problem, not primarily a prompt problem. Qwen's behavior — unfamiliar canonical entity → assumes DLC — is exactly what you'd expect from an LLM being asked to infer ontology membership from priors. **The fix: don't ask the model to infer ontology membership.**

### 3.1 Make universe membership deterministic

Your SQLite database should contain something like:

```sql
CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    game_version TEXT NOT NULL,
    availability TEXT NOT NULL
        CHECK (availability IN ('BASE', 'DLC', 'UNKNOWN')),
    dlc_id TEXT
);
```

Then: Grand Exchange I → SQLite → BASE. **The router never has to decide whether it's DLC.**

### 3.2–3.3 Three-state epistemic model

Add a `resolve_entity(name)` tool with output like:
```json
{"canonical_name": "Grand Exchange I", "entity_type": "sector", "availability": "BASE"}
```

Use **BASE / DLC / UNKNOWN**. Never just BASE / DLC — because "not known to be BASE" does not imply "DLC." This is exactly the failure mode observed with Qwen.

### 3.4 Router prompt (entity scope rule)

```
ENTITY SCOPE RULE
=================
Never infer DLC status from unfamiliarity.
Unknown entity != DLC.
Use the supplied canonical entity registry when determining scope.

If an entity is:
- BASE: it is in scope.
- DLC: abstain only if that DLC is outside the configured supported scope.
- UNKNOWN: do not classify it as DLC merely because you do not recognize it.
```

### 3.5 Resolve entities before routing

```
User → cheap entity extraction → SQLite entity resolution → Router
```

```python
entities = entity_match(user_query)
resolved = [db.resolve_entity(e) for e in entities]
```

The router then receives an `ENTITY REGISTRY` block (e.g. "Grand Exchange I / type: sector / scope: BASE") — the LLM doesn't need to know that from pretraining.

### 3.6 Few-shot demonstrations — include negative pairs deliberately

Don't provide only "DLC example → DLC." That teaches a dangerous surface pattern. Include:
```
"Tell me about Grand Exchange I." → BASE
"Tell me about Kingdom End." → DLC   [CORRECTED: Kingdom End is a real DLC expansion,
                                       not base-game content — see eval_dlc_002 in the
                                       M6 evaluation corpus, which expects
                                       OUT_OF_SCOPE_DLC for this exact entity. The
                                       original ChatGPT response incorrectly listed this
                                       as a BASE example; that error was caught during
                                       M7 plan review (F16) and is corrected here.]
"Tell me about [known DLC entity]." → DLC
"Tell me about [fictional/unknown entity]." → UNKNOWN
```

The important lesson: **unfamiliar ≠ DLC** — not "known = base, unknown = DLC."

---

## Intervention 3 — Bounded-Claims Structured Synthesis (targets Gemma's 7.9% UCR and 28.6s latency)

### The core reframe

Your current 7.9% UCR is particularly informative because most unsupported material is connective/framing language. That means you shouldn't primarily tell Gemma "don't hallucinate." Instead: **remove the opportunity to generate unsupported language.**

Bad: *"Answer the user's question using the information below."*
Better: *"Generate only statements that are directly supported by EVIDENCE."*
Best: *"Generate a list of claims. Each claim MUST be supported by one or more evidence IDs. Claims without evidence IDs are invalid."*

That creates a mechanical relationship (claim → evidence) rather than (answer → hopefully evidence-grounded).

### 2.2 Recommended synthesizer schema

```python
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "evidence": {
                        "type": "array",
                        "items": {"type": "string"},
                        "minItems": 1,
                        "maxItems": 3
                    }
                },
                "required": ["text", "evidence"],
                "additionalProperties": False
            }
        }
    },
    "required": ["claims"],
    "additionalProperties": False
}
```

### 2.3 Recommended system prompt (test this first)

```
You are the factual synthesis component of an offline X4: Foundations
knowledge system.

Your task is NOT to chat, explain your reasoning, introduce the answer,
or provide general background.

You may ONLY produce claims supported by the supplied EVIDENCE.

GROUNDING RULES
1. Every claim MUST be supported by at least one EVIDENCE ID.
2. Do not use world knowledge, model memory, assumptions, or likely facts.
3. If the evidence does not establish a fact, do not state it.
4. Do not infer missing quantities, relationships, causes, game mechanics
   not explicitly supported, DLC membership, faction relationships, or
   numerical values.
5. Preserve all numerical values exactly.
6. Preserve negation exactly. "does not", "cannot", "never", "without",
   "only", and "except" MUST NOT be removed or reversed.
7. Do not add introductory sentences such as "Sure", "Certainly", "In X4",
   "Based on the information provided", "Here is the answer", or similar
   framing.
8. Do not add conclusions unless the conclusion itself is explicitly
   supported by evidence.
9. Do not combine two facts into a new causal or comparative claim unless
   the evidence explicitly supports that relationship.
10. If evidence is insufficient, return fewer claims.
11. It is preferable to return no claim rather than an unsupported claim.
12. Maximum 6 claims.
13. Each claim should normally be one sentence.
14. Keep claims concise.

OUTPUT CONTRACT
Return ONLY the JSON object defined by the supplied schema.
No markdown. No prose outside JSON. No reasoning. No commentary.

EVIDENCE
{evidence}

USER QUESTION
{question}
```

This is intentionally repetitive. For a 12B model, redundancy in the control instruction is often cheaper than ambiguity in the output space.

### 2.4 Attribution without generation overhead

Use a hidden/structured evidence mapping rather than having the model quote sources (which increases generation length):

```json
{"claims": [{"text": "A Hull Part production cycle requires 1 Graphene and 1 Energy Cell.", "evidence": ["E17"]}]}
```

The verifier inspects E17; the user sees only the rendered text. Attribution without quotation overhead.

### 2.5 Nonce anchoring — a caution

Nonce presence ≠ semantic entailment. A model can cite a nonce that doesn't actually support its claim. The nonce is the address; the verifier still needs to inspect the content, not just check that a citation exists.

### 2.6 Strict propositional budget

Set `maximum claims = 6`, `maximum one proposition per claim` — rather than "write a concise answer." **"Concise" is a semantic instruction. "Maximum 6 claims" is a structural constraint**, and structural constraints are substantially easier for small models to follow.

### 2.7 Thinking mode

Verify reasoning/thinking is off for this synthesis stage unless testing proves it materially improves grounding. Router: no thinking. Synthesizer: no thinking initially.

### 2.8 Temperature

Start at `temperature = 0` for the invariant-heavy workload. Ollama itself recommends low temperature for more deterministic structured output.

### Output token control (biggest expected latency win)

Hard generation budget, e.g. `max_new_tokens = 128–160`. Six short claims do not require 400 tokens. **The system should never reward the model for filling the context window.**

If the model hits the limit: truncate → verifier, or preferably regenerate with fewer claims.

---

## Instrumentation — measure before assuming the fix worked

Add these fields to every bake-off test record, so the before/after effect is measured, not assumed:

```json
{
  "model": "gemma4:12b",
  "query_id": "Q17",
  "route": "BOTH",
  "retrieval_chunks": 5,
  "retrieval_tokens": 2037,
  "prompt_tokens": 921,
  "output_tokens": 74,
  "prompt_eval_ms": 1420,
  "generation_ms": 9210,
  "input_tok_per_sec": 1440,
  "output_tok_per_sec": 8.0,
  "claims": 4,
  "supported_claims": 4,
  "unsupported_claims": 0,
  "contradicted_claims": 0
}
```

The core hypothesis to confirm or refute empirically: Gemma's 28.6s latency is dominated by **output-token generation**, not input/context processing — autoregressive generation is sequential while prompt ingestion is comparatively parallel. This should be measured, not assumed.
