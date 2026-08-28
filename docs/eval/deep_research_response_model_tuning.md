I treated your benchmark numbers as the empirical baseline rather than independently reproducing them, because the hardware details needed to reproduce the wall-clock measurements—GPU model, CUDA/driver version, Ollama version, context size, prompt token counts, generation token counts, and sampling parameters—aren't specified. Where the literature gives measured results, I distinguish those from projections for your 12 GB/Q4_K_M setup.

The central conclusion is fairly strong:

Do not attack this primarily as a "better RAG prompt" problem. Your largest gains will come from changing the representation and control surface between retrieval and synthesis: deterministic fact selection → protected factual records → short bounded synthesis → structural output constraints → cheap claim verification.

In particular, I would not initially put LLMLingua-2 in the hot path. Your retrieved context is only ~2,000 tokens. The more important anomaly is Gemma's 28.6-second generation: that strongly suggests output-token explosion, not merely context-processing cost. Compressing 2,000 → 400 input tokens cannot by itself solve a 5× generation-time increase if the model is producing hundreds of unnecessary output tokens.

Executive architecture recommendation

I would change the current pipeline from:

User
  │
  ▼
Router LLM
  │
  ├── SQL
  ├── Vector search
  └── BOTH
          │
          ▼
    4–5 raw chunks
          │
          ▼
    Gemma Synthesizer
          │
          ▼
       Answer

to:

                         ┌───────────────┐
                         │ Canonical DB  │
                         └───────┬───────┘
                                 │
User ──► Router ──► retrieval ──►│
                                 │
                         ┌───────▼────────┐
                         │ Fact compiler   │
                         │ deterministic   │
                         └───────┬────────┘
                                 │
                    300–600 token evidence
                    with protected values
                                 │
                         ┌───────▼────────┐
                         │ Gemma 4 12B    │
                         │ bounded output  │
                         └───────┬────────┘
                                 │
                       structured claims
                                 │
                  ┌──────────────▼──────────────┐
                  │ deterministic + MiniCheck   │
                  │ claim verification           │
                  └──────────────┬──────────────┘
                                 │
                           render answer

And, critically:

             MODEL
              │
              ▼
       "What should I say?"
              │
              ▼
     ┌─────────────────┐
     │ deterministic   │
     │ verifier        │
     └─────────────────┘
              │
       ┌──────┴──────┐
       │             │
    supported     unsupported
       │             │
     render       delete /
                  regenerate

This makes the model responsible for language synthesis, rather than making it responsible for determining whether its own language is factual.

Pillar 1 — Context distillation
1.1 The most important distinction: compression vs selection

There are actually three different operations being conflated under "context compression":

A. Retrieval pruning
2,000 tokens
    ↓
select relevant sentences
    ↓
800 tokens

No semantic rewriting occurs.

B. Extractive compression
2,000 tokens
    ↓
remove low-value words/sentences
    ↓
500 tokens

Still preserves original wording.

C. Abstractive compression
2,000 tokens
    ↓
new representation
    ↓
400 tokens

This can introduce information that wasn't actually present.

For a game knowledge system with hard grounding requirements, A+B are much safer than C.

LLMLingua-2 is particularly interesting because it treats compression as token classification rather than simply asking another LLM to summarize the context. Its reported compression ratios are 2–5×, with 1.6–2.9× end-to-end speedups and a 3–6× faster compressor than earlier LLMLingua approaches.

But those results don't establish that LLMLingua-2 is optimal for your 2,000-token retrieval window.

In fact, I expect a simpler method to win.

1.2 Recommended "Fact Compiler"

Before involving an additional neural compressor, turn the retrieved material into atomic evidence units.

For example:

SOURCE: chunk_17
ENTITY: Hull Part
FACT:
  "Hull Part production requires 1 Graphene and 1 Energy Cell."

Rather than:

In X4, Hull Parts are a common intermediate component...

Your compiler should identify:

entity
subject
predicate
object
numerical values
units
qualifiers
negations
source
confidence
provenance

A useful intermediate representation is:

from dataclasses import dataclass
from typing import Optional

@dataclass
class Fact:
    subject: str
    predicate: str
    object: str
    source_id: str

    numbers: tuple[str, ...] = ()
    units: tuple[str, ...] = ()
    negated: bool = False

    original_text: str = ""

Then your synthesis prompt receives:

[EVIDENCE E17]
subject=Hull Part
predicate=requires
object=Graphene
quantity=1
source=chunk_17

[EVIDENCE E18]
subject=Hull Part
predicate=requires
object=Energy Cell
quantity=1
source=chunk_17

[EVIDENCE E19]
subject=Hull Part
predicate=production_time
value=60
unit=seconds
source=chunk_17

This is considerably safer than asking Gemma to digest five prose chunks.

1.3 Numerical and negation preservation

I would make this a hard invariant, not a prompting aspiration.

Before compression:

import re

NUMBER_RE = re.compile(
    r"""
    (?<!\w)
    [-+]?
    (?:\d+(?:\.\d+)?|\.\d+)
    (?:\s*[%x×]|)
    (?!\w)
    """,
    re.VERBOSE | re.IGNORECASE,
)

NEGATION_RE = re.compile(
    r"\b(?:not|never|no|cannot|can't|without|doesn't|does not|"
    r"neither|nor|only|except|unless)\b",
    re.IGNORECASE,
)

Extract a protected signature:

def evidence_signature(text: str):
    numbers = tuple(NUMBER_RE.findall(text))
    negations = tuple(
        m.group(0).lower()
        for m in NEGATION_RE.finditer(text)
    )

    return {
        "numbers": numbers,
        "negations": negations,
    }

Then reject any compressed representation whose signature doesn't cover the original:

def compression_safe(original: str, compressed: str) -> bool:
    a = evidence_signature(original)
    b = evidence_signature(compressed)

    return (
        set(a["numbers"]) <= set(b["numbers"])
        and set(a["negations"]) <= set(b["negations"])
    )

That is deliberately conservative.

For X4, I'd go further:

Never compress these tokens
numbers
units
percentages
time values
quantities
ware names
ship names
faction names
sector names
DLC names
explicit negation
comparative operators
only
except
requires
does not
cannot

In other words, compression should operate on connective language, not game facts.

1.4 Sentence-level relevance scoring

Before LLMLingua-2, I would test a virtually free baseline:

from rank_bm25 import BM25Okapi

def select_sentences(query, chunks, budget=500):
    candidates = []

    for chunk_id, text in chunks:
        sentences = split_sentences(text)

        for sentence in sentences:
            candidates.append((chunk_id, sentence))

    corpus = [s.lower().split() for _, s in candidates]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query.lower().split())

    ranked = sorted(
        zip(scores, candidates),
        reverse=True
    )

    selected = []
    tokens = 0

    for score, (chunk_id, sentence) in ranked:
        n = estimate_tokens(sentence)

        if tokens + n > budget:
            continue

        selected.append((chunk_id, sentence))
        tokens += n

    return selected

But don't use BM25 alone.

Use a weighted score:

score =
    0.40 lexical relevance
  + 0.30 embedding relevance
  + 0.20 entity overlap
  + 0.10 numerical/constraint relevance

Then always include sentences containing protected facts.

1.5 Query-aware atomic proposition extraction

For your 53-chunk corpus, I would actually consider preprocessing the corpus offline.

Instead of doing:

chunk → embedding

store:

chunk
 ├── sentences
 ├── atomic propositions
 ├── entities
 ├── quantities
 ├── negations
 └── embedding

Then retrieval becomes:

query
  ↓
retrieve chunks
  ↓
retrieve propositions
  ↓
rank propositions
  ↓
300–500 token evidence packet

This is much more attractive for your system than dynamically running a compressor.

1.6 Expected latency impact

There is an important caveat here.

The published LLMLingua-2 results are not measurements on your 12 GB GPU with Gemma 4 12B Q4_K_M. They report 1.6–2.9× end-to-end acceleration at 2–5× compression in their benchmark environment.

Your system is different.

With only ~2,000 input tokens, I'd expect roughly:

Technique	Expected effect
Sentence pruning	small–moderate
2,000 → 500 tokens	modest prompt-processing reduction
Output cap	large
Structured output	large
Removing conversational instructions	large
Claim-level generation	large
Speculative decoding	potentially large
LLMLingua-2	uncertain
Abstractive summarizer	probably negative

The reason is simple:

Autoregressive generation is sequential. Prompt ingestion is comparatively parallel.

If Gemma goes from:

100 output tokens

to:

400 output tokens

you can easily lose much more time than you save by deleting 1,500 prompt tokens.

1.7 My recommended compression target

Don't target:

"2,000 → 300 tokens"

Target:

"minimum evidence necessary to answer the question, normally 300–600 tokens."

Some questions need 150.

Others genuinely need 800.

Use a budget:

MAX_EVIDENCE_TOKENS = 600
MIN_EVIDENCE_TOKENS = 150

and stop adding evidence once every required proposition has coverage.

Pillar 2 — Grounding and anti-extrapolation

This is where I think you can make the largest improvement to Gemma.

Your current 7.9% UCR is particularly informative because you report that most unsupported material is connective/framing language.

That means you shouldn't primarily tell Gemma:

"Don't hallucinate."

Instead:

Remove the opportunity to generate unsupported language.

2.1 Don't ask it to "write an answer"

Ask it to produce supported propositions.

Bad:

Answer the user's question using the information below.

Better:

Generate only statements that are directly supported by EVIDENCE.

Best:

Generate a list of claims.
Each claim MUST be supported by one or more evidence IDs.
Claims without evidence IDs are invalid.

That creates a mechanical relationship:

claim → evidence

rather than:

answer → hopefully evidence-grounded
2.2 Recommended Gemma synthesizer schema
SYNTHESIS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string"
                    },
                    "evidence": {
                        "type": "array",
                        "items": {
                            "type": "string"
                        },
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

Ollama supports supplying JSON Schema directly through format, and recommends validating the returned JSON against the schema.

This is preferable to merely saying "return JSON".

2.3 The Gemma prompt I would test first
SYSTEM:

You are the factual synthesis component of an offline X4: Foundations
knowledge system.

Your task is NOT to chat, explain your reasoning, introduce the answer,
or provide general background.

You may ONLY produce claims supported by the supplied EVIDENCE.

GROUNDING RULES
===============

1. Every claim MUST be supported by at least one EVIDENCE ID.

2. Do not use world knowledge, model memory, assumptions, or likely facts.

3. If the evidence does not establish a fact, do not state it.

4. Do not infer:
   - missing quantities
   - missing relationships
   - missing causes
   - game mechanics not explicitly supported
   - DLC membership
   - faction relationships
   - numerical values

5. Preserve all numerical values exactly.

6. Preserve negation exactly.
   "does not", "cannot", "never", "without", "only", and "except"
   MUST NOT be removed or reversed.

7. Do not add introductory sentences such as:
   "Sure", "Certainly", "In X4", "Based on the information provided",
   "Here is the answer", or similar framing.

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
===============

Return ONLY the JSON object defined by the supplied schema.

No markdown.
No prose outside JSON.
No reasoning.
No commentary.

EVIDENCE
========

{evidence}

USER QUESTION
=============

{question}

This is intentionally repetitive.

For a 12B model, redundancy in the control instruction is often cheaper than ambiguity in the output space.

2.4 "Quote then synthesize" — use selectively

I would not have Gemma quote every source sentence.

That increases generation length.

Instead, use a hidden/structured evidence mapping:

{
  "claims": [
    {
      "text": "A Hull Part production cycle requires 1 Graphene and 1 Energy Cell.",
      "evidence": ["E17"]
    }
  ]
}

The verifier can then inspect E17.

The user sees only:

A Hull Part production cycle requires 1 Graphene and 1 Energy Cell.

So you get attribution without generating quotation overhead.

2.5 Nonce anchoring

Nonce anchoring is useful, but I would use it primarily as a verification mechanism, not as the main grounding mechanism.

Example:

E17 = §K7M2
E18 = §Q4P9

Prompt:

Each claim must include the evidence nonce that supports it.

Output:

{
  "claims": [
    {
      "text": "...",
      "evidence": ["§K7M2"]
    }
  ]
}

This makes provenance explicit.

However, don't rely on the nonce itself to prevent hallucination. A model can produce:

"Grand Exchange I is DLC content." [§K7M2]

even when §K7M2 doesn't support it.

Therefore:

nonce presence ≠ semantic entailment

The nonce is the address; the verifier still needs to inspect the content.

2.6 Strict propositional budget

This is one of the techniques I'd strongly recommend.

Set:

maximum claims = 6
maximum one proposition per claim

rather than:

write a concise answer

Why?

Because "concise" is a semantic instruction.

"Maximum 6 claims" is a structural constraint.

And structured constraints are substantially easier for small models to follow.

2.7 The hidden killer: thinking

Gemma 4 introduces a thinking mode in the 2026 generation.

If your Ollama configuration has reasoning enabled, verify that it is off for this synthesis stage unless your tests prove it materially improves grounding.

Ollama explicitly exposes thinking separately from final content, and Qwen3 supports the same conceptual mechanism.

For this application:

Router:       no thinking
Synthesizer:  no thinking initially
Verifier:     tiny model / no thinking

The job is not open-ended reasoning.

It's controlled transformation of retrieved facts into prose.

2.8 Temperature

I'd run the synthesizer approximately:

temperature = 0.0–0.2
top_p       = 0.8–1.0

and benchmark 0 vs 0.1 vs 0.2.

For your invariant-heavy workload, I'd start at:

temperature = 0

Ollama itself recommends low temperature for more deterministic structured output.

Pillar 3 — DLC/base-game boundary

This is actually a knowledge representation problem, not primarily a prompt problem.

Qwen's behaviour:

unfamiliar canonical entity → assumes DLC

is exactly what you'd expect from an LLM being asked to infer ontology membership from priors.

The fix is:

Don't ask the model to infer ontology membership.

3.1 Make universe membership deterministic

Your SQLite database should contain something like:

CREATE TABLE entities (
    entity_id TEXT PRIMARY KEY,
    canonical_name TEXT NOT NULL,
    entity_type TEXT NOT NULL,
    game_version TEXT NOT NULL,
    availability TEXT NOT NULL
        CHECK (availability IN (
            'BASE',
            'DLC',
            'UNKNOWN'
        )),
    dlc_id TEXT
);

Then:

Grand Exchange I
        ↓
SQLite
        ↓
BASE

The router never has to decide whether it's DLC.

3.2 Router should use an ontology tool

Add:

resolve_entity(name)

with output:

{
  "canonical_name": "Grand Exchange I",
  "entity_type": "sector",
  "availability": "BASE"
}

Then your router can distinguish:

KNOWN_BASE
KNOWN_DLC
UNKNOWN

instead of:

BASE
DLC

That third state is crucial.

3.3 Three-state epistemic model

Use:

BASE
DLC
UNKNOWN

Never:

BASE
DLC

because:

not known to be BASE

does not imply:

DLC

This is exactly the failure mode you're observing with Qwen.

3.4 Router prompt
SYSTEM:

You are the routing classifier for an offline X4: Foundations advisor.

Your job is classification, not answering.

VALID ROUTES
============

STRUCTURED
The question requires canonical database facts.

VECTOR
The question requires explanatory/community knowledge.

BOTH
The question requires both canonical database facts and explanatory
knowledge.

ABSTAIN
Use ONLY when the query is explicitly outside the supported X4 knowledge
scope or is explicitly about unsupported DLC content.

ENTITY SCOPE RULE
=================

Never infer DLC status from unfamiliarity.

Unknown entity != DLC.

Use the supplied canonical entity registry when determining scope.

If an entity is:
- BASE: it is in scope.
- DLC: abstain only if that DLC is outside the configured supported scope.
- UNKNOWN: do not classify it as DLC merely because you do not recognize it.

Examples:

INPUT:
"What is the production time of Hull Parts?"

OUTPUT:
STRUCTURED

INPUT:
"Why is it useful to build Hull Parts near a shipyard?"

OUTPUT:
BOTH

INPUT:
"What does the supported DLC X add?"

OUTPUT:
ABSTAIN

INPUT:
"What is Grand Exchange I?"

OUTPUT:
STRUCTURED

Never explain the classification.
Return only the routing object.
3.5 Better still: resolve entities before routing

I would actually alter the order slightly:

User
 ↓
cheap entity extraction
 ↓
SQLite entity resolution
 ↓
Router

For example:

entities = entity_match(user_query)

resolved = [
    db.resolve_entity(e)
    for e in entities
]

Now the router receives:

ENTITY REGISTRY:

Grand Exchange I
type: sector
scope: BASE

The LLM doesn't need to know that from pretraining.

3.6 Few-shot demonstrations

Don't provide examples such as:

DLC example → DLC

only.

That teaches a dangerous surface pattern.

Instead, deliberately include negative pairs:

"Tell me about Grand Exchange I."
→ BASE

"Tell me about Kingdom End."
→ BASE

"Tell me about [known DLC entity]."
→ DLC

"Tell me about [fictional/unknown entity]."
→ UNKNOWN

"Tell me about an unfamiliar sector called X."
→ UNKNOWN

The important training lesson becomes:

unfamiliar ≠ DLC

rather than:

known = base
unknown = DLC
3.7 Router grammar

Here's where Granite should improve dramatically.

Don't let it generate:

{
    "route": "...",
    "tool": "...",
    "arguments": ...
}

with free-form arguments.

Use:

{
    "route": "STRUCTURED",
    "operation": "lookup_entity",
    "entity": "Grand Exchange I"
}

with:

route ∈ {
    STRUCTURED,
    VECTOR,
    BOTH,
    ABSTAIN
}

and:

operation ∈ {
    lookup_entity,
    lookup_ship,
    lookup_ware,
    lookup_sector,
    lookup_production
}

Ollama supports exactly this kind of JSON-schema constrained output.

Pillar 4 — Constrained decoding

This is where Granite's failure mode is particularly amenable to engineering.

4.1 What grammar guarantees

Suppose you define:

{
  "type": "object",
  "properties": {
    "route": {
      "enum": [
        "STRUCTURED",
        "VECTOR",
        "BOTH",
        "ABSTAIN"
      ]
    }
  },
  "required": ["route"],
  "additionalProperties": false
}

The constrained decoder can guarantee:

valid JSON
+
route is one of the four permitted strings
+
required field exists
+
no additional fields

It cannot guarantee:

Grand Exchange I is the correct entity.

That's semantic validation.

This distinction is fundamental.

4.2 GBNF

A simple GBNF router grammar:

root ::= object

object ::= "{" ws
           "\"route\"" ws ":" ws route
           ws "}"

route ::= "\"STRUCTURED\""
        | "\"VECTOR\""
        | "\"BOTH\""
        | "\"ABSTAIN\""

ws ::= [ \t\n]*

For richer routing:

root ::= "{" ws
         "\"route\"" ws ":" ws route "," ws
         "\"operation\"" ws ":" ws operation "," ws
         "\"query\"" ws ":" ws string
         ws "}"

route ::= "\"STRUCTURED\""
        | "\"VECTOR\""
        | "\"BOTH\""
        | "\"ABSTAIN\""

operation ::= "\"lookup_ship\""
            | "\"lookup_ware\""
            | "\"lookup_sector\""
            | "\"lookup_production\""
            | "\"search_knowledge\""
            | "\"none\""

string ::= "\"" chars "\""

chars ::= [^"\\]*

llama.cpp supports GBNF and JSON-schema-to-grammar conversion.

4.3 Ollama implementation

I would use Ollama's JSON Schema interface rather than manually managing GBNF unless you specifically need grammar-level control.

from ollama import chat

response = chat(
    model="granite4.1:8b",
    messages=[
        {
            "role": "system",
            "content": ROUTER_PROMPT,
        },
        {
            "role": "user",
            "content": user_query,
        },
    ],
    format=ROUTER_SCHEMA,
    options={
        "temperature": 0,
    },
)

route = Router.model_validate_json(
    response.message.content
)

Ollama documents schema-based structured outputs and Pydantic validation directly.

4.4 But don't put the whitelist entirely into the grammar

This is a common architectural mistake.

Imagine 30,000 possible X4 ware names.

Don't create:

ware ::= "Energy Cells"
       | "Hull Parts"
       | "Graphene"
       | ...
       | 30,000 names

That creates an enormous grammar and makes maintenance unpleasant.

Instead:

LLM
 ↓
JSON schema
 ↓
"lookup_ware"
"requested_name": "Hull Parts"
 ↓
Python
 ↓
SQLite canonical lookup
 ↓
canonical ID

The application owns the whitelist.

The grammar owns the syntax.

4.5 Strongest architecture for tool parameters

Instead of:

{
  "tool": "lookup_ware",
  "ware": "Hull Parts"
}

prefer:

{
  "tool": "lookup_ware",
  "query": "Hull Parts"
}

Then:

candidate = normalize(tool_call["query"])

entity = db.resolve_exact_or_alias(
    entity_type="ware",
    query=candidate,
)

if entity is None:
    reject_tool_call()

The model cannot directly dictate SQL.

This eliminates an entire class of hallucinated parameters.

4.6 Granite-specific recommendation

For Granite:

temperature = 0
structured output = ON
additionalProperties = false
enum = ON
max generation tokens = very low

Then perform:

schema validation
→ domain validation
→ parameter validation
→ SQL parameterization

IBM specifically positions Granite 4.1 as improved for tool calling and instruction following, and the 8B instruct model has been post-trained specifically for those capabilities.

So your benchmark suggests you are currently leaving some of that capability on the table through unconstrained generation.

4.7 Latency of constrained decoding

This is another place where the evidence is encouraging.

Current llama.cpp's LLGuidance integration reports approximately 50 μs average CPU time per token for JSON-schema token masking on a 128k-token tokenizer, with 0.5 ms p99 and 20 ms p100 in its JSON Schema Bench measurements.

That's effectively irrelevant beside a 10–30 second LLM generation.

So:

Use constrained decoding aggressively for routing.

There is very little reason not to.

4.8 Fast inline verification

This is probably the most interesting improvement for your system.

You don't need another 12B model.

Use a cascade.

Layer 0 — deterministic checks

~microseconds–milliseconds:

claim contains number?
        ↓
does that number occur in evidence?
claim says "not X"?
        ↓
does evidence actually contain negation?
claim names entity?
        ↓
does entity occur in evidence?
claim cites E17?
        ↓
does E17 exist?

This catches surprisingly much.

4.9 Entity/number preservation verifier
def verify_surface_claim(claim, evidence):
    claim_numbers = extract_numbers(claim)
    evidence_numbers = extract_numbers(evidence)

    if not claim_numbers <= evidence_numbers:
        return False, "NUMBER_NOT_SUPPORTED"

    claim_entities = extract_known_entities(claim)
    evidence_entities = extract_known_entities(evidence)

    if not claim_entities <= evidence_entities:
        return False, "ENTITY_NOT_SUPPORTED"

    if introduces_negation(claim):
        if not evidence_supports_negation(claim, evidence):
            return False, "NEGATION_NOT_SUPPORTED"

    return True, None

This isn't semantic entailment.

But it is extremely fast.

4.10 MiniCheck is unusually well matched to your problem

MiniCheck is probably the strongest existing component I'd test for your final verifier.

It was specifically designed for:

determining whether LLM output is grounded in provided documents.

The EMNLP 2024 paper reports a 770M MiniCheck-FT5 model achieving GPT-4-level performance among comparable systems, while being dramatically cheaper.

Even more relevant:

MiniCheck is available locally through Ollama.

That makes it unusually appropriate for your offline architecture.

Pipeline:

Gemma
  ↓
6 claims
  ↓
deterministic filters
  ↓
MiniCheck
  ↓
supported?
4.11 Verification prompt

For a model-based fallback verifier:

SYSTEM:

You are a binary evidence verifier.

Determine whether CLAIM is supported by EVIDENCE.

Return exactly one label:

SUPPORTED
UNSUPPORTED
CONTRADICTED

SUPPORTED:
The evidence entails the claim.

UNSUPPORTED:
The claim contains information not established by the evidence.

CONTRADICTED:
The evidence establishes information incompatible with the claim.

Do not use world knowledge.
Do not infer missing facts.
Do not reward plausibility.

EVIDENCE:
{evidence}

CLAIM:
{claim}

But I would prefer MiniCheck over using Gemma again.

4.12 Can this fit under 500 ms?

Potentially, yes.

But I would not promise <500 ms without benchmarking your actual CPU/GPU.

The architecture makes it plausible:

deterministic checks       ~<10 ms
MiniCheck-FT5 770M         potentially tens–hundreds ms
JSON parsing               negligible

MiniCheck was explicitly designed as a lightweight alternative to LLM-per-claim fact checking.

Your verifier should also operate on:

6 claims × 300–600 tokens

not:

full conversation × 5 chunks

That dramatically reduces the verification workload.

4.13 Don't verify the whole answer

This matters.

Instead of:

answer → fact check

use:

answer
 ↓
claim segmentation
 ↓
claim 1 ──► verifier
claim 2 ──► verifier
claim 3 ──► verifier

Your existing five-class evaluator already gives you the right conceptual abstraction:

FACT
SUPPORTED_INFERENCE
ADVICE
UNSUPPORTED
CONTRADICTED

I'd actually make the production verifier slightly simpler:

SUPPORTED
UNSUPPORTED
CONTRADICTED

and preserve:

FACT / SUPPORTED_INFERENCE / ADVICE

as generation metadata, not verification states.

The critical optimisation: claim-type separation

Your current UCR strongly suggests another issue.

Consider:

"Hull Parts require Graphene and Energy Cells."

versus:

"This makes Hull Parts an excellent choice for an early-game factory."

The first is a factual claim.

The second is advice/inference.

They should not be processed identically.

I recommend:

{
  "claims": [
    {
      "text": "...",
      "type": "FACT",
      "evidence": ["E17"]
    },
    {
      "text": "...",
      "type": "ADVICE",
      "evidence": ["E17", "E21"]
    }
  ]
}

Then impose:

FACT:
must be directly entailed.

SUPPORTED_INFERENCE:
must have explicit supporting facts.

ADVICE:
must be explicitly marked as advice and never presented as fact.

This prevents advice language from contaminating factual grounding metrics.

A better production synthesizer

I would make Gemma output:

{
  "claims": [
    {
      "type": "FACT",
      "text": "Hull Parts require Graphene and Energy Cells.",
      "evidence": ["E17"]
    },
    {
      "type": "FACT",
      "text": "The production cycle is 60 seconds.",
      "evidence": ["E19"]
    }
  ]
}

Then your application renders:

Hull Parts require Graphene and Energy Cells. The production cycle is 60 seconds.

Notice what disappeared:

"Sure!"
"Certainly!"
"In X4..."
"One useful thing to know..."
"Depending on your playstyle..."
"Overall..."

Those are precisely the kinds of tokens your benchmark says are causing the problem.

Output token control

This deserves its own section because I think it is likely to produce your biggest latency win.

Your synthesizer should have a hard generation budget.

For example:

max_new_tokens = 160

or even:

max_new_tokens = 128

depending on your UI.

Six short claims do not require 400 tokens.

If the model hits the limit:

truncate → verifier

or preferably:

regenerate with fewer claims

The system should never reward the model for filling the context window.

Speculative decoding

Once the grounding architecture is fixed, investigate llama.cpp speculative decoding.

Current llama.cpp supports:

draft-model speculation
EAGLE-3
MTP
n-gram methods
several other speculative modes.

For your workload, I'd test n-gram speculation first because it has essentially negligible model-memory overhead and is already implemented in llama.cpp.

This is particularly interesting for repetitive structured output such as:

{
  "claims": [
    ...
  ]
}

and boilerplate tokens.

The important caveat is that speculative decoding helps generation throughput, not grounding quality.

Therefore:

first:
output control

second:
speculative decoding

not the reverse.

Why LLMLingua-2 shouldn't be your first move

This is probably the most counterintuitive conclusion.

LLMLingua-2 is a very good technology. Its published results show 2–5× prompt compression and 1.6–2.9× end-to-end speedup.

But your architecture has:

53 chunks
300–700 tokens each
retrieve only 4–5
≈ 2,000 tokens

That is already tiny by modern LLM standards.

And your observed symptom is:

normal synthesis ≈ 5.4s
retrieved synthesis ≈ 28.6s

If the additional 1,500 input tokens were the primary cause, you'd expect a relatively smooth increase in latency.

Instead you have a huge increase associated with longer generation.

Therefore instrument:

prompt_tokens
prompt_eval_duration
generation_tokens
generation_duration
tokens/sec

for every test.

You want:

T_total =
    T_load
  + T_prompt_eval
  + N_generated / tok_per_sec
  + T_tooling
  + T_verification

Then you can see exactly where the 23 seconds went.

Benchmark instrumentation I strongly recommend

Add these fields to every test record:

{
  "model": "gemma4:12b",
  "query_id": "Q17",

  "route": "BOTH",

  "retrieval_chunks": 5,
  "retrieval_tokens": 2037,
  "compressed_tokens": 487,

  "prompt_tokens": 921,
  "output_tokens": 74,

  "prompt_eval_ms": 1420,
  "generation_ms": 9210,

  "input_tok_per_sec": 1440,
  "output_tok_per_sec": 8.0,

  "claims": 4,
  "supported_claims": 4,
  "unsupported_claims": 0,
  "contradicted_claims": 0,

  "verification_ms": 83
}

Then your experiment becomes scientifically interpretable.

Experimental matrix

I would run the following ablation study, rather than changing everything simultaneously.

Experiment A — Gemma output control
Variant	Context	Output
A0	raw	unconstrained
A1	raw	max 256
A2	raw	max 128
A3	raw	JSON claims
A4	raw	JSON + 128
A5	raw	JSON + 128 + temp 0

I expect A4/A5 to be dramatically better than A0.

Experiment B — evidence compression

Starting with the best A configuration:

Variant	Evidence
B0	raw 2,000
B1	BM25 1,000
B2	BM25 600
B3	proposition 600
B4	proposition 400
B5	LLMLingua-2 2×
B6	LLMLingua-2 4×

Measure:

ground truth pass
UCR
contradictions
input tokens
output tokens
prompt eval
generation time
total time
Experiment C — verifier
C0 = no verifier
C1 = deterministic verifier
C2 = deterministic + MiniCheck
C3 = deterministic + MiniCheck + regenerate

Your target should be:

UCR < 3%
contradictions = 0

rather than trying to force the generator itself to achieve perfect grounding.

Experiment D — router
D0 = unconstrained Gemma/Qwen/Granite
D1 = JSON mode
D2 = JSON schema
D3 = JSON schema + entity resolver
D4 = JSON schema + entity resolver + UNKNOWN state

I expect D3/D4 to be the important combination for Qwen.

Recommended final architecture

If these experiments behave as I expect, I'd settle on:

                     USER QUERY
                         │
                         ▼
                ┌─────────────────┐
                │ Entity resolver │
                │ deterministic   │
                └────────┬────────┘
                         │
                         ▼
                ┌─────────────────┐
                │ Router          │
                │ Granite 8B      │
                │ JSON constrained│
                └────────┬────────┘
                         │
               ┌─────────┼─────────┐
               ▼         ▼         ▼
             SQL       VECTOR     BOTH
               │         │         │
               └─────────┼─────────┘
                         ▼
                 ┌──────────────┐
                 │ Fact compiler│
                 │ deterministic│
                 └──────┬───────┘
                        │
                 300–600 tokens
                        │
                        ▼
               ┌──────────────────┐
               │ Gemma 4 12B      │
               │ temp=0           │
               │ max=128–160      │
               │ JSON schema      │
               │ claims only      │
               └────────┬─────────┘
                        │
                        ▼
               ┌──────────────────┐
               │ Claim verifier   │
               │                  │
               │ 1. exact checks  │
               │ 2. numbers       │
               │ 3. entities      │
               │ 4. MiniCheck     │
               └────────┬─────────┘
                        │
                  ┌─────┴─────┐
                  ▼           ▼
              supported    rejected
                  │           │
                  ▼           ▼
               render      regenerate
Model-specific conclusions
Gemma 4 12B

Keep it as the Synthesizer.

Your own benchmark already establishes that it is the strongest model for this task:

91.7% ground truth
97.2% routing
95% structured precision

Its weakness isn't factual competence so much as verbosity under evidence-rich prompting.

Gemma 4 is a 12B dense model and its 2026 technical report explicitly emphasizes improved inference efficiency and long-context capabilities.

I would therefore attack:

output length
claim budget
structured output
thinking mode

before changing models.

Qwen 3 14B

Use it as a potential alternative synthesizer/verifier, but don't let it decide DLC membership.

Qwen3's technical report describes a family ranging from 0.6B through very large MoE models, and the 14B model is a dense member of that family.

Your empirical result:

0 contradictions

is extremely attractive.

The 50% abstention result, however, suggests its epistemic boundary is poorly calibrated for your game ontology.

That's exactly what the entity-resolution layer should fix.

Granite 4.1 8B

This is the model I'd use for the router, subject to your constrained-decoding experiment.

IBM explicitly reports improvements in tool calling and instruction following for Granite 4.1, and the 8B instruct model is specifically post-trained for tool use and instruction following.

Your current:

77.8% routing

should therefore not be interpreted as:

Granite isn't capable of routing.

I'd interpret it as:

Granite is being given too much freedom in a task that has a finite formal language.

Constrain the output and validate the arguments.

Literature / technology stack I would actually use
Context compression

LLMLingua-2 — strongest directly relevant compression technology to benchmark. Its token-classification approach is particularly attractive for extractive preservation.

LLMLingua-2 paper

Grounding evaluation

RAGTruth — useful for thinking about unsupported/contradictory claims at word and response level. It contains nearly 18,000 annotated RAG responses.

RAGTruth paper

Claim-level factuality

VeriScore — useful conceptual basis for decomposing generation into verifiable claims rather than judging the whole response as a single object.

VeriScore paper

Fast local verification

MiniCheck — probably the single most interesting addition to your architecture. 770M, designed specifically for grounding verification, with local Ollama availability.

MiniCheck repository

Constrained decoding

llama.cpp GBNF / JSON schema — directly applicable to your Ollama/llama.cpp deployment.

llama.cpp GBNF documentation

High-performance constrained decoding

LLGuidance — especially interesting if you eventually move from Ollama to direct llama.cpp. Its current implementation reports ~50 μs average CPU token-mask computation in JSON Schema Bench.

LLGuidance

Structured Ollama output

Ollama's current API directly supports JSON Schema through format, which makes the routing/synthesis design above straightforward without introducing another inference framework.

Ollama structured outputs documentation

Inference acceleration

llama.cpp speculative decoding is worth testing after the generation budget is under control.

llama.cpp speculative decoding documentation

What I would implement first

In order:

1. Constrain the router

Highest confidence / lowest risk.

Granite 8B
+
temperature 0
+
JSON Schema
+
enum routes
+
additionalProperties=false
2. Introduce UNKNOWN into entity scope
BASE
DLC
UNKNOWN

and resolve canonical entities through SQLite.

This should directly attack Qwen's false DLC abstentions.

3. Change Gemma from prose generation to claim generation
max 6 claims
max 128–160 output tokens
JSON Schema
temperature 0
no thinking

This is the change I expect to have the largest impact on your 28.6 s → <15 s objective.

4. Build deterministic evidence compilation

Don't immediately deploy LLMLingua-2.

First:

chunks
 ↓
sentences
 ↓
BM25 + entity + semantic ranking
 ↓
protected facts
 ↓
300–600 token evidence packet
5. Add deterministic claim checks

Numbers, entities, negation, evidence IDs.

6. Add MiniCheck

Only claims surviving deterministic checks need neural verification.

7. Then benchmark LLMLingua-2

If your instrumentation shows prompt evaluation is still a substantial proportion of total latency, introduce LLMLingua-2.

8. Finally test speculative decoding

At that point you're optimizing genuine autoregressive generation rather than masking an architectural problem.

The most important design principle

Your current system is asking a 12–14B model to perform four logically distinct jobs:

ontology resolution
routing
evidence interpretation
language generation

That is too much responsibility for a local model when three of those jobs can be made substantially deterministic.

The high-reliability version should instead look like:

SQLite          = canonical truth
retriever       = evidence selection
compiler        = evidence normalization
grammar         = syntactic validity
LLM             = linguistic synthesis
MiniCheck       = semantic grounding check
application     = final authority

That separation is what makes your existing 91.7% Gemma result potentially capable of becoming a >95% production system without needing a larger model.

And importantly, I would set the production acceptance criterion not merely to your current ≥85% ground-truth pass, but to:

Ground truth pass       ≥95%
UCR                     ≤2%
Contradictions           =0
Router accuracy         ≥98%
DLC/base false rejects   =0
Tool syntax errors       =0
Verifier false accepts   <1%
P90 latency             <15 s

The combination of bounded claim generation + deterministic ontology + constrained routing + post-generation verification is much more likely to achieve those numbers than trying to find a magical prompt that makes a 12B model intrinsically hallucinate less.