# Deep Research Brief: Optimizing Grounding, Routing, and Latency for Sub-15B Local LLMs in a Hybrid RAG Architecture

---

## Instructions for ChatGPT / Claude Deep Research

You are an expert AI Systems Researcher and Inference Optimization Engineer. Conduct a rigorous, empirical investigation into state-of-the-art prompt engineering, constrained decoding, context compression, and inference-time guardrail techniques specifically tailored to **sub-15B open-weights language models running locally via Ollama / llama.cpp on consumer hardware (12GB VRAM budget)**.

Focus on practical, production-grade solutions supported by recent (2024–2026) academic literature, benchmark evidence, and open-source systems engineering (e.g., Guidance, Outlines, llama.cpp grammars, DSPy, LLMLingua-2, SGLang, vLLM).

---

## 1. System Architecture & Operating Constraints

- **Application:** Local-first offline game knowledge advisor (X4: Foundations RAG system).
- **Inference Runtime:** Local Ollama (`llama.cpp` backend), 12GB VRAM GPU budget, Q4_K_M quantization.
- **Data Layers:**
  - *Layer 1 (Structured):* Local SQLite database holding canonical entities (ships, wares, sectors, production chains). Queried via deterministic parameterized SQL functions.
  - *Layer 2 (Unstructured):* `sqlite-vec` vector store containing 53 curated community knowledge chunks (300–700 tokens each) embedded via `qwen3-embedding:0.6b` (cosine threshold $\tau=0.50$).
- **Pipeline Architecture:**
  1. *Router LLM:* Single-turn tool-calling classifier deciding between `STRUCTURED` (SQL tool call), `VECTOR` (knowledge base search tool call), `BOTH` (hybrid tool calls), or `ABSTAIN` (out-of-scope DLC or malicious queries).
  2. *Synthesizer LLM:* Ingests retrieved SQL rows and/or 4–5 vector chunks and generates a concise, natural language response strictly grounded in provided data.

---

## 2. Empirical Benchmark Baseline & Model Comparison

We evaluated three candidate models across a canonical $N=36$ test corpus (11 query templates) using a strict 5-class claim-level verifier (`FACT`, `SUPPORTED_INFERENCE`, `ADVICE`, `UNSUPPORTED`, `CONTRADICTED`):

| Metric | Target Hard Gate | Gemma 4 12B (`gemma4:12b`) | Granite 4.1 8B (`granite4.1:8b`) | Qwen 3 14B (`qwen3:14b`) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Ground Truth Pass Rate** | $\ge 85.0\%$ | **91.7%** (33/36) ✅ | 52.8% (19/36) ❌ | 80.6% (29/36) ❌ |
| **Routing Accuracy** | $\ge 90.0\%$ | **97.2%** (35/36) ✅ | 77.8% (28/36) ❌ | **91.7%** (33/36) ✅ |
| **Structured Precision** | $\ge 90.0\%$ | **95.0%** (19/20) ✅ | 70.0% (14/20) ❌ | **95.0%** (19/20) ✅ |
| **Abstention Accuracy (DLC/No-Ev)**| **100.0%** | 75.0% (3/4) ❌ | **100.0%** (4/4) ✅ | 50.0% (2/4) ❌ |
| **Unsupported Claim Rate (UCR)** | **$\le 3.0\%$** | **7.9%** (18/229) ❌ | 13.5% (25/185) ❌ | 13.5% (23/170) ❌ |
| **Zero Contradictions Invariant** | **0 Contradictions** | **1** ❌ | **4** ❌ | **0** ✅ |
| **Mean / P90 Response Latency** | $<20.0\text{s} / <30.0\text{s}$ | 13.40s / 24.32s | **10.42s** / 20.46s | 16.96s / 24.51s |

---

## 3. Specific Empirical Failure Modes to Investigate

### Challenge 1: The Gemma 4 12B Grounding & Latency Paradox
- **Behavior:** Gemma 4 achieves the highest accuracy (91.7%) and best routing (97.2%). However, when given 4–5 retrieved chunks ($\tau=0.50$), wall-clock synthesis time spikes from ~5.4s to **28.6s** because the model generates lengthy conversational framing and introductory sentences. Furthermore, these connective/framing statements account for nearly all of its **7.9% Unsupported Claim Rate (UCR)**.
- **Goal:** Reduce Gemma 4 generation time from ~28s to $<15\text{s}$ and push UCR from 7.9% down below **3.0%** without losing factual coverage.

### Challenge 2: Qwen 3 14B's Scope Over-Abstention
- **Behavior:** Qwen 3 achieved a perfect **0-contradiction record** across all 36 test cases, but failed with **50% abstention accuracy** because it hallucinated canonical base-game universe entities (e.g., the *Grand Exchange I* sector) as "DLC expansion content" and refused to answer.
- **Goal:** Calibrate Qwen 3's epistemic refusal boundary so it strictly rejects DLC queries without false-positive refusals on base-game content.

### Challenge 3: Granite 4.1 8B's Parameter Incoherence
- **Behavior:** Granite is the fastest model (10.42s mean latency), but suffered from low routing stability (77.8%) and 4 factual contradictions, often hallucinating invalid tool parameters or skipping required query arguments.
- **Goal:** Explore grammar-constrained decoding (GBNF / JSON schema masking) to eliminate routing parameter syntax errors.

---

## 4. Key Research Questions & Deliverables Needed

Please provide deep, technically detailed answers and actionable implementations for the following 4 pillars:

### Pillar 1: Context Distillation & Token-Pruning Before Synthesis
1. What extractive or extractive-compression techniques (e.g., prompt-guard ranking, LLMLingua-2, atomic proposition extraction, BM25 sentence reranking) can compress 4–5 knowledge chunks (~2,000 tokens) down to 300–500 essential factual tokens before feeding the Synthesizer prompt?
2. How can we ensure numerical parameters (recipe inputs, cycle times, cargo volumes) and negation constraints are 100% preserved during context distillation?
3. What is the measured impact of such compression on generation token counts and end-to-end latency for 12B/14B Q4_K_M models?

### Pillar 2: Attributive Grounding & Anti-Extrapolation Prompting
1. What specific prompt patterns (e.g., "Quote-then-Synthesize", Nonce Token Anchoring, Negative-Constraint Directives, Strict Propositional Budgeting) have proven most effective at driving UCR below 3% in sub-15B models?
2. Provide a concrete, battle-tested system prompt template optimized for `gemma4:12b` that strictly penalizes conversational padding, ungrounded introductions, and speculative connective sentences.

### Pillar 3: Calibrated DLC vs. Base-Game Boundary Prompting
1. How can system prompts and few-shot routing demonstrations be structured to teach models the precise difference between in-scope base-game entities and out-of-scope DLC expansions without causing over-generalization?
2. What techniques prevent small models from assuming an unfamiliar canonical entity is "DLC" simply due to low internal pretraining prior probability?

### Pillar 4: Constrained Decoding & Fast Inline Fact-Checking
1. How can GBNF grammars or Ollama JSON schema constraints be integrated into the routing stage to mathematically guarantee 100% syntactically valid and domain-whitelisted tool parameters?
2. Are there ultra-fast, local inline verification methods (e.g., n-gram overlap check against retrieved nonces, fast cross-encoder validation) that can inspect synthesized claims in $<500\text{ms}$ before displaying them to the user?

---

## 5. Required Output Format

For each pillar, please provide:
1. **Theoretical / Algorithmic Mechanism:** Why the technique works on sub-15B transformer architectures.
2. **Concrete Code / Prompt Implementations:** Complete, drop-in prompt templates, Python logic, or GBNF grammar files.
3. **Tradeoff & Failure Analysis:** Potential edge cases (e.g., over-pruning risk, latency overhead, model refusal).
4. **Relevant Literature / Benchmark Citations:** Specific papers, repositories, or benchmarks (2024–2026) validating the approach.
