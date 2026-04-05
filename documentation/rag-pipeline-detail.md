# RAG Pipeline — AI Knowledge Helper (RAG System)

## Overview

This notebook implements the complete **Retrieval-Augmented Generation (RAG)** pipeline for the Pakistani Banking FAQ system. After the data pipeline has processed and embedded all documents, this single notebook handles everything from receiving a user question to returning a grounded, source-cited answer.

```
data/
├── embeddings/
│   ├── faiss_index.bin             ← input: vector index
│   └── chunks_with_metadata.json   ← input: chunk text 
└── rag/
    └── rag_test_results.json       ← output: batch test results
```

**Prerequisite:** All 4 data pipeline notebooks (`01 → 04`) must be fully run before this notebook.

---

## Why a Single Notebook for RAG?

Unlike the data pipeline — which has long-running, one-time steps (extraction, cleaning, chunking, embedding) that each save intermediate files — the RAG pipeline is a **real-time, stateless flow**. Every query goes through all steps in milliseconds. Splitting it across multiple notebooks would break the natural call chain and make it harder to trace a question end-to-end. A single notebook with clearly separated steps is the right structure here.

---

## Notebook — RAG Pipeline

**File:** `rag-pipeline.ipynb`  
**Input:** `data/embeddings/faiss_index.bin`, `data/embeddings/chunks_with_metadata.json`  
**Output:** `data/test/rag_test_results.json`

---

## Step 0 — Setup

### Step 0.1 — Install Required Libraries

All dependencies needed for the RAG pipeline:

```bash
pip install faiss-cpu sentence-transformers anthropic numpy pandas python-dotenv
```

### Step 0.2 — Imports & Configuration

All configuration lives in one place at the top of the notebook. Change values here — nothing else needs to be touched.

| Config Variable | Default Value | Purpose |
|---|---|---|
| `EMBEDDINGS_DIR` | `../data/embeddings` | Where FAISS index and metadata live |
| `RAG_DIR` | `../data/rag` | Where test results are saved |
| `EMBED_MODEL` | `all-MiniLM-L6-v2` | Must match model used in `04_embedding.ipynb` |
| `LLM_MODEL` | `claude-sonnet-4-6` | Anthropic Claude model for answer generation |
| `MAX_TOKENS` | `1024` | Max tokens in Claude's response |
| `TOP_K` | `3` | Number of chunks retrieved per question |
| `SIM_THRESHOLD` | `0.3` | Minimum similarity score to proceed to LLM |

The API key is loaded from a `.env` file in the project root using `python-dotenv`. A setup verification cell confirms the key is loaded and prints all active config values.

### Step 0.3 — Load Embedding Model + FAISS Index + Chunk Metadata

Three things are loaded once at startup and reused for every query:

**Embedding model** — the same `all-MiniLM-L6-v2` model used during `04_embedding.ipynb`. It is critical that this matches exactly — the question vector and chunk vectors must live in the same embedding space, otherwise similarity scores are meaningless.

**FAISS index** — loaded from `faiss_index.bin`. This is a `IndexFlatIP` (inner product) index. Because all vectors were L2-normalized during embedding, inner product search is equivalent to cosine similarity search. Supports exact nearest-neighbor search with no approximation error.

**Chunk metadata** — loaded from `chunks_with_metadata.json`. This is a flat list of all chunk dicts (text + bank name + source file + page number). The list index of each chunk corresponds exactly to its position in the FAISS index — so when FAISS returns index `i`, we look up `chunks[i]` to get the text and metadata.

A sanity check asserts that `index.ntotal == len(chunks)` — if this fails, it means the index and metadata are out of sync and `04_embedding.ipynb` needs to be re-run.

---

## Step 1 + 2 — Question Embedding + Vector Retrieval

**Function:** `retrieve_chunks(question, top_k)`

This step converts the raw user question into a vector and finds the most semantically similar chunks from the entire document collection.

### How retrieval works

```
user question (string)
      │
      ▼  embed_model.encode([question], normalize_embeddings=True)
question vector  (shape: 1 × 384, L2-normalized, float32)
      │
      ▼  index.search(question_vec, top_k)
FAISS returns → scores (cosine similarities), indices (chunk positions)
      │
      ▼  chunks[idx] for each returned index
top-K chunk dicts with similarity scores attached
```

### Why L2 normalization matters

During `04_embedding.ipynb`, all chunk vectors were normalized to unit length before being added to the FAISS index. The question vector is also normalized here using `normalize_embeddings=True`. This means:

- **FAISS `IndexFlatIP` (inner product) = cosine similarity** when both vectors are unit-length
- Scores returned are in range `[0.0, 1.0]` — 1.0 = identical, 0.0 = completely unrelated
- No conversion needed — scores are directly interpretable as similarity percentages

### Output format

Each returned chunk dict contains:

| Field | Description |
|---|---|
| `rank` | Position in results (1 = most similar) |
| `chunk_id` | Unique chunk identifier |
| `bank_name` | Full bank name (e.g. `Habib Bank Limited (HBL)`) |
| `source_file` | Original PDF filename |
| `page_number` | Page this chunk was extracted from |
| `text` | The actual chunk text (400 tokens) |
| `similarity` | Cosine similarity score (0.0–1.0) |

A quick test cell runs a sample question (`"How to open a bank account in Pakistan?"`) and prints all retrieved chunks with their scores and sources to verify retrieval is working.

---

## Step 3 — Relevance Threshold Check

**Function:** `check_relevance(retrieved)`

### The problem this solves

A vector similarity search always returns results — even for completely irrelevant queries. If a user asks *"What is the capital of France?"*, FAISS will still return the 3 most similar banking chunks, but those chunks have nothing to do with the question. Sending those chunks to Claude would waste tokens and risk producing a confidently wrong answer.

### How the threshold works

```
best_similarity = max(chunk["similarity"] for chunk in retrieved)

if best_similarity >= SIM_THRESHOLD (0.3):
    → question is relevant → proceed to prompt + LLM
else:
    → question is out of scope → return fallback, skip LLM entirely
```

### Why 0.3?

A threshold of `0.3` was chosen because:
- In-scope banking questions typically score `0.45–0.85` against relevant chunks
- Completely unrelated questions (capital of France, weather, etc.) score `0.05–0.25`
- The gap between domains is large enough that `0.3` cleanly separates them

The return dict includes `is_relevant` (bool), `best_similarity` (float), and `status` (`"relevant"` or `"below_threshold"`). A test cell checks both an in-scope and an out-of-scope question side by side to confirm the threshold behaves correctly.

---

## Step 4 — Prompt Construction

**Function:** `build_prompt(question, retrieved_chunks)`

### Why prompt design matters for RAG

The prompt is the only thing between the retrieved chunks and Claude's response. A poorly designed prompt can cause:
- Claude to use its own training knowledge instead of the provided context (hallucination)
- Claude to be too verbose or add caveats not in the source
- Claude to ignore specific numbers or amounts mentioned in the context

### Prompt structure

```
[System instruction — role + grounding rules]

=== CONTEXT DOCUMENTS ===
[Source 1: Bank Name | filename.pdf | Page N]
<chunk text>

---

[Source 2: Bank Name | filename.pdf | Page N]
<chunk text>

---

[Source 3: Bank Name | filename.pdf | Page N]
<chunk text>
=== END OF CONTEXT ===

QUESTION: <user question>

ANSWER:
```

### System instruction design decisions

| Instruction | Why it's included |
|---|---|
| *"You are a helpful assistant for Pakistani banking customers"* | Sets domain-specific role, reduces off-topic responses |
| *"Answer ONLY based on the context documents below"* | Core grounding rule — prevents training-knowledge leakage |
| *"If not in context, say: I don't have enough information..."* | Explicit fallback phrase — prevents hallucinated answers |
| *"Do NOT make up any information"* | Reinforces grounding, especially for numbers |
| *"If amounts or limits are mentioned, always include them"* | Ensures specific figures (fees, limits, rates) are preserved |

### Context formatting

Each chunk is prefixed with a labelled header: `[Source N: Bank Name | filename | Page X]`. This serves two purposes — it helps Claude attribute answers to the correct source, and it makes it easy to trace any answer back to its origin document during debugging.

A preview cell prints the first 1000 characters of a sample prompt so the complete structure can be inspected before running live queries.

---

## Step 5 — LLM Call — Anthropic Claude

**Function:** `call_llm(prompt)`

Sends the constructed RAG prompt to Claude using the **Anthropic Python SDK** and returns the plain text answer.

### API call parameters

| Parameter | Value | Reason |
|---|---|---|
| `model` | `claude-sonnet-4-6` | Strong instruction following, accurate grounding |
| `max_tokens` | `1024` | Enough for detailed FAQ answers; prevents runaway generation |
| `messages` | `[{"role": "user", "content": prompt}]` | Single-turn — full context is in the prompt |

The API key is pulled from the environment at call time (`os.getenv("ANTHROPIC_API_KEY")`), not hardcoded. A **connectivity test cell** sends a trivial prompt (`"Say exactly the words: LLM connection successful"`) before running any real queries — this catches auth errors, network issues, or wrong model names before wasting time on the full test batch.

---

## Step 6 — Full RAG Pipeline (`rag_answer`)

**Function:** `rag_answer(question, verbose=True)`

This is the master function that wires all previous steps together. Every query goes through this single entry point.

### Complete flow

```
rag_answer(question)
      │
      ├─ retrieve_chunks(question, top_k=TOP_K)
      │       └─ embed question → FAISS search → top-K chunks with scores
      │
      ├─ check_relevance(retrieved)
      │       └─ best_similarity < SIM_THRESHOLD?
      │               ├─ YES → return fallback answer, skip LLM  (status: "below_threshold")
      │               └─ NO  → continue
      │
      ├─ build_prompt(question, retrieved)
      │       └─ labelled context block + grounding instructions
      │
      ├─ call_llm(prompt)
      │       └─ Claude API call → answer string
      │
      └─ return {
              question, answer, sources,
              best_similarity, status: "answered"
         }
```

### Return dict

| Key | Type | Description |
|---|---|---|
| `question` | `str` | The original user question |
| `answer` | `str` | Claude's grounded answer (or fallback string) |
| `sources` | `list[str]` | Unique source filenames used in the answer |
| `best_similarity` | `float` | Highest similarity score among retrieved chunks |
| `status` | `str` | `"answered"` or `"below_threshold"` |

### `verbose` mode

When `verbose=True`, the function prints a structured trace of every step — retrieved chunks with scores, threshold decision, and the final answer with sources. This makes it easy to debug why a particular answer was generated.

---

## Step 7 — Interactive Mode

A dedicated cell for testing any custom question against the live pipeline:

```python
MY_QUESTION = "What is the minimum age requirement for Alfalah personal loan?"
result = rag_answer(MY_QUESTION, verbose=True)
```

Change `MY_QUESTION` and re-run the cell. The verbose trace shows exactly which chunks were retrieved, what their scores were, and the full answer with sources — useful for manual evaluation and demos.

---

## Step 8 — Batch Testing

Runs the full pipeline on **8 predefined test questions** covering every bank and document type in the dataset, plus one out-of-scope question to validate the threshold filter.

### Test question design

| Question | Bank | Document | Why included |
|---|---|---|---|
| *"Cash transaction limit for HBL home remittance?"* | HBL | HBL-FAQs-Home-Remittance | Tests specific numeric answer retrieval |
| *"Documents for Meezan Roshan Digital Account?"* | Meezan | Meezan-Bank-FAQs-Roshan-Digital-Account | Tests procedural / checklist answers |
| *"Is net metering in Meezan solar financing?"* | Meezan | Meezan-Bank-FAQs-Roshan-Apna-Ghar | Tests yes/no factual retrieval |
| *"Late payment charge on Alfalah personal loan?"* | Bank Alfalah | Bank-Alfalah-FAQs-Personal-Loan | Tests fee/penalty retrieval |
| *"Can a foreign national use ABL myABL?"* | ABL | ABL-FAQs | Tests eligibility criteria retrieval |
| *"Withholding tax rate on Pakistan Investment Bonds?"* | State Bank | State-Bank-FAQs | Tests government/regulatory data retrieval |
| *"Can Islamic banks charge penalty for late payment?"* | HBL | HBL-Islamic-Current-Account | Tests Shariah-specific policy retrieval |
| *"What is the capital of France?"* | — | — | Out-of-scope: must trigger `below_threshold` |

Each test question includes an `expected_source` field (partial filename) used in Step 9 for computing evaluation metrics.

---

## Step 9 — Evaluation — Hit Rate + MRR

Measures **retrieval quality** on the 7 in-scope questions. The out-of-scope question is excluded from metrics (it has no expected source).

### Metrics

**Hit Rate @ K**

The percentage of questions where the correct source document appears anywhere in the top-K retrieved chunks.

```
Hit Rate = number of hits / total in-scope questions
```

A "hit" is when `expected_source` (e.g. `"HBL-FAQs-Home-Remittance"`) is a substring of any retrieved `source_file`.

**MRR — Mean Reciprocal Rank**

Measures how highly the correct source is ranked, not just whether it appears at all. A correct answer at rank 1 scores `1.0`, at rank 2 scores `0.5`, at rank 3 scores `0.33`.

```
MRR = mean(1 / rank_of_first_hit)   for all in-scope questions
    = 1.0  if always rank 1 (perfect)
    = 0.0  if never found
```

MRR penalizes systems that find the right document but bury it at rank 3 — it rewards putting the most relevant chunk first.

### Interpreting results

| Hit Rate | MRR | Interpretation |
|---|---|---|
| 1.00 | 0.9–1.0 | Excellent — correct source at rank 1 almost always |
| 1.00 | 0.6–0.8 | Good — always found, sometimes at rank 2–3 |
| 0.7–0.9 | 0.5–0.7 | Acceptable — most questions answered correctly |
| < 0.7 | < 0.5 | Poor — consider adjusting chunk size, overlap, or TOP_K |

---

## Step 10 — Results Table + Save to JSON

### Results summary table

A pandas DataFrame is built from all test results and printed with the following columns:

| Column | Description |
|---|---|
| `Question` | First 55 characters of the question |
| `RAG Status` | `Answered` or `No Info` (below threshold) |
| `Best Sim` | Best similarity score from retrieved chunks |
| `Retrieval` | `Hit @ rank N` or `Miss` (N/A for out-of-scope) |
| `Sources Used` | Comma-separated list of source filenames (without `.pdf`) |
| `Answer Preview` | First 75 characters of Claude's answer |

### Saved output

All results are saved to `data/rag/rag_test_results.json` as a flat list. Each record contains:

```json
{
  "question":        "What is the cash transaction limit...",
  "answer":          "The cash transaction limit for HBL...",
  "sources":         ["HBL-FAQs-Home-Remittance.pdf"],
  "best_similarity": 0.7812,
  "status":          "answered",
  "hit":             true,
  "hit_rank":        1,
  "rr":              1.0,
  "expected_source": "HBL-FAQs-Home-Remittance"
}
```

A final count cell prints total questions tested, how many were answered, and how many triggered the below-threshold fallback.

---

## Data Flow Summary

```
User Question (string)
      │
      ▼  embed_model.encode([question], normalize_embeddings=True)
Question Vector  (1 × 384, float32, L2-normalized)
      │
      ▼  faiss.index.search(question_vec, TOP_K=3)
Top-3 Chunk Indices + Cosine Similarity Scores
      │
      ▼  check_relevance() → best_similarity >= SIM_THRESHOLD (0.3)?
      │
      ├── NO (below threshold)
      │       └─ return fallback answer, status = "below_threshold"
      │
      └── YES (relevant)
              │
              ▼  build_prompt(question, chunks)
         RAG Prompt  (system instruction + labelled context + question)
              │
              ▼  anthropic.client.messages.create(model, prompt)
         Claude's Answer  (grounded, context-only)
              │
              ▼
         Return dict: { question, answer, sources, similarity, status }
              │
              ▼  (batch mode only)
         data/rag/rag_test_results.json
```

---

## Pipeline Summary

| Component | Choice | Reason |
|---|---|---|
| Embedding model | `all-MiniLM-L6-v2` | Fast, free, strong semantic quality for English FAQ |
| Vector store | FAISS `IndexFlatIP` | No server needed — cosine sim via dot product on L2-normalized vectors |
| LLM | Claude `claude-sonnet-4-6` (Anthropic) | High accuracy, strong instruction following, low hallucination |
| Similarity metric | Cosine similarity | Standard metric for semantic search |
| Threshold | `0.3` | Cleanly separates in-domain banking queries from out-of-domain questions |
| Chunk retrieval | Top-3 | Best balance of context richness vs noise |
| Prompt strategy | Context-only grounding | Prevents LLM from using training knowledge instead of source documents |

---

## Dependencies

```
faiss-cpu            # Vector index and similarity search
sentence-transformers # Embedding model (all-MiniLM-L6-v2)
anthropic            # Claude LLM API
numpy                # Vector math and array operations
pandas               # Results summary table
python-dotenv        # Load ANTHROPIC_API_KEY from .env file
```

Install all at once:

```bash
pip install faiss-cpu sentence-transformers anthropic numpy pandas python-dotenv
```