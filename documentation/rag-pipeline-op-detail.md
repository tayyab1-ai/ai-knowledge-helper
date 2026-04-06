# RAG Pipeline v2 — Complete Documentation

## Table of Contents

1. [What is RAG Pipeline v2](#1-what-is-rag-pipeline-v2)
2. [v1 vs v2 — What Changed and Why](#2-v1-vs-v2--what-changed-and-why)
3. [Complete Architecture](#3-complete-architecture)
4. [Step-by-Step Pipeline Walkthrough](#4-step-by-step-pipeline-walkthrough)
5. [Step 1 — Loading Artifacts](#5-step-1--loading-artifacts)
6. [Step 2 — Building Cluster Index Structures](#6-step-2--building-cluster-index-structures)
7. [Step 3 — Cluster Prediction](#7-step-3--cluster-prediction)
8. [Step 4 — Cluster-Aware Retrieval](#8-step-4--cluster-aware-retrieval)
9. [Step 5 — Relevance Threshold Check](#9-step-5--relevance-threshold-check)
10. [Step 6 — Prompt Construction](#10-step-6--prompt-construction)
11. [Step 7 — LLM Call via Groq](#11-step-7--llm-call-via-groq)
12. [Fallback Mechanism](#12-fallback-mechanism)
13. [Evaluation — Hit Rate and MRR](#13-evaluation--hit-rate-and-mrr)
14. [v1 vs v2 Comparison](#14-v1-vs-v2-comparison)
15. [Output Files](#15-output-files)
16. [Configuration Reference](#16-configuration-reference)
17. [Design Decisions Summary](#17-design-decisions-summary)
18. [Full Project File Map](#18-full-project-file-map)

---

## 1. What is RAG Pipeline v2

RAG Pipeline v2 is an upgraded version of `rag_pipeline_groq.ipynb`. It keeps every component of v1 — the embedding model, FAISS index, Groq LLM call, prompt structure, relevance threshold, evaluation metrics — and adds one significant improvement: **cluster-aware retrieval**.

Instead of searching all 207 chunks every time a question is asked, v2 first predicts which topic cluster the question belongs to using the cluster centroids computed in `05_clustering.ipynb`, then scopes its FAISS search to only the chunks in that cluster. This reduces the search space from 207 chunks to approximately 30-40 chunks per query.

The result is a retrieval step that is faster, more focused, and less likely to surface chunks from unrelated topics.

---

## 2. v1 vs v2 — What Changed and Why

### The Problem with v1 Retrieval

In v1, every question triggers a full FAISS search across all 207 chunks. FAISS is fast, so this is not a performance bottleneck for 207 vectors. The problem is **retrieval precision**.

Consider a question like "What is the profit rate for Meezan investment account?" The correct answer lives in the "Profit Rates and Investment Products" cluster. But the full search also scores and ranks chunks from the "Remittance" cluster, the "Digital Banking" cluster, and the "Loan and Financing" cluster. If any of those chunks happen to score slightly higher due to shared vocabulary — the word "account" appears in many banking contexts — they can crowd out the genuinely relevant chunks in the top-3.

### What v2 Changes

v2 adds a cluster prediction step before the FAISS search. The question vector is compared against the 6 cluster centroids and the closest centroid is identified. Then FAISS only searches the 30-40 chunks that belong to that cluster.

This means:
- Chunks from irrelevant topics are never considered
- The top-3 results come from a semantically coherent neighborhood of the embedding space
- The LLM receives context that is more focused on the actual topic of the question

### What v2 Does Not Change

Everything downstream of retrieval is identical:
- Relevance threshold check — same logic and threshold value
- Prompt format and system instructions — unchanged
- Groq API call — same model, same parameters
- Evaluation metrics — same Hit Rate and MRR calculations
- Test questions — same 8 questions for fair comparison

### Why Not Use a Separate FAISS Index Per Cluster

One alternative design would be to build K separate FAISS indexes, one per cluster, and search only the relevant index at query time. This would be cleaner in theory but adds complexity:

- K index files to save, load, and keep synchronized with the chunk data
- More code to manage index creation and loading
- No real performance benefit over the approach used — for clusters of 30-40 vectors, manual dot product scoring is equally fast as FAISS search

Instead, v2 uses `index.reconstruct(position)` to retrieve stored vectors from the full FAISS index by position, then scores them manually. This keeps one index file and avoids extra file management.

---

## 3. Complete Architecture

```
INPUT: User question (string)
         |
         v
+-------------------------+
|   Embedding Model       |
|   all-MiniLM-L6-v2      |
|   device = cpu          |
|   max_seq_length = 128  |
+-------------------------+
         |
         | question_vec (shape: 1 x 384, L2-normalized)
         v
+-------------------------+
|   Cluster Prediction    |
|                         |
|   For each cluster k:   |
|     score_k = dot(      |
|       question_vec,     |
|       centroid_k        |
|     )                   |
|   predicted = argmax(k) |
+-------------------------+
         |
         | predicted_cluster_id, cluster_size, fallback flag
         v
+-----------------------------+
|   Cluster-Aware Retrieval   |
|                             |
|   if fallback:              |
|     FAISS search(207 vecs)  |
|   else:                     |
|     Get ~30-40 positions    |
|     Reconstruct vectors     |
|     Score vs question_vec   |
|     Return Top-K            |
+-----------------------------+
         |
         | top-3 chunk dicts with similarity scores
         v
+-------------------------+
|   Relevance Check       |
|   best_sim >= 0.3?      |
+-------------------------+
    |             |
   Yes            No
    |             |
    v             v
Build Prompt   Return FALLBACK_ANSWER
    |           (skip LLM call)
    v
+-------------------------+
|   Groq API Call         |
|   llama-3.1-8b-instant  |
|   max_tokens = 300      |
|   temperature = 0.1     |
|   top_p = 0.9           |
+-------------------------+
         |
         v
OUTPUT: Structured result dict
  {
    question, answer, sources,
    best_similarity, status,
    cluster_info, _retrieved
  }
```

---

## 4. Step-by-Step Pipeline Walkthrough

The `rag_answer_v2(question, verbose=True)` function executes the following steps in sequence:

```
1. Embed question          → question_vec (1 x 384 float32)
2. Predict cluster         → predicted_cluster_id, fallback flag
3. Retrieve chunks         → top-3 chunk dicts from predicted cluster
4. Check relevance         → is best_similarity >= SIM_THRESHOLD?
5. Build prompt            → context string + question + instructions
6. Call Groq LLM           → raw answer string
7. Return result dict      → question, answer, sources, cluster_info
```

Each step is isolated in its own function. The `rag_answer_v2` function calls them in sequence and handles the fallback branches.

---

## 5. Step 1 — Loading Artifacts

The notebook loads four artifacts at startup. These are loaded once and reused for every query.

### Embedding Model

```python
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
embed_model.max_seq_length = 128
```

`device="cpu"` prevents slow CUDA initialization probing when no GPU is available. `max_seq_length=128` caps the tokenization length — FAQ chunks are short and 128 tokens is sufficient. This reduces compute on every `encode()` call.

### FAISS Index

```python
index = faiss.read_index("data/embeddings/faiss_index.bin")
```

The same index built in `04_embedding.ipynb`. It is an `IndexFlatIP` (flat inner product index) storing 207 normalized 384-dimensional vectors. This index is used both for fallback full-index search and for reconstructing stored vectors via `index.reconstruct(position)`.

### Clustered Chunks

```python
with open("data/clustering/chunks_with_clusters.json") as f:
    chunks = json.load(f)
```

This is the output of `05_clustering.ipynb` — the same 207 chunk dicts as `chunks_with_metadata.json` but with `cluster_id` and `cluster_label` added to every entry.

### Cluster Index Structures

Two data structures are built from the loaded chunks:

**`cluster_to_indices`** — a dict mapping each `cluster_id` to the list of FAISS index positions (chunk array positions) belonging to that cluster:

```python
cluster_to_indices = {
    0: [3, 7, 12, 19, ...],   # positions of chunks in cluster 0
    1: [0, 5, 8, 14, ...],    # positions of chunks in cluster 1
    ...
}
```

**`cluster_centroids`** — a dict mapping each `cluster_id` to its centroid vector:

```python
# Centroid = mean of all chunk vectors in that cluster, then re-normalized
for k in range(K_FINAL):
    positions   = cluster_to_indices[k]
    vecs        = [index.reconstruct(pos) for pos in positions]
    centroid    = mean(vecs)
    centroid    = centroid / ||centroid||_2   # re-normalize to unit length
    cluster_centroids[k] = centroid
```

Centroids are reconstructed from the FAISS index at startup rather than loaded from a saved file. This avoids needing to save the fitted K-Means model separately and keeps the number of required files minimal.

---

## 6. Step 2 — Building Cluster Index Structures

This step happens once at notebook startup — not at query time.

The number of clusters `K_FINAL` is determined automatically from the data:

```python
K_FINAL = max(c["cluster_id"] for c in chunks) + 1
```

This means no configuration change is needed in v2 if `K_FINAL` was changed in `05_clustering.ipynb` — v2 reads it directly from the data.

The `cluster_to_indices` and `cluster_centroids` dicts are printed with a summary at startup so it is easy to verify the cluster sizes match what was expected from the clustering notebook.

---

## 7. Step 3 — Cluster Prediction

The `predict_cluster(question_vec)` function determines which cluster to search for a given question.

**Algorithm:**

```python
def predict_cluster(question_vec):
    q = question_vec[0]   # shape (384,)
    
    # Compute cosine similarity to each centroid
    # Both q and centroid_k are L2-normalized, so dot product = cosine similarity
    scores = { k: dot(q, centroid_k) for k in range(K_FINAL) }
    
    # Predicted cluster = highest similarity
    predicted_id = argmax(scores)
    
    # Check if cluster has enough chunks for retrieval
    fallback = len(cluster_to_indices[predicted_id]) < MIN_CLUSTER_CHUNKS
    
    return predicted_id, scores, fallback
```

**Why dot product equals cosine similarity here:**

Both the question vector and all centroid vectors are L2-normalized to unit length. For two unit vectors a and b:

```
cosine_similarity(a, b) = dot(a, b) / (||a|| * ||b||) = dot(a, b) / (1 * 1) = dot(a, b)
```

The division cancels out, so the dot product directly gives cosine similarity. This is the same reason the FAISS index uses `IndexFlatIP` (inner product) rather than L2 distance.

**Computational cost:**

For K=6, this is 6 dot products of 384-dimensional vectors — 6 × 384 = 2,304 multiplications. This is negligible compared to any other operation in the pipeline.

**Output includes all cluster scores:**

The `similarity_scores` dict in the prediction result shows scores for all K clusters. This is logged when `verbose=True` and saved in the result dict, making it easy to see how confident the prediction was (large gap between top score and second score = confident prediction; small gap = ambiguous question).

---

## 8. Step 4 — Cluster-Aware Retrieval

The `retrieve_chunks_v2(question, top_k)` function handles retrieval.

**Normal path (no fallback):**

```python
# Get positions of chunks in the predicted cluster
candidate_positions = cluster_to_indices[predicted_cluster_id]  # e.g., 35 positions

# Reconstruct stored vectors for each position from FAISS index
candidate_vecs = [index.reconstruct(pos) for pos in candidate_positions]
# shape: (35, 384)

# Score each candidate against the question vector
raw_scores = candidate_vecs.dot(question_vec[0])  # shape: (35,)

# Sort descending, take top-K
top_indices = argsort(raw_scores)[::-1][:top_k]
```

**Why `index.reconstruct()` instead of a separate numpy array:**

The FAISS `IndexFlatIP` index stores the original vectors verbatim (no compression, no quantization). `index.reconstruct(i)` retrieves the vector that was added at position `i`. This means we can reuse the same index for both full search and targeted vector retrieval without storing the embeddings array separately at runtime.

**Fallback path:**

If the predicted cluster has fewer than `MIN_CLUSTER_CHUNKS = TOP_K = 3` chunks, the function falls back to a standard FAISS search across all 207 chunks:

```python
if fallback:
    scores, indices = index.search(question_vec, top_k)
    # same as v1
```

This protects against edge cases where a cluster is very small and cannot supply enough distinct results for top-K retrieval.

**Result format:**

Each retrieved chunk dict has two fields added before being returned:

```python
chunk["rank"]       = rank         # 1, 2, 3
chunk["similarity"] = score        # cosine similarity, rounded to 4 decimals
```

The `cluster_info` dict is returned alongside the chunk list so the caller can log which cluster was used and whether fallback was triggered.

---

## 9. Step 5 — Relevance Threshold Check

Identical to v1. The `check_relevance(retrieved)` function checks whether the best similarity score among the retrieved chunks exceeds the `SIM_THRESHOLD=0.3`.

**Why this check is still needed in v2:**

Even with cluster-aware retrieval, an out-of-scope question (e.g., "What is the capital of France?") will still be assigned to the closest cluster. The cluster assignment will have a low similarity score, and the chunks retrieved from that cluster will also have low similarity scores. The threshold check catches this case and returns the fallback answer without calling the LLM.

**Return values:**

| Condition | `is_relevant` | `status` |
|---|---|---|
| `best_sim >= 0.3` | `True` | `"relevant"` |
| `best_sim < 0.3` | `False` | `"irrelevant"` |
| No chunks retrieved | `False` | `"no_results"` |

---

## 10. Step 6 — Prompt Construction

Identical to v1. The `build_prompt(question, retrieved_chunks)` function formats the retrieved chunks as labelled context blocks and wraps them in a structured prompt.

**Prompt structure:**

```
You are a helpful assistant for Pakistani banking customers.
Answer the question ONLY based on the context documents provided below.
If the answer is not in the context, say exactly: "I don't have enough information..."
Do NOT make up any information. Be concise and accurate.
If amounts, limits, or numbers are mentioned in the context, always include them.

=== CONTEXT DOCUMENTS ===
[Source 1: Bank Name | source_file.pdf | Page N]
chunk text...

---

[Source 2: Bank Name | source_file.pdf | Page N]
chunk text...

---

[Source 3: Bank Name | source_file.pdf | Page N]
chunk text...
=== END OF CONTEXT ===

QUESTION: user question here

ANSWER
```

**Why this prompt works:**

The explicit "ONLY based on the context" instruction forces the LLM to stay grounded. The labelled source headers tell the model where each piece of information came from. The fallback phrase is specified verbatim so the response can be checked programmatically. The numbered sources make it easy to trace which chunk contributed to the answer.

**No change from v1:**

The only practical difference is that in v2, the three chunks passed to this function come from a smaller, more focused cluster rather than the full index. The prompt itself is identical.

---

## 11. Step 7 — LLM Call via Groq

The `call_llm(prompt)` function sends the prompt to the Groq API and returns the answer string.

**Model used:**

```
llama-3.1-8b-instant
```

This is the current production Llama 3.1 8B model on Groq. The original `llama3-8b-8192` used in early development was decommissioned by Groq in May 2025. `llama-3.1-8b-instant` is the official replacement and is equivalent in capability.

**API call:**

```python
response = groq_client.chat.completions.create(
    model      = LLM_MODEL,
    messages   = [{"role": "user", "content": prompt}],
    max_tokens = 300,
    temperature= 0.1,
    top_p      = 0.9,
)
return response.choices[0].message.content.strip()
```

**Parameter choices:**

| Parameter | Value | Reason |
|---|---|---|
| `max_tokens` | 300 | FAQ answers are short — 300 tokens is sufficient and avoids padding |
| `temperature` | 0.1 | Very low temperature keeps answers factual and grounded; reduces hallucination risk |
| `top_p` | 0.9 | Slightly restricted nucleus sampling — works with low temperature to produce focused outputs |

**Why Groq:**

Groq runs inference on custom LPU (Language Processing Unit) hardware. Response time for this model with 300 max tokens is typically 1-3 seconds per query, compared to 30-40 minutes for cold-start Ollama with `llama3:8b` locally. No local model download, no server management, no GPU required.

---

## 12. Fallback Mechanism

v2 has two distinct fallback layers:

### Fallback 1 — Small Cluster Fallback (Retrieval Level)

If the predicted cluster contains fewer than `MIN_CLUSTER_CHUNKS` (= `TOP_K` = 3) chunks, the retrieval step falls back to a full FAISS search across all 207 chunks. This ensures that top-K retrieval always has enough candidates to fill the top-3 result slots.

Logged as `fallback: True` in the `cluster_info` dict.

### Fallback 2 — Relevance Threshold Fallback (Pipeline Level)

If the best similarity score among all retrieved chunks is below `SIM_THRESHOLD = 0.3`, the question is considered out-of-scope and the pipeline returns:

```python
FALLBACK_ANSWER = "No relevant information found in the available banking documents."
```

The LLM call is skipped entirely. This prevents the model from hallucinating answers to questions the dataset cannot answer.

Logged as `status: "below_threshold"` in the result dict.

---

## 13. Evaluation — Hit Rate and MRR

v2 uses the same two retrieval quality metrics as v1, computed on the same 7 in-scope test questions.

### Hit Rate

For each in-scope question, check whether the expected source file appears anywhere in the top-K retrieved chunks. Hit Rate is the fraction of questions where this is true.

```
Hit Rate = number of questions with at least one hit / total in-scope questions
```

A Hit Rate of 1.0 means the correct source was retrieved for every question. A Hit Rate of 0.857 (6/7) means one question did not retrieve the expected source.

### MRR — Mean Reciprocal Rank

MRR measures not just whether the correct source was retrieved, but how high it was ranked. A hit at rank 1 is rewarded more than a hit at rank 3.

```
For each question:
    if hit at rank r: RR = 1/r
    if no hit:        RR = 0.0

MRR = mean(RR across all in-scope questions)
```

| Hit rank | Reciprocal Rank |
|---|---|
| Rank 1 | 1.000 |
| Rank 2 | 0.500 |
| Rank 3 | 0.333 |
| No hit | 0.000 |

### Why Reuse `_retrieved` Instead of Re-Running Retrieval

After the batch test run, the evaluation step needs the retrieved chunks for each question to check which sources were returned. Instead of calling `retrieve_chunks_v2()` again for each question (which would re-embed and re-search), the result dict stores the retrieved chunks under the `_retrieved` key during the main run. The evaluation step reads from there:

```python
retrieved_for_eval = r.get("_retrieved") or retrieve_chunks_v2(r["question"])[0]
```

This eliminates 7 redundant embed+search operations during evaluation.

---

## 14. v1 vs v2 Comparison

### What the Comparison Measures

Step 11 in the notebook loads the v1 results from `data/test/rag_test_results.json` and computes a side-by-side comparison on four metrics:

| Metric | Description |
|---|---|
| Hit Rate | Did the correct source appear in top-3? |
| MRR | How highly was the correct source ranked? |
| Chunks searched (avg) | Average number of chunks scored per question |
| Search space reduction | How much smaller is the v2 search space vs v1 |

### Expected Outcomes

In most cases v2 Hit Rate and MRR will be equal to or better than v1. If they are worse, possible causes:

- The cluster assignment for a particular question type is incorrect — the centroid for the predicted cluster is not close enough to that question's embedding, pointing to a cluster boundary issue
- `K_FINAL` is too high, causing the correct chunks to be split across two neighboring clusters
- The cluster labels in `05_clustering.ipynb` were not well-matched to the actual topic distribution of the data

In those cases, re-run `05_clustering.ipynb` with a different `K_FINAL`, update `CLUSTER_LABELS`, and re-run v2.

---

## 15. Output Files

| File | Contents | Location |
|---|---|---|
| `rag_test_results_v2.json` | All test question results with cluster info added | `data/test/` |

**Fields in `rag_test_results_v2.json`:**

```json
{
  "question":          "What documents are needed to open a Meezan Roshan account?",
  "answer":            "According to the context...",
  "sources":           ["Meezan-Bank-FAQs-Roshan-Digital-Account.pdf"],
  "best_similarity":   0.7234,
  "status":            "answered",
  "hit":               true,
  "hit_rank":          1,
  "rr":                1.0,
  "expected_source":   "Meezan-Bank-FAQs-Roshan-Digital-Account",
  "predicted_cluster": 0,
  "cluster_label":     "Account Opening & Requirements",
  "cluster_size":      35,
  "fallback_used":     false
}
```

The v1 results file (`rag_test_results.json`) is not modified.

---

## 16. Configuration Reference

| Parameter | Default | Description |
|---|---|---|
| `EMBED_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model — must match the one used in `04_embedding.ipynb` |
| `LLM_MODEL` | `llama-3.1-8b-instant` | Groq model name |
| `MAX_TOKENS` | `300` | Maximum tokens in LLM response |
| `TOP_K` | `3` | Number of chunks to retrieve per question |
| `SIM_THRESHOLD` | `0.3` | Minimum similarity score for a question to be considered in-scope |
| `MIN_CLUSTER_CHUNKS` | `3` (= `TOP_K`) | Minimum cluster size before falling back to full-index search |

---

## 17. Design Decisions Summary

| Component | Decision | Reason |
|---|---|---|
| Cluster prediction method | Centroid dot product | No model inference; K dot products is essentially free; consistent with L2-normalized embedding space |
| Vector reconstruction | `index.reconstruct()` | Avoids saving embeddings array separately; reuses existing FAISS index |
| Fallback threshold | `MIN_CLUSTER_CHUNKS = TOP_K` | A cluster with fewer chunks than TOP_K cannot produce enough distinct results |
| Centroids built at runtime | Computed from FAISS + chunk data | No need to save/load K-Means model; fewer required files |
| K_FINAL read from data | `max(cluster_id) + 1` | v2 automatically adapts if K_FINAL was changed in `05_clustering.ipynb` |
| Prompt unchanged from v1 | Same system instructions | The improvement is in retrieval scope, not in prompt engineering |
| Same test questions as v1 | Identical 8 questions | Fair comparison — same inputs, different retrieval |
| `_retrieved` stored in result | Avoids re-embedding at eval time | Eliminates 7 redundant encode+search calls during evaluation |

---

## 18. Full Project File Map

```
project/
|
├── notebooks/
|   ├── 01_extraction.ipynb           <- PDF text extraction
|   ├── 02_cleaning.ipynb             <- text cleaning and normalization
|   ├── 03_chunking.ipynb             <- text splitting into chunks
|   ├── 04_embedding.ipynb            <- embedding generation + FAISS index
|   ├── 05_clustering.ipynb           <- K-Means clustering on chunk embeddings
|   ├── rag_pipeline_groq.ipynb       <- RAG v1 — full index search
|   └── rag_pipeline_v2.ipynb         <- RAG v2 — cluster-aware retrieval (this notebook)
|
├── data/
|   ├── raw/                          <- original PDF files
|   ├── cleaned/                      <- cleaned text files
|   ├── chunks/                       <- chunked text files
|   |
|   ├── embeddings/
|   |   ├── faiss_index.bin           <- FAISS IndexFlatIP, 207 vectors x 384 dims
|   |   ├── embeddings.npy            <- raw float32 array, shape (207, 384)
|   |   └── chunks_with_metadata.json <- 207 chunks with bank/source/page fields
|   |
|   ├── clustering/
|   |   ├── chunks_with_clusters.json <- same 207 chunks + cluster_id + cluster_label
|   |   ├── cluster_summary.json      <- per-cluster stats
|   |   ├── cluster_ids.npy           <- raw cluster assignment array, shape (207,)
|   |   ├── elbow_plot.png            <- inertia vs K curve
|   |   ├── tsne_plot.png             <- 2D cluster visualization
|   |   └── cluster_distribution.png <- bar chart of cluster sizes
|   |
|   └── test/
|       ├── rag_test_results.json     <- v1 batch test results
|       └── rag_test_results_v2.json  <- v2 batch test results with cluster info
|
└── .env                              <- GROQ_API_KEY=your_key_here
```

**Run order for the complete project:**

```
01_extraction.ipynb
      |
      v
02_cleaning.ipynb
      |
      v
03_chunking.ipynb
      |
      v
04_embedding.ipynb
      |
      +---> rag_pipeline_groq.ipynb  (v1 baseline — run first to generate v1 results)
      |
      v
05_clustering.ipynb
      |
      v
rag_pipeline_v2.ipynb               (v2 — run after clustering to compare against v1)
```
