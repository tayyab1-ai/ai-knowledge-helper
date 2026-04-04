# Data Pipeline — AI Knowledge Helper (RAG System)

## Overview

This project builds a **Retrieval-Augmented Generation (RAG)** pipeline over Pakistani banking FAQ documents. The pipeline is split into **5 separate notebooks**, each responsible for one stage. Every notebook reads from the previous stage's output and writes its results to the `data/` folder.

```
data/
├── raw-data/       ← original PDFs (source)
├── extracted/      ← raw text per page       [Notebook 01]
├── cleaned/        ← cleaned text per page   [Notebook 02]
├── chunks/         ← text chunks             [Notebook 03]
└── embeddings/     ← vectors + metadata      [Notebook 04]
```

**Run order:** `01 → 02 → 03 → 04 → 05`

---

## Notebook 01 — Data Extraction

**File:** `01_data_extraction.ipynb`  
**Input:** `data/raw-data/*.pdf`  
**Output:** `data/extracted/<filename>.json`

### What it does

Reads every PDF in `raw-data/` and extracts text from each page using **PyMuPDF (`fitz`)**. The extractor processes one page at a time and stores the raw text along with metadata.

**Each extracted page record contains:**
- `bank_name` — readable bank name (e.g. `"Habib Bank Limited (HBL)"`)
- `source_file` — original PDF filename
- `page_number` — page number within the document
- `total_pages` — total pages in the document
- `raw_text` — raw extracted text (uncleaned)

A **bank name mapper** converts filename prefixes (`HBL`, `Meezan`, `ABL`, `Bank-Alfalah`, `State-Bank`) to full readable bank names. Each PDF is saved as its own JSON file in `data/extracted/`. The notebook ends with a summary table showing how many pages were extracted from each file.

**Libraries used:** `fitz` (PyMuPDF), `json`, `pathlib`

---

## Notebook 02 — Data Cleaning

**File:** `02_data_cleaning.ipynb`  
**Input:** `data/extracted/*.json`  
**Output:** `data/cleaned/<filename>.json`

### What it does

Applies a text cleaning pipeline to every raw page extracted in Notebook 01. PDF text extraction introduces many artifacts — this notebook fixes them before chunking.

**Cleaning steps applied to each page:**

| Step | What it fixes |
|---|---|
| Fix hyphenation | Rejoins words broken across lines (e.g. `remit-\ntance` → `remittance`) |
| Remove special chars | Strips non-printable unicode control characters |
| Remove page numbers | Removes lines like `Page 1 of 10` or standalone digit lines |
| Normalize whitespace | Collapses multiple spaces/tabs, reduces 3+ newlines to 2 |

After cleaning, pages with fewer than **80 characters** are discarded (near-empty pages — usually cover images, blank pages, or decorative pages with no useful text).

The output JSON has the same structure as extracted, but with `raw_text` replaced by `cleaned_text`. A before/after comparison cell is included to verify cleaning quality visually. The notebook reports total pages in, pages filtered out, and retention rate.

**Libraries used:** `re`, `json`, `pathlib`

---

## Notebook 03 — Chunking

**File:** `03_chunking.ipynb`  
**Input:** `data/cleaned/*.json`  
**Output:** `data/chunks/chunks.json`

### What it does

Splits each cleaned page's text into smaller, overlapping chunks using a **token-based sliding window**. Chunking is necessary because embedding models have a token limit, and smaller focused chunks retrieve more precisely than whole pages.

**Chunking parameters:**

| Parameter | Value | Reason |
|---|---|---|
| Chunk size | 400 tokens | Fits within embedding model limits; covers ~1–2 FAQ answers |
| Overlap | 50 tokens | Preserves context across chunk boundaries |
| Tokenizer | `cl100k_base` (tiktoken) | Same tokenizer used by OpenAI and most embedding models |

The sliding window moves forward by `chunk_size - overlap` tokens each step, so consecutive chunks share 50 tokens of context. All chunks from all files are merged into a **single flat list** saved as `chunks.json`.

**Each chunk record contains:**
- `chunk_id` — unique ID (`chunk_00000`, `chunk_00001`, ...)
- `bank_name`, `source_file`, `page_number` — inherited metadata
- `chunk_index` — position of chunk within its source page
- `token_count` — actual token count of this chunk
- `text` — the chunk text

The notebook ends with stats: total chunks, average/min/max token counts, and a per-bank breakdown.

**Libraries used:** `tiktoken`, `json`, `pathlib`, `collections`

---

## Notebook 04 — Embedding

**File:** `04_embedding.ipynb`  
**Input:** `data/chunks/chunks.json`  
**Output:** `data/embeddings/embeddings.npy`, `data/embeddings/chunks_with_metadata.json`, `data/embeddings/faiss_index.bin` *(optional)*

### What it does

Converts every text chunk into a **dense vector embedding** using a sentence transformer model. These vectors capture the semantic meaning of each chunk and enable similarity-based retrieval.

**Default model:** `all-MiniLM-L6-v2`
- 384-dimensional embeddings
- Fast inference, good quality for English FAQ text
- Embeddings are **L2-normalized** (cosine similarity = dot product, faster retrieval)

Chunks are embedded in **batches of 64** with a progress bar. The resulting matrix has shape `(N_chunks, 384)`.

**What gets saved:**

| File | Contents | Purpose |
|---|---|---|
| `embeddings.npy` | NumPy array `(N, 384)` float32 | Fast vector loading for retrieval |
| `chunks_with_metadata.json` | All chunk dicts (no vectors) | Text + metadata lookup by index |
| `faiss_index.bin` *(optional)* | FAISS IndexFlatIP | Fast approximate search for large collections |

A **sanity check cell** runs a test query (e.g. `"What is the daily cash withdrawal limit?"`) against the embeddings using cosine similarity and prints the top-3 most similar chunks to verify the embeddings are working correctly.

**Libraries used:** `sentence-transformers`, `numpy`, `faiss-cpu` (optional)

---


## Data Flow Summary

```
raw-data PDFs
     │
     ▼  [01_data_extraction] 

data/extracted/*.json       (raw text + metadata per page)
     │
     ▼  [02_data_cleaning] 

data/cleaned/*.json         (cleaned text + metadata per page)
     │
     ▼  [03_chunking] 

data/chunks/chunks.json     (flat list of all chunks across all docs)
     │
     ▼  [04_embedding]       

data/embeddings/
      ├── embeddings.npy             (N × 384 float32 matrix)
      ├── chunks_with_metadata.json  (text + metadata, no vectors)
      └── faiss_index.bin            (optional FAISS index)

```

---

## Dependencies

```
pymupdf                   # PDF text extraction
tiktoken                  # Token counting and chunking
sentence-transformers     # Embedding model
numpy                     # Embedding matrix operations
faiss-cpu                 # (Optional) fast vector search
```

Install all at once:

```bash
pip install pymupdf tiktoken sentence-transformers numpy faiss-cpu anthropic openai
```
