# AI Knowledge Helper — RAG-Based Pakistani Banking FAQ System

---

## Overview

This project implements a complete **Retrieval-Augmented Generation (RAG)** system designed to answer questions from **Pakistani banking FAQ documents**.

Instead of relying on a Large Language Model (LLM) alone, the system retrieves **relevant, domain-specific knowledge** from a curated dataset and uses it to generate **grounded, accurate, and context-aware responses**.

The system is built as a **full pipeline**, starting from raw FAQ documents and ending with a working question-answering interface.

---

## Problem Statement

Modern LLMs (like GPT or Claude) are powerful but suffer from two major limitations:

- ❌ They **do not have access to private or domain-specific data**
- ❌ They may **hallucinate incorrect answers**

This project solves both problems by implementing a **RAG architecture**, where:

1. Relevant information is retrieved from a trusted dataset  
2. The LLM generates answers using only that retrieved context  

---

## Why Pakistani Banking FAQs?

This dataset was chosen intentionally as a **perfect RAG use case**:

- ✅ LLMs have **zero prior knowledge** of internal banking policies  
- ✅ Data is already in **Question → Answer format**  
- ✅ Content is **dense, structured, and domain-specific**  
- ✅ Strong **real-world application** (banking chatbots)  

---

## System Architecture

### High-Level Flow

```
User Question
      ↓
Query Embedding (Sentence Transformer)
      ↓
FAISS Vector Search
      ↓
Relevant Chunks (cluster) Retrieved
      ↓
Prompt Construction
      ↓
LLM (Groq / LLaMA 3)
      ↓
Final Answer (Grounded Response)
```

---

## Project Structure

```
ai-knowledge-helper/
│
├── data/
│   ├── raw/  
│   ├── extracted/    
│   ├── cleaned/ 
│   ├── chunks/                     
│   ├── embeddings/
│   │     ├── faiss_index.bin         
│   │     └── chunks_with_metadata.json
│   ├── clusters/   
│   └── rag/
│         ├── rag_test_results.json   
│         └── rag_test_results-op.json   
│
├── notebooks/
│   ├── data-pipeline/  
│   │     ├── data_extraction.ipynb
│   │     ├── data_cleaning.ipynb
│   │     ├── chunking.ipynb
│   │     └── embedding.ipynb
│   ├── rag-pipeline/  
│   │     ├── rag-pipeline.ipynb        
│   │     └── rag-pipeline-op.ipynb
│   └── optimization/ 
│         └── clustering.ipynb
│
├── documentation/
│   ├── dataset-detail.md
│   ├── data-pipeline-detail.md
│   ├── clustering-detail.md
│   ├── rag-pipeline-detail.md
│   └── rag-pipeline-op-detail.md
│
├── api/ 
│   └── main.py
│
├── frontend/                       
│   └── app.py
│
├── requrements.txt
│
└── README.md
```

---

## Data Pipeline (Offline Processing)

The data pipeline transforms raw FAQ documents into a format usable by the RAG system.

### Steps:

### 1. Data Extraction
- Load raw banking FAQ documents  

### 2. Data Cleaning
- Remove noise  
- Normalize formatting  
- Ensure consistency  

### 3. Chunking
- Split text into **300–500 token chunks**  
- Preserve semantic meaning  

### 4. Embedding
- Model: `all-MiniLM-L6-v2`  
- Output: **384-dimensional vectors**  

### Output:

```
data/embeddings/
├── faiss_index.bin
└── chunks_with_metadata.json
```

---

## Clustering Pipeline (Optimization Layer)

Clustering is used to improve retrieval efficiency.

### Why Clustering?

Instead of searching all chunks:
- Group similar chunks into **topic clusters**  
- Reduce search space during retrieval  

### Method Used:

- Algorithm: **K-Means**  
- Initialization: **k-means++**  
- Embedding normalization applied  

### Benefits:

- Faster retrieval  
- More relevant results  
- Reduced noise  

---

## RAG Pipeline (Core System)

The RAG pipeline handles real-time question answering.

### Steps:

1. User query received  
2. Query converted to embedding  
3. FAISS retrieves top-k similar chunks  
4. Relevant context selected  
5. Prompt constructed  
6. LLM generates answer  

---

## RAG Pipeline v2 (Optimized)

An improved version of the base pipeline with:

### 🔹 Cluster-Aware Retrieval

Instead of searching all chunks:

1. Predict query cluster  
2. Restrict search to that cluster  
3. Retrieve only relevant subset  

### Benefits:

- Faster response time  
- Higher precision  
- Reduced irrelevant context  

---

## LLM Selection

### Final Choice: Groq (LLaMA 3.1 8B Instant)

### Why?

- ✅ Free and fast inference  
- ✅ Low latency  
- ✅ Easy integration  
- ✅ Good grounding with proper prompts  

### Alternatives Considered:

- GPT-4o → rejected (cost)  
- Claude → rejected (rate limits / access issues)  

---

## Evaluation Metrics

The system is evaluated using:

- **Hit Rate** → Was the correct chunk retrieved?  
- **MRR (Mean Reciprocal Rank)** → Ranking quality  

---

## Key Features

- ✅ End-to-end RAG pipeline  
- ✅ FAISS-based semantic search  
- ✅ Cluster-aware retrieval (v2)  
- ✅ Grounded LLM responses  
- ✅ Modular pipeline design  
- ✅ Fully documented system  

---

## Installation

### 1. Clone Repository

```bash
git clone <your-repo-link>
cd ai-knowledge-helper
```

### 2. Install Dependencies

```bash
pip install faiss-cpu sentence-transformers numpy pandas python-dotenv
```

---

## Usage

### Step 1 — Run Data Pipeline

Run notebooks in order:

```
01 → 02 → 03 → 04 → 05
```

### Step 2 — Run RAG Pipeline

```
rag-pipeline.ipynb
```

or

```
rag-pipeline-v2.ipynb
```

---

## Example Query

```
What is the daily withdrawal limit for Meezan Bank ATM?
```

### Output:

- Retrieved relevant chunk  
- Generated grounded answer using LLM  

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment
Create a `.env` file in the root directory and add your Groq API key:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. Run the Backend (FastAPI)
```bash
cd API
uvicorn main:app --reload --port 8000
```

### 4. Run the Frontend (Streamlit)
In a new terminal:
```bash
cd frontend
streamlit run app.py
```

---

## Design Decisions Summary

- Use of **RAG instead of fine-tuning**  
- Selection of **MiniLM embeddings for efficiency**  
- Adoption of **FAISS for fast similarity search**  
- Use of **K-Means clustering for optimization**  
- Choice of **Groq LLM for cost-effective inference**  

---

## Future Improvements

- 🔹 Hybrid search (BM25 + embeddings)  
- 🔹 Better re-ranking models  
- 🔹 UI enhancements  
- 🔹 Real-time API deployment  
- 🔹 Multi-document ingestion  

---

## Conclusion

This project demonstrates a **complete, production-style RAG system**, solving a real-world problem using:

- Information Retrieval  
- Machine Learning  
- Large Language Models  

It is designed to be:
- Scalable  
- Modular  
- Practical for real-world deployment  

---

## Author

- Muhammad Tayyab

---