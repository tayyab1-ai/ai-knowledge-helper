import os
import json
import numpy as np
import faiss
from groq import Groq
from pathlib import Path
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict

# Load environment variables
load_dotenv()

app = FastAPI(title="RAG AI Knowledge Helper API")

# --- Configuration ---
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL = "llama-3.1-8b-instant"
TOP_K = 3
SIM_THRESHOLD = 0.3
MIN_CLUSTER_CHUNKS = TOP_K
FALLBACK_ANSWER = "No relevant information found in the available banking documents."

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
CLUSTERING_DIR = BASE_DIR / "data" / "clustering"

# Global variables for models and data
embed_model = None
index = None
chunks = None
cluster_to_indices = {}
cluster_centroids = {}
groq_client = None

class QueryRequest(BaseModel):
    question: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    best_similarity: float
    status: str
    cluster_info: Optional[Dict] = None

@app.on_event("startup")
def startup_event():
    global embed_model, index, chunks, cluster_to_indices, cluster_centroids, groq_client
    
    print("Initializing RAG Pipeline...")
    
    # Initialize Groq Client
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("WARNING: GROQ_API_KEY not found in environment.")
    groq_client = Groq(api_key=api_key)

    # Load Embedding Model
    print(f"Loading embedding model: {EMBED_MODEL}")
    embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    embed_model.max_seq_length = 128

    # Load FAISS Index
    faiss_path = EMBEDDINGS_DIR / "faiss_index.bin"
    if faiss_path.exists():
        index = faiss.read_index(str(faiss_path))
        print(f"FAISS index loaded with {index.ntotal} vectors.")
    else:
        print(f"WARNING: FAISS index not found at {faiss_path}")

    # Load Clustered Chunks
    clusters_path = CLUSTERING_DIR / "chunks_with_clusters.json"
    if clusters_path.exists():
        with open(clusters_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        print(f"Loaded {len(chunks)} clustered chunks.")
        
        # Build cluster index structures
        all_cluster_ids = [c["cluster_id"] for c in chunks]
        k_final = max(all_cluster_ids) + 1
        cluster_to_indices = {k: [] for k in range(k_final)}
        for position, chunk in enumerate(chunks):
            cluster_to_indices[chunk["cluster_id"]].append(position)
            
        # Reconstruct centroids
        if index:
            for k in range(k_final):
                positions = cluster_to_indices[k]
                cluster_vecs = np.array([index.reconstruct(int(pos)) for pos in positions], dtype=np.float32)
                centroid = cluster_vecs.mean(axis=0)
                centroid = centroid / np.linalg.norm(centroid)
                cluster_centroids[k] = centroid
            print(f"Cluster centroids reconstructed for {k_final} clusters.")
    else:
        print(f"WARNING: Clustered chunks not found at {clusters_path}")

def predict_cluster(question_vec: np.ndarray):
    q_vec = question_vec[0]
    scores = {k: float(np.dot(q_vec, cluster_centroids[k])) for k in cluster_centroids}
    if not scores:
        return None
    
    predicted_id = max(scores, key=scores.get)
    predicted_label = next((c["cluster_label"] for c in chunks if c["cluster_id"] == predicted_id), "Unknown")
    cluster_size = len(cluster_to_indices[predicted_id])
    fallback = cluster_size < MIN_CLUSTER_CHUNKS
    
    return {
        "predicted_cluster_id": predicted_id,
        "predicted_cluster_label": predicted_label,
        "similarity_scores": scores,
        "cluster_size": cluster_size,
        "fallback": fallback
    }

def retrieve_chunks(question: str, top_k: int = TOP_K):
    question_vec = embed_model.encode(
        [question], normalize_embeddings=True, batch_size=1, show_progress_bar=False, convert_to_numpy=True
    ).astype(np.float32)

    cluster_pred = predict_cluster(question_vec)
    
    if not cluster_pred or cluster_pred["fallback"] or not index:
        if not index: return [], cluster_pred
        scores, indices = index.search(question_vec, top_k)
        result_positions = list(indices[0])
        result_scores = list(scores[0])
    else:
        candidate_positions = cluster_to_indices[cluster_pred["predicted_cluster_id"]]
        candidate_vecs = np.array([index.reconstruct(int(pos)) for pos in candidate_positions], dtype=np.float32)
        raw_scores = candidate_vecs.dot(question_vec[0])
        top_local_indices = np.argsort(raw_scores)[::-1][:top_k]
        result_positions = [candidate_positions[i] for i in top_local_indices]
        result_scores = [float(raw_scores[i]) for i in top_local_indices]

    retrieved = []
    for rank, (pos, score) in enumerate(zip(result_positions, result_scores), 1):
        if pos == -1: continue
        chunk_data = chunks[pos].copy()
        chunk_data["rank"] = rank
        chunk_data["similarity"] = round(float(score), 4)
        retrieved.append(chunk_data)
    
    return retrieved, cluster_pred

def build_prompt(question: str, retrieved_chunks: List[Dict]):
    context_parts = []
    for chunk in retrieved_chunks:
        header = f"[Source {chunk['rank']}: {chunk['bank_name']} | {chunk['source_file']} | Page {chunk['page_number']}]"
        context_parts.append(f"{header}\n{chunk['text']}")
    
    context = "\n\n---\n\n".join(context_parts)
    return f"""You are a helpful assistant for Pakistani banking customers.
Answer the question ONLY based on the context documents provided below.
If the answer is not in the context, say exactly: "I don't have enough information to answer this from the available documents."
Do NOT make up any information. Be concise and accurate.
If amounts, limits, or numbers are mentioned in the context, always include them in your answer.

=== CONTEXT DOCUMENTS ===
{context}
=== END OF CONTEXT ===

QUESTION: {question}

ANSWER"""

@app.post("/query", response_model=QueryResponse)
async def query_rag(request: QueryRequest):
    if not index or not chunks:
        raise HTTPException(status_code=503, detail="RAG system not fully initialized. Data files missing.")
    
    question = request.question
    retrieved, cluster_pred = retrieve_chunks(question)
    
    best_similarity = max([c["similarity"] for c in retrieved]) if retrieved else 0.0
    
    if best_similarity < SIM_THRESHOLD:
        return QueryResponse(
            question=question,
            answer=FALLBACK_ANSWER,
            sources=[],
            best_similarity=best_similarity,
            status="below_threshold",
            cluster_info=cluster_pred
        )
    
    prompt = build_prompt(question, retrieved)
    try:
        response = groq_client.chat.completions.create(
            model=LLM_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
            temperature=0.1
        )
        answer = response.choices[0].message.content.strip()
        sources = list(set(c["source_file"] for c in retrieved))
        
        return QueryResponse(
            question=question,
            answer=answer,
            sources=sources,
            best_similarity=best_similarity,
            status="success",
            cluster_info=cluster_pred
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "index_loaded": index is not None,
        "chunks_loaded": chunks is not None,
        "groq_api_key": os.getenv("GROQ_API_KEY") is not None
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
