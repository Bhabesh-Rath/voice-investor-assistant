import os
import logging
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

CHROMA_PATH = "./data/chroma"
EMBED_MODEL = "all-MiniLM-L6-v2"

def init_vector_store():
    os.makedirs(CHROMA_PATH, exist_ok=True)
    client = PersistentClient(path=CHROMA_PATH)
    try:
        collection = client.get_collection("faq_kb")
    except Exception:
        collection = client.create_collection("faq_kb")
    return collection

def embed_and_store(chunks: list[dict]):
    collection = init_vector_store()
    if collection.count() > 0:
        logger.info("Vector store already populated. Skipping ingest.")
        return
        
    logger.info("Embedding & storing chunks...")
    model = SentenceTransformer(EMBED_MODEL)
    
    texts = [c["text"] for c in chunks]
    ids = [c["id"] for c in chunks]
    metas = [c["metadata"] for c in chunks]
    
    embeddings = model.encode(texts, show_progress_bar=False).tolist()
    collection.add(ids=ids, embeddings=embeddings, metadatas=metas, documents=texts)
    logger.info(f"Stored {len(ids)} chunks in ChromaDB")

def query_kb(transcript: str, top_k: int = 3) -> str:
    if not transcript.strip():
        raise ValueError("Empty transcript provided for retrieval")
        
    collection = init_vector_store()
    model = SentenceTransformer(EMBED_MODEL)
    
    query_emb = model.encode([transcript]).tolist()
    results = collection.query(query_embeddings=query_emb, n_results=top_k)
    
    context_parts = []
    for i, doc in enumerate(results["documents"][0]):
        faq_num = results["metadatas"][0][i].get("faq_number", "?")
        context_parts.append(f"[FAQ Q{faq_num}]\n{doc.strip()}")
        
    return "\n\n".join(context_parts) if context_parts else "No relevant context found in the knowledge base."