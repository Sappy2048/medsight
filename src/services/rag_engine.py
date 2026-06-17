import json
import logging
import os
from pathlib import Path
from typing import Optional
from xmlrpc import client

from qdrant_client import QdrantClient, models
import re
from src.config import (
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)

logger = logging.getLogger(__name__)

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384

_client: Optional[QdrantClient] = None

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        # Lock in the FastEmbed model at the client level
        _client.set_model(FASTEMBED_MODEL) 
        logger.info("Qdrant client initialised and model set")
    return _client

def _ensure_collection(client):
    pass

def ingest_guidelines():
    # ... (Keep your existing JSON loading logic here) ...
    
    client = _get_client()
    
    REBUILD_COLLECTION = True
    if REBUILD_COLLECTION:
        try:
            client.delete_collection(collection_name=QDRANT_COLLECTION)
            print(f"🧹 Dropped old collection: {QDRANT_COLLECTION}")
        except Exception:
            pass # Collection doesn't exist yet

    documents = []
    metadata_payloads = []

    with open("data/icmr_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    for chunk in chunks:
        raw_text = chunk.get("text", "")
        # The safer Regex: fixes "4.12.8Negative" without breaking other formatting
        cleaned_text = re.sub(r'(?<=[a-z0-9])(?=[A-Z])', ' ', raw_text)
        
        drug = chunk.get("drug", "Unknown")
        category = chunk.get("category", "None")
        
        # The Rich Embedding: Massive boost to retrieval accuracy
        rich_context = f"Drug: {drug}\nCategory: {category}\nText: {cleaned_text}"
        documents.append(rich_context)
        
        # Clean Metadata Payload (No mutation)
        metadata_payloads.append({
            "drug": drug,
            "category": category,
            "text": cleaned_text,
            "source": chunk.get("source", "ICMR_Guidelines")
        })

    # Let Qdrant completely handle the math and collection creation under the hood
    client.add(
        collection_name=QDRANT_COLLECTION,
        documents=documents,             # <-- Uses our new Rich Embeddings!
        metadata=metadata_payloads,      # <-- Uses our clean metadata!
        ids=list(range(len(chunks)))
    )

    logger.info(f"Ingestion complete — {len(chunks)} points stored")
    
    # Put this right below client.add(...)
    try:
        client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name="drug",     field_schema=models.PayloadSchemaType.KEYWORD)
        client.create_payload_index(collection_name=QDRANT_COLLECTION, field_name="category", field_schema=models.PayloadSchemaType.KEYWORD)
        print("✅ Rebuilt payload indexes!")
    except Exception as e:
        print(f"Index creation note: {e}")

# ── Public: Retrieval ─────────────────────────────────────────────────────────

# ── Public: Retrieval ─────────────────────────────────────────────────────────

def retrieve_context(
    drug_name: str,
    query: str,
    limit: int = 3,
    client: Optional[QdrantClient] = None,
) -> list[dict]:
    """
    Retrieve relevant ICMR chunks for a drug + query.
    Filters by drug name, ranks by cosine similarity.
    Falls back to unfiltered search if no drug-specific chunks found.
    """
    if client is None:
        client = _get_client()
    
    # Ensure FastEmbed model is set if not already
    try:
        # Check if model is initialized (internal check for QdrantClient)
        # Use getattr to avoid Pylance error on private attribute access
        model_name = getattr(client, "_model_name", None)
        if model_name is None:
            client.set_model(FASTEMBED_MODEL)
    except Exception:
        pass

    drug_filter = models.Filter(
        must=[
            models.FieldCondition(
                key="drug",
                match=models.MatchValue(value=drug_name),
            )
        ]
    )

    try:
        # Using the FastEmbed-specific query method
        results = client.query(
            collection_name=QDRANT_COLLECTION,
            query_text=query,
            query_filter=drug_filter,
            limit=limit,
        )
    except Exception as e:
        logger.warning(f"Filtered search failed for {drug_name}: {e}")
        results = []

    # Fallback — no drug-specific chunks
    if not results:
        logger.info(f"No chunks for {drug_name}, using fallback search")
        try:
            results = client.query(
                collection_name=QDRANT_COLLECTION,
                query_text=query,
                limit=limit,
            )
        except Exception as e:
            logger.warning(f"Fallback search failed: {e}")
            return []

    # FastEmbed stores our payload inside the metadata object
    # Return the full payload so the Impact Agent has context
    chunks = [
        {
            "text": r.metadata.get("text", ""),
            "drug": r.metadata.get("drug", "unknown"),
            "category": r.metadata.get("category", "general"),
            "source": r.metadata.get("source", "ICMR")
        }
        for r in results if r.metadata
    ]
    logger.info(f"Retrieved {len(chunks)} chunks for {drug_name}")
    return chunks

# ── CLI for testing ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) > 1 and sys.argv[1] == "ingest":
        ingest_guidelines()
        print("Done.")

    elif len(sys.argv) > 2 and sys.argv[1] == "search":
        drug  = sys.argv[2]
        query = sys.argv[3] if len(sys.argv) > 3 else f"drug interactions {drug}"
        hits = retrieve_context(drug, query)
        print(f"\n--- Results for {drug} ---")
        for i, chunk in enumerate(hits, 1):  # <--- Changed to hits
            # Safely grab the text and metadata from the new dictionary
            text_snippet = chunk.get("text", "")[:300]
            chunk_drug = chunk.get("drug", "Unknown")  # <--- Changed to chunk_drug
            category = chunk.get("category", "None")
            
            print(f"\n[{i}] DRUG: {chunk_drug} | CATEGORY: {category}")
            print(f"    TEXT: {text_snippet}...")
    else:
        print("Usage:")
        print("  python rag_engine.py ingest")
        print("  python rag_engine.py search Azithromycin 'drug interaction warning'")