import json
import logging
import os
from pathlib import Path
from typing import Optional

from qdrant_client import QdrantClient, models

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

def _ensure_collection() -> None:
    client = _get_client()
    existing = [c.name for c in client.get_collections().collections]

    if QDRANT_COLLECTION not in existing:
        # Create it using Qdrant's exact FastEmbed specifications
        client.create_collection(
            collection_name=QDRANT_COLLECTION,
            vectors_config=client.get_fastembed_vector_params(),
        )
        logger.info(f"Created collection: {QDRANT_COLLECTION}")
    else:
        logger.info(f"Collection exists: {QDRANT_COLLECTION}")

def ingest_guidelines(chunks_path: str = "data/icmr_chunks.json") -> None:
    """
    Modern ingestion using FastEmbed native add() method.
    """
    path = Path(chunks_path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {chunks_path}")

    with open(path, "r", encoding="utf-8") as f:
        chunks: list[dict] = json.load(f)

    if not chunks:
        raise ValueError("icmr_chunks.json is empty")

    client = _get_client()

    logger.info(f"Ingesting {len(chunks)} chunks...")

    # Let Qdrant completely handle the math and collection creation under the hood
    client.add(
        collection_name=QDRANT_COLLECTION,
        documents=[chunk["text"] for chunk in chunks],
        metadata=[
            {
                "drug":     chunk.get("drug", "unknown"),
                "category": chunk.get("category", "general"),
                "text":     chunk["text"],
                "source":   chunk.get("source", "ICMR_Guidelines"),
            }
            for chunk in chunks
        ],
        ids=list(range(len(chunks))),
    )

    logger.info(f"Ingestion complete — {len(chunks)} points stored")


# ── Public: Retrieval ─────────────────────────────────────────────────────────

# ── Public: Retrieval ─────────────────────────────────────────────────────────

def retrieve_context(
    drug_name: str,
    query: str,
    limit: int = 3,
) -> list[str]:
    """
    Retrieve relevant ICMR chunks for a drug + query.
    Filters by drug name, ranks by cosine similarity.
    Falls back to unfiltered search if no drug-specific chunks found.
    """
    client = _get_client()

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
    chunks = [r.metadata["text"] for r in results if r.metadata]
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
        hits  = retrieve_context(drug, query)
        print(f"\n--- Results for {drug} ---")
        for i, chunk in enumerate(hits, 1):
            print(f"\n[{i}] {chunk[:300]}...")
    else:
        print("Usage:")
        print("  python rag_engine.py ingest")
        print("  python rag_engine.py search Azithromycin 'drug interaction warning'")