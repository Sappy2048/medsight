# ── Standard Library ──────────────────────────────────────────────────────────
import asyncio
import json
import logging
import re
import sys
from typing import Optional

# ── Third-Party ───────────────────────────────────────────────────────────────
from qdrant_client import QdrantClient, models

# ── Internal ──────────────────────────────────────────────────────────────────
from src.config import (
    QDRANT_COLLECTION,
    QDRANT_API_KEY,
    QDRANT_URL,
)

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────

FASTEMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBEDDING_DIM   = 384

# Module-level singleton — avoids re-initialising the FastEmbed model on every call
_client: Optional[QdrantClient] = None


# ── Client Initialisation ─────────────────────────────────────────────────────

def _get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        _client.set_model(FASTEMBED_MODEL)
        logger.info("Qdrant client initialised and FastEmbed model set: %s", FASTEMBED_MODEL)
    return _client


def _ensure_collection(client: QdrantClient) -> None:
    """
    Idempotently guarantee that QDRANT_COLLECTION exists with the correct
    FastEmbed vector parameters.  Safe to call before every ingest run.
    """
    existing = {c.name for c in client.get_collections().collections}
    if QDRANT_COLLECTION in existing:
        logger.info("Collection '%s' already exists — skipping creation.", QDRANT_COLLECTION)
        return

    # Let the client generate the correct named vector schema for our specific model
    client.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=client.get_fastembed_vector_params(),
    )
    logger.info(
        "Created collection '%s' (FastEmbed named vectors configured).",
        QDRANT_COLLECTION,
    )

# ── Ingestion ─────────────────────────────────────────────────────────────────

def ingest_guidelines(chunks_path: str = "data/icmr_chunks.json") -> None:
    """
    Load pre-processed ICMR chunks from *chunks_path*, embed them via
    FastEmbed, and upsert into Qdrant.  Drops and rebuilds the collection on
    every run so the index is always consistent with the source file.
    """
    client = _get_client()

    # ── Drop & recreate for a clean rebuild ──────────────────────────────────
    try:
        client.delete_collection(collection_name=QDRANT_COLLECTION)
        logger.info("Dropped old collection: %s", QDRANT_COLLECTION)
    except Exception:
        pass  # Collection did not exist yet — that is fine

    _ensure_collection(client)

    # ── Load source data ──────────────────────────────────────────────────────
    try:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
    except FileNotFoundError:
        logger.error(
            "ICMR chunks file not found at '%s'. "
            "Run the pre-processing script first, then retry ingestion.",
            chunks_path,
        )
        return

    # ── Build rich documents & payloads ──────────────────────────────────────
    documents: list[str] = []
    metadata_payloads: list[dict] = []

    for chunk in chunks:
        raw_text     = chunk.get("text", "")
        # Safely insert spaces between camelCase/number boundaries without
        # breaking acronyms (e.g. "4.12.8NegativeFinding" → "4.12.8 Negative Finding")
        cleaned_text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", raw_text)

        drug     = chunk.get("drug",     "Unknown")
        category = chunk.get("category", "None")

        # Prepend structured context so the embedding captures drug/category
        # semantics in addition to the raw text — improves retrieval precision.
        rich_context = f"Drug: {drug}\nCategory: {category}\nText: {cleaned_text}"
        documents.append(rich_context)

        metadata_payloads.append({
            "drug":     drug,
            "category": category,
            "text":     cleaned_text,
            "source":   chunk.get("source", "ICMR_Guidelines"),
        })

    # ── Upsert via FastEmbed ──────────────────────────────────────────────────
    client.add(
        collection_name=QDRANT_COLLECTION,
        documents=documents,
        metadata=metadata_payloads,
        ids=list(range(len(chunks))),
    )
    logger.info("Ingestion complete — %d points stored in '%s'.", len(chunks), QDRANT_COLLECTION)

    # ── Payload indexes for fast drug/category filtering ─────────────────────
    for field in ("drug", "category"):
        try:
            client.create_payload_index(
                collection_name=QDRANT_COLLECTION,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )
        except Exception as e:
            logger.warning("Payload index creation note for field '%s': %s", field, e)

    logger.info("Payload indexes created for 'drug' and 'category'.")


# ── Public: Synchronous Retrieval ─────────────────────────────────────────────

def retrieve_context(
    drug_name: str,
    query: str,
    limit: int = 3,
    client: Optional[QdrantClient] = None,
) -> list[dict]:
    """
    Retrieve the top-*limit* ICMR chunks for *drug_name* ranked by cosine
    similarity to *query*.

    Strategy:
      1. Drug-filtered semantic search (preferred — high precision).
      2. Unfiltered fallback if no drug-specific chunks exist.
    """
    if client is None:
        client = _get_client()

    # Ensure the FastEmbed model is set on any externally supplied client
    if getattr(client, "_model_name", None) is None:
        try:
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

    # ── Primary: drug-filtered search ────────────────────────────────────────
    try:
        results = client.query(
            collection_name=QDRANT_COLLECTION,
            query_text=query,
            query_filter=drug_filter,
            limit=limit,
        )
    except Exception as e:
        logger.warning("Filtered search failed for '%s': %s", drug_name, e)
        results = []

    # ── Fallback: unfiltered search ───────────────────────────────────────────
    if not results:
        logger.info("No drug-specific chunks for '%s' — falling back to unfiltered search.", drug_name)
        try:
            results = client.query(
                collection_name=QDRANT_COLLECTION,
                query_text=query,
                limit=limit,
            )
        except Exception as e:
            logger.warning("Fallback search failed: %s", e)
            return []

    chunks = [
        {
            "text":     r.metadata.get("text",     ""),
            "drug":     r.metadata.get("drug",     "unknown"),
            "category": r.metadata.get("category", "general"),
            "source":   r.metadata.get("source",   "ICMR"),
        }
        for r in results
        if r.metadata
    ]
    logger.info("Retrieved %d chunk(s) for '%s'.", len(chunks), drug_name)
    return chunks


# ── Public: Async LangGraph Wrapper ──────────────────────────────────────────

async def get_clinical_context(drug_name: str, query: str) -> str:
    """
    Async wrapper around :func:`retrieve_context` for use inside a LangGraph
    node or async tool.

    * Offloads the synchronous Qdrant call to a thread pool via
      ``asyncio.to_thread`` so the event loop is never blocked.
    * Formats the returned chunks into a single, LLM-readable string payload.
    * Returns ``""`` on failure or when no relevant chunks are found, acting as
      a safe no-op for downstream agents.

    Example output::

        [Category: interaction] Drug: Warfarin | Text: Concurrent use with...
        [Category: contraindication] Drug: Warfarin | Text: Avoid in patients...
    """
    try:
        chunks: list[dict] = await asyncio.to_thread(retrieve_context, drug_name, query)
    except Exception as e:
        logger.warning("get_clinical_context failed for '%s': %s", drug_name, e)
        return ""

    if not chunks:
        return ""

    lines = [
        f"[Category: {c.get('category', 'general')}] "
        f"Drug: {c.get('drug', drug_name)} | "
        f"Text: {c.get('text', '').strip()}"
        for c in chunks
    ]
    return "\n".join(lines)


# ── CLI for quick testing ─────────────────────────────────────────────────────

if __name__ == "__main__":
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
            print(f"\n[{i}] DRUG: {chunk.get('drug', 'Unknown')} | CATEGORY: {chunk.get('category', 'None')}")
            print(f"    TEXT: {chunk.get('text', '')[:300]}...")
    else:
        print("Usage:")
        print("  python -m src.services.rag_engine ingest")
        print("  python -m src.services.rag_engine search Azithromycin 'drug interaction warning'")