"""
etl_pipeline.py
───────────────
Offline ingestion script: ICMR PDF guidelines → `data/icmr_chunks.json`.

Pipeline stages
  1. PDF  → Markdown  (pymupdf4llm  — structure-aware, table-preserving)
  2. Markdown → JSON  (Qwen 2.5 via Together AI — structured extraction)
  3. Aggregation      (flat list of ClinicalChunk dicts → icmr_chunks.json)

Usage (from project root):
    python -m src.services.etl_pipeline
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import asyncio
import json
import logging
import re
import sys
import time
from pathlib import Path
from typing import List, Literal, cast

# ── Third-Party ───────────────────────────────────────────────────────────────
import pymupdf4llm
from openai import AsyncOpenAI
from pydantic import BaseModel, ValidationError

# ── Internal ──────────────────────────────────────────────────────────────────
from src.config import (
    LLM_MODEL,
    TOGETHER_API_KEY,
    TOGETHER_BASE_URL,
)

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("etl_pipeline")

# ── Together AI Client ────────────────────────────────────────────────────────

_llm_client = AsyncOpenAI(
    api_key=TOGETHER_API_KEY,
    base_url=TOGETHER_BASE_URL,
)

# ── Constants ─────────────────────────────────────────────────────────────────

# Approximate character budget per LLM call.  Qwen 2.5-7B has a 32 k token
# context; at ~4 chars/token that gives ~128 k chars — we stay well under with
# 12 k to leave headroom for the system prompt and JSON response.
MAX_CHARS_PER_CHUNK = 12_000

# Retry settings for LLM calls (exponential back-off)
MAX_RETRIES   = 3
RETRY_BASE_S  = 2       # seconds — doubles each attempt

# Output paths
OUTPUT_DIR  = Path("data")
OUTPUT_FILE = OUTPUT_DIR / "icmr_chunks.json"
PDF_DIR     = Path("data/raw_pdfs")


# ── Pydantic Schemas ──────────────────────────────────────────────────────────

class ClinicalChunk(BaseModel):
    drug:     str
    category: Literal["recommendation", "dosage_caution", "contraindication"]
    text:     str


class ExtractedDocument(BaseModel):
    chunks: List[ClinicalChunk]


# ── Stage 1: PDF → Markdown ───────────────────────────────────────────────────

def extract_markdown_from_pdf(pdf_path: Path) -> str:
    """
    Convert *pdf_path* to a single Markdown string using pymupdf4llm.

    pymupdf4llm preserves table structure, headers, and list formatting —
    critical for the dense ICMR guideline layout that standard text extractors
    mangle into noise.
    """
    logger.info("Extracting Markdown from: %s", pdf_path.name)
    try:
        markdown: str = cast(str, pymupdf4llm.to_markdown(str(pdf_path)))
        logger.info(
            "  → %d characters extracted from %s",
            len(markdown),
            pdf_path.name,
        )
        return markdown
    except Exception as exc:
        logger.error("Failed to extract PDF '%s': %s", pdf_path.name, exc)
        return ""


# ── Stage 2: Markdown → Structured JSON (LLM) ────────────────────────────────

def _split_markdown(markdown: str, max_chars: int = MAX_CHARS_PER_CHUNK) -> List[str]:
    """
    Split *markdown* into segments ≤ *max_chars* characters.

    Strategy (priority order):
      1. Split on '## ' level-2 headers — keeps related clinical content
         about one drug in the same segment wherever possible.
      2. If a resulting section still exceeds *max_chars* (e.g., giant tables),
         hard-split it at the nearest preceding blank line within the budget.

    This avoids the catastrophic context overflow that happens when you naïvely
    dump a 200-page PDF into a 7B parameter model.
    """
    # Split on major headers — keep the delimiter attached to the following text
    sections = re.split(r"(?=^## )", markdown, flags=re.MULTILINE)

    segments: List[str] = []
    for section in sections:
        if not section.strip():
            continue

        if len(section) <= max_chars:
            segments.append(section)
            continue

        # Hard-split oversized section at blank-line boundaries
        cursor = 0
        while cursor < len(section):
            end = cursor + max_chars
            if end >= len(section):
                segments.append(section[cursor:])
                break
            # Walk back to the nearest blank line so we don't cut mid-sentence
            split_at = section.rfind("\n\n", cursor, end)
            if split_at == -1:
                split_at = end  # No blank line found — hard cut
            segments.append(section[cursor:split_at])
            cursor = split_at

    logger.debug("Document split into %d segment(s).", len(segments))
    return segments


_SYSTEM_PROMPT = (
    "You are a strict clinical data extractor. Read the following medical guideline text.\n"
    "Identify any specific drugs or drug classes mentioned.\n"
    "IF NO SPECIFIC DRUGS ARE MENTIONED, YOU MUST RETURN AN EMPTY ARRAY: {\"chunks\": []}\n"
    "Do NOT create placeholder entries like 'No specific drugs mentioned'.\n"
    "For each drug found, extract the clinical advice and classify it strictly as one of:\n"
    "  - 'recommendation'   : general best-practice advice, treatment pathways\n"
    "  - 'dosage_caution'   : dose adjustment, renal/hepatic caution, age-related\n"
    "  - 'contraindication' : situations where the drug must not be used\n"
    "CRITICAL INSTRUCTIONS FOR THE 'text' FIELD:\n"
    "1. You MUST include the exact clinical preconditions (e.g., specific HbA1c percentages, lab values, or patient symptoms) that trigger the use of the drug.\n"
    "2. Synthesize the extraction into a single, highly readable, grammatically correct sentence.\n"
    "3. Aggressively ignore and remove any irrelevant layout noise, unrelated lab targets (like unrelated BP/LDL goals), or page numbers that bled into the text.\n"
    "Return ONLY a valid JSON object with a 'chunks' array containing these 3 keys: 'drug', 'category', 'text'."
)


async def _call_llm_with_retry(segment: str, segment_idx: int) -> ExtractedDocument:
    """
    Call the Together AI LLM for a single markdown *segment*.
    Retries up to MAX_RETRIES times with exponential back-off on transient
    failures (rate limits, JSON parse errors, validation errors).

    Returns an :class:`ExtractedDocument` (possibly with an empty chunk list).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = await _llm_client.chat.completions.create(
                model=LLM_MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": segment},
                ],
                temperature=0.1,   # Low temperature → deterministic, structured output
                max_tokens=2048,
            )

            raw_json = response.choices[0].message.content or "{}"
            raw_json = raw_json.strip()
            
            # DEFENSIVE PARSING: Clean any accidental markdown code blocks
            if raw_json.startswith("```json"):
                raw_json = raw_json[7:]
            elif raw_json.startswith("```"):
                raw_json = raw_json[3:]
            if raw_json.endswith("```"):
                raw_json = raw_json[:-3]
            raw_json = raw_json.strip()

            # DEFENSIVE PARSING: If the LLM returned a raw array, wrap it in our expected object
            if raw_json.startswith("[") and raw_json.endswith("]"):
                raw_json = f'{{"chunks": {raw_json}}}'

            parsed = ExtractedDocument.model_validate_json(raw_json)
            logger.debug(
                "  Segment %d → %d chunk(s) extracted.",
                segment_idx,
                len(parsed.chunks),
            )
            return parsed

        except ValidationError as exc:
            logger.warning(
                "  Segment %d | attempt %d/%d | Pydantic validation failed: %s",
                segment_idx, attempt, MAX_RETRIES, exc,
            )
        except json.JSONDecodeError as exc:
            logger.warning(
                "  Segment %d | attempt %d/%d | JSON decode error: %s",
                segment_idx, attempt, MAX_RETRIES, exc,
            )
        except Exception as exc:
            logger.warning(
                "  Segment %d | attempt %d/%d | LLM call error: %s",
                segment_idx, attempt, MAX_RETRIES, exc,
            )

        if attempt < MAX_RETRIES:
            wait = RETRY_BASE_S ** attempt
            logger.info("  Retrying segment %d in %.1fs…", segment_idx, wait)
            await asyncio.sleep(wait)

    logger.error("  Segment %d exhausted all retries — skipping.", segment_idx)
    return ExtractedDocument(chunks=[])


async def process_markdown_with_llm(
    markdown: str,
    source_name: str,
) -> List[dict]:
    """
    Split *markdown* into manageable segments, run each through the LLM
    concurrently, validate with Pydantic, and return a flat list of chunk dicts
    (with the *source* field injected).
    """
    segments = _split_markdown(markdown)
    if not segments:
        logger.warning("No segments produced for '%s' — skipping.", source_name)
        return []

    logger.info(
        "Processing '%s': %d segment(s) → LLM …",
        source_name,
        len(segments),
    )

    # Fan-out: all segments for a single PDF run concurrently.
    # Together AI's free tier has generous parallel request limits; if you hit
    # 429s the per-segment retry logic handles them.
    tasks = [
        _call_llm_with_retry(seg, idx)
        for idx, seg in enumerate(segments, start=1)
    ]
    results: List[ExtractedDocument] = await asyncio.gather(*tasks)

    # Flatten + inject source
    flat_chunks: List[dict] = []
    for doc in results:
        for chunk in doc.chunks:
            flat_chunks.append({
                "drug":     chunk.drug,
                "category": chunk.category,
                "text":     chunk.text,
                "source":   source_name,
            })

    logger.info(
        "  '%s' → %d valid chunk(s) extracted.",
        source_name,
        len(flat_chunks),
    )
    return flat_chunks


# ── Stage 3: Aggregation & Persistence ───────────────────────────────────────

async def run_etl_pipeline(pdf_directory: str = str(PDF_DIR)) -> None:
    """
    Entry point for the full ETL run.

    1. Discover all PDFs under *pdf_directory*.
    2. For each PDF: extract Markdown → call LLM → collect chunks.
    3. Write the aggregated list to ``data/icmr_chunks.json``.
    """
    pdf_dir = Path(pdf_directory)
    if not pdf_dir.exists():
        logger.error(
            "PDF directory '%s' does not exist. "
            "Create it and add ICMR guideline PDFs, then rerun.",
            pdf_dir,
        )
        sys.exit(1)

    pdf_files = sorted(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        logger.error(
            "No PDF files found in '%s'. "
            "Add ICMR guideline PDFs and rerun.",
            pdf_dir,
        )
        sys.exit(1)

    logger.info(
        "ETL pipeline started | %d PDF(s) found in '%s'",
        len(pdf_files),
        pdf_dir,
    )

    all_chunks: List[dict] = []
    t0 = time.monotonic()

    for pdf_path in pdf_files:
        logger.info("━━━ Processing: %s", pdf_path.name)

        # Stage 1 — synchronous (CPU-bound via C extension; not worth threading for 1 file)
        markdown = extract_markdown_from_pdf(pdf_path)
        if not markdown.strip():
            logger.warning("Empty markdown for '%s' — skipping.", pdf_path.name)
            continue

        # Stage 2 — async LLM calls
        chunks = await process_markdown_with_llm(markdown, source_name=pdf_path.name)
        all_chunks.extend(chunks)

    elapsed = time.monotonic() - t0
    logger.info(
        "━━━ All PDFs processed | %d total chunk(s) | %.1fs elapsed",
        len(all_chunks),
        elapsed,
    )

    # ── Write output ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)

    logger.info("✅ Saved %d chunk(s) → %s", len(all_chunks), OUTPUT_FILE)


# ── CLI Entry Point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    asyncio.run(run_etl_pipeline("data/raw_pdfs"))
