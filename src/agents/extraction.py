"""
extraction.py — FDA Label Interaction Extraction Agent

Responsibility:
    Takes a FDALabelVersion (output of fda_client) and extracts all
    drug interaction records from the 4 interaction-relevant sections:
        - boxed_warning
        - contraindications
        - warnings
        - drug_interactions

    Single-phase LLM extraction: the model extracts drug mentions,
    recommendation text, surrounding warning context, raw severity phrase,
    AND severity score in one pass. No post-hoc fuzzy matching.

    The SEVERITY_ONTOLOGY is injected directly into the prompt so the LLM
    scores against the same 0-5 scale used by temporal.py — no translation
    layer needed.

Exposed function:
    async def extract_interactions(
        label:       FDALabelVersion,
        source_drug: str,
        llm_client: AsyncOpenAI,
    ) -> ExtractionResult
"""

import json
import logging
from typing import Optional, Literal, cast
import asyncio

from openai import AsyncOpenAI

from src.config import LLM_MODEL, SEVERITY_ONTOLOGY
from src.schemas.fda_schema import FDALabelVersion
from src.schemas.diff_schema import ExtractionResult, InteractionRecord

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2

# Maximum concurrent LLM calls when a section is split into multiple chunks.
# Keeps burst traffic under Together AI's RPM limit.
# Raise to 3-4 only if your API tier supports higher throughput.
_CHUNK_CONCURRENCY = 2

_INTERACTION_SECTIONS: list[
    Literal["boxed_warning", "contraindications", "warnings", "drug_interactions"]
] = ["boxed_warning", "contraindications", "warnings", "drug_interactions"]

InteractionSection = Literal[
    "boxed_warning",
    "contraindications",
    "warnings",
    "drug_interactions",
]

# ─── Prompt Builder ────────────────────────────────────────────────────────────

def _build_system_prompt() -> str:
    """
    Build the extraction system prompt with SEVERITY_ONTOLOGY injected.

    Injecting the ontology at build time (not hardcoded) means if config.py
    changes the scale, the prompt automatically stays in sync — no manual
    prompt editing required.
    """
    ontology_lines = "\n".join(
        f'  {score} = "{phrase}"'
        for phrase, score in sorted(SEVERITY_ONTOLOGY.items(), key=lambda x: x[1])
    )

    return f"""\
You are a clinical data extraction engine embedded in an FDA drug safety system.

You will receive a single section of an FDA drug label for a specific source drug.
Your job is to extract every named drug interaction mentioned in this text as a
structured JSON array.

SEVERITY SCALE (you must use this exact scale for severity_score):
{ontology_lines}

FIELD DEFINITIONS:
- target_drug:          The exact name of the interacting drug as written in the label.
- recommendation_text:  The single most actionable sentence describing what to do
                        about this interaction. Direct quote from label only.
- warning_text:         The full surrounding paragraph or warning block that contains
                        this interaction. Provides clinical context beyond the
                        recommendation sentence. Direct quote. May equal
                        recommendation_text if the warning is a single sentence.
- severity_text:        The exact phrase from the label that best captures severity
                        (e.g. "contraindicated", "avoid", "monitor closely").
                        Do NOT paraphrase. Copy verbatim from the text.
- severity_score:       Integer 0-5 using the severity scale above. You are
                        responsible for mapping the full clinical meaning of
                        severity_text to the correct score — read the surrounding
                        context, not just the keyword in isolation.

RULES (non-negotiable):
1. Extract ONLY interactions where a specific drug name is explicitly mentioned.
   Drug class warnings (e.g. "NSAIDs", "anticoagulants") without a specific
   named drug do NOT qualify — skip them.
2. severity_text must be copied verbatim from the label. Do not paraphrase.
3. severity_score must reflect the true clinical severity of the interaction
   as described in the full context — not just the literal keyword match.
   Example: "use is not recommended due to risk of fatal hemorrhage" should
   score 4 ("avoid"), not 1 ("monitor"), even if the word "avoid" is absent.
4. If no specific named drug interactions are present, return an empty array.
5. Never invent a drug name not present in the text.

OUTPUT FORMAT (strict JSON, no markdown, no explanation outside JSON):
{{
  "interactions": [
    {{
      "target_drug":         "<exact drug name from label>",
      "recommendation_text": "<direct quote — actionable sentence>",
      "warning_text":        "<direct quote — full surrounding paragraph>",
      "severity_text":       "<verbatim severity phrase from label>",
      "severity_score":      <integer 0-5>
    }}
  ]
}}
"""


# Build once at module load — ontology is static at runtime
_SYSTEM_PROMPT = _build_system_prompt()


# ─── Phase 1: Text Chunking ─────────────────────────────────────────────────

def _chunk_text(text: str, max_chars: int = 4000) -> list[str]:
    """
    Splits a large section string into chunks bounded by paragraph breaks (\n\n).

    Paragraph-level splitting is preferred because the extraction prompt asks the
    LLM to copy the "full surrounding paragraph" as `warning_text`. Splitting
    mid-sentence would destroy that context window.

    Fallback: a single paragraph larger than `max_chars` is emitted as-is so we
    never silently drop text. The LLM will still parse what it can and the retry
    logic will recover if it fails.

    Args:
        text:      Raw section text from FDALabelVersion.
        max_chars: Soft ceiling per chunk (~4000 chars ≈ ~1 000 tokens, safely
                   below the generation budget so the JSON response has room).

    Returns:
        List of non-empty chunk strings, in document order.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    for para in paragraphs:
        # Oversized single paragraph: flush current bucket first, then emit alone
        if len(para) > max_chars:
            if current_parts:
                chunks.append("\n\n".join(current_parts))
                current_parts = []
                current_len = 0
            chunks.append(para)
            continue

        if current_len + len(para) > max_chars and current_parts:
            chunks.append("\n\n".join(current_parts))
            current_parts = [para]
            current_len = len(para)
        else:
            current_parts.append(para)
            current_len += len(para)

    if current_parts:
        chunks.append("\n\n".join(current_parts))

    return chunks


# ─── Phase 2: Per-Chunk LLM Extraction ───────────────────────────────────────

async def _extract_chunk(
    section_name: InteractionSection,
    chunk_text: str,
    source_drug: str,
    llm_client: AsyncOpenAI,
) -> list[dict]:
    """
    Single LLM call for one text chunk of a label section.

    Separated from _extract_section so that chunking logic stays in the
    orchestrator and this function stays a pure "one LLM call → list[dict]".

    Returns:
        Raw list of interaction dicts on success.
        Empty list after all retries — never raises, section failure is non-fatal.
    """
    user_message = (
        f"Source drug: {source_drug}\n"
        f"Label section: {section_name}\n\n"
        f"Section text chunk:\n{chunk_text}"
    )

    last_error: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=4096,    # ~4x headroom now that each chunk is ~4 000 chars
            )

            content = response.choices[0].message.content or ""
            raw = json.loads(content)

            if isinstance(raw, list):
                interactions = raw
            elif isinstance(raw, dict):
                interactions = raw.get("interactions", [])
            else:
                raise ValueError(f"Unexpected JSON structure returned from LLM: {type(raw)}")

            if not isinstance(interactions, list):
                raise ValueError(
                    f"Expected 'interactions' list, got {type(interactions)}"
                )

            logger.debug(
                f"Chunk of '{section_name}' → {len(interactions)} interactions"
            )
            return interactions

        except (json.JSONDecodeError, ValueError, Exception) as e:
            last_error = e
            logger.warning(
                f"Chunk extraction attempt {attempt + 1}/{_MAX_RETRIES + 1} "
                f"for section '{section_name}' failed: {e}"
            )

    logger.error(
        f"Chunk of section '{section_name}' failed after all retries: {last_error}. "
        f"Skipping chunk — pipeline continues."
    )
    return []


# ─── Phase 3: Section Orchestrator ───────────────────────────────────────────

def _deduplicate_interactions(raw_list: list[dict]) -> list[dict]:
    """
    Deduplicates interactions that were extracted from multiple chunks of the
    same section — keeps the record with the highest `severity_score`.

    Deduplication key: (target_drug.lower(), section_name).
    This prevents a Warfarin→Aspirin interaction appearing 3 times because
    the same paragraph was caught in two overlapping chunks.

    Pure function — does not mutate input.
    """
    best: dict[tuple[str, str], dict] = {}

    for item in raw_list:
        drug_key = str(item.get("target_drug", "")).lower().strip()
        section_key = str(item.get("_section", ""))  # injected by orchestrator
        key = (drug_key, section_key)

        # Defensive score parsing — LLM may return null, empty string, or a
        # quoted number ("2"). A bare int() on any of those would ValueError
        # here, crashing aggregation before _assemble_records can act as firewall.
        try:
            raw_score = item.get("severity_score")
            current_score = int(raw_score) if raw_score is not None else 0
        except (ValueError, TypeError):
            current_score = 0

        existing = best.get(key)
        if existing is None:
            best[key] = item
        else:
            try:
                existing_score = int(existing.get("severity_score", 0))
            except (ValueError, TypeError):
                existing_score = 0

            if current_score > existing_score:
                best[key] = item

    return list(best.values())


async def _extract_section(
    section_name: InteractionSection,
    section_text: str,
    source_drug: str,
    llm_client: AsyncOpenAI,
) -> list[dict]:
    """
    Orchestrates extraction over one label section.

    For short sections (fits in a single chunk) this is a transparent pass-through
    to `_extract_chunk` — no overhead.

    For large sections (e.g. Warfarin drug_interactions) it:
        1. Splits text into paragraph-bounded chunks of ≤4 000 chars.
        2. Fans out all chunks concurrently via asyncio.gather.
        3. Flattens the results.
        4. Deduplicates by (target_drug, section) keeping the highest severity.

    Returns:
        Aggregated, deduplicated list of raw interaction dicts.
        Empty list on complete failure — never raises.
    """
    chunks = _chunk_text(section_text, max_chars=4000)

    if len(chunks) == 1:
        # Fast path — no chunking overhead for standard-sized sections
        return await _extract_chunk(section_name, chunks[0], source_drug, llm_client)

    logger.info(
        f"Section '{section_name}' split into {len(chunks)} chunks "
        f"(total chars: {len(section_text)}) — "
        f"concurrency cap: {_CHUNK_CONCURRENCY}."
    )

    # Semaphore is created per-section, not per-module, so concurrent
    # calls to extract_interactions (different label sections) each get
    # their own independent pool — they don't starve each other.
    sem = asyncio.Semaphore(_CHUNK_CONCURRENCY)

    async def _throttled_chunk(chunk: str) -> list[dict]:
        async with sem:
            return await _extract_chunk(section_name, chunk, source_drug, llm_client)

    tasks = [_throttled_chunk(chunk) for chunk in chunks]
    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)

    aggregated: list[dict] = []
    for i, result in enumerate(chunk_results):
        if isinstance(result, BaseException):
            logger.error(
                f"Chunk {i + 1}/{len(chunks)} of section '{section_name}' "
                f"raised: {result} — skipping chunk."
            )
            continue
        # Tag each raw dict with section name for the deduplication key
        for item in result:
            item["_section"] = section_name
        aggregated.extend(result)

    deduped = _deduplicate_interactions(aggregated)

    if len(aggregated) != len(deduped):
        logger.info(
            f"Section '{section_name}': {len(aggregated)} raw → "
            f"{len(deduped)} after deduplication."
        )

    # Strip the internal _section tag — _assemble_records receives
    # section_name as an explicit argument, not from the dict.
    for item in deduped:
        item.pop("_section", None)

    return deduped


# ─── Assembler ────────────────────────────────────────────────────────────────

def _assemble_records(
    raw_interactions: list[dict],
    section_name: InteractionSection,
    source_drug: str,
    version_date: str,
    spl_id: str,
) -> list[InteractionRecord]:
    """
    Converts raw LLM output dicts into validated InteractionRecord objects.

    severity_score is taken directly from the LLM output — no post-hoc
    mapping. Pydantic enforces int type. If the LLM returns a score outside
    0-5, it is clamped here before Pydantic sees it, rather than letting
    Pydantic raise a validation error and discard an otherwise valid record.

    Pydantic is the final firewall — missing required fields cause the record
    to be logged and skipped, never silently corrupted.
    """
    records: list[InteractionRecord] = []
    clean_source = source_drug.lower().strip()

    for raw in raw_interactions:
        try:
            target_drug_raw = raw.get("target_drug", "")

            # Skip records with no target drug
            if not target_drug_raw:
                continue

            clean_target = str(target_drug_raw).lower().strip()

            # Guardrail: 7B models can mistake the source drug for a target drug
            # when parsing self-referential warnings (e.g. "Azithromycin is
            # contraindicated in patients with prior use of azithromycin").
            # Drop any record where target is a substring of source or vice-versa.
            if clean_target == clean_source or clean_source in clean_target or clean_target in clean_source:
                logger.debug(
                    f"Filtered self-referential record: target='{target_drug_raw}' "
                    f"source='{source_drug}' section='{section_name}'"
                )
                continue

            # Clamp LLM score to valid ontology range — defensive, not trusting
            raw_score = int(raw.get("severity_score", 0))
            severity_score = max(0, min(5, raw_score))

            if raw_score != severity_score:
                logger.warning(
                    f"LLM returned out-of-range severity_score={raw_score} "
                    f"for target='{target_drug_raw}' — clamped to {severity_score}"
                )

            record = InteractionRecord(
                source_drug=source_drug,
                target_drug=raw["target_drug"].strip(),
                recommendation_text=raw["recommendation_text"].strip(),
                warning_text=raw.get("warning_text", "").strip() or None,
                severity_text=raw["severity_text"].strip(),
                severity_score=severity_score,
                version_date=version_date,
                spl_id=spl_id,
                section=section_name,
            )
            records.append(record)

        except (KeyError, ValueError, Exception) as e:
            logger.warning(
                f"Skipping malformed record from section '{section_name}': "
                f"{e} — raw={raw}"
            )

    return records


# ─── Public Entry Point ───────────────────────────────────────────────────────

async def extract_interactions(
    label: FDALabelVersion,
    source_drug: str,
    llm_client: AsyncOpenAI,
) -> ExtractionResult:
    """
    Main entry point for the Extraction Agent.

    Args:
        label:       FDALabelVersion from fda_client — contains spl_id,
                     effective_time, and the 4 parsed label sections.
        source_drug: The drug whose label this is (e.g. "Warfarin").
        llm_client: Shared AsyncOpenAI client — injected, never created here.

    Returns:
        ExtractionResult with all InteractionRecord objects found across
        all 4 sections. Empty interactions list is valid — not an error.

    Concurrency:
        All non-null sections are extracted concurrently via asyncio.gather.
        Section failures are isolated — one bad LLM call never kills the job.
    """
    section_tasks: list[tuple[InteractionSection, asyncio.Task[list[dict]]]] = []

    for section_name in _INTERACTION_SECTIONS:
        section_text: Optional[str] = getattr(label.sections, section_name, None)

        if not section_text:
            logger.debug(f"Section '{section_name}' is null — skipping")
            continue

        task = asyncio.create_task(
            _extract_section(
                section_name=section_name,
                section_text=section_text,
                source_drug=source_drug,
                llm_client=llm_client,
            )
        )
        section_tasks.append((section_name, task))

    results = await asyncio.gather(
        *[task for _, task in section_tasks],
        return_exceptions=True,
    )

    all_records: list[InteractionRecord] = []

    for (section_name, _), result in zip(section_tasks, results):
        if isinstance(result, BaseException):
            logger.error(
                f"Section '{section_name}' gather raised: {result} — skipping"
            )
            continue

        raw_list = cast(list[dict], result)
        records = _assemble_records(
            raw_interactions=raw_list,
            section_name=section_name,
            source_drug=source_drug,
            version_date=label.effective_time,
            spl_id=label.spl_id,
        )
        all_records.extend(records)

    logger.info(
        f"Extraction complete: {source_drug} | {label.spl_id} | "
        f"{len(all_records)} total records across {len(section_tasks)} sections"
    )

    return ExtractionResult(
        source_drug=source_drug,
        version_date=label.effective_time,
        spl_id=label.spl_id,
        interactions=all_records,
    )

