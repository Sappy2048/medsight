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


# ─── Phase 1: LLM Extraction ─────────────────────────────────────────────────

async def _extract_section(
    section_name: InteractionSection,
    section_text: str,
    source_drug: str,
    llm_client: AsyncOpenAI,
) -> list[dict]:
    """
    Single LLM call to extract all drug interactions from one label section.

    The LLM is responsible for:
        - Identifying named drug mentions
        - Extracting recommendation_text (actionable sentence)
        - Extracting warning_text (surrounding paragraph context)
        - Copying severity_text verbatim
        - Mapping severity_score using the injected ontology

    Returns raw list of dicts — Pydantic validation happens in _assemble_records.
    Returns empty list on complete failure — section failure is non-fatal.
    """
    user_message = (
        f"Source drug: {source_drug}\n"
        f"Label section: {section_name}\n\n"
        f"Section text:\n{section_text}"
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
                temperature=0.1,    # slight flexibility for severity judgment
                max_tokens=2048,    # handles sections with 8-10 drug mentions
            )

            content = response.choices[0].message.content or ""
            raw = json.loads(content)

            # Prompt enforces {"interactions": [...]} wrapper — extract the list
            interactions = raw.get("interactions", [])

            if not isinstance(interactions, list):
                raise ValueError(
                    f"Expected 'interactions' to be a list, got {type(interactions)}"
                )

            logger.debug(
                f"Section '{section_name}' → {len(interactions)} interactions extracted"
            )
            return interactions

        except (json.JSONDecodeError, ValueError, Exception) as e:
            last_error = e
            logger.warning(
                f"Extraction attempt {attempt + 1}/{_MAX_RETRIES + 1} "
                f"for section '{section_name}' failed: {e}"
            )

    logger.error(
        f"Section '{section_name}' extraction failed after all retries: {last_error}. "
        f"Skipping — pipeline continues with other sections."
    )
    return []


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

    for raw in raw_interactions:
        try:
            # Clamp LLM score to valid ontology range — defensive, not trusting
            raw_score = int(raw.get("severity_score", 0))
            severity_score = max(0, min(5, raw_score))

            if raw_score != severity_score:
                logger.warning(
                    f"LLM returned out-of-range severity_score={raw_score} "
                    f"for target='{raw.get('target_drug')}' — clamped to {severity_score}"
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
 )
