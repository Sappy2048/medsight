"""
temporal.py — Temporal Diff Agent (Layer 2, the core differentiator)

Two-phase design:
    Phase 1 (deterministic): match interactions, classify change, compute delta
    Phase 2 (LLM reasoning):  generate clinical reasoning for Agent 5 (Patient Impact)

Phase 1 NEVER hallucinates — it's pure Python comparison logic.
Phase 2 reasons over Phase 1's verified output — it cannot invent facts,
only interpret the facts it's given.

Exposed function:
    async def compute_temporal_diff(
        past: ExtractionResult,
        present: ExtractionResult,
        target_drug: str,
        llm_client: AsyncOpenAI,
    ) -> DiffResult
"""

import json
import logging
from typing import Optional,Literal
from rapidfuzz import fuzz
from openai import AsyncOpenAI

from src.config import LLM_MODEL, SEVERITY_ONTOLOGY
from src.schemas.diff_schema import ExtractionResult, InteractionRecord, DiffResult

logger = logging.getLogger(__name__)

_MAX_RETRIES = 2

ChangeType = Literal["ADDED", "REMOVED", "STRENGTHENED", "WEAKENED", "UNCHANGED"]

# ─── Phase 1: Deterministic matching + classification ─────────────────────────

# Minimum character length for substring matching — prevents "mycin" matching
# "azithromycin". Both strings must be at least this long before substring
# logic is allowed to fire.
MIN_SUBSTRING_MATCH_LEN = 6

# Fuzzy match threshold — token_sort_ratio is used (not WRatio) because:
#   - It handles salt suffixes: "azithromycin" vs "azithromycin anhydrous" → 100
#   - It is NOT fooled by short shared substrings between different drugs
#   - WRatio at 85 fires on "metformin" vs "metamizole" (genuinely different)
FUZZY_MATCH_THRESHOLD = 88


def is_drug_match(query: str, target: str) -> bool:
    """
    Core three-tier matching logic reused across agents.
    
    Tier 1: Exact match (case-insensitive)
    Tier 2: Substring match (bidirectional, length-guarded)
    Tier 3: Fuzzy match (token_sort_ratio >= FUZZY_MATCH_THRESHOLD)
    """
    query_lower = query.lower().strip()
    target_lower = target.lower().strip()

    # Tier 1: exact match
    if query_lower == target_lower:
        return True

    # Tier 2: substring match — bidirectional with length guard
    len_q = len(query_lower)
    len_t = len(target_lower)

    if (
        min(len_q, len_t) >= MIN_SUBSTRING_MATCH_LEN
        and (query_lower in target_lower or target_lower in query_lower)
    ):
        logger.warning(
            f"Substring match fired: query='{query}' → target='{target}'"
        )
        return True

    # Tier 3: fuzzy match — token_sort_ratio for salt/form suffix tolerance
    score = fuzz.token_sort_ratio(query_lower, target_lower)
    if score >= FUZZY_MATCH_THRESHOLD:
        logger.warning(
            f"Fuzzy match fired: query='{query}' → target='{target}' "
            f"(token_sort_ratio={score})"
        )
        return True

    return False


def _find_interaction(
    extraction: ExtractionResult,
    target_drug: str,
    section: Optional[str] = None,
) -> Optional[InteractionRecord]:
    """
    Find the best-matching InteractionRecord for target_drug within one
    ExtractionResult using the three-tier matching strategy.

    If multiple sections mention the same drug, the highest-severity record
    is returned (most conservative clinical posture).
    """
    matches: list[InteractionRecord] = []

    for interaction in extraction.interactions:
        # Gate on section filter first — avoids unnecessary string ops
        if section is not None and interaction.section != section:
            continue

        if is_drug_match(target_drug, interaction.target_drug):
            matches.append(interaction)

    if not matches:
        return None

    # Most conservative posture: return highest severity match across sections
    return max(matches, key=lambda r: r.severity_score)

def _classify_change(
    past_match: Optional[InteractionRecord],
    present_match: Optional[InteractionRecord],
) -> tuple[ChangeType, int]:
    """
    Pure deterministic classification — no LLM involved.

    Returns:
        (change_type, severity_delta)

    Logic:
        past=None,    present=exists  → ADDED,        delta = present_score
        past=exists,  present=None    → REMOVED,       delta = -past_score
        both exist, present > past    → STRENGTHENED,  delta = present - past
        both exist, present < past    → WEAKENED,       delta = present - past
        both exist, present == past   → UNCHANGED,      delta = 0
    """
    if past_match is None and present_match is None:
        # Neither version mentions this drug pair — should not reach this
        # function in practice, but handle gracefully rather than crash.
        return "UNCHANGED", 0

    if past_match is None and present_match is not None:
        return "ADDED", present_match.severity_score

    if past_match is not None and present_match is None:
        return "REMOVED", -past_match.severity_score
    
    assert past_match is not None
    assert present_match is not None

    # Both exist — compare severity scores
    delta = present_match.severity_score - past_match.severity_score

    if delta > 0:
        return "STRENGTHENED", delta
    elif delta < 0:
        return "WEAKENED", delta
    else:
        return "UNCHANGED", 0


def _build_base_diff(
    target_drug: str,
    source_drug: str,
    past: ExtractionResult,
    present: ExtractionResult,
    past_match: Optional[InteractionRecord],
    present_match: Optional[InteractionRecord],
    data_unavailable: bool = False,          
) -> DiffResult:
    """
    Assembles the DiffResult from Phase 1 outputs only.
    No LLM involved — every field here is either a direct copy from
    ExtractionResult/InteractionRecord, or simple arithmetic.
    """
    change_type, severity_delta = _classify_change(past_match, present_match)

    is_significant = (
        abs(severity_delta) >= 2
        or change_type in ["ADDED", "REMOVED"]
    ) and not data_unavailable              # ← never flag as significant if data missing

    source_drug = past.source_drug or present.source_drug

    return DiffResult(
        drug_pair=f"{source_drug} + {target_drug}",
        change_type=change_type,
        past_recommendation=past_match.recommendation_text if past_match else None,
        present_recommendation=present_match.recommendation_text if present_match else None,
        past_severity_score=past_match.severity_score if past_match else None,
        present_severity_score=present_match.severity_score if present_match else None,
        severity_delta=severity_delta,
        past_version_date=past.version_date,
        present_version_date=present.version_date,
        is_clinically_significant=is_significant,
        past_spl_id=past.spl_id,
        present_spl_id=present.spl_id,
        data_unavailable=data_unavailable,   # ← NEW
    )


# ─── Phase 2: LLM clinical reasoning ────────────────────────────────────────────

_REASONING_SYSTEM_PROMPT = """\
You are a clinical reasoning assistant embedded in a drug safety system.

You will receive a verified, structured diff between two FDA label versions
for a specific drug interaction. This diff has already been computed
deterministically — every fact (severity scores, dates, change type) is
ground truth. Your job is NOT to re-derive these facts. Your job is to
explain their clinical significance in plain language for a downstream
Patient Impact Agent.

RULES (non-negotiable):
1. NEVER invent a fact not present in the input JSON.
2. NEVER change or contradict severity_score, change_type, or dates given.
3. If past_recommendation or present_recommendation is null, say so plainly
   — do not guess what it might have said.
4. Your output must be a single clinical reasoning paragraph (3-5 sentences)
   that explains: what changed, why it matters clinically, and what the
   downstream agent should pay attention to.
5. Be specific about the mechanism if evident from the recommendation text
   (e.g. bleeding risk, QT prolongation, renal impairment) — but only if
   that mechanism is explicitly mentioned in the input text. Do not infer
   a mechanism that isn't stated.

OUTPUT FORMAT (strict JSON, no markdown, no explanation outside the JSON):
{
  "clinical_reasoning": "<3-5 sentence paragraph>",
  "key_concern": "<one short phrase capturing the single most important risk, or null if UNCHANGED>",
  "confidence": "high" | "medium" | "low"
}

Set confidence to "low" if either recommendation text is null or very short.
Set confidence to "high" only if both versions have clear, specific recommendation text.
"""


async def _generate_clinical_reasoning(
    diff: DiffResult,
    llm_client: AsyncOpenAI,
) -> dict:
    """
    Phase 2: LLM reasons over the verified DiffResult to produce a clinical
    explanation. The LLM is given the diff as context — it cannot alter the
    underlying facts, only interpret them.

    Returns dict with keys: clinical_reasoning, key_concern, confidence
    Falls back to a templated reasoning string if LLM call fails after retries,
    so the pipeline never breaks on this non-critical enrichment step.
    """
    diff_context = diff.model_dump_json(indent=2)

    last_error: Optional[Exception] = None

    for attempt in range(_MAX_RETRIES + 1):
        try:
            response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _REASONING_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Diff to reason about:\n{diff_context}"},
                ],
                response_format={"type": "json_object"},
                temperature=0.2,  # low but not zero — allow natural phrasing
                max_tokens=512,
            )

            content = response.choices[0].message.content
            parsed = json.loads(content or "{}")

            # Basic shape validation — don't trust LLM blindly even here
            if "clinical_reasoning" not in parsed:
                raise ValueError("Missing clinical_reasoning in LLM response")

            return parsed

        except (json.JSONDecodeError, ValueError, Exception) as e:
            last_error = e
            logger.warning(
                f"Clinical reasoning attempt {attempt + 1}/{_MAX_RETRIES + 1} failed: {e}"
            )

    # Fallback — pipeline must not crash on enrichment failure
    logger.error(f"Clinical reasoning failed after all retries: {last_error}")
    return {
        "clinical_reasoning": (
            f"Interaction warning for {diff.drug_pair} changed from "
            f"'{diff.past_recommendation or 'no prior record'}' to "
            f"'{diff.present_recommendation or 'no current record'}' "
            f"({diff.change_type}, severity delta {diff.severity_delta}). "
            f"Automated reasoning unavailable — review raw diff directly."
        ),
        "key_concern": diff.change_type if diff.is_clinically_significant else None,
        "confidence": "low",
    }


# ─── Public entry point ──────────────────────────────────────────────────────


async def compute_temporal_diff(
    past: Optional[ExtractionResult],        # ← NOW Optional
    present: ExtractionResult,
    target_drug: str,
    llm_client: AsyncOpenAI,
    prescription_date: Optional[str] = None, # ← NEW: for logging/tracing
) -> tuple[DiffResult, dict]:
    """
    Main entry point for the Temporal Diff Agent.

    Args:
        past:              ExtractionResult from the label active at prescription_date.
                           Pass None if label history predates the prescription date —
                           data_unavailable will be set on the returned DiffResult.
        present:           ExtractionResult from the latest label version.
        target_drug:       The specific interacting drug to diff (e.g. "Azithromycin").
        llm_client:       Shared AsyncOpenAI client (injected, never created internally).
        prescription_date: Optional ISO date string for logging/tracing only.

    Returns:
        (DiffResult, reasoning_dict)
        DiffResult       — the verified, deterministic diff (ground truth)
        reasoning_dict   — {clinical_reasoning, key_concern, confidence} from Phase 2

    The two are returned separately so that Agent 5 (Patient Impact) can use
    DiffResult fields directly for any rule-based logic, while also having
    access to the LLM's clinical narrative for richer synthesis.
    """
    # ── Edge case: label history predates prescription_date ──────────────────
    data_unavailable = past is None

    if data_unavailable:
        logger.warning(
            f"No historical label found for '{target_drug}' "
            f"at prescription_date={prescription_date}. "
            f"Emitting data_unavailable=True — skipping Phase 1 match."
        )
        # Build a minimal DiffResult that signals the gap cleanly
        diff = DiffResult(
            drug_pair=f"{present.source_drug} + {target_drug}",
            change_type="UNCHANGED",          # safest neutral default
            past_version_date="UNAVAILABLE",
            present_version_date=present.version_date,
            is_clinically_significant=False,
            past_spl_id=None,
            present_spl_id=present.spl_id,
            data_unavailable=True,            # ← the real signal
        )
        reasoning = {
            "clinical_reasoning": (
                f"Historical FDA label for {present.source_drug} + {target_drug} "
                f"predates available version history. Temporal comparison cannot "
                f"be performed. Manual review of current label is advised."
            ),
            "key_concern": "HISTORICAL_DATA_UNAVAILABLE",
            "confidence": "low",
        }
        return diff, reasoning

    # ── Normal path ──────────────────────────────────────────────────────────
    past_match    = _find_interaction(past, target_drug)
    present_match = _find_interaction(present, target_drug)

    diff = _build_base_diff(
        target_drug=target_drug,
        source_drug=past.source_drug,
        past=past,
        present=present,
        past_match=past_match,
        present_match=present_match,
        data_unavailable=False,
    )

    logger.info(
        f"Phase 1 complete: {diff.drug_pair} → {diff.change_type} "
        f"(delta={diff.severity_delta}, significant={diff.is_clinically_significant})"
    )

    reasoning = await _generate_clinical_reasoning(diff, llm_client)

    logger.info(f"Phase 2 complete: confidence={reasoning.get('confidence')}")

    return diff, reasoning



# ─── CLI helper for testing ───────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import os

    async def _test():
        logging.basicConfig(level=logging.INFO)

        client = AsyncOpenAI(
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1"),
            api_key=os.getenv("OLLAMA_API_KEY", "ollama")
        )

        # Mock data for testing without waiting on extraction.py
        past = ExtractionResult(
            source_drug="Warfarin",
            version_date="2019-03-15",
            spl_id="mock-spl::v3",
            interactions=[
                InteractionRecord(
                    source_drug="Warfarin",
                    target_drug="Azithromycin",
                    recommendation_text="Monitor INR levels closely",
                    severity_text="monitor closely",
                    severity_score=SEVERITY_ONTOLOGY["monitor closely"],
                    version_date="2019-03-15",
                    spl_id="mock-spl::v3",
                )
            ],
        )

        present = ExtractionResult(
            source_drug="Warfarin",
            version_date="2023-08-01",
            spl_id="mock-spl::v7",
            interactions=[
                InteractionRecord(
                    source_drug="Warfarin",
                    target_drug="Azithromycin",
                    recommendation_text="Avoid concomitant use due to risk of fatal bleeding",
                    severity_text="avoid",
                    severity_score=SEVERITY_ONTOLOGY["avoid"],
                    version_date="2023-08-01",
                    spl_id="mock-spl::v7",
                )
            ],
        )

        diff, reasoning = await compute_temporal_diff(
            past, present, target_drug="Azithromycin", llm_client=client
        )

        print("\n--- DiffResult ---")
        print(diff.model_dump_json(indent=2))
        print("\n--- Clinical Reasoning ---")
        print(json.dumps(reasoning, indent=2))

    asyncio.run(_test())))