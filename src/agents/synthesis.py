"""
synthesizer.py — Patient Synthesis Agent (Agent 6)

Role:
    The final safety and verification layer. It cross-checks the clinical claims 
    made by Agent 5 (Impact) against the deterministic facts from Agent 4 (Temporal).

Three-phase architecture:
    Phase 1 (deterministic): Risk Level Integrity Guard, Empty Diff Guard.
    Phase 2 (LLM): Grounding verification against "Anchor Facts".
    Phase 3 (Assembly): Integrity score calculation and report finalization.

"""

import json
import logging
from datetime import datetime, timezone
from typing import Tuple, Optional, List, Dict, Any
from openai import AsyncOpenAI

from src.config import LLM_MODEL
from src.schemas.diff_schema import DiffResult
from src.schemas.impact_schema import PatientImpactReport
from src.schemas.synthesizer_schema import MedSightFinalReport

logger = logging.getLogger("medsight.synthesizer")

# ─── Phase 2: LLM Verification Prompt ────────────────────────────────────────

_VERIFICATION_SYSTEM_PROMPT = """\
You are a clinical grounding verification assistant. 
Your task is to verify two clinical claims (Summary and Recommended Action) against 
a provided list of "Anchor Facts" (deterministic FDA interaction data).

ANCHOR FACTS:
{anchor_facts}

CLAIMS TO VERIFY:
1. Summary: {summary}
2. Recommended Action: {recommended_action}

INSTRUCTIONS:
- Grounding Check: Every drug name, specific severity level (e.g., "strengthened", "contraindicated"), 
  and date in the claims MUST have a corresponding anchor in the facts.
- Clinical Interpretation: General medical phrases that describe the *nature* of the risk 
  (e.g., "fatal bleeding", "QT prolongation") are PERMITTED if the anchor facts show a high 
  severity score (4 or 5) or if the recommendation text in the anchor implies it.
- Drift Check: Identify "drifted claims": verbatim phrases that are explicitly 
  contradicted by the facts OR mention drugs/facts not present in the anchor list.
- Patching: If drift is detected, provide a "patched" version that is strictly grounded.
- Output: Return ONLY a strict JSON object.

OUTPUT FORMAT:
{{
  "all_claims_grounded": bool,
  "drifted_claims": ["phrase 1", "phrase 2"],
  "patched_summary": "string or null",
  "patched_recommended_action": "string or null"
}}
"""

# ─── Implementation ───────────────────────────────────────────────────────────

async def synthesize_final_report(
    report: PatientImpactReport,
    diff_results: List[DiffResult],
    llm_client: AsyncOpenAI
) -> MedSightFinalReport:
    """
    The main entry point for the Synthesis Agent.
    """
    logger.info(f"Starting synthesis for report dated {report.prescription_date}")

    # ─── Phase 1: Deterministic Pre-Checks ────────────────────────────────────
    violation_notes = []
    integrity_flag = False
    
    # Check 1: Risk Level Integrity Guard
    max_severity = 0
    if diff_results:
        scores = [d.present_severity_score for d in diff_results if d.present_severity_score is not None]
        if scores:
            max_severity = max(scores)
            
    if report.overall_risk_level == "CRITICAL" and max_severity < 5:
        note = f"Risk level CRITICAL but max severity is {max_severity}"
        violation_notes.append(note)
        integrity_flag = True
        logger.warning(f"Phase 1 Violation: {note}")
        
    if report.overall_risk_level == "HIGH" and max_severity < 4:
        note = f"Risk level HIGH but max severity is {max_severity}"
        violation_notes.append(note)
        integrity_flag = True
        logger.warning(f"Phase 1 Violation: {note}")

    # Check 2: Empty Diff Guard
    if not diff_results and report.flagged_pairs_count > 0:
        note = "Flagged pairs count > 0 but no diff results provided"
        violation_notes.append(note)
        integrity_flag = True
        logger.warning(f"Phase 1 Violation: {note}")

    # Check 3: Severity Badge Derivation
    # Deterministic 1:1 mapping
    severity_badge = report.overall_risk_level

    # ─── Phase 2: LLM Grounding Verification ──────────────────────────────────
    llm_result: Dict[str, Any] = {
        "all_claims_grounded": True,
        "drifted_claims": [],
        "patched_summary": None,
        "patched_recommended_action": None
    }
    llm_failed = False

    if report.flagged_pairs_count > 0:
        logger.debug("Initiating Phase 2 LLM verification")
        
        # Build Anchor Facts Block
        anchors = []
        for d in diff_results:
            if d.is_clinically_significant:
                anchors.append({
                    "drug_pair": d.drug_pair,
                    "change_type": d.change_type,
                    "severity_delta": d.severity_delta,
                    "present_severity_score": d.present_severity_score,
                    "present_version_date": d.present_version_date,
                    "present_recommendation": d.present_recommendation
                })
        
        user_content = f"""\
ANCHOR FACTS:
{json.dumps(anchors, indent=2)}

CLAIMS TO VERIFY:
1. Summary: {report.summary}
2. Recommended Action: {report.recommended_action}
"""

        try:
            response = await llm_client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _VERIFICATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_content}
                ],
                response_format={"type": "json_object"},
                temperature=0.0,
                max_tokens=512
            )
            
            content = response.choices[0].message.content
            if content:
                llm_result = json.loads(content)
                if not llm_result.get("all_claims_grounded"):
                    logger.warning(f"Phase 2 drift detected: {llm_result.get('drifted_claims')}")
                else:
                    logger.info("Phase 2 LLM verification passed cleanly")
            else:
                raise ValueError("Empty LLM response")
                
        except Exception as e:
            logger.warning(f"Phase 2 fallback triggered: {e}")
            llm_failed = True
    else:
        logger.info("Skipping Phase 2 (zero flagged pairs)")

    # ─── Phase 3: Final Assembly ──────────────────────────────────────────────
    
    # 1. verified field
    verified = (not integrity_flag) and llm_result.get("all_claims_grounded", True)
    if llm_failed:
        verified = False

    # 2. verification_notes
    final_notes = violation_notes + llm_result.get("drifted_claims", [])
    if llm_failed:
        final_notes.append("LLM verification unavailable — skipped")

    # 3. integrity_score calculation
    score = 1.0
    if report.flagged_pairs_count > 0:
        if llm_failed:
            score = 0.5
        else:
            # Deduct 0.3 per Phase 1 flag
            if integrity_flag:
                # We count flags, but violation_notes might have multiple entries. 
                # Integrity guard has 2 specific rules.
                # Simplification: -0.3 per Phase 1 check failure
                p1_deductions = 0
                if any("Risk level" in n for n in violation_notes): p1_deductions += 0.3
                if any("Flagged pairs" in n for n in violation_notes): p1_deductions += 0.3
                score -= p1_deductions
            
            # Deduct 0.1 per drifted claim
            drift_deductions = len(llm_result.get("drifted_claims", [])) * 0.1
            score -= drift_deductions
    else:
        score = 1.0
        
    score = max(0.0, round(score, 2))

    # 4. override_applied and final_* fields
    final_summary = llm_result.get("patched_summary") or report.summary
    final_recommended_action = llm_result.get("patched_recommended_action") or report.recommended_action
    
    override_applied = (
        (llm_result.get("patched_summary") is not None) or 
        (llm_result.get("patched_recommended_action") is not None)
    )

    result = MedSightFinalReport(
        report=report,
        verified=verified,
        integrity_score=score,
        severity_badge=severity_badge,
        verification_notes=final_notes,
        override_applied=override_applied,
        final_summary=final_summary,
        final_recommended_action=final_recommended_action,
        generated_at=datetime.now(timezone.utc).isoformat()
    )

    logger.info(f"Final report assembled: score={score}, verified={verified}")
    return result
