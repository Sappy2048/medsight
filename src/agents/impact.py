"""
impact.py — Patient Impact Agent (Agent 5, the final synthesis layer)

Role:
    Produces a prioritized, patient-contextual clinical alert for the treating physician
    by aggregating temporal diffs, original prescription details, and ICMR guidelines.

Three-phase architecture:
    Phase 1 (deterministic): Filter, exposure calculation, dose context, risk classification.
    Phase 2 (RAG): Retrieval of relevant ICMR guidelines from Qdrant.
    Phase 3 (LLM): Synthesis of the final clinical summary and recommended actions.

Graceful degradation:
    - Qdrant timeout/failure: icmr_context remains None.
    - LLM timeout/failure: Emits templated fallback narrative.
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional, Literal, Tuple
from groq import AsyncGroq
from qdrant_client import QdrantClient

from src.config import GROQ_MODEL, QDRANT_COLLECTION
from src.schemas.diff_schema import DiffResult
from src.schemas.resolution_schema import ResolvedDrug
from src.schemas.impact_schema import DrugPairAlert, PatientImpactReport
from src.agents.temporal import is_drug_match
from src.services.rag_engine import retrieve_context

logger = logging.getLogger(__name__)

# Constants
_QDRANT_SIMILARITY_THRESHOLD = 0.75
_MAX_LLM_RETRIES = 2

# ─── Phase 1: Deterministic Enrichment ────────────────────────────────────────

def _calculate_exposure_days(prescription_date: str, present_version_date: str) -> Optional[int]:
    """
    Computes days since warning changed relative to prescription_date.
    Returns None if dates are invalid or unavailable.
    
    Logic: (warning_date - prescription_date). 
    If prescription is older than warning, result is positive.
    """
    try:
        p_date = datetime.fromisoformat(prescription_date)
        v_date = datetime.fromisoformat(present_version_date)
        delta = (v_date - p_date).days
        return max(0, delta)
    except (ValueError, TypeError):
        return None

def _get_dose_context(drug_name: str, resolved_drugs: list[ResolvedDrug]) -> Optional[str]:
    """
    Matches generic drug name to a ResolvedDrug to extract clinical context.
    Reuses the three-tier matcher from temporal.py.
    """
    for drug in resolved_drugs:
        # ResolvedDrug has generic_names (list)
        for gen_name in drug.generic_names:
            if is_drug_match(drug_name, gen_name):
                context = []
                if drug.formulated_strength:
                    context.append(f"Strength: {drug.formulated_strength}")
                if drug.prescribed_dose:
                    context.append(f"Dose: {drug.prescribed_dose}")
                if drug.route_of_administration:
                    context.append(f"Route: {drug.route_of_administration}")
                
                return " — ".join(context) if context else None
    return None

RiskLevel = Literal["CRITICAL", "HIGH", "MODERATE", "LOW", "NONE"]

def _classify_overall_risk(alerts: list[DrugPairAlert]) -> RiskLevel:
    """
    Pure rule-based risk classification.
    """
    if not alerts:
        return "NONE"
    
    max_severity = max((a.present_severity_score or 0) for a in alerts)
    max_delta    = max(a.severity_delta for a in alerts)
    has_added    = any(a.change_type == "ADDED" for a in alerts)

    if max_severity == 5:
        return "CRITICAL"
    if max_severity >= 4 or (has_added and max_delta >= 3):
        return "HIGH"
    
    # MODERATE: any is_clinically_significant alert that didn't hit HIGH
    if any(a.severity_delta >= 2 or a.change_type in ("ADDED", "REMOVED") for a in alerts):
        return "MODERATE"
    
    # LOW: significant pairs exist but delta < 2
    if alerts:
        return "LOW"
    
    return "NONE"

# ─── Phase 3: LLM Synthesis ───────────────────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior clinical pharmacist specializing in drug safety.
Your job is to synthesize a final clinical alert for a treating physician based on 
newly strengthened FDA warnings detected for their patient's prescription.

INPUTS PROVIDED:
1. List of DrugPairAlerts (verified facts, severity scores, and clinical reasoning)
2. ICMR Guideline context (if available)
3. Prescription metadata

RULES:
- Be concise, professional, and action-oriented.
- Focus on the clinical significance of the changes.
- Do NOT alter any pre-computed facts (risk level, dates, scores).
- If multiple alerts exist, highlight the compounding risk.
- Address the physician directly.

OUTPUT FORMAT (strict JSON):
{
  "summary": "2-3 sentence plain-English summary of the overall situation.",
  "recommended_action": "Top-line clinical action (e.g., 'Discontinue X and check Y')."
}
"""

async def _generate_synthesis(
    report_data: dict,
    groq_client: AsyncGroq,
    compounding_notes: Optional[list[str]] = None
) -> Tuple[str, str]:
    """
    LLM phase to generate narrative summary and action recommendation.
    """
    try:
        response = await groq_client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": _SYNTHESIS_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(report_data, indent=2)},
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=512,
        )
        response_text = response.choices[0].message.content
        if not response_text:
            raise ValueError("Groq returned an empty response")
            
        content = json.loads(response_text)
        return content.get("summary", ""), content.get("recommended_action", "")
    except Exception as e:
        logger.error(f"LLM synthesis failed: {e}")
        # Fallback strings
        fallback_summary = "Automated clinical summary unavailable. Please review individual alerts below."
        if compounding_notes:
            fallback_summary += f" Note: {'; '.join(compounding_notes)}"
            
        return (
            fallback_summary,
            "Review flagged drug interactions and adjust prescription as per clinical judgement."
        )

# ─── Public Entry Point ───────────────────────────────────────────────────────

async def analyze_patient_impact(
    diffs:             list[Tuple[DiffResult, dict]],
    resolved_drugs:    list[ResolvedDrug],
    prescription_date: str,
    groq_client:       AsyncGroq,
    qdrant_client:     Optional[QdrantClient] = None,
) -> PatientImpactReport:
    """
    Main entry point for Agent 5.
    """
    alerts: list[DrugPairAlert] = []
    
    # Phase 1: Deterministic Enrichment
    source_counts = {}
    for diff, reasoning in diffs:
        # Phase 1 Filter: two independent gates
        if diff.data_unavailable:
            logger.info(f"Skipping {diff.drug_pair} — historical data unavailable")
            continue
            
        if not diff.is_clinically_significant and diff.change_type == "UNCHANGED":
            logger.info(f"Skipping {diff.drug_pair} — unchanged and not clinically significant")
            continue
            
        if not diff.is_clinically_significant:
            logger.info(f"Skipping {diff.drug_pair} — not clinically significant (delta={diff.severity_delta})")
            continue
            
        # Parse drugs from pair "Source + Target"
        drugs = diff.drug_pair.split(" + ")
        source_drug = drugs[0]
        target_drug = drugs[1]
        
        source_counts[source_drug] = source_counts.get(source_drug, 0) + 1

        alert = DrugPairAlert(
            drug_pair=diff.drug_pair,
            change_type=diff.change_type,
            severity_delta=diff.severity_delta,
            present_severity_score=diff.present_severity_score,
            clinical_reasoning=reasoning.get("clinical_reasoning", ""),
            key_concern=reasoning.get("key_concern"),
            confidence=reasoning.get("confidence", "low"),
            exposure_days=_calculate_exposure_days(prescription_date, diff.present_version_date),
            dose_context=_get_dose_context(target_drug, resolved_drugs)
        )
        alerts.append(alert)

    # Sort alerts by severity (highest score first, then delta)
    alerts.sort(key=lambda x: (x.present_severity_score or 0, x.severity_delta), reverse=True)

    # Phase 2: RAG Retrieval (Top alert only)
    icmr_guideline_used = False
    if alerts:
        top_alert = alerts[0]
        try:
            # Semantic risk query driving toward mechanism-level matching
            source_drug, target_drug = top_alert.drug_pair.split(" + ")
            concern = top_alert.key_concern or "drug interaction risk"
            query = (
                f"{concern} {source_drug} {target_drug} "
                f"interaction warning {top_alert.change_type.lower()}"
            )
            
            # Pass injected qdrant_client through Option B
            chunks = retrieve_context(source_drug, query, limit=1, client=qdrant_client)
            if not chunks:
                 chunks = retrieve_context(target_drug, query, limit=1, client=qdrant_client)
            
            if chunks:
                top_alert.icmr_context = chunks[0]["text"]
                icmr_guideline_used = True
        except Exception as e:
            logger.warning(f"ICMR RAG retrieval failed: {e}")

    # Detect compounding
    compounding_notes = []
    for drug, count in source_counts.items():
        if count > 1:
            compounding_notes.append(f"{drug} has {count} simultaneously strengthened interactions.")

    # Rule-based Risk
    overall_risk = _classify_overall_risk(alerts)

    # Phase 3: LLM Synthesis
    report_context = {
        "prescription_date": prescription_date,
        "overall_risk_level": overall_risk,
        "alerts": [a.model_dump() for a in alerts],
        "compounding_notes": compounding_notes,
        "total_evaluated": len(diffs)
    }
    
    summary, recommended_action = await _generate_synthesis(report_context, groq_client, compounding_notes)

    return PatientImpactReport(
        prescription_date=prescription_date,
        report_generated_at=datetime.now(timezone.utc).isoformat(),
        overall_risk_level=overall_risk,
        summary=summary,
        alerts=alerts,
        recommended_action=recommended_action,
        flagged_pairs_count=len(alerts),
        total_pairs_evaluated=len(diffs),
        icmr_guideline_used=icmr_guideline_used
    )
