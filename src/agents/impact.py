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
from typing import Optional, Literal, Tuple, Dict, Any
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from src.config import LLM_MODEL, QDRANT_COLLECTION
from src.schemas.diff_schema import DiffResult
from src.schemas.resolution_schema import ResolvedDrug
from src.schemas.impact_schema import DrugPairAlert, PatientImpactReport
from src.agents.temporal import is_drug_match
from src.services.rag_engine import retrieve_context

logger = logging.getLogger(__name__)

# Constants
_QDRANT_SIMILARITY_THRESHOLD = 0.75
_MAX_LLM_RETRIES = 2

# Minimum present_severity_score that triggers a baseline (non-temporal) alert.
# 3 = "Use caution" — captures Avoid(4) and Contraindicated(5) as well.
_BASELINE_RISK_THRESHOLD = 3

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
    Handles both temporal-change alerts and STABLE_RISK baseline alerts.
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
    
    # MODERATE: significant temporal change OR any stable interaction at score >= 3
    if any(
        a.severity_delta >= 2
        or a.change_type in ("ADDED", "REMOVED")
        or (a.change_type == "STABLE_RISK" and (a.present_severity_score or 0) >= 3)
        for a in alerts
    ):
        return "MODERATE"
    
    # LOW: some alerts exist but none hit above thresholds
    if alerts:
        return "LOW"
    
    return "NONE"

# ─── Phase 3: LLM Synthesis ───────────────────────────────────────────────────

_SYNTHESIS_SYSTEM_PROMPT = """\
You are a senior clinical pharmacist specializing in drug safety.
Your job is to synthesize a final clinical alert for a treating physician based on
driving drug safety findings for their patient's prescription.

INPUTS PROVIDED:
1. List of DrugPairAlerts with two categories:
   a. TEMPORAL CHANGES: Interactions whose FDA severity was strengthened/weakened since prescription.
      (change_type: ADDED, REMOVED, STRENGTHENED, WEAKENED)
   b. STABLE RISKS: Interactions with a high baseline severity score that were already present
      when the drug was prescribed and remain active today — these are NOT label changes,
      but are clinically critical adverse interactions that must be communicated.
      (change_type: STABLE_RISK)
2. ICMR Guideline context (if available)
3. Prescription metadata

RULES:
- Be concise, professional, and action-oriented.
- STABLE_RISK alerts with score >= 4 (Avoid/Contraindicated) are HIGH clinical priority.
  Mention them explicitly in both summary and recommended_action.
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
    llm_client: AsyncOpenAI,
    compounding_notes: Optional[list[str]] = None
) -> Tuple[str, str]:
    """
    LLM phase to generate narrative summary and action recommendation.
    """
    try:
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
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
            raise ValueError("LLM returned an empty response")
            
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

# ─── Phase 1.5: Baseline Interaction Alerts ──────────────────────────────────

async def _build_baseline_alerts(
    extraction_results: Dict[str, Any],
    resolved_drugs:     list[ResolvedDrug],
    existing_pairs:     set[str],
    llm_client:         AsyncOpenAI,
) -> list[DrugPairAlert]:
    """
    Scans the *present* ExtractionResult for each drug and creates a DrugPairAlert
    for any high-severity interaction that:
      (a) has present_severity_score >= _BASELINE_RISK_THRESHOLD, and
      (b) involves another drug in this prescription (matched semantically), and
      (c) has not already been reported as a temporal-change alert.

    These are "STABLE_RISK" alerts — clinically significant even with no label revision.
    """
    from src.services.rxnorm_client import get_drug_classes
    from src.agents.temporal import is_semantic_drug_match

    # Pre-fetch drug classes for all generics in the prescription to avoid repeated API calls
    prescription_classes: dict[str, list[str]] = {}
    for drug in resolved_drugs:
        for gen_name in drug.generic_names:
            try:
                classes = await get_drug_classes(gen_name)
                prescription_classes[gen_name] = classes
            except Exception as e:
                logger.warning(f"Failed to fetch drug classes for '{gen_name}': {e}")
                prescription_classes[gen_name] = []

    baseline_alerts: list[DrugPairAlert] = []

    for source_drug, versions in extraction_results.items():
        present_ext = versions.get("present") if isinstance(versions, dict) else None
        if present_ext is None:
            continue

        # Support both ExtractionResult objects and plain dicts (defensive)
        if hasattr(present_ext, "interactions"):
            interactions = present_ext.interactions
        elif isinstance(present_ext, dict):
            interactions = present_ext.get("interactions", [])
        else:
            continue

        for record in interactions:
            # Handle both InteractionRecord objects and plain dicts
            if hasattr(record, "target_drug"):
                target_drug      = record.target_drug
                severity_score   = record.severity_score
                recommendation   = record.recommendation_text
                warning_text     = record.warning_text
            elif isinstance(record, dict):
                target_drug      = record.get("target_drug", "")
                severity_score   = record.get("severity_score", 0)
                recommendation   = record.get("recommendation_text", "")
                warning_text     = record.get("warning_text")
            else:
                continue

            if not target_drug or severity_score < _BASELINE_RISK_THRESHOLD:
                continue

            # Semantically match the label's target_drug against all other drugs in this prescription
            matched_generic = None
            for drug in resolved_drugs:
                for gen_name in drug.generic_names:
                    # Prevent matching a drug to itself
                    if is_drug_match(gen_name, source_drug):
                        continue

                    classes = prescription_classes.get(gen_name)
                    if await is_semantic_drug_match(
                        patient_drug=gen_name,
                        label_target=target_drug,
                        llm_client=llm_client,
                        patient_drug_classes=classes,
                    ):
                        matched_generic = gen_name
                        break
                if matched_generic:
                    break

            if not matched_generic:
                continue

            # Deduplicate — canonical sorted key matches both orderings
            pair_key = " + ".join(sorted([source_drug, matched_generic]))
            forward  = f"{source_drug} + {matched_generic}"
            if pair_key in existing_pairs or forward in existing_pairs:
                continue
            existing_pairs.add(pair_key)

            # Build clinical reasoning from raw label text
            parts: list[str] = []
            if recommendation:
                parts.append(recommendation)
            if warning_text:
                parts.append(f"Warning: {warning_text}")
            clinical_reasoning = (
                " | ".join(parts)
                if parts
                else f"Active {source_drug}–{matched_generic} interaction documented in current FDA label (severity {severity_score}/5)."
            )

            baseline_alerts.append(DrugPairAlert(
                drug_pair=forward,
                change_type="STABLE_RISK",
                severity_delta=0,
                present_severity_score=severity_score,
                clinical_reasoning=clinical_reasoning,
                key_concern=(
                    f"{source_drug} + {matched_generic}: active FDA interaction at severity {severity_score}/5 — "
                    "no recent label change, but interaction is clinically active."
                ),
                confidence="high",  # Deterministic — sourced directly from FDA label
                dose_context=_get_dose_context(matched_generic, resolved_drugs),
                exposure_days=None,  # No temporal change to measure exposure from
            ))

    return baseline_alerts


# ─── Public Entry Point ───────────────────────────────────────────────────────

async def analyze_patient_impact(
    diffs:              list[Tuple[DiffResult, dict]],
    resolved_drugs:     list[ResolvedDrug],
    prescription_date:  str,
    llm_client:         AsyncOpenAI,
    qdrant_client:      Optional[QdrantClient] = None,
    extraction_results: Optional[Dict[str, Any]] = None,
) -> PatientImpactReport:
    """
    Main entry point for Agent 5.

    Two-pass alert generation:
      Phase 1  — Temporal alerts: diffs where the FDA label severity CHANGED since prescription.
      Phase 1.5 — Baseline alerts: interactions where the severity is HIGH *now* but was not
                  tracked as a diff (UNCHANGED or data_unavailable). These are clinically
                  critical adverse interactions the physician must still be informed of.
    """
    alerts: list[DrugPairAlert] = []
    seen_pairs: set[str] = set()  # Deduplication key shared across both passes
    
    # ── Phase 1: Temporal Diff Alerts ─────────────────────────────────────────
    source_counts: dict[str, int] = {}
    for diff, reasoning in diffs:
        if diff.data_unavailable:
            logger.info(f"Skipping {diff.drug_pair} — historical data unavailable")
            continue

        if not diff.is_clinically_significant and diff.change_type == "UNCHANGED":
            logger.info(f"Skipping {diff.drug_pair} — unchanged, not clinically significant")
            continue

        if not diff.is_clinically_significant:
            logger.info(f"Skipping {diff.drug_pair} — not clinically significant (delta={diff.severity_delta})")
            continue

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
        seen_pairs.add(" + ".join(sorted(drugs)))

    # ── Phase 1.5: Baseline Interaction Alerts ────────────────────────────────
    # Surface high-severity interactions that are clinically active but had no
    # temporal label change (UNCHANGED diffs or data_unavailable cases).
    if extraction_results:
        baseline = await _build_baseline_alerts(
            extraction_results=extraction_results,
            resolved_drugs=resolved_drugs,
            existing_pairs=seen_pairs,  # mutated in-place to prevent duplicates
            llm_client=llm_client,
        )
        if baseline:
            logger.info(f"Phase 1.5: {len(baseline)} baseline STABLE_RISK alert(s) added")
        alerts.extend(baseline)

    # Sort: temporal changes first (severity_delta > 0), then baseline by score
    alerts.sort(
        key=lambda x: (x.present_severity_score or 0, x.severity_delta),
        reverse=True
    )

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
    
    summary, recommended_action = await _generate_synthesis(report_context, llm_client, compounding_notes)

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
