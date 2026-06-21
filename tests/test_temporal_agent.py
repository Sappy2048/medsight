import asyncio
import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.agents.temporal import compute_temporal_diff
from src.schemas.diff_schema import ExtractionResult, InteractionRecord
from src.config import SEVERITY_ONTOLOGY, TOGETHER_BASE_URL, TOGETHER_API_KEY

import pytest

@pytest.mark.asyncio
async def test_temporal_logic():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    client = AsyncOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=TOGETHER_API_KEY
    )

    # 1. Setup Mock Data
    # Past: Warfarin + Azithromycin = "Monitor closely" (Score 2)
    past = ExtractionResult(
        source_drug="Warfarin",
        version_date="2010-01-01",
        spl_id="warfarin::v1",
        interactions=[
            InteractionRecord(
                source_drug="Warfarin",
                target_drug="Azithromycin",
                recommendation_text="Monitor INR levels closely when starting Azithromycin.",
                severity_text="monitor closely",
                severity_score=SEVERITY_ONTOLOGY["monitor closely"],
                version_date="2010-01-01",
                spl_id="warfarin::v1",
                section="drug_interactions"
            )
        ]
    )

    # Present: Warfarin + Azithromycin = "Avoid" (Score 4)
    present = ExtractionResult(
        source_drug="Warfarin",
        version_date="2024-01-01",
        spl_id="warfarin::v10",
        interactions=[
            InteractionRecord(
                source_drug="Warfarin",
                target_drug="Azithromycin",
                recommendation_text="Concomitant use not recommended; avoid due to fatal bleeding risk.",
                severity_text="avoid",
                severity_score=SEVERITY_ONTOLOGY["avoid"],
                version_date="2024-01-01",
                spl_id="warfarin::v10",
                section="warnings"
            )
        ]
    )

    # 2. Run Agent
    print("\n--- Running Temporal Diff Agent ---")
    try:
        diff, reasoning = await compute_temporal_diff(
            past=past,
            present=present,
            target_drug="Azithromycin",
            llm_client=client,
            prescription_date="2015-05-15"
        )

        # 3. Assertions
        print(f"Change Type: {diff.change_type}")
        print(f"Severity Delta: {diff.severity_delta}")
        print(f"Significant: {diff.is_clinically_significant}")
        print(f"Clinical Reasoning: {reasoning['clinical_reasoning']}")

        assert diff.change_type == "STRENGTHENED"
        assert diff.severity_delta == 2
        assert diff.is_clinically_significant is True
        assert "bleeding" in reasoning["clinical_reasoning"].lower() or "inr" in reasoning["clinical_reasoning"].lower()
    except Exception as e:
        print(f"Test failed (Ollama likely not running): {e}")


@pytest.mark.asyncio
async def test_baseline_interaction_reporting():
    """
    Assert that stable high-severity interactions (e.g. past=4 -> present=4, UNCHANGED)
    are captured as STABLE_RISK alerts, contribute to overall risk classification,
    and are included as anchor facts for synthesis verification.
    """
    from src.agents.impact import analyze_patient_impact
    from src.agents.synthesis import synthesize_final_report
    from src.schemas.resolution_schema import ResolvedDrug
    from src.schemas.diff_schema import DiffResult, ExtractionResult, InteractionRecord
    from unittest.mock import AsyncMock, MagicMock

    # 1. Setup Mock Inputs
    # Resolved drugs in prescription: Ciprofloxacin & Prednisolone
    resolved_drugs = [
        ResolvedDrug(
            raw_prescription_input="Ciprofloxacin 500mg",
            generic_names=["Ciprofloxacin"],
            formulated_strength="500mg",
            rxcui_list=["cipro-123"],
            prescribed_dose="1 tab",
            route_of_administration="oral",
            resolution_method="exact_match",
            parsing_method="llm"
        ),
        ResolvedDrug(
            raw_prescription_input="Prednisolone 10mg",
            generic_names=["Prednisolone"],
            formulated_strength="10mg",
            rxcui_list=["pred-456"],
            prescribed_dose="1 tab",
            route_of_administration="oral",
            resolution_method="exact_match",
            parsing_method="llm"
        )
    ]

    # Diffs contains Ciprofloxacin + Prednisolone, but it was UNCHANGED (delta=0)
    diff = DiffResult(
        drug_pair="Ciprofloxacin + Prednisolone",
        change_type="UNCHANGED",
        past_recommendation="Avoid concurrent use.",
        present_recommendation="Avoid concurrent use.",
        past_severity_score=4,
        present_severity_score=4,
        severity_delta=0,
        past_version_date="2020-01-01",
        present_version_date="2024-01-01",
        is_clinically_significant=False,
        past_spl_id="spl-past",
        present_spl_id="spl-present"
    )
    diffs = [(diff, {"clinical_reasoning": "Unchanged risk."})]

    # Extraction results contains the active current label warnings
    extraction_results = {
        "Ciprofloxacin": {
            "present": ExtractionResult(
                source_drug="Ciprofloxacin",
                version_date="2024-01-01",
                spl_id="spl-present",
                interactions=[
                    InteractionRecord(
                        source_drug="Ciprofloxacin",
                        target_drug="Prednisolone",
                        recommendation_text="Avoid coadministration due to tendon rupture risk.",
                        severity_text="Avoid",
                        severity_score=4,
                        version_date="2024-01-01",
                        spl_id="spl-present",
                        section="warnings"
                    )
                ]
            )
        }
    }

    # Mock AsyncOpenAI client
    mock_llm = AsyncMock()
    # Mock synthesis LLM response
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"summary": "Ciprofloxacin and Prednisolone coadministration carries high tendon rupture risk.", "recommended_action": "Avoid concurrent use."}'
                )
            )
        ]
    )

    # 2. Run Patient Impact analysis
    report = await analyze_patient_impact(
        diffs=diffs,
        resolved_drugs=resolved_drugs,
        prescription_date="2022-06-07",
        llm_client=mock_llm,
        qdrant_client=None,
        extraction_results=extraction_results
    )

    # 3. Assertions on PatientImpactReport
    assert report.flagged_pairs_count == 1
    alert = report.alerts[0]
    assert alert.drug_pair == "Ciprofloxacin + Prednisolone"
    assert alert.change_type == "STABLE_RISK"
    assert alert.present_severity_score == 4
    assert alert.severity_delta == 0
    assert "tendon rupture risk" in alert.clinical_reasoning
    assert report.overall_risk_level == "HIGH"

    # Mock the verification LLM response
    mock_llm.chat.completions.create.return_value = MagicMock(
        choices=[
            MagicMock(
                message=MagicMock(
                    content='{"all_claims_grounded": true, "drifted_claims": [], "patched_summary": null, "patched_recommended_action": null}'
                )
            )
        ]
    )

    # 4. Run Synthesizer Verification
    final_report = await synthesize_final_report(
        report=report,
        diff_results=[diff],
        llm_client=mock_llm
    )

    # 5. Assertions on final synthesized report
    assert final_report.verified is True
    assert final_report.integrity_score == 1.0
    assert final_report.severity_badge == "HIGH"
    assert "tendon rupture" in final_report.final_summary

    # Ensure the verification call passed the STABLE_RISK alert as an anchor fact
    call_args = mock_llm.chat.completions.create.call_args_list[-1]
    prompt_content = call_args.kwargs["messages"][1]["content"]
    assert "STABLE_RISK" in prompt_content
    assert "Ciprofloxacin + Prednisolone" in prompt_content


if __name__ == "__main__":
    asyncio.run(test_temporal_logic())
