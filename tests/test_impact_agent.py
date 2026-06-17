import asyncio
import os
import logging
from groq import AsyncGroq
from qdrant_client import QdrantClient

from src.agents.impact import analyze_patient_impact
from src.schemas.diff_schema import DiffResult
from src.schemas.resolution_schema import ResolvedDrug
from src.schemas.impact_schema import PatientImpactReport

async def test_impact_synthesis():
    logging.basicConfig(level=logging.INFO)
    
    # 1. Setup Clients
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Error: GROQ_API_KEY not found in environment")
        return

    groq_client = AsyncGroq(api_key=groq_api_key)
    # Qdrant client will be initialized from config via rag_engine if not passed, 
    # but we pass None here to test graceful degradation or use real if available.
    qdrant_client = None 
    try:
        from src.config import QDRANT_URL, QDRANT_API_KEY
        if QDRANT_URL:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            print("Using real Qdrant client for test")
    except Exception:
        print("Skipping real Qdrant for test")

    # 2. Mock Data
    # Warning date is 2023-12-01 in mock diff below.
    # Set Rx to 2023-06-01 to test retrospective exposure.
    prescription_date = "2023-06-01"
    
    # Mock ResolvedDrugs (Agent 1 Output)
    resolved_drugs = [
        ResolvedDrug(
            raw_prescription_input="Azee 500mg OD",
            generic_names=["Azithromycin"],
            formulated_strength="500mg",
            prescribed_dose="1 tablet",
            route_of_administration="oral",
            frequency="OD",
            rxcui_list=["12345"],
            resolution_method="exact_match",
            parsing_method="llm"
        ),
        ResolvedDrug(
            raw_prescription_input="Warfarin 5mg BD",
            generic_names=["Warfarin"],
            formulated_strength="5mg",
            prescribed_dose="1 tablet",
            route_of_administration="oral",
            frequency="BD",
            rxcui_list=["67890"],
            resolution_method="exact_match",
            parsing_method="llm"
        )
    ]

    # Mock DiffResults (Agent 4 Output)
    diff_war_azi = DiffResult(
        drug_pair="Warfarin + Azithromycin",
        change_type="STRENGTHENED",
        past_recommendation="Monitor INR levels",
        present_recommendation="Avoid concomitant use; fatal bleeding risk increased",
        past_severity_score=2,
        present_severity_score=4,
        severity_delta=2,
        past_version_date="2019-01-01",
        present_version_date="2023-12-01",
        is_clinically_significant=True,
        past_spl_id="past-spl",
        present_spl_id="present-spl"
    )
    
    reasoning_war_azi = {
        "clinical_reasoning": "The warning for Warfarin and Azithromycin has been significantly strengthened. Previously only monitoring was required, but the latest FDA label advises avoiding the combination entirely due to high risk of fatal bleeding.",
        "key_concern": "Fatal bleeding risk",
        "confidence": "high"
    }

    diffs = [
        (diff_war_azi, reasoning_war_azi)
    ]

    # 3. Execute
    print("\n--- Running Patient Impact Analysis ---")
    report = await analyze_patient_impact(
        diffs=diffs,
        resolved_drugs=resolved_drugs,
        prescription_date=prescription_date,
        groq_client=groq_client,
        qdrant_client=qdrant_client
    )

    # 4. Assertions / Printing
    print("\n--- Patient Impact Report ---")
    print(f"Risk Level: {report.overall_risk_level}")
    print(f"Summary: {report.summary}")
    print(f"Recommended Action: {report.recommended_action}")
    print(f"ICMR Used: {report.icmr_guideline_used}")
    
    for alert in report.alerts:
        print(f"\nAlert: {alert.drug_pair}")
        print(f"  Change: {alert.change_type} (Delta: {alert.severity_delta})")
        print(f"  Dose Context: {alert.dose_context}")
        print(f"  Exposure Days: {alert.exposure_days}")
        if alert.icmr_context:
            print(f"  ICMR Context: {alert.icmr_context[:100]}...")

if __name__ == "__main__":
    asyncio.run(test_impact_synthesis())
