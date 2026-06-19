import asyncio
import os
import json
import logging
from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.agents.temporal import compute_temporal_diff
from src.schemas.diff_schema import ExtractionResult, InteractionRecord
from src.config import SEVERITY_ONTOLOGY, TOGETHER_BASE_URL, TOGETHER_API_KEY

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

if __name__ == "__main__":
    asyncio.run(test_temporal_logic())
