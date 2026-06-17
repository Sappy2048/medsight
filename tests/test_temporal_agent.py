import asyncio
import os
import logging
from groq import AsyncGroq
from src.schemas.diff_schema import ExtractionResult, InteractionRecord
from src.agents.temporal import compute_temporal_diff

logging.basicConfig(level=logging.WARNING)

def make_record(drug: str, score: int, text: str = "Test text") -> InteractionRecord:
    return InteractionRecord(
        source_drug="TestSource",
        target_drug=drug,
        recommendation_text=text,
        severity_text="test_severity",
        severity_score=score,
        version_date="2026-01-01",
        spl_id="test-spl"
    )

def make_extraction(interactions: list) -> ExtractionResult:
    return ExtractionResult(
        source_drug="TestSource",
        version_date="2026-01-01",
        spl_id="test-spl",
        interactions=interactions
    )

async def run_tests():
    client = AsyncGroq(api_key=os.getenv("GROQ_API_KEY"))
    target = "Azithromycin"
    
    print("Running Temporal Agent Boundary Tests...\n")

    # 1. WEAKENED Boundary (-1 Delta) -> Should NOT be clinically significant
    diff, _ = await compute_temporal_diff(
        make_extraction([make_record(target, 4)]), 
        make_extraction([make_record(target, 3)]), 
        target, client
    )
    assert diff.change_type == "WEAKENED" and diff.severity_delta == -1 and not diff.is_clinically_significant, "❌ -1 WEAKENED Boundary failed"
    print("✅ Passed: -1 WEAKENED Boundary (Not Significant)")
    
    # 2. STRENGTHENED Boundary (+1 Delta)
    diff, _ = await compute_temporal_diff(
        make_extraction([make_record(target, 2)]),
        make_extraction([make_record(target, 3)]),
        target, client
    )
    assert (
        diff.change_type == "STRENGTHENED"
        and diff.severity_delta == 1
        and not diff.is_clinically_significant
    )
    print("✅ Passed: +1 STRENGTHENED Boundary (Not Significant)")
    
    # 3. ADDED Low Severity (None -> 1) -> Should BE clinically significant because it's ADDED
    diff, _ = await compute_temporal_diff(
        make_extraction([]), 
        make_extraction([make_record(target, 1)]), 
        target, client
    )
    assert diff.change_type == "ADDED" and diff.severity_delta == 1 and diff.is_clinically_significant, "❌ Low Severity ADDED failed"
    print("✅ Passed: Low Severity ADDED (Significant)")

    # 3. REMOVED Low Severity (1 -> None) -> Should BE clinically significant because it's REMOVED
    diff, _ = await compute_temporal_diff(
        make_extraction([make_record(target, 1)]), 
        make_extraction([]), 
        target, client
    )
    assert diff.change_type == "REMOVED" and diff.severity_delta == -1 and diff.is_clinically_significant, "❌ Low Severity REMOVED failed"
    print("✅ Passed: Low Severity REMOVED (Significant)")

    # 4. Empty String Text Test (Pydantic safety check)
    diff, reasoning = await compute_temporal_diff(
        make_extraction([make_record(target, 4, "Avoid use")]), 
        make_extraction([make_record(target, 4, "")]), # Empty string instead of None
        target, client
    )
    assert reasoning["confidence"] == "low", "❌ Missing text confidence failed"
    print("✅ Passed: Missing text handled safely via empty string")

    print("\n🎉 Boundary tests passed! We are officially bulletproof.")

if __name__ == "__main__":
    asyncio.run(run_tests())