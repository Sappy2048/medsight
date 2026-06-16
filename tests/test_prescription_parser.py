import asyncio
import pytest
import json
import os
from dotenv import load_dotenv
from pydantic import ValidationError
from unittest.mock import AsyncMock, MagicMock
from src.agents.prescription_parser import PrescriptionParsingAgent
from src.schemas.prescription_schema import ParsedPrescription

load_dotenv(dotenv_path=".env")  # Load environment variables from .env file for testing

def _mock_groq_response(json_str: str) -> MagicMock:
    """Builds a mock that mimics the Groq response structure."""
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json_str
    return mock_response


@pytest.mark.asyncio
async def test_single_drug_parse():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response("""
    {
      "drugs": [{"target_brand_name": "Ascoril LS", "prescribed_dose": "10ml",
                 "route": "oral", "frequency": "BID", "duration_days": null}],
      "prescription_date": null,
      "patient_age": null,
      "raw_input": "Give pt Ascoril LS 10 ml po twice daily",
      "extraction_confidence": "high"
    }
    """)

    agent = PrescriptionParsingAgent(groq_client=mock_client)
    result = await agent.parse("Give pt Ascoril LS 10 ml po twice daily")

    assert isinstance(result, ParsedPrescription)
    assert result.drugs[0].target_brand_name == "Ascoril LS"
    assert result.drugs[0].route == "oral"
    assert result.drugs[0].frequency == "BID"


@pytest.mark.asyncio
async def test_multi_drug_parse():
    mock_client = AsyncMock()
    mock_client.chat.completions.create.return_value = _mock_groq_response("""
    {
      "drugs": [
        {"target_brand_name": "Augmentin", "prescribed_dose": "625mg",
         "route": "oral", "frequency": "BID", "duration_days": 5},
        {"target_brand_name": "Dolo", "prescribed_dose": "650mg",
         "route": "oral", "frequency": "TID", "duration_days": 5}
      ],
      "prescription_date": null,
      "patient_age": null,
      "raw_input": "Tab Augmentin 625 BD + Dolo 650 TDS for 5 days",
      "extraction_confidence": "high"
    }
    """)

    agent = PrescriptionParsingAgent(groq_client=mock_client)
    result = await agent.parse("Tab Augmentin 625 BD + Dolo 650 TDS for 5 days")

    assert len(result.drugs) == 2
    assert result.drugs[1].target_brand_name == "Dolo"
    assert result.drugs[1].frequency == "TID"

# Helper to mock Groq's response structure
def _build_mock_response(json_content: str) -> MagicMock:
    mock_response = MagicMock()
    mock_response.choices[0].message.content = json_content
    return mock_response

@pytest.mark.asyncio
async def test_retry_on_pydantic_validation_error():
    """
    Forces the LLM to return invalid schema twice, then valid schema on the third try.
    Verifies the agent retries and eventually succeeds.
    """
    mock_client = AsyncMock()
    
    # Attempt 1: 'duration_days' is a string instead of an integer (Triggers ValidationError)
    bad_json_1 = json.dumps({
        "drugs": [{"target_brand_name": "Aspirin", "route": "oral", "frequency": "OD", "duration_days": "five"}],
        "prescription_date": None, "patient_age": None, "extraction_confidence": "high"
    })
    
    # Attempt 2: 'route' is an invalid enum value (Triggers ValidationError)
    bad_json_2 = json.dumps({
        "drugs": [{"target_brand_name": "Aspirin", "route": "rectal", "frequency": "OD", "duration_days": 5}],
        "prescription_date": None, "patient_age": None, "extraction_confidence": "high"
    })
    
    # Attempt 3: Perfect JSON
    good_json = json.dumps({
        "drugs": [{"target_brand_name": "Aspirin", "route": "oral", "frequency": "OD", "duration_days": 5}],
        "prescription_date": None, "patient_age": None, "extraction_confidence": "high"
    })
    
    # Set the side_effect to return these sequentially
    mock_client.chat.completions.create.side_effect = [
        _build_mock_response(bad_json_1),
        _build_mock_response(bad_json_2),
        _build_mock_response(good_json)
    ]
    
    agent = PrescriptionParsingAgent(groq_client=mock_client)
    result = await agent.parse("Take Aspirin once daily for 5 days")
    
    # Assert it eventually succeeded
    assert result.drugs[0].target_brand_name == "Aspirin"
    
    # Assert the LLM was called exactly 3 times (1 initial + 2 retries)
    assert mock_client.chat.completions.create.call_count == 3


@pytest.mark.asyncio
async def test_exhaust_retries_raises_runtime_error():
    """
    Forces the LLM to consistently return garbage.
    Verifies that the agent raises a RuntimeError after exhausting retries.
    """
    mock_client = AsyncMock()
    
    # Provide completely broken JSON that triggers JSONDecodeError
    mock_client.chat.completions.create.return_value = _build_mock_response("```json \n Oops I forgot how to JSON")
    
    agent = PrescriptionParsingAgent(groq_client=mock_client)
    
    # We expect a RuntimeError when retries run out
    with pytest.raises(RuntimeError) as exc_info:
        await agent.parse("Tab Paracetamol 500mg")
        
    # Verify the error message contains helpful context
    assert "failed after 3 attempts" in str(exc_info.value)
    
    # Verify it tried exactly MAX_RETRIES + 1 times (2 retries + 1 initial = 3)
    assert mock_client.chat.completions.create.call_count == 3

@pytest.fixture
def live_client():
    """Fixture to provide a real AsyncGroq client if API key is present."""
    api_key = os.environ.get("GROQ_API_KEY", "YOUR_GROQ_API_KEY_HERE")
    if api_key == "YOUR_GROQ_API_KEY_HERE":
        pytest.skip("Skipping live tests: No Groq API key found.")
    from groq import AsyncGroq
    return AsyncGroq(api_key=api_key)


@pytest.mark.asyncio
async def test_live_messy_clinical_prose(live_client):
    """Tests extraction from a dense paragraph of clinical notes with mixed instructions."""
    agent = PrescriptionParsingAgent(groq_client=live_client)
    raw_text = (
        "Pt presented with acute pharyngitis. BP 120/80. Temp 101F. "
        "Start Tab Augmentin 625mg po twice daily for 7 days. "
        "Also take Dolo 650 TDS for 3 days. Use Betadine gargles SOS."
    )
    
    result = await agent.parse(raw_text)
    
    # It should ignore the BP/Temp and extract exactly 3 drugs
    assert len(result.drugs) == 3
    
    drug_names = [d.target_brand_name.lower() for d in result.drugs]
    assert "augmentin" in drug_names
    assert "dolo" in drug_names
    assert "betadine" in drug_names

    # Check Betadine PRN/SOS normalization
    betadine = next(d for d in result.drugs if d.target_brand_name.lower() == "betadine")
    assert betadine.frequency == "PRN" # "SOS" should ideally map to PRN (as needed)


@pytest.mark.asyncio
async def test_live_extreme_abbreviations(live_client):
    """Tests handling of heavy shorthand and missing parameters."""
    agent = PrescriptionParsingAgent(groq_client=live_client)
    raw_text = "PCM 500 QID x 5d + Amox 250 TDS"
    
    result = await agent.parse(raw_text)
    assert len(result.drugs) == 2
    
    pcm = result.drugs[0]
    assert "500" in str(pcm.prescribed_dose)
    assert pcm.frequency == "QID"
    assert pcm.duration_days == 5
    
    amox = result.drugs[1]
    assert amox.frequency == "TID" # "TDS" maps to "TID"
    assert amox.duration_days is None # No duration specified for Amox


@pytest.mark.asyncio
async def test_live_irrelevant_input(live_client):
    """Tests how the LLM/Schema handle a prompt with absolutely no drugs."""
    agent = PrescriptionParsingAgent(groq_client=live_client)
    raw_text = "Patient feels better today. Advised to drink plenty of water and rest."
    
    result = await agent.parse(raw_text)
    
    # Should return an empty list, not hallucinate drugs or break the schema
    assert isinstance(result.drugs, list)
    assert len(result.drugs) == 0
    assert result.extraction_confidence in ["low", "high"] # Usually low or N/A for empty