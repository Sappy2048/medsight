import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import date

from src.agents.graph import run_medsight
from src.schemas.prescription_schema import ParsedPrescription
from src.schemas.synthesizer_schema import MedSightFinalReport
from src.schemas.resolution_schema import ResolvedDrug
from src.schemas.impact_schema import PatientImpactReport

# ─── Mocks & Fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_client():
    return AsyncMock()

@pytest.fixture
def mock_qdrant_client():
    return MagicMock()

@pytest.fixture
def mock_db_pool():
    return MagicMock()

@pytest.fixture
def dummy_prescription():
    return ParsedPrescription(
        drugs=[],
        prescription_date=date(2023, 10, 5),
        patient_age=45,
        raw_input="Dummy input",
        extraction_confidence="high"
    )

@pytest.fixture
def dummy_final_report():
    """
    Creates a valid MedSightFinalReport using model_construct to bypass 
    deep validation of the nested PatientImpactReport during graph testing.
    """
    return MedSightFinalReport.model_construct(
        report=MagicMock(spec=PatientImpactReport),
        verified=True,
        integrity_score=0.9,
        severity_badge="LOW",
        verification_notes=[],
        override_applied=False,
        final_summary="All checks passed successfully.",
        final_recommended_action="Proceed with current prescription."
    )

# ─── Test Cases ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
@patch("src.agents.graph.preflight_validate", new_callable=AsyncMock)
async def test_early_exit_clarification_required(
    mock_preflight, 
    mock_llm_client, 
    mock_qdrant_client, 
    mock_db_pool,
    dummy_prescription
):
    """
    EDGE CASE 1: Copilot requests clarification. 
    Graph ends gracefully with an early exit dictionary structure.
    """
    mock_preflight.return_value = (
        dummy_prescription, 
        "Please provide the missing prescription date."
    )

    with patch("src.agents.graph.resolve_prescription", new_callable=AsyncMock) as mock_resolve:
        result = await run_medsight(
            "Take 500mg paracetamol", 
            mock_llm_client, 
            mock_qdrant_client, 
            mock_db_pool
        )

        assert isinstance(result, dict)
        assert result["status"] == "clarification_required"
        assert result["message"] == "Please provide the missing prescription date."
        mock_resolve.assert_not_called()


@pytest.mark.asyncio
@patch("src.agents.graph.preflight_validate", new_callable=AsyncMock)
@patch("src.agents.graph.resolve_prescription", new_callable=AsyncMock)
@patch("src.agents.graph.get_past_and_present_labels", new_callable=AsyncMock)
@patch("src.agents.graph.extract_interactions", new_callable=AsyncMock)
@patch("src.agents.graph.compute_temporal_diff", new_callable=AsyncMock)
@patch("src.agents.graph.analyze_patient_impact", new_callable=AsyncMock)
@patch("src.agents.graph.synthesize_final_report", new_callable=AsyncMock)
@patch("src.agents.graph.oversee_report", new_callable=AsyncMock)
async def test_successful_pipeline_execution(
    mock_oversee,
    mock_synthesize,
    mock_impact,
    mock_compute,
    mock_extract,
    mock_labels,
    mock_resolve,
    mock_preflight,
    mock_llm_client,
    mock_qdrant_client,
    mock_db_pool,
    dummy_prescription,
    dummy_final_report
):
    """
    EDGE CASE 2: Pipeline proceeds perfectly end-to-end.
    """
    mock_preflight.return_value = (dummy_prescription, None)
    
    mock_drug = MagicMock(spec=ResolvedDrug)
    mock_drug.generic_names = ["Paracetamol"]
    mock_resolve.return_value = [mock_drug]
    
    mock_labels.return_value = ("past_label", "present_label")
    mock_extract.return_value = {}
    mock_compute.return_value = ({}, {})
    mock_impact.return_value = AsyncMock(spec=PatientImpactReport)
    mock_synthesize.return_value = dummy_final_report
    mock_oversee.return_value = (False, "Looks good.")

    result = await run_medsight(
        "Take 500mg Dolo OD", 
        mock_llm_client, 
        mock_qdrant_client, 
        mock_db_pool
    )

    # Asserts validated against the wrapper dictionary structure
    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["report"] == dummy_final_report


@pytest.mark.asyncio
@patch("src.agents.graph.preflight_validate", new_callable=AsyncMock)
@patch("src.agents.graph.resolve_prescription", new_callable=AsyncMock)
@patch("src.agents.graph.get_past_and_present_labels", new_callable=AsyncMock)
@patch("src.agents.graph.extract_interactions", new_callable=AsyncMock)
@patch("src.agents.graph.compute_temporal_diff", new_callable=AsyncMock)
@patch("src.agents.graph.analyze_patient_impact", new_callable=AsyncMock)
@patch("src.agents.graph.synthesize_final_report", new_callable=AsyncMock)
async def test_runtime_error_on_missing_report(
    mock_synthesize,
    mock_impact,
    mock_compute,
    mock_extract,
    mock_labels,
    mock_resolve,
    mock_preflight,
    mock_llm_client,
    mock_qdrant_client,
    mock_db_pool,
    dummy_prescription
):
    """
    EDGE CASE 3: Synthesizer yields None.
    """
    mock_preflight.return_value = (dummy_prescription, None)
    
    mock_drug = MagicMock(spec=ResolvedDrug)
    mock_drug.generic_names = ["Paracetamol"]
    mock_resolve.return_value = [mock_drug]
    
    mock_labels.return_value = ("past", "present")
    mock_extract.return_value = {}
    mock_compute.return_value = ({}, {})
    mock_impact.return_value = AsyncMock(spec=PatientImpactReport)
    mock_synthesize.return_value = None

    with pytest.raises(ValueError, match="Final report is None — cannot oversee report."):
        await run_medsight(
            "Valid input but failing pipeline", 
            mock_llm_client, 
            mock_qdrant_client, 
            mock_db_pool
        )


@pytest.mark.asyncio
@patch("src.agents.graph.preflight_validate", new_callable=AsyncMock)
@patch("src.agents.graph.resolve_prescription", new_callable=AsyncMock)
@patch("src.agents.graph.get_past_and_present_labels", new_callable=AsyncMock)
@patch("src.agents.graph.extract_interactions", new_callable=AsyncMock)
@patch("src.agents.graph.compute_temporal_diff", new_callable=AsyncMock)
@patch("src.agents.graph.analyze_patient_impact", new_callable=AsyncMock)
@patch("src.agents.graph.synthesize_final_report", new_callable=AsyncMock)
@patch("src.agents.graph.oversee_report", new_callable=AsyncMock)
async def test_overseer_loop_trigger(
    mock_oversee,
    mock_synthesize,
    mock_impact,
    mock_compute,
    mock_extract,
    mock_labels,
    mock_resolve,
    mock_preflight,
    mock_llm_client,
    mock_qdrant_client,
    mock_db_pool,
    dummy_prescription,
    dummy_final_report
):
    """
    EDGE CASE 4: Quality overseer triggers an engine re-run.
    """
    mock_preflight.return_value = (dummy_prescription, None)
    
    mock_drug_1 = MagicMock(spec=ResolvedDrug)
    mock_drug_1.generic_names = ["Paracetamol"]
    
    mock_drug_2 = MagicMock(spec=ResolvedDrug)
    mock_drug_2.generic_names = ["Paracetamol"]

    mock_resolve.side_effect = [
        [mock_drug_1],
        [mock_drug_2]
    ]
    
    mock_labels.return_value = ("past", "present")
    mock_extract.return_value = {}
    mock_compute.return_value = ({}, {})
    mock_impact.return_value = AsyncMock()
    mock_synthesize.return_value = dummy_final_report
    
    # Clean 1-pass side effect sequence since graph double-execution is eliminated
    mock_oversee.side_effect = [
        (True, "Integrity score too low, rerunning."),
        (False, "Verified on second pass.")
    ]

    result = await run_medsight(
        "Trigger loop test", 
        mock_llm_client, 
        mock_qdrant_client, 
        mock_db_pool
    )

    assert isinstance(result, dict)
    assert result["status"] == "success"
    assert result["report"] == dummy_final_report
    assert mock_oversee.call_count == 2
    assert mock_resolve.call_count == 2