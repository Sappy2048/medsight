import asyncio
import pytest
import httpx
from unittest.mock import patch, MagicMock

from src.agents.resolution import resolve_prescription
from src.schemas.prescription_schema import ParsedPrescription, ParsedDrug
from src.schemas.resolution_schema import ResolvedDrug, ResolutionError


@pytest.fixture
def mock_http_client() -> httpx.AsyncClient:
    """Returns a mock httpx.AsyncClient (lifecycle managed by test)."""
    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_popular_rxcui_resolver():
    """
    Patches get_rxcui to return deterministic RxCUIs for the generics
    of popular Indian drugs.
    """
    rxcui_map = {
        "paracetamol": "161",
        "amoxycillin": "723",
        "clavulanic acid": "429",
        "ibuprofen": "5640",
        "pantoprazole": "40254",
        "domperidone": "3616",
        "pheniramine": "8156",
        "bromelain": "1802",
        "diclofenac": "3355",
        "glimepiride": "2566",
        "metformin": "6809",
        "levocetirizine": "358265",
        "montelukast": "122858",
    }
    
    async def fake_get_rxcui(drug_name: str, client: httpx.AsyncClient) -> str | None:
        return rxcui_map.get(drug_name.lower().strip())
    
    with patch("src.agents.resolution.get_rxcui", side_effect=fake_get_rxcui):
        yield None


@pytest.mark.asyncio
async def test_popular_indian_drug_dolo_resolution(
    mock_http_client: httpx.AsyncClient,
    mock_popular_rxcui_resolver,
):
    """
    Verify that Dolo 650 resolves correctly to Paracetamol 650mg and fetches the correct RxCUI.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Dolo 650",
                prescribed_dose="1 tablet",
                route="oral",
                frequency="TID",
                duration_days=5,
            )
        ],
        raw_input="Dolo 650 1 tab TID for 5 days",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    assert resolved.raw_prescription_input == "Dolo 650"
    assert resolved.generic_names == ["Paracetamol"]
    assert resolved.formulated_strength == "650mg"
    assert resolved.rxcui_list == ["161"]
    assert resolved.prescribed_dose == "1 tablet"
    assert resolved.route_of_administration == "oral"
    assert resolved.frequency == "TID"
    assert resolved.resolution_method in ("exact_match", "fuzzy_match")
    
    # Metadata assertions
    assert resolved.dataset_metadata is not None
    assert resolved.dataset_metadata.medicine_type is not None


@pytest.mark.asyncio
async def test_popular_indian_drug_augmentin_resolution(
    mock_http_client: httpx.AsyncClient,
    mock_popular_rxcui_resolver,
):
    """
    Verify that Augmentin 625 Duo resolves to Amoxycillin and Clavulanic Acid.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625 Duo",
                prescribed_dose="1 tablet",
                route="oral",
                frequency="BID",
                duration_days=7,
            )
        ],
        raw_input="Augmentin 625 Duo BID",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    assert resolved.raw_prescription_input == "Augmentin 625 Duo"
    assert "Amoxycillin" in resolved.generic_names
    assert "Clavulanic Acid" in resolved.generic_names
    assert len(resolved.generic_names) == 2
    assert resolved.rxcui_list == ["723", "429"] or resolved.rxcui_list == ["429", "723"]
    assert resolved.resolution_method in ("exact_match", "fuzzy_match")


@pytest.mark.asyncio
async def test_popular_indian_drug_combiflam_resolution(
    mock_http_client: httpx.AsyncClient,
    mock_popular_rxcui_resolver,
):
    """
    Verify that Combiflam resolves to Ibuprofen and Paracetamol.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Combiflam",
                prescribed_dose="1 tablet",
                route="oral",
                frequency="PRN",
            )
        ],
        raw_input="Combiflam 1 tab SOS",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    assert resolved.raw_prescription_input == "Combiflam"
    assert "Ibuprofen" in resolved.generic_names
    assert "Paracetamol" in resolved.generic_names
    assert len(resolved.generic_names) == 2
    assert "5640" in resolved.rxcui_list
    assert "161" in resolved.rxcui_list
    assert resolved.resolution_method in ("exact_match", "fuzzy_match")


@pytest.mark.asyncio
async def test_popular_indian_drug_pantocid_resolution(
    mock_http_client: httpx.AsyncClient,
    mock_popular_rxcui_resolver,
):
    """
    Verify that Pantocid resolves to its generic components (Domperidone + Pantoprazole).
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Pantocid",
                prescribed_dose="1 tablet",
                route="oral",
                frequency="OD",
            )
        ],
        raw_input="Pantocid 1 tab OD",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    assert resolved.raw_prescription_input == "Pantocid"
    assert "Pantoprazole" in resolved.generic_names or "Domperidone" in resolved.generic_names
    assert "40254" in resolved.rxcui_list or "3616" in resolved.rxcui_list


@pytest.mark.asyncio
async def test_multiple_popular_drugs_concurrent_resolution(
    mock_http_client: httpx.AsyncClient,
    mock_popular_rxcui_resolver,
):
    """
    Verify that multiple popular drugs are resolved concurrently in the correct order.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(target_brand_name="Dolo 650", route="oral", frequency="TID"),
            ParsedDrug(target_brand_name="Augmentin 625 Duo", route="oral", frequency="BID"),
            ParsedDrug(target_brand_name="Combiflam", route="oral", frequency="PRN"),
            ParsedDrug(target_brand_name="Avil", route="oral", frequency="BID"),
        ],
        raw_input="Dolo 650 + Augmentin + Combiflam + Avil",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 4
    
    # Verify order and generic names
    assert results[0].raw_prescription_input == "Dolo 650"
    assert results[0].generic_names == ["Paracetamol"]
    
    assert results[1].raw_prescription_input == "Augmentin 625 Duo"
    assert "Amoxycillin" in results[1].generic_names
    
    assert results[2].raw_prescription_input == "Combiflam"
    assert "Ibuprofen" in results[2].generic_names
    
    assert results[3].raw_prescription_input == "Avil"
    assert "Pheniramine" in results[3].generic_names
