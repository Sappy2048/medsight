"""
test_resolution_pipeline.py
───────────────────────────
End-to-end tests for Agent 1 (Drug Resolution Pipeline).

Tests the complete flow from ParsedPrescription input through
dataset extraction, composition parsing, and RxCUI resolution.

The RxNorm API is fully mocked — no external network calls.
"""

import asyncio
import pytest
import pytest_asyncio
from typing import Generator
from unittest.mock import AsyncMock, patch, MagicMock

import httpx

from src.agents.resolution import resolve_prescription, _resolve_rxcui_list, _resolve_single_drug
from src.schemas.prescription_schema import ParsedPrescription, ParsedDrug
from src.schemas.resolution_schema import ResolvedDrug, ResolutionError


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures & Mocks
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_http_client() -> httpx.AsyncClient:
    """Returns a mock httpx.AsyncClient (lifecycle managed by test)."""
    return MagicMock(spec=httpx.AsyncClient)


@pytest.fixture
def mock_rxcui_resolver() -> Generator[None, None, None]:
    """
    Patches get_rxcui to return deterministic fake RxCUIs.
    
    Mapping:
        - Amoxycillin -> "723"
        - Clavulanic Acid -> "429"
        - Ambroxol -> "12345"
        - Levosalbutamol -> "12346"
        - Guaifenesin -> "12347"
        - Paracetamol -> "161"
        - Povidone Iodine -> "6666"
        - Unknown -> None
    """
    rxcui_map = {
        "amoxycillin": "723",
        "clavulanic acid": "429",
        "ambroxol": "12345",
        "levosalbutamol": "12346",
        "guaifenesin": "12347",
        "paracetamol": "161",
        "povidone iodine": "6666",
        "povidone": "6666",
        "iodine": "6666",
        "aspirin": "1191",
        "metformin": "6809",
        "unknownbrand123": "99999",  # Raw fallback gets a fake CUI
        "totallyunknownbrandxyz": "88888",  # Another fake CUI for raw fallback
    }
    
    async def fake_get_rxcui(drug_name: str, client: httpx.AsyncClient) -> str | None:
        return rxcui_map.get(drug_name.lower().strip())
    
    with patch("src.agents.resolution.get_rxcui", side_effect=fake_get_rxcui):
        yield None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Dataset Extraction & Resolution Methods
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_exact_match_with_salt_composition(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Full pipeline: "Augmentin 625 Duo Tablet" should hit exact match,
    use salt_composition (Priority 1), extract 2 generics, resolve both RxCUIs.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625 Duo Tablet",
                prescribed_dose="1 tablet",
                route="oral",
                frequency="BID",
                duration_days=7,
            )
        ],
        raw_input="Tab Augmentin 625 Duo Tablet 1 tab BD for 7 days",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    # Verify resolution path
    assert resolved.resolution_method == "exact_match"
    assert resolved.fuzzy_score is None
    
    # Verify generics extracted from salt_composition
    assert "Amoxycillin" in resolved.generic_names
    assert "Clavulanic Acid" in resolved.generic_names
    assert len(resolved.generic_names) == 2
    
    # Verify dosages parsed
    assert resolved.formulated_strength == "500mg + 125mg"
    
    # Verify RxCUIs resolved
    assert "723" in resolved.rxcui_list  # Amoxycillin
    assert "429" in resolved.rxcui_list  # Clavulanic Acid
    assert len(resolved.rxcui_list) == 2
    
    # Verify metadata populated
    assert resolved.dataset_metadata is not None
    assert resolved.dataset_metadata.manufacturer is not None


@pytest.mark.asyncio
async def test_fuzzy_match_brand_name(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Parser outputs "Augmentin 625" (shortened), dataset has full name.
    Should fuzzy match and extract correct generics.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625",  # Parser strips form factor
                prescribed_dose="1 tablet",
                route="oral",
                frequency="BID",
            )
        ],
        raw_input="Tab Augmentin 625 BD",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    # Should fuzzy match, not exact
    assert resolved.resolution_method == "fuzzy_match"
    assert resolved.fuzzy_score is not None
    assert resolved.fuzzy_score >= 85.0
    
    # Should still find correct drug
    assert "Amoxycillin" in resolved.generic_names
    assert "Clavulanic Acid" in resolved.generic_names


@pytest.mark.asyncio
async def test_three_component_fdc_salt_priority(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    "Ascoril LS Syrup" has 3 components in salt_composition but only 2 in short columns.
    Priority 1 should capture all 3 components, not truncated 2.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Ascoril LS Syrup",
                prescribed_dose="10ml",
                route="oral",
                frequency="TID",
            )
        ],
        raw_input="Syp Ascoril LS 10ml TID",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    # Should have ALL 3 components from salt_composition
    assert len(resolved.generic_names) == 3
    assert "Ambroxol" in resolved.generic_names
    assert "Levosalbutamol" in resolved.generic_names
    assert "Guaifenesin" in resolved.generic_names
    
    # All 3 should have resolved RxCUIs
    assert len(resolved.rxcui_list) == 3
    assert "12345" in resolved.rxcui_list  # Ambroxol
    assert "12346" in resolved.rxcui_list  # Levosalbutamol
    assert "12347" in resolved.rxcui_list  # Guaifenesin
    
    # Dosages should include liquid formulations
    assert resolved.formulated_strength is not None
    assert "30mg/5ml" in resolved.formulated_strength
    assert "1mg/5ml" in resolved.formulated_strength
    assert "50mg/5ml" in resolved.formulated_strength


@pytest.mark.asyncio
async def test_short_composition_truncation_fallback(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Drug without salt_composition falls back to short_composition1/2.
    Even if real drug has 3 components, we'll only get 2 (known limitation).
    """
    # Use a drug that definitely has no salt_composition
    # We'll verify the waterfall behavior by checking the output structure
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Dolo 650 Tablet",  # Has no salt_comp in dataset
                prescribed_dose="1 tablet",
                route="oral",
                frequency="QID",
            )
        ],
        raw_input="Dolo 650 1 tab QID",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    # Should resolve via short_composition1 (Paracetamol 650mg)
    assert "Paracetamol" in resolved.generic_names
    assert resolved.formulated_strength == "650mg"
    assert "161" in resolved.rxcui_list


# ─────────────────────────────────────────────────────────────────────────────
# Test: Raw Fallback & Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_raw_fallback_unknown_brand(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Completely unknown brand should fall back to raw_generic mode,
    treating the brand name as the generic name.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="UnknownBrand123",
                prescribed_dose="500mg",
                route="oral",
                frequency="OD",
            )
        ],
        raw_input="Take UnknownBrand123 500mg daily",
        extraction_confidence="low",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 1
    resolved = results[0]
    
    assert resolved.resolution_method == "raw_fallback"
    assert resolved.fuzzy_score is None
    assert resolved.dataset_metadata is None  # No dataset match
    
    # Should use brand name as generic (dosage stripped by _parse_single_component)
    assert "UnknownBrand123" in resolved.generic_names
    assert resolved.formulated_strength is None or resolved.formulated_strength == ""


@pytest.mark.asyncio
async def test_rxcui_partial_failure_allowed(
    mock_http_client: httpx.AsyncClient,
):
    """
    If 2 generics resolve and 1 fails, we should still get a ResolvedDrug
    with partial rxcui_list (not a total failure).
    """
    # Custom mock: one generic fails
    async def partial_fail(drug_name: str, client: httpx.AsyncClient) -> str | None:
        if "Guaifenesin" in drug_name:
            return None  # Simulate API miss
        return {"Ambroxol": "12345", "Levosalbutamol": "12346"}.get(drug_name)
    
    with patch("src.agents.resolution.get_rxcui", side_effect=partial_fail):
        parsed = ParsedPrescription(
            drugs=[
                ParsedDrug(
                    target_brand_name="Ascoril LS Syrup",
                    prescribed_dose="10ml",
                    route="oral",
                    frequency="TID",
                )
            ],
            raw_input="Syp Ascoril LS 10ml TID",
            extraction_confidence="high",
        )
        
        results = await resolve_prescription(parsed, mock_http_client)
        
        assert len(results) == 1
        resolved = results[0]
        
        # Should have 3 generics but only 2 RxCUIs
        assert len(resolved.generic_names) == 3
        assert len(resolved.rxcui_list) == 2
        assert "12345" in resolved.rxcui_list
        assert "12346" in resolved.rxcui_list


@pytest.mark.asyncio
async def test_rxcui_total_failure_raises_error(
    mock_http_client: httpx.AsyncClient,
):
    """
    If NO generics resolve to RxCUIs, ResolutionError should be raised.
    """
    async def total_fail(drug_name: str, client: httpx.AsyncClient) -> None:
        return None  # Everything fails
    
    with patch("src.agents.resolution.get_rxcui", side_effect=total_fail):
        parsed = ParsedPrescription(
            drugs=[
                ParsedDrug(
                    target_brand_name="Augmentin 625 Duo Tablet",
                    prescribed_dose="1 tablet",
                    route="oral",
                    frequency="BID",
                )
            ],
            raw_input="Tab Augmentin 625 BD",
            extraction_confidence="high",
        )
        
        with pytest.raises(ResolutionError) as exc_info:
            await resolve_prescription(parsed, mock_http_client)
        
        assert "RxNorm could not resolve any generic" in str(exc_info.value)


@pytest.mark.asyncio
async def test_empty_prescription_raises_error(
    mock_http_client: httpx.AsyncClient,
):
    """
    ParsedPrescription with no drugs should raise ResolutionError.
    """
    parsed = ParsedPrescription(
        drugs=[],
        raw_input="Patient feels better today",
        extraction_confidence="low",
    )
    
    with pytest.raises(ResolutionError) as exc_info:
        await resolve_prescription(parsed, mock_http_client)
    
    assert "no drugs to resolve" in str(exc_info.value).lower()


# ─────────────────────────────────────────────────────────────────────────────
# Test: Multi-Drug Concurrency
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_multi_drug_concurrent_resolution(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    3 drugs should be resolved concurrently, all succeed, order preserved.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625 Duo Tablet",
                prescribed_dose="1 tab",
                route="oral",
                frequency="BID",
            ),
            ParsedDrug(
                target_brand_name="Dolo 650 Tablet",
                prescribed_dose="1 tab",
                route="oral",
                frequency="QID",
            ),
            ParsedDrug(
                target_brand_name="Ascoril LS Syrup",
                prescribed_dose="10ml",
                route="oral",
                frequency="TID",
            ),
        ],
        raw_input="Augmentin BD + Dolo QID + Ascoril TID",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    
    assert len(results) == 3
    
    # Order preserved
    assert results[0].raw_prescription_input == "Augmentin 625 Duo Tablet"
    assert results[1].raw_prescription_input == "Dolo 650 Tablet"
    assert results[2].raw_prescription_input == "Ascoril LS Syrup"
    
    # All have resolved generics
    assert len(results[0].generic_names) == 2  # Augmentin
    assert len(results[1].generic_names) == 1  # Dolo
    assert len(results[2].generic_names) == 3  # Ascoril
    
    # All have RxCUIs
    assert all(len(r.rxcui_list) >= 1 for r in results)


# ─────────────────────────────────────────────────────────────────────────────
# Test: Clinical Field Handling
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_parsed_clinical_fields_preserved(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    prescribed_dose, route, frequency from parser should be preserved
    in ResolvedDrug (with route normalization).
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625 Duo Tablet",
                prescribed_dose="1 tablet after meals",
                route="oral",
                frequency="BID",
                duration_days=5,
            )
        ],
        raw_input="Augmentin 1 tab BD x 5 days",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    resolved = results[0]
    
    assert resolved.prescribed_dose == "1 tablet after meals"
    assert resolved.frequency == "BID"
    assert resolved.route_of_administration == "oral"


@pytest.mark.asyncio
async def test_route_normalization_invalid_route(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Invalid/unknown route from parser should be normalized to None by Pydantic validator.
    The validator only accepts specific routes; 'unknown' is not in the valid set.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625 Duo Tablet",
                prescribed_dose="1 tab",
                route="unknown",  # Not in _VALID_ROUTES, so becomes None
                frequency="BID",
            )
        ],
        raw_input="Augmentin 1 tab",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    # "unknown" is NOT in _VALID_ROUTES, so it gets normalized to None
    assert results[0].route_of_administration is None


@pytest.mark.asyncio
async def test_metadata_populated_on_match(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    DatasetMetadata should be populated with manufacturer, type, etc.
    on exact or fuzzy match. Should be None on raw fallback.
    """
    # Exact match - should have metadata
    parsed_exact = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Augmentin 625 Duo Tablet",
                prescribed_dose="1 tab",
                route="oral",
                frequency="BID",
            )
        ],
        raw_input="Augmentin",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed_exact, mock_http_client)
    assert results[0].dataset_metadata is not None
    assert results[0].dataset_metadata.manufacturer is not None
    
    # Raw fallback - should have no metadata
    parsed_raw = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="TotallyUnknownBrandXYZ",
                prescribed_dose="500mg",
                route="oral",
                frequency="OD",
            )
        ],
        raw_input="Unknown drug",
        extraction_confidence="low",
    )
    
    results = await resolve_prescription(parsed_raw, mock_http_client)
    assert results[0].dataset_metadata is None


# ─────────────────────────────────────────────────────────────────────────────
# Test: Dosage Parsing Edge Cases
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_liquid_formulation_dosage_parsing(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Liquid formulations like "30mg/5ml" should be preserved in formulated_strength.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Ascoril LS Syrup",
                prescribed_dose="10ml",  # Patient takes 10ml
                route="oral",
                frequency="TID",
            )
        ],
        raw_input="Ascoril LS 10ml TID",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    resolved = results[0]
    
    # Prescribed dose is what patient takes
    assert resolved.prescribed_dose == "10ml"
    
    # Formulated strength is what's in the syrup (per 5ml)
    assert resolved.formulated_strength is not None
    assert "30mg/5ml" in resolved.formulated_strength


@pytest.mark.asyncio
async def test_percentage_concentration_parsing(
    mock_http_client: httpx.AsyncClient,
    mock_rxcui_resolver,
):
    """
    Topical preparations with % w/v or % w/w should parse correctly.
    """
    parsed = ParsedPrescription(
        drugs=[
            ParsedDrug(
                target_brand_name="Betadine 2% Gargle Mint",
                prescribed_dose="15ml diluted",
                route="topical",
                frequency="PRN",
            )
        ],
        raw_input="Betadine gargle SOS",
        extraction_confidence="high",
    )
    
    results = await resolve_prescription(parsed, mock_http_client)
    resolved = results[0]
    
    assert "Povidone Iodine" in resolved.generic_names
    assert resolved.formulated_strength is not None
    assert "2% w/v" in resolved.formulated_strength or "2%" in resolved.formulated_strength
