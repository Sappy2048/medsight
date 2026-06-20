import pytest
from unittest.mock import MagicMock
from src.services.indian_drug_loader import extract_drug_data
from src.agents.temporal import _find_interaction, _classify_change
from src.agents.extraction import _assemble_records
from src.schemas.diff_schema import ExtractionResult, InteractionRecord

# ─── Task 1.1: Agent 1 FDC Extraction ────────────────────────────────────────

def test_agent1_fdc_extraction_ascoril_ls():
    """
    Vulnerability: FDC drugs with 3+ ingredients are often truncated in 
    short_composition columns.
    Assert: Ascoril LS Syrup correctly hits salt_composition fallback.
    """
    generics, dosages, method, score, metadata = extract_drug_data("Ascoril LS Syrup")
    
    # Ascoril LS Syrup in dataset: Ambroxol + Levosalbutamol + Guaifenesin
    assert len(generics) == 3
    assert "Ambroxol" in generics
    assert "Levosalbutamol" in generics
    assert "Guaifenesin" in generics
    assert "30mg/5ml" in dosages
    assert method in ["exact_match", "fuzzy_match"]


# ─── Task 1.2: Substring & Fuzzy Matcher ─────────────────────────────────────

@pytest.mark.asyncio
async def test_temporal_find_interaction_matching_logic():
    """
    Vulnerability: Over-eager matching (mycin -> azithromycin) or 
    under-eager matching (azithromycin -> azithromycin anhydrous).
    """
    mock_extraction = ExtractionResult(
        source_drug="Warfarin",
        version_date="2023-01-01",
        spl_id="test-spl",
        interactions=[
            InteractionRecord(
                source_drug="Warfarin",
                target_drug="Azithromycin Anhydrous",
                recommendation_text="Avoid",
                severity_text="Avoid",
                severity_score=5,
                version_date="2023-01-01",
                spl_id="test-spl",
                section="drug_interactions"
            )
        ]
    )

    # Mock the LLM client responses
    mock_llm = MagicMock()
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"is_match": false}'
    
    async def mock_create(*args, **kwargs):
        return mock_response
    mock_llm.chat.completions.create = mock_create

    # 1. Test Tier 2 Substring: "Azithromycin" should match "Azithromycin Anhydrous"
    match_substring = await _find_interaction(mock_extraction, "Azithromycin", mock_llm)
    assert match_substring is not None
    assert match_substring.target_drug == "Azithromycin Anhydrous"

    # 2. Test Tier 2 Guard: "mycin" should NOT match "Azithromycin Anhydrous" (too short)
    match_short = await _find_interaction(mock_extraction, "mycin", mock_llm)
    assert match_short is None

    # 3. Test Tier 3 Fuzzy: "Azitromycin" (typo) should match "Azithromycin"
    mock_extraction_typo = ExtractionResult(
        source_drug="Warfarin",
        version_date="2023-01-01",
        spl_id="test-spl",
        interactions=[
            InteractionRecord(
                source_drug="Warfarin",
                target_drug="Azithromycin",
                recommendation_text="Avoid",
                severity_text="Avoid",
                severity_score=5,
                version_date="2023-01-01",
                spl_id="test-spl",
                section="drug_interactions"
            )
        ]
    )
    match_fuzzy = await _find_interaction(mock_extraction_typo, "Azitromycin", mock_llm)
    assert match_fuzzy is not None
    assert match_fuzzy.target_drug == "Azithromycin"



# ─── Task 1.3: Severity Clamping ─────────────────────────────────────────────

def test_extraction_severity_clamping():
    """
    Vulnerability: LLM hallucinations may output scores outside 0-5.
    Assert: Python firewall clamps severity_score before Pydantic validation.
    """
    raw_llm_output = [
        {
            "target_drug": "Aspirin",
            "recommendation_text": "Do not mix",
            "severity_text": "Severe",
            "severity_score": 7  # Hallucinated score
        }
    ]
    
    records = _assemble_records(
        raw_interactions=raw_llm_output,
        section_name="drug_interactions",
        source_drug="Warfarin",
        version_date="2023-01-01",
        spl_id="test-spl"
    )
    
    assert len(records) == 1
    assert records[0].severity_score == 5  # Clamped to 5


# ─── Task 1.4: Diff Logic ────────────────────────────────────────────────────

@pytest.mark.parametrize("past_score, present_score, expected_type, expected_delta", [
    (None, 3, "ADDED", 3),
    (4, None, "REMOVED", -4),
    (2, 5, "STRENGTHENED", 3),
    (5, 2, "WEAKENED", -3),
    (3, 3, "UNCHANGED", 0),
])
def test_temporal_classify_change_branches(past_score, present_score, expected_type, expected_delta):
    """
    Test all 5 logical branches of the deterministic diff classifier.
    """
    past_match = None
    if past_score is not None:
        past_match = MagicMock(spec=InteractionRecord)
        past_match.severity_score = past_score
        
    present_match = None
    if present_score is not None:
        present_match = MagicMock(spec=InteractionRecord)
        present_match.severity_score = present_score
        
    change_type, delta = _classify_change(past_match, present_match)
    
    assert change_type == expected_type
    assert delta == expected_delta


# ─── Task 3.0: Semantic Target Expansion Tests ───────────────────────────────

from src.agents.temporal import is_semantic_drug_match
from src.services.rxnorm_client import get_drug_classes

@pytest.mark.asyncio
async def test_get_drug_classes():
    """
    Assert that get_drug_classes retrieves classes for a common drug.
    """
    classes = await get_drug_classes("Oxycodone")
    assert isinstance(classes, list)
    assert len(classes) > 0
    assert any(isinstance(c, str) for c in classes)


@pytest.mark.asyncio
async def test_is_semantic_drug_match():
    """
    Verify 5-tier matching works semantically.
    """
    mock_llm = MagicMock()
    
    # 1. Tier 1-3 String check
    assert await is_semantic_drug_match("Oxycodone", "Oxycodone", mock_llm) is True
    
    # 2. Tier 4 RxClass check: "Oxycodone" is in opioid-related drug classes
    classes = await get_drug_classes("Oxycodone")
    if classes:
        target_class = classes[0]
        assert await is_semantic_drug_match("Oxycodone", target_class, mock_llm) is True
        assert await is_semantic_drug_match("Oxycodone", target_class.lower(), mock_llm) is True

    # 3. Tier 5 LLM check: fall back to LLM semantic match
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = '{"is_match": true}'
    
    async def mock_create(*args, **kwargs):
        return mock_response
    mock_llm.chat.completions.create = mock_create

    assert await is_semantic_drug_match("SomeGenericDrug", "SomeGenericClass", mock_llm) is True

