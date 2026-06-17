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

def test_temporal_find_interaction_matching_logic():
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

    # 1. Test Tier 2 Substring: "Azithromycin" should match "Azithromycin Anhydrous"
    match_substring = _find_interaction(mock_extraction, "Azithromycin")
    assert match_substring is not None
    assert match_substring.target_drug == "Azithromycin Anhydrous"

    # 2. Test Tier 2 Guard: "mycin" should NOT match "Azithromycin Anhydrous" (too short)
    match_short = _find_interaction(mock_extraction, "mycin")
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
    match_fuzzy = _find_interaction(mock_extraction_typo, "Azitromycin")
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
