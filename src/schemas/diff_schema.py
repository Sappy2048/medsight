from typing import Optional, Literal
from pydantic import BaseModel, Field


class InteractionRecord(BaseModel):
    """Single interaction extracted from one label version."""
    source_drug:         str
    target_drug:         str
    recommendation_text: str
    warning_text:        Optional[str] = None
    severity_text:       str           # raw FDA text: "avoid", "monitor" etc.
    severity_score:      int           # mapped via SEVERITY_ONTOLOGY in config.py
    version_date:        str
    spl_id:              str
    
    # --- Addition 2: Section attribution ---
    section: Literal[
        "boxed_warning",
        "contraindications",
        "warnings_and_precautions",
        "drug_interactions"
    ] = "drug_interactions"


class ExtractionResult(BaseModel):
    """Full output of Extraction Agent for one label version."""
    source_drug:  str
    version_date: str
    spl_id:       str
    interactions: list[InteractionRecord]


class DiffResult(BaseModel):
    """Output of Temporal Diff Agent comparing past vs present label."""
    drug_pair: str  # e.g., "Aspirin + Warfarin"

    # Locked down to prevent LLM hallucinations
    change_type: Literal[
        "ADDED", "REMOVED", "STRENGTHENED", "WEAKENED", "UNCHANGED"
    ]

    past_recommendation:    Optional[str] = None
    present_recommendation: Optional[str] = None
    past_severity_score:    Optional[int] = None
    present_severity_score: Optional[int] = None

    # Safe default prevents math crashes if past/present values are None
    severity_delta: int = 0

    past_version_date:    str
    present_version_date: str
    is_clinically_significant: bool  # True if delta >= 2 or ADDED

    # Split into past/present so the UI can cite both FDA labels
    past_spl_id:    Optional[str] = None
    present_spl_id: Optional[str] = None

    # --- Addition 1: Graceful handling of missing label history ---
    data_unavailable: bool = False  # True if label history predates prescription_date
