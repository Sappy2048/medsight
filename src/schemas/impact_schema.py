from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime

class DrugPairAlert(BaseModel):
    """Contextual alert for a specific drug pair interaction change."""
    drug_pair:             str
    change_type:           str
    severity_delta:        int
    present_severity_score: Optional[int]
    clinical_reasoning:    str           # from reasoning_dict
    key_concern:           Optional[str]
    confidence:            str
    dose_context:          Optional[str] = None # e.g. "High-dose Warfarin 10mg — amplifies bleeding risk"
    exposure_days:         Optional[int] = None # days since warning changed vs prescription_date
    icmr_context:          Optional[str] = None # relevant ICMR guideline snippet if found

class PatientImpactReport(BaseModel):
    """The final synthesized report for the treating physician."""
    prescription_date:     str
    report_generated_at:   str = Field(default_factory=lambda: datetime.now().isoformat())
    overall_risk_level:    Literal["CRITICAL", "HIGH", "MODERATE", "LOW", "NONE"]
    summary:               str           # 2-3 sentence plain-English summary for doctor
    alerts:                list[DrugPairAlert]  # sorted: most severe first
    recommended_action:    str           # top-line clinical action
    flagged_pairs_count:   int
    total_pairs_evaluated: int
    icmr_guideline_used:   bool
