from pydantic import BaseModel, Field, model_validator
from typing import Literal, Optional
from datetime import date


# ─── Controlled Vocabularies ──────────────────────────────────────────────────

RouteType = Literal[
    "oral", "intravenous", "topical", "subcutaneous",
    "intramuscular", "sublingual", "inhalation", "unknown"
]

FrequencyType = Literal[
    "OD",    # Once daily
    "BID",   # Twice daily
    "TID",   # Three times daily
    "QID",   # Four times daily
    "PRN",   # As needed
    "STAT",  # Immediately / one-time
    "QHS",   # At bedtime
    "Q4H",   # Every 4 hours
    "Q6H",   # Every 6 hours
    "Q8H",   # Every 8 hours
    "Q12H",  # Every 12 hours
    "unknown"
]


# ─── Per-Drug Extraction Unit ─────────────────────────────────────────────────

class ParsedDrug(BaseModel):
    """Represents one drug extracted from the prescription string."""

    target_brand_name: str = Field(
        description=(
            "The medicine brand name only, stripped of all dosages, frequencies, "
            "and instructions. E.g. 'Ascoril LS', 'Dolo', 'Augmentin'. "
            "If unknown, return an empty string."
        )
    )
    prescribed_dose: Optional[str] = Field(
        default=None,
        description="Exact dose per instance: '10ml', '500mg', '2 tablets'. Null if not stated."
    )
    route: RouteType = Field(
        default="unknown",
        description=(
            "Normalized route. Map 'po'/'by mouth'/'oral' → 'oral', "
            "'iv'/'intravenous' → 'intravenous', etc. Default 'unknown'."
        )
    )
    frequency: FrequencyType = Field(
        default="unknown",
        description=(
            "Normalized frequency code. Map 'twice daily'/'BD'/'BID' → 'BID', "
            "'once daily'/'OD'/'QD' → 'OD', 'three times'/'TID'/'TDS' → 'TID', etc. "
            "Default 'unknown'."
        )
    )
    duration_days: Optional[int] = Field(
        default=None,
        description="Duration in days if mentioned. '5 days' → 5, '2 weeks' → 14. Null if absent."
    )

    @model_validator(mode="after")
    def strip_brand_name(self) -> "ParsedDrug":
        """Ensure brand name has no trailing whitespace or dosage leakage."""
        self.target_brand_name = self.target_brand_name.strip()
        return self


# ─── Top-Level Prescription Parse Result ─────────────────────────────────────

class ParsedPrescription(BaseModel):
    """
    Full output of the Prescription Parsing Agent.
    One prescription string → one ParsedPrescription containing N drugs.
    """

    drugs: list[ParsedDrug] = Field(
        description="All drugs identified in the prescription. Always a list, even for one drug."
    )
    prescription_date: Optional[date] = Field(
        default=None,
        description=(
            "The date the prescription was written, if mentioned. "
            "ISO format: YYYY-MM-DD. Null if absent — will be requested from user."
        )
    )
    patient_age: Optional[int] = Field(
        default=None,
        description="Patient age in years if mentioned. Null if absent."
    )
    raw_input: str = Field(
        description="The original unmodified input string. Preserved for audit trail."
    )
    extraction_confidence: Literal["high", "medium", "low"] = Field(
        default="medium",
        description=(
            "Self-assessed confidence: 'high' if all fields populated cleanly, "
            "'low' if significant ambiguity exists, 'medium' otherwise."
        )
    )
