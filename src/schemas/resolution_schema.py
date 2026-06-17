from typing import Optional
from pydantic import BaseModel, Field, field_validator


class DatasetMetadata(BaseModel):
    """Non-critical pharmacological context from the Indian Medicine Dataset."""
    manufacturer:        Optional[str] = None
    medicine_type:       Optional[str] = None
    pack_size:           Optional[str] = None
    medicine_desc:       Optional[str] = None
    static_side_effects: Optional[str] = None


class ResolvedDrug(BaseModel):
    """
    Validated output of Agent 1 (Drug Resolution).
    Passed downstream to Agent 2 (FDA Label Fetcher).
    """

    # ── Input traceability ────────────────────────────────────────────────────
    raw_prescription_input: str

    # ── Pharmacological context (from dataset) ────────────────────────────────
    generic_names:       list[str] = Field(..., min_length=1)
    formulated_strength: Optional[str] = Field(
        None,
        description="Strength per unit from dataset, e.g. '500mg' or '30mg/5ml'",
    )

    # ── RxNorm resolution (populated in Agent 1, used by Agent 2) ────────────
    rxcui_list: list[str] = Field(..., min_length=1)

    # ── Clinical context (from LLM prescription parser) ───────────────────────
    prescribed_dose:         Optional[str] = Field(
        None,
        description="What the patient actually takes per instance: '10ml', '2 tablets'",
    )
    route_of_administration: Optional[str] = None
    frequency:               Optional[str] = None

    # ── Traceability & confidence ─────────────────────────────────────────────
    resolution_method: str = Field(
        pattern="^(exact_match|fuzzy_match|raw_fallback)$"
    )
    parsing_method: str = Field(
        pattern="^(llm)$",
    )
    fuzzy_score: Optional[float] = Field(
        None,
        description="RapidFuzz WRatio score (0-100). Only set when resolution_method=fuzzy_match.",
        ge=0.0,
        le=100.0,
    )
    dataset_metadata: Optional[DatasetMetadata] = None

    @field_validator("route_of_administration", mode="before")
    @classmethod
    def validate_route(cls, v: Optional[str]) -> Optional[str]:
        if v is None:
            return None
        _VALID_ROUTES = {
            "oral", "intravenous", "topical",
            "subcutaneous", "intramuscular",
            "sublingual", "inhalation",
        }
        normalized = v.lower().strip()
        if normalized not in _VALID_ROUTES:
            return None
        return normalized


class ResolutionError(Exception):
    """Raised when a drug cannot be resolved through any available path."""
    pass
