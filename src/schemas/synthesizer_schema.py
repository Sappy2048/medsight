from datetime import datetime, timezone
from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator

from src.schemas.impact_schema import PatientImpactReport

class MedSightFinalReport(BaseModel):
    """The final, verified clinical report produced by Agent 6."""
    
    # Embedded original report
    report: PatientImpactReport
    
    # Verification and integrity metadata
    verified: bool = Field(
        ...,
        description="True only if both deterministic and LLM verification passed cleanly."
    )
    integrity_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Calculated score based on countable violations (1.0 = perfect)."
    )
    severity_badge: Literal["CRITICAL", "HIGH", "MODERATE", "LOW", "NONE"] = Field(
        ...,
        description="Deterministic 1:1 mapping from report.overall_risk_level."
    )
    verification_notes: list[str] = Field(
        default_factory=list,
        description="Flat list of Phase 1 integrity violations and Phase 2 drift claims."
    )
    override_applied: bool = Field(
        default=False,
        description="True if the LLM patched either the summary or the recommended action."
    )
    
    # Final narrative outputs (either original or patched)
    final_summary: str = Field(
        ...,
        description="The clinical summary presented to the doctor."
    )
    final_recommended_action: str = Field(
        ...,
        description="The clinical action presented to the doctor."
    )
    
    generated_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO timestamp of report assembly."
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "verified": True,
                "integrity_score": 1.0,
                "severity_badge": "HIGH",
                "final_summary": "Risk of bleeding increased...",
                "final_recommended_action": "Monitor INR...",
            }
        }
    }
