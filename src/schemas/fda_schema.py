from typing import List, Optional
from datetime import date
from pydantic import BaseModel, Field

class SPLVersion(BaseModel):
    """Represents version control and tracking metadata for a Structured Product Labeling (SPL)."""
    spl_id: str = Field(..., description="The unique identifier for this specific version of the SPL.")
    spl_set_id: str = Field(..., description="The stable identifier for the drug product set across versions.")
    spl_version: int = Field(..., description="The integer version number of this SPL.")
    effective_time: str = Field(..., description="The effective date of this version (human-readable format from DailyMed).")
    title: Optional[str] = Field(None, description="The title of the labeling document.")


class LabelSections(BaseModel):
    """Contains the specific text sections extracted from the FDA drug label.
    
    Note: openFDA returns text sections as lists of strings (paragraphs).
    """
    boxed_warning: Optional[str] = Field(None, description="Black box warning information.")
    contraindications: Optional[str] = Field(None, description="Situations where the drug should not be used.")
    warnings: Optional[str] = Field(None, description="General warnings and precautions.")
    drug_interactions: Optional[str] = Field(None, description="Information on how the drug interacts with other substances.")


class FDALabelVersion(BaseModel):
    """The complete structured payload returned by the FDA client containing product identifiers and label content."""
    spl_id: str = Field(..., description="The unique identifier for this specific version of the SPL.")
    spl_set_id: str = Field(..., description="The stable identifier for the drug product set.")
    effective_time: str = Field(..., description="The effective date of this version.")
    sections: LabelSections = Field(..., description="The parsed textual sections of the drug label.")