import json
import logging
from pydantic import ValidationError
from openai import AsyncOpenAI

from src.config import LLM_MODEL
from src.schemas.prescription_schema import ParsedPrescription, ParsedDrug

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
You are a clinical NLP extraction engine embedded in a drug safety system.

TASK: Parse a raw prescription or clinical query string and return a JSON object.

RULES (non-negotiable):
1. Extract ALL drugs mentioned. Return them in the "drugs" array.
   - CRITICAL: Include chronic medications, background maintenance therapies, 
     and drugs the patient is "stabilized on" or "currently taking" if they are 
     mentioned in the query text. These are active components of the patient's 
     medication profile and must not be filtered out as clinical history.

2. Brand Name Parsing & Dosage Stripping:
   - Strip ONLY numeric values and explicit strength units (e.g., mg, ml, mcg, %). 
   - Retain all non-numeric brand suffixes (e.g., "Duo", "Forte", "Plus", "XR", "SR", "LS") as they are critical to brand identity.
   - Parenthetical Guard: If a generic name or explanation is provided inside parentheses after a brand name—e.g., "BrandX (Ibuprofen)"—extract ONLY the preceding brand name ("BrandX"). Do not include the parentheses or their contents in the target_brand_name string.
   - Examples: "Augmentin 625 Duo" → "Augmentin Duo" | "Brand X (Ibuprofen 400mg)" → "Brand X".

3. Route Normalization & Form Inference:
   - Map EXACTLY to one of: oral | intravenous | topical | subcutaneous | intramuscular | sublingual | inhalation | unknown
   - Mappings: "po", "by mouth", "orally" → "oral" | "iv", "intravenously" → "intravenous"
   - Inference Rule: Infer route from administration form keywords if not explicitly stated:
     "drops", "gargles", "rinse", "cream", "gel", "ointment", "patch" → "topical"
     "inhaler", "nebulisation", "respules" → "inhalation"
     "injection", "infusion", "ampoule", "vial" → "intravenous" (default if site unspecified)

4. Frequency Normalization:
   - Map EXACTLY to one of: OD | BID | TID | QID | PRN | STAT | QHS | Q4H | Q6H | Q8H | Q12H | unknown
   - Mappings: 
     "once daily", "qd", "od", "1-0-0", "0-1-0", "0-0-1" → "OD"
     "twice daily", "bd", "bid", "1-0-1" → "BID"
     "three times", "tds", "tid", "1-1-1" → "TID"
     "four times", "qid" → "QID"
     "as needed", "prn", "sos" → "PRN"

5. Duration: Convert to integer days. "2 weeks" → 14. "5 days" → 5. Absent → null.

6. Prescription Date: If mentioned with an absolute date, return ISO format YYYY-MM-DD. 
   - CRITICAL: If relative terms like "today", "yesterday", or "now" are used, or if date elements are missing (e.g., "Oct 2023"), DO NOT guess a day, and DO NOT use placeholder characters like "XX" or "??". Set it to null.

7. Extraction Confidence:
   - "high": All extracted drugs have name + route + frequency clearly identified.
   - "medium": Most drugs have name + frequency, but route or dose is missing or inferred for one or more.
   - "low": Ambiguous clinical text, non-prescription text, or inputs with high uncertainty.

8. Prescribed Dose Formatting:
   - If a unit is explicitly stated in the input (mg, ml, mcg, units), preserve it exactly.
   - If no unit is stated, store the numeric value as a plain string without inferring a unit.
   - Examples: "Dolo 650mg" → "650mg" | "PCM 500" → "500" | "Syrup 5ml" → "5ml" | "Augmentin 875/125" → "875/125"

9. Non-Prescription / Guardrail Mode:
   - If the input contains no prescribable drug instructions (e.g., general clinical questions like "What is metformin used for?"), return an empty drugs array. Do NOT hallucinate drug entries.
   - Output exact format: {"drugs": [], "prescription_date": null, "patient_age": null, "raw_input": "<input>", "extraction_confidence": "low"}

OUTPUT FORMAT (strict JSON, no markdown, no explanation):
{
  "drugs": [
    {
      "target_brand_name": "...",
      "prescribed_dose": "..." or null,
      "route": "...",
      "frequency": "...",
      "duration_days": integer or null
    }
  ],
  "prescription_date": string (ISO format YYYY-MM-DD) or null,
  "patient_age": integer or null,
  "raw_input": "<copy the exact input here>",
  "extraction_confidence": "high" | "medium" | "low"
}

EXAMPLE 1 (Standard with Retained Suffix):
Input: "Tab Augmentin 625 Duo BD + Dolo 650 TDS for 5 days"
Output:
{
  "drugs": [
    {"target_brand_name": "Augmentin Duo", "prescribed_dose": "625", "route": "oral", "frequency": "BID", "duration_days": 5},
    {"target_brand_name": "Dolo", "prescribed_dose": "650", "route": "oral", "frequency": "TID", "duration_days": 5}
  ],
  "prescription_date": null,
  "patient_age": null,
  "raw_input": "Tab Augmentin 625 Duo BD + Dolo 650 TDS for 5 days",
  "extraction_confidence": "high"
}

EXAMPLE 2 (Incomplete data, Form Inference, and Strict Units):
Input: "PCM 500mg QID x 5d + Betadine gargles SOS"
Output:
{
  "drugs": [
    {"target_brand_name": "PCM", "prescribed_dose": "500mg", "route": "oral", "frequency": "QID", "duration_days": 5},
    {"target_brand_name": "Betadine", "prescribed_dose": null, "route": "topical", "frequency": "PRN", "duration_days": null}
  ],
  "prescription_date": null,
  "patient_age": null,
  "raw_input": "PCM 500mg QID x 5d + Betadine gargles SOS",
  "extraction_confidence": "medium"
}
""".strip()


# ─── The Parser Agent ─────────────────────────────────────────────────────────

class PrescriptionParsingAgent:
    """
    Agent 0 — Prescription Parser.
    Converts raw clinical text into a validated ParsedPrescription object.
    Uses Local LLM with JSON mode.
    Receives the shared AsyncOpenAI client via constructor injection.
    """

    # How many times to retry on ValidationError before raising
    _MAX_RETRIES: int = 2

    def __init__(self, llm_client: AsyncOpenAI) -> None:
        self._client = llm_client

    async def parse(self, raw_text: str) -> ParsedPrescription:
        """
        Main entry point. Returns a validated ParsedPrescription.
        Retries up to _MAX_RETRIES times on schema validation failure.
        """
        last_error: Exception | None = None
        error_context: str | None = None

        for attempt in range(1, self._MAX_RETRIES + 2):  # 1 initial + _MAX_RETRIES
            try:
                raw_json = await self._call_llm(raw_text, attempt, error_context)
                return self._validate(raw_json, raw_text)

            except ValidationError as e:
                last_error = e
                error_context = f"Pydantic ValidationError: {str(e.errors())}"
                logger.warning(
                    "PrescriptionParser attempt %d/%d — ValidationError parsed",
                    attempt, self._MAX_RETRIES + 1
                )

            except json.JSONDecodeError as e:
                last_error = e
                error_context = f"JSONDecodeError: {str(e)}"
                logger.warning(
                    "PrescriptionParser attempt %d/%d — JSONDecodeError",
                    attempt, self._MAX_RETRIES + 1
                )

        raise RuntimeError(
            f"PrescriptionParser failed after {self._MAX_RETRIES + 1} attempts "
            f"on input: '{raw_text}'. Last error: {last_error}"
        ) from last_error

    async def _call_llm(self, raw_text: str, attempt: int, error_context: str | None = None) -> dict:
        """
        Sends the raw text to Local LLM. On retry attempts, appends the exact
        Pydantic error tracking data to instruct the model what to fix.
        """
        user_message = f"Extract from this prescription: {raw_text}"

        if attempt > 1 and error_context:
            user_message += (
                f"\n\nCRITICAL: Your previous response failed structural validation with this error:\n{error_context}\n"
                "Please fix this mistake. If a clear, fully formatted date cannot be extracted, "
                "set 'prescription_date' directly to null. Do not under any circumstances output placeholders like 'XX' or '??'."
            )

        response = await self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
            max_tokens=1024,
        )

        content = response.choices[0].message.content
        return json.loads(content or '{}')

    @staticmethod
    def _validate(raw_json: dict, original_input: str) -> ParsedPrescription:
        """
        Pydantic firewall validation layer.
        """
        raw_json["raw_input"] = original_input
        return ParsedPrescription(**raw_json)