import json
import logging
from pydantic import ValidationError
from openai import AsyncOpenAI

from src.schemas.prescription_schema import ParsedPrescription, ParsedDrug

logger = logging.getLogger(__name__)


_SYSTEM_PROMPT = """
You are a clinical NLP extraction engine embedded in a drug safety system.

TASK: Parse a raw prescription or clinical query string and return a JSON object.

RULES (non-negotiable):
1. Extract ALL drugs mentioned. Return them in the "drugs" array.
2. Strip ALL dosage values from the brand name. "Dolo 650" → target_brand_name = "Dolo".
3. Route normalization — map EXACTLY to one of:
   oral | intravenous | topical | subcutaneous | intramuscular | sublingual | inhalation | unknown
   Mappings: "po", "by mouth", "orally" → "oral" | "iv", "intravenously" → "intravenous"
4. Frequency normalization — map EXACTLY to one of:
   OD | BID | TID | QID | PRN | STAT | QHS | Q4H | Q6H | Q8H | Q12H | unknown
   Mappings: "once daily","qd","od" → "OD" | "twice daily","bd","bid" → "BID" |
   "three times","tds","tid" → "TID" | "four times","qid" → "QID" | "as needed","prn" → "PRN"
5. Duration: convert to integer days. "2 weeks" → 14. "5 days" → 5. Absent → null.
6. If prescription_date is mentioned, return ISO format YYYY-MM-DD. Absent → null.
7. Set extraction_confidence: "high" if all drugs have name+route+frequency, "low" if ambiguous.

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
  "prescription_date": "YYYY-MM-DD" or null,
  "patient_age": integer or null,
  "raw_input": "<copy the exact input here>",
  "extraction_confidence": "high" | "medium" | "low"
}

EXAMPLE 1 (Standard):
Input: "Tab Augmentin 625 BD + Dolo 650 TDS for 5 days"
Output:
{
  "drugs": [
    {"target_brand_name": "Augmentin", "prescribed_dose": "625mg", "route": "oral", "frequency": "BID", "duration_days": 5},
    {"target_brand_name": "Dolo", "prescribed_dose": "650mg", "route": "oral", "frequency": "TID", "duration_days": 5}
  ],
  ...
}

EXAMPLE 2 (Incomplete durations & SOS/PRN mapping):
Input: "PCM 500 QID x 5d + Amox 250 TDS + Betadine gargles SOS"
Output:
{
  "drugs": [
    {"target_brand_name": "PCM", "prescribed_dose": "500", "route": "oral", "frequency": "QID", "duration_days": 5},
    {"target_brand_name": "Amox", "prescribed_dose": "250", "route": "oral", "frequency": "TID", "duration_days": null},
    {"target_brand_name": "Betadine", "prescribed_dose": null, "route": "topical", "frequency": "PRN", "duration_days": null}
  ],
  ...
}
""".strip()


# ─── The Parser Agent ─────────────────────────────────────────────────────────

class PrescriptionParsingAgent:
    """
    Agent 0 — Prescription Parser.
    Converts raw clinical text into a validated ParsedPrescription object.
    Uses Local LLM with JSON mode.
    Receives the shared AsyncOpenAI client via constructor injection (never instantiates its own).
    """

    # How many times to retry on ValidationError before raising
    _MAX_RETRIES: int = 2

    def __init__(self, llm_client: AsyncOpenAI) -> None:
        # Shared client — injected, never created internally (guardrail compliance)
        self._client = llm_client

    async def parse(self, raw_text: str) -> ParsedPrescription:
        """
        Main entry point. Returns a validated ParsedPrescription.
        Retries up to _MAX_RETRIES times on schema validation failure.
        Raises RuntimeError if all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, self._MAX_RETRIES + 2):  # +2: 1 initial + MAX_RETRIES
            try:
                raw_json = await self._call_llm(raw_text, attempt)
                return self._validate(raw_json, raw_text)

            except ValidationError as e:
                # LLM returned valid JSON but wrong schema — retry with error context
                last_error = e
                logger.warning(
                    "PrescriptionParser attempt %d/%d — ValidationError: %s",
                    attempt, self._MAX_RETRIES + 1, e.errors()
                )

            except json.JSONDecodeError as e:
                # LLM returned non-JSON — retry
                last_error = e
                logger.warning(
                    "PrescriptionParser attempt %d/%d — JSONDecodeError: %s",
                    attempt, self._MAX_RETRIES + 1, str(e)
                )

        raise RuntimeError(
            f"PrescriptionParser failed after {self._MAX_RETRIES + 1} attempts "
            f"on input: '{raw_text}'. Last error: {last_error}"
        ) from last_error

    async def _call_llm(self, raw_text: str, attempt: int) -> dict:
        """
        Sends the raw text to Local LLM. On retry attempts, appends the schema
        as an additional reminder to reduce repeated hallucination patterns.
        """
        user_message = f"Extract from this prescription: {raw_text}"

        # On retries, add reinforcement so the LLM doesn't repeat the same mistake
        if attempt > 1:
            user_message += (
                "\n\nIMPORTANT: Your previous response had a schema error. "
                "Ensure 'drugs' is a list, 'route' and 'frequency' match "
                "the exact allowed values, and all required fields are present."
            )

        response = await self._client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_message},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,   # Deterministic — we want extraction, not creativity
            max_tokens=1024,
        )

        content = response.choices[0].message.content
        return json.loads(content or '{}')

    @staticmethod
    def _validate(raw_json: dict, original_input: str) -> ParsedPrescription:
        """
        Pydantic firewall. Also ensures raw_input is always preserved
        from the original caller, not from whatever the LLM echoed back.
        """
        # Enforce raw_input from the actual caller — don't trust the LLM's echo
        raw_json["raw_input"] = original_input
        return ParsedPrescription(**raw_json)
"] = original_input
        return ParsedPrescription(**raw_json)
