"""
copilot.py — Patient Copilot Agent (Agent 0)

Role:
    Three-mode LLM agent. Entry gate, quality overseer, and conversational interface.
    
Modes:
    1. preflight_validate(): Extracts ParsedPrescription from raw text.
    2. oversee_report(): Deterministic logic to decide on re-runs based on integrity.
    3. answer_question(): Stateful Q&A grounded on the final clinical report.
"""

import json
import logging
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional, cast, Iterable
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletionMessageParam

from src.config import LLM_MODEL
from src.schemas.prescription_schema import ParsedPrescription, ParsedDrug
from src.schemas.synthesizer_schema import MedSightFinalReport
from src.agents.prescription_parser import PrescriptionParsingAgent

logger = logging.getLogger(__name__)

# ─── Mode 1: Pre-flight Validation ──────────────────────────────────────────

async def preflight_validate(
    raw_input: str,
    llm_client: AsyncOpenAI,
) -> ParsedPrescription:
    """
    Mode 1: Entry gate.
    Wraps the existing PrescriptionParsingAgent to extract structured data.
    Handles errors by returning a low-confidence empty prescription.
    """
    parser = PrescriptionParsingAgent(llm_client)
    try:
        # PrescriptionParsingAgent internally handles normalization and extraction
        # It also preserves raw_input
        return await parser.parse(raw_input)
    except Exception as e:
        logger.error(f"Copilot preflight_validate failed: {e}")
        return ParsedPrescription(
            drugs=[],
            prescription_date=None,
            patient_age=None,
            raw_input=raw_input,
            extraction_confidence="low"
        )

# ─── Mode 2: Overseer Logic ──────────────────────────────────────────────────

async def oversee_report(
    report: MedSightFinalReport,
    llm_client: AsyncOpenAI, # Included for API consistency per brief
) -> Tuple[bool, str]:
    """
    Mode 2: Quality overseer.
    Deterministic decision logic for pipeline re-runs.
    Returns: (should_rerun: bool, explanation_for_physician: str)
    """
    # Deterministic logic per brief
    should_rerun = False
    
    # We don't check loop_count here; graph.py enforces the cap.
    if not report.verified and report.integrity_score < 0.7:
        should_rerun = True
    
    # Rerave should_rerun if override was applied or score is high enough
    if report.verified or report.integrity_score >= 0.7 or report.override_applied:
        should_rerun = False

    # Explanation generation
    if not should_rerun:
        if report.verified:
            explanation = "Report verified. Proceeding to delivery."
        else:
            # Degraded but accepted
            notes = "; ".join(report.verification_notes) if report.verification_notes else "None"
            explanation = f"Report delivered with minor integrity flags: {notes}"
    else:
        explanation = "Critical integrity failure detected. Initiating automated pipeline re-run."

    return should_rerun, explanation

# ─── Mode 3: Grounded Q&A ────────────────────────────────────────────────────

_QA_SYSTEM_PROMPT = """\
You are a senior clinical pharmacist assisting a physician. 
Your job is to answer questions about the generated MedSight drug safety report.

STRICT RULES:
1. Answer ONLY from data present in the report JSON provided below. 
2. If the question cannot be answered from the report, respond exactly: 
   "I can only answer questions based on the generated MedSight report. Please consult clinical references for broader guidance."
3. Address the physician directly in a professional, concise tone.
4. NEVER suggest alternative drugs not present in the report.
5. NEVER speculate on clinical outcomes or mechanisms not explicitly stated in the report.

REPORT DATA (JSON):
{report_json}
"""

async def answer_question(
    question: str,
    report: MedSightFinalReport,
    history: List[Dict[str, str]],
    llm_client: AsyncOpenAI,
) -> str:
    """
    Mode 3: Conversational interface.
    Multi-turn Q&A grounded strictly on the MedSightFinalReport.
    """
    try:
        # Inject report into system prompt
        system_prompt = _QA_SYSTEM_PROMPT.format(
            report_json=report.model_dump_json(indent=2)
        )
        
        # Build message history: [system] + history + [user]
        messages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": system_prompt}
        ]
        messages.extend(cast(List[ChatCompletionMessageParam], history))
        messages.append({"role": "user", "content": question})
        
        response = await llm_client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages,
            temperature=0.2,
            max_tokens=512,
        )
        
        return response.choices[0].message.content or "No response from AI assistant."
        
    except Exception as e:
        logger.error(f"Copilot answer_question failed: {e}")
        return "Unable to process your question at this time. Please review the report directly."
