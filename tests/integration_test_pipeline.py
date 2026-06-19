"""
integration_test_pipeline.py
─────────────────────────────
Live integration test for MedSight pipeline stages:
  Stage 1 — Prescription Parser    (Real LLM via Together API)
  Stage 2 — Drug Resolution        (Real Indian-Medicine-Dataset + RxNorm API)
  Stage 3 — FDA Label Fetching     (Real DailyMed API)
  Stage 4 — Interaction Extraction (Real LLM — structured JSON from label text)

Run:
    PYTHONPATH=. venv/bin/python tests/integration_test_pipeline.py

No mocks. All network calls are real.
"""

import asyncio
import logging
import sys
import textwrap
from datetime import date
from typing import Optional

import httpx
from openai import AsyncOpenAI

# ── Project imports ───────────────────────────────────────────────────────────
from src.config import TOGETHER_BASE_URL, TOGETHER_API_KEY
from src.agents.copilot import preflight_validate
from src.agents.resolution import resolve_prescription
from src.agents.extraction import extract_interactions
from src.services.fda_client import get_past_and_present_labels
from src.schemas.fda_schema import FDALabelVersion
from src.schemas.resolution_schema import ResolvedDrug
from src.schemas.diff_schema import ExtractionResult

# ── Logging — suppress httpx noise, keep medsight INFO ───────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(levelname)-8s %(name)s | %(message)s",
    stream=sys.stdout,
)
logging.getLogger("medsight").setLevel(logging.INFO)
logging.getLogger("integration_test").setLevel(logging.INFO)
logger = logging.getLogger("integration_test")

# ── Terminal helpers ──────────────────────────────────────────────────────────
_SEP_MAJOR = "═" * 72
_SEP_MINOR = "─" * 72
_SEP_THIN  = "·" * 68

SEVERITY_BADGE = {5: "🔴 CONTRAINDICATED", 4: "🟠 AVOID", 3: "🟡 USE CAUTION",
                  2: "🔵 MONITOR CLOSELY", 1: "⚪ MONITOR", 0: "✅ NO INTERACTION"}

def _header(title: str):
    print(f"\n{_SEP_MAJOR}")
    print(f"  {title}")
    print(_SEP_MAJOR)

def _section(title: str):
    print(f"\n{_SEP_MINOR}")
    print(f"  {title}")
    print(_SEP_MINOR)

def _subsection(title: str):
    print(f"\n  ▸ {title}")
    print("  " + _SEP_THIN)

def _ok(msg: str):   print(f"  ✅  {msg}")
def _warn(msg: str): print(f"  ⚠️   {msg}")
def _fail(msg: str): print(f"  ❌  {msg}")

def _print_label_section(name: str, content: Optional[str]):
    if not content:
        print(f"\n    [{name.upper()}]: (not present in this label version)")
        return
    preview = content[:800]
    wrapped = textwrap.indent(textwrap.fill(preview, width=68), prefix="      ")
    print(f"\n    [{name.upper()}]:")
    print(wrapped)
    if len(content) > 800:
        print(f"      ... (+{len(content) - 800} chars truncated)")

def _print_interaction_record(idx: int, rec):
    badge = SEVERITY_BADGE.get(rec.severity_score, "❓ UNKNOWN")
    print(f"\n    [{idx}] Target Drug : {rec.target_drug}")
    print(f"         Severity    : {badge}  (score={rec.severity_score})")
    print(f"         Sev. text   : \"{rec.severity_text}\"")
    print(f"         Section     : {rec.section}")
    rec_wrapped = textwrap.indent(
        textwrap.fill(rec.recommendation_text[:300], width=64), prefix="         "
    )
    print(f"         Recommend.  :\n{rec_wrapped}")

# ── Test queries ──────────────────────────────────────────────────────────────
TEST_CASES = [
    {
        "label": "Classic high-risk pair (Warfarin + Azithromycin, 2015)",
        "query": "Patient (62M) was prescribed Warfarin 5mg OD and Azithromycin 500mg OD for 5 days on 2015-05-12.",
    },
    {
        "label": "Missing date — copilot should request clarification",
        "query": "Patient on Metformin 500mg BID and Ibuprofen 400mg TID.",
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Stage runners
# ─────────────────────────────────────────────────────────────────────────────

async def run_stage1(query: str, llm_client: AsyncOpenAI):
    """Stage 1: Prescription Parsing via real LLM."""
    _section("STAGE 1 — Prescription Parser")
    print(f"  Input: \"{query}\"\n")

    parsed, clarification_msg = await preflight_validate(query, llm_client)

    if clarification_msg:
        _warn("Copilot triggered clarification request.")
        print(f"\n  Clarification message:\n")
        print(textwrap.indent(clarification_msg, "    "))
        print()
        return parsed, clarification_msg

    _ok(f"Parsed {len(parsed.drugs)} drug(s) — confidence: {parsed.extraction_confidence}")
    for i, drug in enumerate(parsed.drugs, 1):
        print(f"    Drug {i}: {drug.target_brand_name}  |  Dose: {drug.prescribed_dose}")
        print(f"            Route: {drug.route}  |  Freq: {drug.frequency}  |  Duration: {drug.duration_days}d")
    print(f"    Prescription date : {parsed.prescription_date}")
    print(f"    Patient age       : {parsed.patient_age}")
    print()
    return parsed, None


async def run_stage2(parsed, http_client: httpx.AsyncClient) -> list[ResolvedDrug]:
    """Stage 2: Drug Resolution via dataset + RxNorm API."""
    _section("STAGE 2 — Drug Resolution")

    if not parsed.drugs:
        _warn("No drugs to resolve (parser returned empty list).")
        return []

    resolved_list = await resolve_prescription(parsed, http_client)

    for resolved in resolved_list:
        _ok(f"{resolved.raw_prescription_input}")
        print(f"    Resolution method : {resolved.resolution_method}  (score={resolved.fuzzy_score})")
        print(f"    Generic name(s)   : {', '.join(resolved.generic_names)}")
        print(f"    Formulated str.   : {resolved.formulated_strength}")
        print(f"    RxCUI(s)          : {', '.join(resolved.rxcui_list) or '(none resolved)'}")
        if resolved.dataset_metadata:
            print(f"    Manufacturer      : {resolved.dataset_metadata.manufacturer}")
        print()

    return resolved_list


async def run_stage3(
    resolved_list: list[ResolvedDrug],
    parsed,
    http_client: httpx.AsyncClient,
) -> dict[str, tuple[Optional[FDALabelVersion], Optional[FDALabelVersion]]]:
    """Stage 3: FDA Label Fetching via DailyMed API. Returns label pairs keyed by generic."""
    _section("STAGE 3 — FDA Label Fetching")

    if not resolved_list:
        _warn("No resolved drugs — skipping FDA label fetch.")
        return {}

    raw_date = parsed.prescription_date
    if raw_date is None:
        prescription_date = "2024-01-01"
        _warn(f"No prescription date — defaulting to {prescription_date}")
    elif isinstance(raw_date, date):
        prescription_date = raw_date.isoformat()
    else:
        prescription_date = str(raw_date)

    print(f"  Prescription date for historical lookup: {prescription_date}\n")

    label_history: dict[str, tuple[Optional[FDALabelVersion], Optional[FDALabelVersion]]] = {}

    for resolved in resolved_list:
        primary_generic = resolved.generic_names[0]
        print(f"\n  Drug: {primary_generic}")
        print("  " + _SEP_THIN)

        try:
            past_label, present_label = await get_past_and_present_labels(
                primary_generic, prescription_date, http_client
            )
        except Exception as e:
            _fail(f"Failed to fetch labels for {primary_generic}: {e}")
            continue

        label_history[primary_generic] = (past_label, present_label)

        # ── PAST LABEL ──────────────────────────────────────────────────────
        if past_label:
            print(f"\n  ◀ HISTORICAL LABEL (at {prescription_date})")
            print(f"    SPL ID         : {past_label.spl_id}")
            print(f"    Effective time : {past_label.effective_time}")
            _print_label_section("boxed_warning",     past_label.sections.boxed_warning)
            _print_label_section("contraindications", past_label.sections.contraindications)
            _print_label_section("warnings",          past_label.sections.warnings)
            _print_label_section("drug_interactions", past_label.sections.drug_interactions)
        else:
            _warn(f"No past label found for {primary_generic} at {prescription_date}")

        # ── PRESENT LABEL ────────────────────────────────────────────────────
        if present_label:
            print(f"\n  ▶ CURRENT LABEL (latest)")
            print(f"    SPL ID         : {present_label.spl_id}")
            print(f"    Effective time : {present_label.effective_time}")
            _print_label_section("boxed_warning",     present_label.sections.boxed_warning)
            _print_label_section("contraindications", present_label.sections.contraindications)
            _print_label_section("warnings",          present_label.sections.warnings)
            _print_label_section("drug_interactions", present_label.sections.drug_interactions)
        else:
            _warn(f"No current label found for {primary_generic}")

        print()

    return label_history


async def run_stage4(
    label_history: dict[str, tuple[Optional[FDALabelVersion], Optional[FDALabelVersion]]],
    llm_client: AsyncOpenAI,
):
    """Stage 4: LLM-driven interaction extraction from past AND present label versions."""
    _section("STAGE 4 — Interaction Extraction (LLM)")

    if not label_history:
        _warn("No label data available — skipping extraction.")
        return

    for generic, (past_label, present_label) in label_history.items():
        print(f"\n  Drug: {generic}")
        print("  " + _SEP_THIN)

        # ── Extract from PAST label ──────────────────────────────────────────
        if past_label:
            _subsection(f"HISTORICAL  ({past_label.effective_time})")
            try:
                past_result: ExtractionResult = await extract_interactions(
                    past_label, generic, llm_client
                )
                if past_result.interactions:
                    _ok(f"{len(past_result.interactions)} interaction(s) extracted")
                    for idx, rec in enumerate(past_result.interactions, 1):
                        _print_interaction_record(idx, rec)
                else:
                    _warn("No named drug interactions found in historical label.")
            except Exception as e:
                _fail(f"Extraction failed for historical {generic}: {e}")
        else:
            _warn(f"No historical label for {generic} — skipping past extraction.")

        # ── Extract from PRESENT label ───────────────────────────────────────
        if present_label:
            _subsection(f"CURRENT     ({present_label.effective_time})")
            try:
                present_result: ExtractionResult = await extract_interactions(
                    present_label, generic, llm_client
                )
                if present_result.interactions:
                    _ok(f"{len(present_result.interactions)} interaction(s) extracted")
                    for idx, rec in enumerate(present_result.interactions, 1):
                        _print_interaction_record(idx, rec)
                else:
                    _warn("No named drug interactions found in current label.")
            except Exception as e:
                _fail(f"Extraction failed for current {generic}: {e}")
        else:
            _warn(f"No current label for {generic} — skipping present extraction.")

        print()


# ─────────────────────────────────────────────────────────────────────────────
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    if not TOGETHER_API_KEY:
        print("ERROR: TOGETHER_API_KEY is not set in .env. Aborting.")
        sys.exit(1)

    llm_client = AsyncOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=TOGETHER_API_KEY,
    )

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        for i, test_case in enumerate(TEST_CASES, 1):
            _header(f"TEST CASE {i}/{len(TEST_CASES)}: {test_case['label']}")

            # ── Stage 1: Parse ────────────────────────────────────────────
            parsed, clarification = await run_stage1(test_case["query"], llm_client)
            if clarification:
                print("  ↳ Pipeline halted at Stage 1 — clarification required.\n")
                continue

            # ── Stage 2: Resolve ──────────────────────────────────────────
            try:
                resolved_list = await run_stage2(parsed, http_client)
            except Exception as e:
                _fail(f"Stage 2 failed: {e}")
                continue

            # ── Stage 3: FDA Labels ───────────────────────────────────────
            try:
                label_history = await run_stage3(resolved_list, parsed, http_client)
            except Exception as e:
                _fail(f"Stage 3 failed: {e}")
                continue

            # ── Stage 4: Extraction ───────────────────────────────────────
            try:
                await run_stage4(label_history, llm_client)
            except Exception as e:
                _fail(f"Stage 4 failed: {e}")
                continue

    print(f"\n{_SEP_MAJOR}")
    print("  Integration test complete.")
    print(_SEP_MAJOR + "\n")


if __name__ == "__main__":
    asyncio.run(main())
