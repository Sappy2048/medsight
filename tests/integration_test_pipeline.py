"""
integration_test_pipeline.py
─────────────────────────────
Full end-to-end integration test for ALL MedSight pipeline stages,
driven entirely through the LangGraph orchestration engine (run_medsight).

Stages covered:
  Stage 1 — Copilot Preflight   (prescription parsing + clarification routing)
  Stage 2 — Drug Resolution     (Indian-Medicine-Dataset + RxNorm API)
  Stage 3 — FDA Label Fetching  (DailyMed API — historical + current)
  Stage 4 — Interaction Extract (LLM structured JSON from label sections)
  Stage 5 — Temporal Diff       (past vs present warning change detection)
  Stage 6 — Patient Impact      (patient-specific risk synthesis)
  Stage 7 — Final Report        (integrity verification + clinical narrative)

Run:
    PYTHONPATH=. venv/bin/python tests/integration_test_pipeline.py
    PYTHONPATH=. venv/bin/python tests/integration_test_pipeline.py "Your prescription text"

No mocks. All network calls are real.
"""

import asyncio
import logging
import sys
import textwrap
from datetime import date
from typing import Optional, Any

from openai import AsyncOpenAI
from qdrant_client import QdrantClient

# ── Project imports ───────────────────────────────────────────────────────────
from src.config import TOGETHER_BASE_URL, TOGETHER_API_KEY, QDRANT_URL, QDRANT_API_KEY
from src.agents.graph import run_medsight, build_medsight_graph, MedSightState

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
_SEP_MAJOR = "═" * 80
_SEP_MINOR = "─" * 80
_SEP_THIN  = "·" * 76

SEVERITY_BADGE = {
    5: "🔴 CONTRAINDICATED",
    4: "🟠 AVOID",
    3: "🟡 USE CAUTION",
    2: "🔵 MONITOR CLOSELY",
    1: "⚪ MONITOR",
    0: "✅ NO INTERACTION",
}

CHANGE_BADGE = {
    "ADDED":       "🆕 ADDED",
    "REMOVED":     "🗑  REMOVED",
    "STRENGTHENED":"⬆️  STRENGTHENED",
    "WEAKENED":    "⬇️  WEAKENED",
    "UNCHANGED":   "➖ UNCHANGED",
}

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
def _info(msg: str): print(f"  ℹ️   {msg}")

def _wrap(text: str, width: int = 72, indent: str = "      ") -> str:
    return textwrap.indent(textwrap.fill(text.strip(), width=width), prefix=indent)


# ─────────────────────────────────────────────────────────────────────────────
# Stage display functions (operating on final_state dict)
# ─────────────────────────────────────────────────────────────────────────────

def display_stage1(state: dict):
    """Display Stage 1: Copilot Preflight / Prescription Parsing."""
    _section("STAGE 1 — Copilot Preflight (Prescription Parsing)")

    prescription = state.get("prescription")
    clarification = state.get("clarification_message")

    if clarification:
        _warn("Copilot triggered a clarification request — pipeline halted here.")
        print(f"\n  Clarification message:\n")
        print(textwrap.indent(clarification, "    "))
        print()
        return False  # signal: pipeline did not continue

    if prescription is None:
        _fail("Prescription object is None — parsing failed completely.")
        return False

    _ok(f"Parsed {len(prescription.drugs)} drug(s)  —  confidence: {prescription.extraction_confidence}")
    for i, drug in enumerate(prescription.drugs, 1):
        print(f"    Drug {i}: {drug.target_brand_name}  |  Dose: {drug.prescribed_dose}")
        print(f"            Route: {drug.route}  |  Freq: {drug.frequency}  |  Duration: {drug.duration_days}d")
    print(f"    Prescription date : {prescription.prescription_date}")
    print(f"    Patient age       : {prescription.patient_age}")
    print()
    return True


def display_stage2(state: dict):
    """Display Stage 2: Drug Resolution."""
    _section("STAGE 2 — Drug Resolution")

    resolved_drugs = state.get("resolved_drugs", [])
    if not resolved_drugs:
        _warn("No resolved drugs in state — resolution may have failed or returned empty.")
        return

    _ok(f"{len(resolved_drugs)} drug(s) resolved to canonical generics.")
    for resolved in resolved_drugs:
        print(f"\n  Drug: {resolved.raw_prescription_input}")
        print(f"    Resolution method : {resolved.resolution_method}  (score={resolved.fuzzy_score})")
        print(f"    Generic name(s)   : {', '.join(resolved.generic_names)}")
        print(f"    Formulated str.   : {resolved.formulated_strength}")
        print(f"    RxCUI(s)          : {', '.join(resolved.rxcui_list) or '(none resolved)'}")
        if resolved.dataset_metadata:
            print(f"    Manufacturer      : {resolved.dataset_metadata.manufacturer}")
    print()


def display_stage3(state: dict):
    """Display Stage 3: FDA Label History."""
    _section("STAGE 3 — FDA Label Fetching")

    label_history = state.get("label_history", {})
    if not label_history:
        _warn("No label history in state — fetch may have failed or produced no results.")
        return

    _ok(f"Labels fetched for {len(label_history)} generic(s): {', '.join(label_history.keys())}")

    for generic, (past_label, present_label) in label_history.items():
        print(f"\n  Drug: {generic}")
        print("  " + _SEP_THIN)

        if past_label:
            print(f"\n  ◀  HISTORICAL LABEL")
            print(f"     SPL ID         : {past_label.spl_id}")
            print(f"     Effective time : {past_label.effective_time}")
            _print_label_preview("boxed_warning",     past_label.sections.boxed_warning)
            _print_label_preview("drug_interactions", past_label.sections.drug_interactions)
        else:
            _warn(f"No historical label found for {generic}.")

        if present_label:
            print(f"\n  ▶  CURRENT LABEL")
            print(f"     SPL ID         : {present_label.spl_id}")
            print(f"     Effective time : {present_label.effective_time}")
            _print_label_preview("boxed_warning",     present_label.sections.boxed_warning)
            _print_label_preview("drug_interactions", present_label.sections.drug_interactions)
        else:
            _warn(f"No current label found for {generic}.")
    print()


def _print_label_preview(name: str, content: Optional[str], max_chars: int = 600):
    if not content:
        print(f"\n     [{name.upper()}]: (not present)")
        return
    preview = content[:max_chars]
    wrapped = textwrap.indent(textwrap.fill(preview, width=70), prefix="       ")
    print(f"\n     [{name.upper()}]:")
    print(wrapped)
    if len(content) > max_chars:
        print(f"       ... (+{len(content) - max_chars} chars truncated)")


def display_stage4(state: dict):
    """Display Stage 4: Extraction results per generic (from extraction_results in state)."""
    _section("STAGE 4 — Interaction Extraction (LLM)")

    extraction_results = state.get("extraction_results", {})
    label_history      = state.get("label_history", {})

    if not label_history:
        _warn("No label history in state — extraction was never attempted.")
        return

    if not extraction_results:
        _warn(
            f"label_history has {len(label_history)} generic(s) but extraction_results is empty. "
            "The extraction agent may have failed silently — check logs for WARNING/ERROR lines."
        )
        return

    _ok(f"Extraction completed for {len(extraction_results)} generic(s).")

    for generic, pair in extraction_results.items():
        past_ext    = pair.get("past")
        present_ext = pair.get("present")

        print(f"\n  Drug: {generic}")
        print("  " + _SEP_THIN)

        # ── Historical extraction ──────────────────────────────────────────────
        if past_ext:
            past_date = past_ext.version_date
            n = len(past_ext.interactions)
            if n:
                _ok(f"HISTORICAL ({past_date}) — {n} interaction(s) extracted")
                for idx, rec in enumerate(past_ext.interactions, 1):
                    badge = SEVERITY_BADGE.get(rec.severity_score, "❓ UNKNOWN")
                    print(f"\n    [{idx}] Target       : {rec.target_drug}")
                    print(f"         Severity     : {badge}  (score={rec.severity_score})")
                    print(f"         Sev. text    : \"{rec.severity_text}\"")
                    print(f"         Section      : {rec.section}")
                    print(f"         Recommend.   :")
                    print(_wrap(rec.recommendation_text[:300], width=68, indent="           "))
            else:
                _warn(f"HISTORICAL ({past_date}) — no named drug interactions found.")
        else:
            _warn(f"No historical extraction result for {generic}.")

        # ── Current extraction ─────────────────────────────────────────────────
        if present_ext:
            present_date = present_ext.version_date
            n = len(present_ext.interactions)
            if n:
                _ok(f"CURRENT     ({present_date}) — {n} interaction(s) extracted")
                for idx, rec in enumerate(present_ext.interactions, 1):
                    badge = SEVERITY_BADGE.get(rec.severity_score, "❓ UNKNOWN")
                    print(f"\n    [{idx}] Target       : {rec.target_drug}")
                    print(f"         Severity     : {badge}  (score={rec.severity_score})")
                    print(f"         Sev. text    : \"{rec.severity_text}\"")
                    print(f"         Section      : {rec.section}")
                    print(f"         Recommend.   :")
                    print(_wrap(rec.recommendation_text[:300], width=68, indent="           "))
            else:
                _warn(f"CURRENT     ({present_date}) — no named drug interactions found.")
        else:
            _warn(f"No current extraction result for {generic}.")

    print()


def display_stage5(state: dict):
    """Display Stage 5: Temporal diff results."""
    _section("STAGE 5 — Temporal Diff (Past vs Present Warning Changes)")

    diffs          = state.get("diffs", [])
    reasoning_list = state.get("reasoning", [])
    label_history  = state.get("label_history", {})
    extraction_results = state.get("extraction_results", {})

    # ── Diagnostics — help user understand why diffs may be empty ─────────────
    print(f"  Label history generics : {list(label_history.keys()) or '(none)'}")
    print(f"  Extraction results     : {list(extraction_results.keys()) or '(none)'}")
    print(f"  Diffs computed         : {len(diffs)}")

    if not diffs:
        if label_history and not extraction_results:
            _fail(
                "Diffs are empty AND extraction_results is empty — "
                "the extraction agent returned nothing. Check STAGE 4 warnings above."
            )
        elif extraction_results:
            total_past = sum(
                len(v["past"].interactions) if v.get("past") else 0
                for v in extraction_results.values()
            )
            total_present = sum(
                len(v["present"].interactions) if v.get("present") else 0
                for v in extraction_results.values()
            )
            if total_past == 0 and total_present == 0:
                _warn(
                    "Extraction ran but found 0 interactions in both past and present labels. "
                    "Temporal diff has no data to compare — all pairs would be UNCHANGED."
                )
            else:
                _warn("Diffs list is empty despite extraction finding interactions. Check temporal agent.")
        else:
            _warn("No diffs — label history and extraction both empty.")
        return

    significant = [d for d in diffs if d.is_clinically_significant]
    _ok(
        f"{len(diffs)} pair(s) evaluated — "
        f"🚨 {len(significant)} clinically significant  |  "
        f"➖ {len(diffs) - len(significant)} unchanged/minor."
    )

    for i, (diff, reasoning) in enumerate(zip(diffs, reasoning_list), 1):
        change_badge = CHANGE_BADGE.get(diff.change_type, f"❓ {diff.change_type}")
        sig_marker   = "🚨" if diff.is_clinically_significant else "  "

        print(f"\n  {sig_marker} [{i}] {diff.drug_pair}")
        print(f"       Change type     : {change_badge}")
        print(
            f"       Severity delta  : {diff.severity_delta:+d}  "
            f"(past={diff.past_severity_score} → present={diff.present_severity_score})"
        )
        print(f"       Past date       : {diff.past_version_date}")
        print(f"       Present date    : {diff.present_version_date}")

        if diff.data_unavailable:
            _warn("   Label history predates prescription date — data unavailable.")

        if diff.past_recommendation:
            print(f"\n       Past rec.:")
            print(_wrap(diff.past_recommendation[:250], width=68, indent="         "))
        if diff.present_recommendation:
            print(f"\n       Present rec.:")
            print(_wrap(diff.present_recommendation[:250], width=68, indent="         "))

        if reasoning and isinstance(reasoning, dict):
            clinical = reasoning.get("clinical_reasoning", "")
            if clinical:
                print(f"\n       Clinical reasoning:")
                print(_wrap(clinical[:400], width=68, indent="         "))
    print()


def display_stage6(state: dict):
    """Display Stage 6: Patient Impact Report."""
    _section("STAGE 6 — Patient Impact Analysis")

    impact = state.get("impact_report")
    if impact is None:
        _warn("No impact report in state — impact agent may have failed or been skipped.")
        return

    RISK_BADGE = {
        "CRITICAL": "🔴 CRITICAL",
        "HIGH":     "🟠 HIGH",
        "MODERATE": "🟡 MODERATE",
        "LOW":      "🔵 LOW",
        "NONE":     "✅ NONE",
    }

    print(f"\n  Overall risk level  : {RISK_BADGE.get(impact.overall_risk_level, impact.overall_risk_level)}")
    print(f"  Flagged pairs       : {impact.flagged_pairs_count} / {impact.total_pairs_evaluated}")
    print(f"  ICMR guideline used : {'Yes' if impact.icmr_guideline_used else 'No'}")
    print(f"  Prescription date   : {impact.prescription_date}")
    print(f"\n  Summary:")
    print(_wrap(impact.summary, width=74, indent="    "))
    print(f"\n  Recommended action:")
    print(_wrap(impact.recommended_action, width=74, indent="    "))

    if impact.alerts:
        _subsection(f"Drug Pair Alerts ({len(impact.alerts)})")
        for i, alert in enumerate(impact.alerts, 1):
            print(f"\n    [{i}] {alert.drug_pair}")
            print(f"         Change type  : {CHANGE_BADGE.get(alert.change_type, alert.change_type)}")
            print(f"         Delta        : {alert.severity_delta:+d}  "
                  f"(present score={alert.present_severity_score})")
            print(f"         Confidence   : {alert.confidence}")
            if alert.key_concern:
                print(f"         Key concern  : {alert.key_concern[:200]}")
            if alert.dose_context:
                print(f"         Dose context : {alert.dose_context[:200]}")
            if alert.icmr_context:
                print(f"         ICMR context : {alert.icmr_context[:200]}")
            print(f"         Reasoning:")
            print(_wrap(alert.clinical_reasoning[:300], width=70, indent="           "))
    print()


def display_stage7(state: dict):
    """Display Stage 7: Final Synthesized Report."""
    _section("STAGE 7 — Final Report (Synthesis + Integrity Verification)")

    report = state.get("final_report")
    if report is None:
        _warn("No final report in state — synthesizer may have failed.")
        return

    INT_BAR = "█" * int(report.integrity_score * 20) + "░" * (20 - int(report.integrity_score * 20))

    print(f"\n  Severity badge      : {report.severity_badge}")
    print(f"  Verified            : {'✅ Yes' if report.verified else '❌ No'}")
    print(f"  Integrity score     : {report.integrity_score:.2f}  [{INT_BAR}]")
    print(f"  Override applied    : {'Yes' if report.override_applied else 'No'}")
    print(f"  Generated at        : {report.generated_at}")

    if report.verification_notes:
        print(f"\n  Verification notes  :")
        for note in report.verification_notes:
            print(f"    • {note}")

    print(f"\n  ── Clinical Summary ──────────────────────────────────────────")
    print(_wrap(report.final_summary, width=74, indent="    "))

    print(f"\n  ── Recommended Action ───────────────────────────────────────")
    print(_wrap(report.final_recommended_action, width=74, indent="    "))
    print()


def display_errors(state: dict):
    """Display any pipeline errors accumulated in state."""
    errors = state.get("errors", [])
    if errors:
        _section("PIPELINE ERRORS ACCUMULATED")
        for e in errors:
            _fail(e)
        print()


# ─────────────────────────────────────────────────────────────────────────────
# Mock Qdrant for environments without a live Qdrant instance
# ─────────────────────────────────────────────────────────────────────────────

class _MockQdrantClient:
    """Minimal no-op Qdrant client for integration tests."""
    def search(self, *args, **kwargs): return []
    def upsert(self, *args, **kwargs): pass


def _make_qdrant_client() -> Any:
    if QDRANT_URL:
        try:
            return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
        except Exception as e:
            logger.warning(f"Could not connect to Qdrant ({e}) — using mock client.")
    return _MockQdrantClient()


# ─────────────────────────────────────────────────────────────────────────────
# Instrumented graph runner — captures intermediate state per node
# ─────────────────────────────────────────────────────────────────────────────

async def run_instrumented(
    raw_input: str,
    llm_client: AsyncOpenAI,
    qdrant_client: Any,
    db_pool: Any,
) -> dict:
    """
    Runs the LangGraph pipeline and streams state snapshots per node so that
    each stage can be displayed as it completes, with precise error attribution.
    
    Returns the final accumulated state dict.
    """
    app = build_medsight_graph(llm_client, qdrant_client, db_pool)

    initial_state = MedSightState(
        raw_input=raw_input,
        prescription=None,
        resolved_drugs=[],
        label_history={},
        extraction_results={},
        diffs=[],
        reasoning=[],
        impact_report=None,
        final_report=None,
        copilot_session=[],
        loop_count=0,
        awaiting_input=False,
        should_rerun=False,
        clarification_message=None,
        errors=[],
    )

    NODE_STAGE_MAP = {
        "copilot_preflight": "STAGE 1 — Copilot Preflight",
        "resolver":          "STAGE 2 — Drug Resolution",
        "label_fetcher":     "STAGE 3 — FDA Label Fetching",
        "temporal":          "STAGE 4+5 — Extraction & Temporal Diff",
        "impact":            "STAGE 6 — Patient Impact",
        "synthesizer":       "STAGE 7a — Report Synthesis",
        "copilot_overseer":  "STAGE 7b — Integrity Overseer",
        "persist":           "STAGE 8 — Persistence",
    }

    accumulated: dict = dict(initial_state)

    print()
    _info("Streaming LangGraph node execution…\n")

    try:
        async for chunk in app.astream(initial_state, stream_mode="updates"):
            for node_name, node_output in chunk.items():
                stage_label = NODE_STAGE_MAP.get(node_name, node_name)
                _ok(f"Node completed: {stage_label}")

                # Merge node output into accumulated state
                if isinstance(node_output, dict):
                    accumulated.update(node_output)

                # Early-exit signal: copilot requested clarification
                if node_name == "copilot_preflight" and accumulated.get("awaiting_input"):
                    _warn("Pipeline halted — awaiting clarification.")
                    return accumulated

    except Exception as e:
        active_stages = [
            NODE_STAGE_MAP.get(n, n)
            for n in NODE_STAGE_MAP
            if n not in [
                "copilot_preflight", "resolver", "label_fetcher",
                "temporal", "impact", "synthesizer",
                "copilot_overseer", "persist"
            ]
        ]
        _fail(f"Pipeline crashed with exception: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    return accumulated


# ─────────────────────────────────────────────────────────────────────────────
# Default test cases
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_TEST_CASES = [
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
# Main runner
# ─────────────────────────────────────────────────────────────────────────────

async def main():
    if not TOGETHER_API_KEY:
        print("ERROR: TOGETHER_API_KEY is not set in .env. Aborting.")
        sys.exit(1)

    # ── CLI input override ────────────────────────────────────────────────────
    if len(sys.argv) > 1:
        user_query = " ".join(sys.argv[1:])
        test_cases = [{"label": "User-provided query", "query": user_query}]
    else:
        test_cases = DEFAULT_TEST_CASES

    llm_client = AsyncOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=TOGETHER_API_KEY,
    )
    qdrant_client = _make_qdrant_client()

    for i, test_case in enumerate(test_cases, 1):
        _header(f"TEST CASE {i}/{len(test_cases)}: {test_case['label']}")
        print(f"  Input: \"{test_case['query']}\"\n")

        # ── Run the full instrumented graph ───────────────────────────────────
        state = await run_instrumented(
            raw_input=test_case["query"],
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            db_pool=None,
        )

        # ── Display each stage from state ─────────────────────────────────────
        pipeline_continued = display_stage1(state)

        if not pipeline_continued:
            print("  ↳ Pipeline halted at Stage 1.\n")
            continue

        display_stage2(state)
        display_stage3(state)
        display_stage4(state)
        display_stage5(state)
        display_stage6(state)
        display_stage7(state)
        display_errors(state)

    print(f"\n{_SEP_MAJOR}")
    print("  Integration test complete.")
    print(_SEP_MAJOR + "\n")


if __name__ == "__main__":
    asyncio.run(main())
