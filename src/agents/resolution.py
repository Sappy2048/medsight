"""
resolution.py — Agent 1: Drug Resolution
─────────────────────────────────────────
Resolves drug brand names from a ParsedPrescription into fully structured
ResolvedDrug objects, ready for Agent 2 (FDA Label Fetcher).

Entry point: resolve_prescription(parsed_prescription, http_client)

Input:  ParsedPrescription  — output of Agent 0 (LLM Prescription Parser)
Output: list[ResolvedDrug]  — one per drug in the prescription

Resolution pipeline (per drug):
    1. Query Indian Medicine Dataset (exact → fuzzy → raw fallback)
       via indian_drug_loader.extract_drug_data()
    2. Parse generics + dosages from composition waterfall
    3. Resolve each generic name → RxCUI via RxNorm API (concurrent)
    4. Assemble and validate ResolvedDrug via Pydantic firewall

Guardrails:
    - Dataset index loaded ONCE at module level in indian_drug_loader.py
    - RxNorm resolution is fully async and concurrent per drug
    - Shared httpx.AsyncClient injected — never instantiated here
    - All drugs in a prescription resolved concurrently via asyncio.gather
    - ResolutionError raised (never swallowed) on pipeline failure
"""

import asyncio
import logging
from typing import Optional, cast

import httpx

from src.schemas.resolution_schema import (
    DatasetMetadata,
    ResolvedDrug,
    ResolutionError,
)
from src.schemas.prescription_schema import ParsedDrug, ParsedPrescription
from src.services.indian_drug_loader import extract_drug_data
from src.services.rxnorm_client import get_rxcui

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — RxNorm Resolution
# ─────────────────────────────────────────────────────────────────────────────

async def _resolve_rxcui_list(
    generics: list[str],
    http_client: httpx.AsyncClient,
) -> list[str]:
    """
    Resolves a list of generic INN names to RxCUI strings concurrently.

    Each generic is dispatched to the RxNorm API in parallel via
    asyncio.gather. Failures on individual generics are logged and skipped
    rather than crashing the entire resolution — a partial RxCUI list is
    still valid and useful downstream.

    Args:
        generics:    Cleaned INN generic names from the composition waterfall.
                     e.g. ["Ambroxol", "Levosalbutamol", "Guaifenesin"]
        http_client: Shared httpx.AsyncClient — injected, never created here.

    Returns:
        List of resolved RxCUI strings. At least one must resolve or
        ResolutionError is raised.

    Raises:
        ResolutionError: If NO generics resolve to an RxCUI. A drug with
                         zero RxCUIs cannot be processed by Agent 2.
    """
    # return_exceptions=True means exceptions are returned as values in the
    # results list rather than immediately raised. Each item is therefore
    # Optional[str] | BaseException — we filter below.
    raw_results: list[str | None | BaseException] = await asyncio.gather(  # type: ignore[assignment]
        *[get_rxcui(generic, http_client) for generic in generics],
        return_exceptions=True,
    )

    rxcuis: list[str] = []
    for generic, result in zip(generics, raw_results):
        if isinstance(result, BaseException):
            logger.warning(
                "RxNorm lookup raised an exception for '%s': %s",
                generic, result,
            )
        elif result is None:
            logger.warning(
                "RxNorm returned no CUI for '%s'. "
                "Verify this is a valid INN generic name.",
                generic,
            )
        else:
            # result is narrowed to str here by the two guards above,
            # but Pylance cannot infer through gather's return type —
            # cast makes the assignment unambiguous.
            rxcuis.append(cast(str, result))

    if not rxcuis:
        raise ResolutionError(
            f"RxNorm could not resolve any generic in {generics} to an RxCUI. "
            "Ensure these are valid INN generic names and that the RxNorm API "
            "is reachable."
        )

    return rxcuis


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Single Drug Resolution Pipeline
# ─────────────────────────────────────────────────────────────────────────────

async def _resolve_single_drug(
    parsed_drug: ParsedDrug,
    http_client: httpx.AsyncClient,
) -> ResolvedDrug:
    """
    Full resolution pipeline for a single ParsedDrug.

    Steps:
        1. Query Indian Medicine Dataset via extract_drug_data()
        2. Receive generics, dosages, resolution metadata, and raw metadata dict
        3. Resolve generics → RxCUIs concurrently via RxNorm
        4. Construct DatasetMetadata from the raw metadata dict
        5. Assemble and return a validated ResolvedDrug

    Args:
        parsed_drug: One drug entry from ParsedPrescription.drugs.
                     Carries target_brand_name, prescribed_dose, route, frequency.
        http_client: Shared httpx.AsyncClient — injected, never created here.

    Returns:
        A fully validated ResolvedDrug.

    Raises:
        ResolutionError: Propagated from _resolve_rxcui_list if no RxCUI
                         can be resolved for any generic in this drug.
    """
    brand_name = parsed_drug.target_brand_name

    # ── Step 1 & 2: Dataset Lookup + Composition Waterfall ───────────────────
    generics, dosages, resolution_method, fuzzy_score, metadata_dict = (
        extract_drug_data(brand_name)
    )

    logger.info(
        "Dataset resolution for '%s': method=%s, generics=%s",
        brand_name, resolution_method, generics,
    )

    # ── Step 3: RxNorm Resolution ─────────────────────────────────────────────
    rxcui_list = await _resolve_rxcui_list(generics, http_client)

    # ── Step 4: Build DatasetMetadata ─────────────────────────────────────────
    metadata: Optional[DatasetMetadata] = (
        DatasetMetadata(**metadata_dict)
        if metadata_dict is not None
        else None
    )

    # ── Step 5: Assemble formulated_strength ──────────────────────────────────
    # dosages is a parallel list to generics. Join non-empty entries into a
    # single human-readable strength string for the ResolvedDrug schema.
    # e.g. ["30mg/5ml", "1mg/5ml", "50mg/5ml"] → "30mg/5ml + 1mg/5ml + 50mg/5ml"
    formulated_strength: Optional[str] = (
        " + ".join(d for d in dosages if d) or None
    )

    # ── Step 6: Extract route and frequency ───────────────────────────────────
    # ParsedDrug.route and .frequency are typed as Literal["oral", ...] etc.
    # Literal types are plain str subtypes — they have no .value attribute
    # (that only exists on enum members). Pass them directly; Pydantic's
    # validate_route field_validator on ResolvedDrug handles normalisation.
    route:     Optional[str] = parsed_drug.route
    frequency: Optional[str] = parsed_drug.frequency

    # ── Step 7: Pydantic Firewall — Assemble ResolvedDrug ────────────────────
    return ResolvedDrug(
        raw_prescription_input   = brand_name,
        generic_names            = generics,
        formulated_strength      = formulated_strength,
        rxcui_list               = rxcui_list,
        prescribed_dose          = parsed_drug.prescribed_dose,
        route_of_administration  = route,
        frequency                = frequency,
        resolution_method        = resolution_method,
        parsing_method           = "llm",
        fuzzy_score              = fuzzy_score,
        dataset_metadata         = metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Agent Entry Point
# ─────────────────────────────────────────────────────────────────────────────

async def resolve_prescription(
    parsed_prescription: ParsedPrescription,
    http_client: httpx.AsyncClient,
) -> list[ResolvedDrug]:
    """
    Agent 1 entry point.

    Resolves ALL drugs in a ParsedPrescription concurrently and returns
    a list of validated ResolvedDrug objects for Agent 2.

    Concurrency model:
        All drugs are resolved in parallel via asyncio.gather. For a typical
        prescription of 2-4 drugs, this means RxNorm API calls for all drugs
        are in-flight simultaneously rather than sequentially.

    Error handling:
        If ANY single drug fails to resolve (e.g. RxNorm total miss),
        ResolutionError is raised immediately. The pipeline does not produce
        partial results — a prescription with an unresolved drug cannot be
        safely analysed for interactions.

    Args:
        parsed_prescription: Output of Agent 0 (LLM Prescription Parser).
                             Contains a list of ParsedDrug objects.
        http_client:         Shared httpx.AsyncClient — injected, never
                             created here. Caller owns the lifecycle.

    Returns:
        list[ResolvedDrug] — one entry per drug in parsed_prescription.drugs,
        in the same order.

    Raises:
        ResolutionError: If parsed_prescription contains no drugs, or if
                         any single drug fails to resolve end-to-end.
    """
    if not parsed_prescription.drugs:
        raise ResolutionError(
            "ParsedPrescription contains no drugs to resolve. "
            "Ensure Agent 0 (Prescription Parser) produced at least one ParsedDrug."
        )

    logger.info(
        "Agent 1 — resolving %d drug(s): %s",
        len(parsed_prescription.drugs),
        [d.target_brand_name for d in parsed_prescription.drugs],
    )

    # Dispatch all drugs concurrently.
    # return_exceptions=True so one drug failure does not cancel others —
    # we inspect and surface failures in the loop below.
    raw_results: list[ResolvedDrug | BaseException] = await asyncio.gather(  # type: ignore[assignment]
        *[
            _resolve_single_drug(drug, http_client)
            for drug in parsed_prescription.drugs
        ],
        return_exceptions=True,
    )

    resolved: list[ResolvedDrug] = []
    for drug, result in zip(parsed_prescription.drugs, raw_results):
        if isinstance(result, BaseException):
            raise ResolutionError(
                f"Failed to resolve '{drug.target_brand_name}': {result}"
            ) from result
        # result is narrowed to ResolvedDrug by the guard above.
        resolved.append(cast(ResolvedDrug, result))

    logger.info(
        "Agent 1 — successfully resolved %d drug(s): %s",
        len(resolved),
        [r.generic_names for r in resolved],
    )

    return resolved
