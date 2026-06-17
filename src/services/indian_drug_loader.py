"""
indian_drug_loader.py
─────────────────────
Agent 1 — Indian Drug Dataset Extraction Engine

Responsibilities:
    1. Load the junioralive/Indian-Medicine-Dataset once at module import.
    2. Resolve an Indian brand name → matched dataset row (exact or fuzzy).
    3. Execute the composition waterfall to extract generic names + dosages.
    4. Return components directly for ResolvedDrug assembly in resolution.py.

Dataset: 253k+ rows of Indian medications.
Source:  https://huggingface.co/datasets/junioralive/Indian-Medicine-Dataset

Column Schema:
    name                — Brand name (e.g. "Augmentin 625 Duo Tablet")
    short_composition1  — Always populated. One ingredient + dosage.
    short_composition2  — Frequently populated. Second ingredient + dosage.
    salt_composition    — Sparse (~3%). Full FDC string when present. Source of Truth.
    manufacturer_name   — Manufacturer name.
    type                — Medicine type (e.g. "allopathy").
    pack_size_label     — Pack size description.
    medicine_desc       — Medicine description for downstream reporting.
    side_effects        — Static side effects for downstream reporting.

⚠️  CRITICAL GUARDRAIL:
    The `drug_interactions` column is PERMANENTLY EXCLUDED from all operations.
    MedSight computes interactions via live FDA API temporal diffs (Agent 3).
    Reading static CSV interaction data causes Context Poisoning downstream.
"""

import logging
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — Dataset Schema Definition
# ─────────────────────────────────────────────────────────────────────────────

# Maps internal keys → actual CSV column names.
# If the upstream dataset schema changes, only update this dict.
_COLUMNS = {
    "name":             "name",
    "salt_composition": "salt_composition",
    "short_comp1":      "short_composition1",
    "short_comp2":      "short_composition2",
    "manufacturer":     "manufacturer_name",
    "type":             "type",
    "pack_size":        "pack_size_label",
    "medicine_desc":    "medicine_desc",
    "side_effects":     "side_effects",
    # ☠️  "drug_interactions" is intentionally absent — see module docstring.
}

# Columns that MUST exist in the CSV. Optional enrichment columns
# (medicine_desc, side_effects, salt_composition, short_composition2)
# are tolerated gracefully if absent.
_REQUIRED_COLUMNS = {
    _COLUMNS["name"],
    _COLUMNS["short_comp1"],
}


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — Composition Parsing (Pure Functions, No I/O)
# ─────────────────────────────────────────────────────────────────────────────

# Matches dosage tokens within a single component string.
# Handles: 500mg | 30mg/5ml | 0.5% w/v | 2.5% w/w | 1mg/5ml | 10 IU
_DOSAGE_RE = re.compile(
    r"""
    \d+(?:\.\d+)?                   # integer or decimal: 500, 30, 0.5
    \s*
    (?:mg|mcg|ml|g|iu|units?)       # standard pharmaceutical unit
    (?:                             # optional ratio denominator: /5ml, /ml
        \s*/\s*
        \d*\s*(?:ml|g)?
    )?
    (?:\s*w/[vw])?                  # optional w/v or w/w concentration notation
    |
    \d+(?:\.\d+)?                   # bare percentage with optional w/v or w/w
    \s*%
    (?:\s*w/[vw])?
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Removes bracketed parenthetical residue after dosage stripping.
# e.g. "Amoxycillin (as Trihydrate)" → "Amoxycillin"
_BRACKET_RESIDUE_RE = re.compile(r"\s*\(.*?\)\s*|\s*\[.*?\]\s*")


def _parse_single_component(raw: str) -> tuple[str, str]:
    """
    Parses one raw component string into (generic_name, dosage).

    Args:
        raw: e.g. "Amoxycillin (500mg)" or "Levosalbutamol (1mg/5ml)"

    Returns:
        ("Amoxycillin", "500mg") or ("Levosalbutamol", "1mg/5ml")
        If no dosage found, dosage is returned as an empty string "".

    Pure function. No I/O. No side effects.
    """
    raw = raw.strip()

    # Extract the first dosage token found
    dosage_match = _DOSAGE_RE.search(raw)
    dosage = dosage_match.group(0).strip() if dosage_match else ""

    # Strip dosage and surrounding brackets to isolate the generic name
    name_only = _DOSAGE_RE.sub("", raw)
    name_only = _BRACKET_RESIDUE_RE.sub(" ", name_only)
    name_only = name_only.strip().rstrip(",;.")

    return name_only, dosage


def _extract_from_composition_string(
    composition: str,
) -> tuple[list[str], list[str]]:
    """
    Splits a full composition string and parses each component into
    parallel (generics, dosages) lists.

    Args:
        composition: e.g.
            "Ambroxol (30mg/5ml) + Levosalbutamol (1mg/5ml) + Guaifenesin (50mg/5ml)"

    Returns:
        (
            ["Ambroxol", "Levosalbutamol", "Guaifenesin"],
            ["30mg/5ml", "1mg/5ml", "50mg/5ml"]
        )

        Both lists are parallel — dosages[i] corresponds to generics[i].
        If a component has no parseable dosage, dosages[i] is "".

    Pure function. No I/O. No side effects.
    """
    if not isinstance(composition, str) or not composition.strip():
        return [], []

    composition = composition.strip()
    if composition.lower() in ("nan", "none", ""):
        return [], []

    # Split on + delimiter — standard separator in both salt_composition
    # and short_composition columns
    raw_components = [c.strip() for c in composition.split("+") if c.strip()]

    generics: list[str] = []
    dosages:  list[str] = []

    for component in raw_components:
        name, dosage = _parse_single_component(component)
        if name:
            generics.append(name)
            dosages.append(dosage)

    return generics, dosages


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — Dataset Index (Module-Level Singleton)
# ─────────────────────────────────────────────────────────────────────────────

_LOOKUP: dict[str, dict] = {}   # normalized_brand_name → row dict
_CORPUS: list[str]       = []   # flat key list for RapidFuzz corpus


def _is_empty(value: object) -> bool:
    """
    Returns True if a value from a Pandas dtype=str read is meaningfully empty.
    Guards against NaN-as-string artifacts produced by pd.read_csv(dtype=str).
    """
    if value is None:
        return True
    return str(value).strip().lower() in ("nan", "none", "")


def _normalize(text: str) -> str:
    """
    Lowercase + strip. Applied uniformly to both index keys and query strings
    to ensure consistent matching regardless of input casing.
    """
    return text.strip().lower()


def _load_dataset(csv_path: Path) -> None:
    """
    Loads the CSV into the module-level _LOOKUP index and _CORPUS list.
    Called ONCE at module import time. Never called per-request.

    Fails loudly (raises) on:
        - Missing CSV file
        - Missing required columns

    Excluded columns:
        drug_interactions — dropped at load time (context poisoning guardrail).
        Its absence from _LOOKUP is structural, not just conventional.
    """
    global _LOOKUP, _CORPUS

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Indian Medicine Dataset not found at: {csv_path}\n"
            "Download from: "
            "https://huggingface.co/datasets/junioralive/Indian-Medicine-Dataset"
        )

    # Read all columns as strings to prevent Pandas type-inference issues
    # with mixed-type columns (e.g. numeric pack sizes, sparse NaN fields)
    df = pd.read_csv(csv_path, dtype=str, low_memory=False)

    # ── Validate required columns ─────────────────────────────────────────────
    missing = _REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}\n"
            f"Columns found: {sorted(df.columns.tolist())}"
        )

    # ── Permanently drop drug_interactions if present ─────────────────────────
    # This is a hard guardrail. Even if someone adds a .get("drug_interactions")
    # call downstream, they will receive None — the data never enters _LOOKUP.
    if "drug_interactions" in df.columns:
        df = df.drop(columns=["drug_interactions"])
        logger.info(
            "drug_interactions column detected and permanently excluded "
            "(context poisoning prevention)."
        )

    # ── Build the in-memory index ─────────────────────────────────────────────
    name_col = _COLUMNS["name"]
    indexed  = 0

    for _, row in df.iterrows():
        raw_name = row.get(name_col)
        if _is_empty(raw_name):
            continue

        key = _normalize(str(raw_name))

        # Store only the columns defined in _COLUMNS.
        # Optional columns fall back to "" if absent from this specific CSV version.
        _LOOKUP[key] = {
            "name":          str(raw_name).strip(),
            "salt_comp":     row.get(_COLUMNS["salt_composition"], ""),
            "short_comp1":   row.get(_COLUMNS["short_comp1"],      ""),
            "short_comp2":   row.get(_COLUMNS["short_comp2"],      ""),
            "manufacturer":  row.get(_COLUMNS["manufacturer"],     ""),
            "type":          row.get(_COLUMNS["type"],             ""),
            "pack_size":     row.get(_COLUMNS["pack_size"],        ""),
            "medicine_desc": row.get(_COLUMNS["medicine_desc"],    ""),
            "side_effects":  row.get(_COLUMNS["side_effects"],     ""),
        }
        indexed += 1

    _CORPUS = list(_LOOKUP.keys())
    logger.info(
        "Indian Medicine Dataset loaded: %d / %d rows indexed.",
        indexed,
        len(df),
    )


# ── Load at module import time ────────────────────────────────────────────────
_load_dataset(Path("data/updated_indian_medicine_data.csv"))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — Composition Waterfall
# ─────────────────────────────────────────────────────────────────────────────

def _run_composition_waterfall(row: dict) -> str:
    """
    Executes the Priority 1 → Priority 2 extraction waterfall to obtain
    the best available raw composition string for a matched dataset row.

    Priority 1 — salt_composition (Source of Truth for FDCs):
        Sparse (~3% populated), but when present it contains ALL active
        ingredients for Fixed-Dose Combinations — including 3rd and 4th
        components that are permanently truncated in the short_composition
        columns. Always prefer this.

    Priority 2 — short_composition1 + short_composition2 (Fallback):
        Always populated, but hard-capped at two components by the dataset
        designers. FDCs with 3+ ingredients are silently truncated here.
        Use only when Priority 1 is unavailable.

    Returns:
        Best available raw composition string, ready for parsing.
        Returns "" if all sources are empty — never returns None.
    """
    # Priority 1: salt_composition — FDC-safe source of truth
    salt = row.get("salt_comp", "")
    if not _is_empty(salt):
        logger.debug("Waterfall: resolved via salt_composition (FDC-safe).")
        return str(salt).strip()

    # Priority 2: Concatenate short_composition columns
    comp1 = row.get("short_comp1", "")
    comp2 = row.get("short_comp2", "")

    parts = [
        str(c).strip()
        for c in (comp1, comp2)
        if not _is_empty(c)
    ]

    if parts:
        logger.debug(
            "Waterfall: resolved via short_composition (%d column(s)). "
            "⚠️  FDC truncation risk if drug has >2 active ingredients.",
            len(parts),
        )
        # Join with + so _extract_from_composition_string splits correctly
        return " + ".join(parts)

    logger.warning(
        "Waterfall exhausted — no composition data found in row for '%s'.",
        row.get("name", "unknown"),
    )
    return ""


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — Public Interface
# ─────────────────────────────────────────────────────────────────────────────

def extract_drug_data(brand_name: str) -> tuple[
    list[str],          # generic_names
    list[str],          # dosages_extracted (parallel to generic_names)
    str,                # resolution_method
    Optional[float],    # fuzzy_score
    Optional[dict],     # raw metadata dict (caller builds DatasetMetadata)
]:
    """
    Main public entry point for the Indian Drug Dataset Extraction Engine.

    Takes an Indian brand name, resolves it against the in-memory index,
    runs the composition waterfall, and returns structured components
    for ResolvedDrug assembly in resolution.py.

    Resolution path:
        1. Exact match  — O(1) normalized dict lookup
        2. Fuzzy match  — RapidFuzz WRatio, score_cutoff=85
        3. Raw fallback — brand name treated as a bare generic name

    Args:
        brand_name: Indian brand name from the parsed prescription.
                    e.g. "Ascoril LS Syrup", "Augmentin 625", "Dolo 650"

    Returns:
        (generic_names, dosages_extracted, resolution_method, fuzzy_score, metadata_dict)

        generic_names     — Cleaned INN names, dosage-stripped, ready for RxNorm.
        dosages_extracted — Parallel list. dosages[i] belongs to generics[i].
        resolution_method — "exact_match" | "fuzzy_match" | "raw_fallback"
        fuzzy_score       — RapidFuzz WRatio score, or None if not fuzzy match.
        metadata_dict     — Raw dict with manufacturer, type, etc. None on raw_fallback.
                            Caller (resolution.py) constructs DatasetMetadata from this.

    Never raises on a no-match — returns a raw_fallback result so the pipeline
    can continue for known generics typed directly (e.g. "Warfarin", "Metformin").
    """
    key = _normalize(brand_name)

    # ── Step 1: Exact Match ───────────────────────────────────────────────────
    row          = _LOOKUP.get(key)
    method       = "exact_match"
    fuzzy_score: Optional[float] = None

    # ── Step 2: Fuzzy Match ───────────────────────────────────────────────────
    if row is None and _CORPUS:
        result = process.extractOne(
            key,
            _CORPUS,
            scorer=fuzz.WRatio,
            score_cutoff=85,        # Reject below 85 — prevents false positives
        )
        if result:
            matched_key, score, _ = result
            row         = _LOOKUP[matched_key]
            method      = "fuzzy_match"
            fuzzy_score = float(score)
            logger.info(
                "Fuzzy match: '%s' → '%s' (score=%.1f)",
                brand_name, row["name"], score,
            )

    # ── Step 3: Raw Fallback ──────────────────────────────────────────────────
    if row is None:
        logger.warning(
            "No dataset match for '%s'. Treating as raw generic (e.g. Warfarin, Metformin).",
            brand_name,
        )
        # _parse_single_component handles the no-dosage case cleanly
        name, dosage = _parse_single_component(brand_name)
        return (
            [name] if name else [brand_name],
            [dosage],
            "raw_fallback",
            None,
            None,
        )

    # ── Step 4: Composition Waterfall ─────────────────────────────────────────
    composition_raw          = _run_composition_waterfall(row)
    generics, dosages        = _extract_from_composition_string(composition_raw)

    # Guard: if parsing returned nothing, fall back to brand name as generic
    if not generics:
        logger.warning(
            "Composition parsing returned no generics for '%s' "
            "(raw composition: '%s'). Using brand name as fallback generic.",
            brand_name, composition_raw,
        )
        generics = [row["name"]]
        dosages  = [""]

    # ── Step 5: Build Raw Metadata Dict ──────────────────────────────────────
    # drug_interactions is never present in _LOOKUP — excluded at load time.
    # Caller (resolution.py) wraps this in DatasetMetadata().
    metadata_dict = {
        "manufacturer":        row["manufacturer"]  if not _is_empty(row["manufacturer"])  else None,
        "medicine_type":       row["type"]           if not _is_empty(row["type"])           else None,
        "pack_size":           row["pack_size"]      if not _is_empty(row["pack_size"])      else None,
        "medicine_desc":       row["medicine_desc"]  if not _is_empty(row["medicine_desc"])  else None,
        "static_side_effects": row["side_effects"]   if not _is_empty(row["side_effects"])   else None,
    }

    return generics, dosages, method, fuzzy_score, metadata_dict
