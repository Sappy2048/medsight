import asyncio
import io
import logging
import zipfile
from datetime import datetime
from typing import Optional

import httpx
import lxml.etree as etree

from src.config import DAILYMED_BASE_URL, LOINC_SECTIONS
from src.schemas.fda_schema import FDALabelVersion, LabelSections, SPLVersion

logger = logging.getLogger(__name__)

DAILYMED_DOWNLOAD_URL = "https://dailymed.nlm.nih.gov/dailymed/getFile.cfm"
SPL_XML_NAMESPACE = "urn:hl7-org:v3"

# Locale-independent month mapping for date parsing
_MONTH_MAP = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# Default HTTP timeout for all requests
DEFAULT_TIMEOUT = 30.0

# Maximum number of candidates to probe when resolving historical lineage
# Increased to 15 to cast a wider historical net
_HISTORICAL_PROBE_LIMIT = 50


# ─── Internal Helpers ─────────────────────────────────────────────────────────


def _parse_dailymed_date(date_str: str) -> datetime:
    """
    Parse a date string from DailyMed into a datetime object.
    Uses locale-independent parsing to handle month abbreviations correctly.
    """
    normalized = " ".join(date_str.strip().lower().split())
    parts = normalized.split()
    if len(parts) != 3:
        raise ValueError(f"Unexpected date format: {date_str}")

    month_abbr, day_str, year_str = parts
    month = _MONTH_MAP.get(month_abbr)
    if month is None:
        raise ValueError(f"Unknown month abbreviation: {month_abbr}")

    day = int(day_str.rstrip(","))
    year = int(year_str)

    return datetime(year, month, day)


def _extract_text_recursive(element: etree._Element, ns: dict) -> list[str]:
    """
    Recursively extracts text from an hl7:section and all its nested sub-sections.
    Preserves lists and tables while flattening them into the result list.
    """
    chunks = []
    
    # 1. Extract text from the current section's hl7:text node
    text_node = element.find("hl7:text", namespaces=ns)
    if text_node is not None:
        for child in text_node:
            tag = etree.QName(child.tag).localname
            if tag == "paragraph":
                text = "".join(child.itertext()).strip()
                if text:
                    chunks.append(text)
            elif tag == "list":
                items = child.findall(".//hl7:item", namespaces=ns)
                for item in items:
                    text = "".join(item.itertext()).strip()
                    if text:
                        chunks.append(f"• {text}")
            elif tag == "table":
                for row in child.findall(".//hl7:tr", namespaces=ns):
                    cell_texts = ["".join(cell.itertext()).strip() for cell in row if "".join(cell.itertext()).strip()]
                    if cell_texts:
                        chunks.append(" | ".join(cell_texts))

    # 2. Recurse into nested sections via hl7:component/hl7:section
    for component in element.findall("hl7:component", namespaces=ns):
        sub_section = component.find("hl7:section", namespaces=ns)
        if sub_section is not None:
            chunks.extend(_extract_text_recursive(sub_section, ns))
            
    return chunks


def _extract_section_text(xml_tree: etree._Element, loinc_code: str) -> Optional[str]:
    """
    Extract all text from an FDA SPL XML section identified by LOINC code, 
    including all nested sub-sections.
    """
    ns = {"hl7": SPL_XML_NAMESPACE}
    sections = xml_tree.xpath(f".//hl7:section[hl7:code[@code='{loinc_code}']]", namespaces=ns)

    if not sections:
        return None
    
    all_chunks = []
    for section in sections:
        all_chunks.extend(_extract_text_recursive(section, ns))

    return "\n\n".join(all_chunks) if all_chunks else None


def _parse_label_xml(
    xml_bytes: bytes, spl_set_id: str, spl_version: int, published_date: str
) -> FDALabelVersion:
    """
    Parse raw SPL XML bytes into a structured FDALabelVersion.
    Extracts exactly the 4 LOINC sections defined in config.LOINC_SECTIONS.
    """
    # Use secure parser to prevent XXE attacks
    parser = etree.XMLParser(resolve_entities=False, no_network=True)
    try:
        tree = etree.fromstring(xml_bytes, parser=parser)
    except etree.XMLSyntaxError as e:
        raise ValueError(f"Failed to parse XML for {spl_set_id} v{spl_version}: {e}") from e

    sections_data = {}
    for field_name, loinc_code in LOINC_SECTIONS.items():
        sections_data[field_name] = _extract_section_text(tree, loinc_code)

    return FDALabelVersion(
        spl_id=f"{spl_set_id}::v{spl_version}",
        spl_set_id=spl_set_id,
        effective_time=published_date,
        sections=LabelSections(**sections_data),
    )


def _select_active_modern_set_id(candidates: list[dict]) -> str:
    """
    Select the spl_set_id whose lineage best reflects modern regulatory updates.

    Selection priority (all sync — no network calls needed):
      1. Highest spl_version count  — proxy for active, well-maintained lineage
      2. Most recent published_date — tie-breaker for equal version counts

    This is the selector used for the *present* label.
    """
    if not candidates:
        raise ValueError("No candidates provided to _select_active_modern_set_id")

    def sort_key(c: dict):
        try:
            spl_version = int(c.get("spl_version", 0))
        except (ValueError, TypeError):
            spl_version = 0
        try:
            pub_date = _parse_dailymed_date(c.get("published_date", "Jan 01, 2000"))
        except ValueError:
            pub_date = datetime(2000, 1, 1)
        return (spl_version, pub_date)

    sorted_candidates = sorted(candidates, key=sort_key, reverse=True)
    chosen = sorted_candidates[0]["setid"]
    logger.debug(
        f"[modern selector] chose setid={chosen} "
        f"(spl_version={sorted_candidates[0].get('spl_version')}, "
        f"published={sorted_candidates[0].get('published_date')})"
    )
    return chosen


async def _select_deepest_historical_set_id(
    candidates: list[dict], client: httpx.AsyncClient, prescription_date: datetime
) -> str:
    """
    Select the spl_set_id whose lineage has the active label closest to
    (but not after) the prescription date.

    This anchors historical lineage selection to the exact prescription moment,
    avoiding defunct lineages that may have stopped updating years before the
    prescription was written.

    Strategy:
      - Cap to the top _HISTORICAL_PROBE_LIMIT candidates (by published date) to
        limit extra network round-trips.
      - Concurrently fetch each candidate's version history.
      - For each, find the latest version whose published_date <= prescription_date.
      - Return the setid whose closest-prior date is the maximum (i.e. most
        recently active lineage relative to the prescription date).
      - Falls back to the modern selector result if history probing fails entirely.
    """
    if not candidates:
        raise ValueError("No candidates provided to _select_deepest_historical_set_id")

    # Pre-filter: Sort by the candidate's latest published_date ASCENDING.
    # Lineages that were discontinued or haven't been updated in years
    # will naturally float to the top of this list, giving us the perfect
    # pool of legacy lines to probe for older records.
    def presort_key(c: dict):
        pub_date_str = c.get("published_date")
        if not pub_date_str:
            return datetime(9999, 12, 31)  # Sink invalid dates to the bottom
        try:
            return _parse_dailymed_date(pub_date_str)
        except ValueError:
            return datetime(9999, 12, 31)

    # Sort ascending (oldest first) and slice
    top_candidates = sorted(candidates, key=presort_key)[:_HISTORICAL_PROBE_LIMIT]

    async def probe_closest_date(candidate: dict) -> tuple[str, datetime]:
        """Fetch history for one candidate and return the closest valid date before or on the prescription date."""
        setid = candidate["setid"]
        url = f"{DAILYMED_BASE_URL}/spls/{setid}/history.json"
        try:
            response = await client.get(url, timeout=DEFAULT_TIMEOUT)
            response.raise_for_status()
            history_entries = response.json().get("data", {}).get("history", [])

            # Collect all dates that are on or before the prescription date
            valid_dates = []
            for entry in history_entries:
                pub_date_str = entry.get("published_date")
                if not pub_date_str:
                    continue
                try:
                    pub_date = _parse_dailymed_date(pub_date_str)
                except ValueError:
                    continue
                if pub_date <= prescription_date:
                    valid_dates.append(pub_date)

            if valid_dates:
                # The closest date to the prescription date (largest value <= rx_date)
                closest_date = max(valid_dates)
                return setid, closest_date
        except Exception as exc:
            logger.debug(f"[historical probe] failed for setid={setid}: {exc}")

        # Sentinel: no valid versions before prescription date, or request failed.
        # Use a far-past date so this candidate sinks to the bottom of the descending sort.
        return setid, datetime(1000, 1, 1)

    # Probe all candidates concurrently
    probe_results = await asyncio.gather(*[probe_closest_date(c) for c in top_candidates])

    # Sort descending: the candidate whose active label is closest to the prescription date wins
    probe_results_sorted = sorted(probe_results, key=lambda t: t[1], reverse=True)
    best_setid, best_date = probe_results_sorted[0]

    if best_date == datetime(1000, 1, 1):
        # All probes failed or no candidate had a version before the prescription date
        logger.warning(
            "[historical selector] all history probes failed or no valid pre-prescription "
            "versions found; falling back to modern selector for past lineage."
        )
        return _select_active_modern_set_id(candidates)

    logger.debug(
        f"[historical selector] chose setid={best_setid} "
        f"(closest pre-prescription date={best_date.date()})"
    )
    return best_setid


# ─── Public API Functions ──────────────────────────────────────────────────────


async def _resolve_candidates(drug_name: str, client: httpx.AsyncClient) -> list[dict]:
    """
    Fetch the raw candidate list for a drug name from DailyMed /spls.
    Returns the raw list[dict] so callers can apply whichever selector they need.
    """
    url = f"{DAILYMED_BASE_URL}/spls.json"

    for params in [
        {"drug_name": drug_name, "pagesize": 100, "doctype": "34391-3"},  # Rx first
        {"drug_name": drug_name, "pagesize": 100},                         # Fallback: all types
    ]:
        response = await client.get(url, params=params, timeout=DEFAULT_TIMEOUT)
        response.raise_for_status()
        candidates = response.json().get("data", [])
        if candidates:
            logger.info(f"Resolved '{drug_name}' with params: {params} → {len(candidates)} candidates")
            return candidates

    raise ValueError(f"No SPL found in DailyMed for drug: '{drug_name}'")


async def get_spl_set_id(drug_name: str, client: httpx.AsyncClient) -> str:
    """
    Resolve a drug name to its canonical spl_set_id via DailyMed /spls search.

    Uses the modern (active) selector — suitable for cases where a single
    set_id is required (e.g. direct version history lookups).
    """
    candidates = await _resolve_candidates(drug_name, client)
    return _select_active_modern_set_id(candidates)


async def get_version_history(
    spl_set_id: str, client: httpx.AsyncClient
) -> list[SPLVersion]:
    """
    Fetch the complete version history for a given spl_set_id.

    Returns versions sorted ascending by published_date (oldest first),
    so index 0 = first ever version, index -1 = latest version.

    DailyMed history response shape:
    {
      "data": {
        "history": [ { "spl_version": "3", "published_date": "Sep 26, 2012" }, ... ],
        "spl":     { "setid": "...", "title": "..." }
      }
    }
    """

    url = f"{DAILYMED_BASE_URL}/spls/{spl_set_id}/history.json"
    response = await client.get(url, timeout=DEFAULT_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    history_entries = data.get("data", {}).get("history", [])
    spl_meta = data.get("data", {}).get("spl", {})
    title = spl_meta.get("title")

    if not history_entries:
        raise ValueError(f"No version history found for spl_set_id: '{spl_set_id}'")

    versions = []
    for entry in history_entries:
        # spl_version comes back as a string from the API ("3"), cast to int
        spl_version = int(entry["spl_version"])
        published_date = entry["published_date"]

        versions.append(
            SPLVersion(
                # Synthesized composite ID — matches what _parse_label_xml produces
                spl_id=f"{spl_set_id}::v{spl_version}",
                spl_set_id=spl_set_id,
                spl_version=spl_version,
                effective_time=published_date,
                title=title,
            )
        )

    # Sort ascending: oldest version first
    versions.sort(key=lambda v: _parse_dailymed_date(v.effective_time))
    return versions


async def get_label_content(
    spl_set_id: str,
    spl_version: int,
    published_date: str,
    client: httpx.AsyncClient,
) -> FDALabelVersion:
    """
    Fetch and parse the full SPL XML label for a specific version of a drug label.

    DailyMed does NOT expose historical label XML via a JSON endpoint.
    Historical versions are available as ZIP files via getFile.cfm.
    The ZIP contains a single XML file which is the full SPL document.

    Args:
        spl_set_id:     Canonical set ID UUID
        spl_version:    Integer version number (from history endpoint)
        published_date: Human-readable date string from history (e.g. "Sep 26, 2012")
        client:         Shared httpx.AsyncClient

    Returns:
        FDALabelVersion with all 4 LOINC sections extracted

    Raises:
        ValueError: If ZIP contains no XML file
        httpx.HTTPStatusError: On API errors
    """

    params = {
        "type": "zip",
        "setid": spl_set_id,
        "version": spl_version,
    }

    response = await client.get(
        DAILYMED_DOWNLOAD_URL, params=params, timeout=DEFAULT_TIMEOUT
    )
    response.raise_for_status()

    # Unzip in memory — no disk I/O
    with zipfile.ZipFile(io.BytesIO(response.content)) as zf:
        # SPL ZIP should contain one main XML file alongside any image assets
        xml_filenames = [name for name in zf.namelist() if name.endswith(".xml")]

        if not xml_filenames:
            raise ValueError(
                f"No XML file found in ZIP for spl_set_id={spl_set_id}, version={spl_version}"
            )

        # Prefer the main SPL XML (usually named with setid or just the label)
        # Filter out auxiliary XMLs that might contain metadata
        main_xml = None
        for name in xml_filenames:
            # Main SPL files are typically at root level and larger
            if not name.startswith("_") and "/" not in name:
                main_xml = name
                break

        # Fallback to first XML if no clear main file found
        xml_name = main_xml if main_xml else xml_filenames[0]
        xml_bytes = zf.read(xml_name)

    return _parse_label_xml(xml_bytes, spl_set_id, spl_version, published_date)


async def get_past_and_present_labels(
    drug_name: str,
    prescription_date: str,  # "YYYY-MM-DD"
    client: httpx.AsyncClient,
) -> tuple[FDALabelVersion, FDALabelVersion]:
    """
    Orchestrator: given a drug name and prescription date, return the label
    that was active ON that date (past) and the current latest label (present).

    Key design: past and present labels are now resolved from INDEPENDENT
    spl_set_id lineages, each optimised for its temporal objective:

      - present_set_id  →  _select_active_modern_set_id()
          Picks the lineage with the highest version count and most recent
          publication. Ensures modern safety mandates are never missed due to
          an older brand lineage going inactive.

      - past_set_id     →  _select_deepest_historical_set_id()
          Concurrently probes the top candidate history ledgers and selects the
          lineage whose active label is closest to the prescription date.

    Version selection logic within each lineage:
      - past    = latest version whose published_date <= prescription_date,
                  falling back to version 1 if the rx date predates the registry.
      - present = absolute latest version (highest spl_version in its lineage).

    Both XML downloads are initiated concurrently via asyncio.gather.

    Args:
        drug_name:         Generic or brand name string
        prescription_date: ISO format date string "YYYY-MM-DD"
        client:            Shared httpx.AsyncClient

    Returns:
        (past_label, present_label) — both FDALabelVersion instances

    Raises:
        ValueError: If drug not found or history is empty
    """
    rx_date = datetime.strptime(prescription_date, "%Y-%m-%d")

    # ── Step 1: Resolve candidate pool once (single network call) ────────────
    candidates = await _resolve_candidates(drug_name, client)

    # ── Step 2: Resolve both lineages concurrently ────────────────────────────
    # present selector is synchronous; historical selector needs async probe calls.
    # We run the historical probe alongside fetching the present lineage's history.
    present_set_id = _select_active_modern_set_id(candidates)

    past_set_id = await _select_deepest_historical_set_id(candidates, client, rx_date)

    logger.info(
        f"'{drug_name}' lineage split — "
        f"present_set_id={present_set_id}, past_set_id={past_set_id}"
    )

    # ── Step 3: Fetch both version histories concurrently ─────────────────────
    if present_set_id == past_set_id:
        # Same lineage selected — one history fetch suffices
        present_versions = await get_version_history(present_set_id, client)
        past_versions = present_versions
    else:
        present_versions, past_versions = await asyncio.gather(
            get_version_history(present_set_id, client),
            get_version_history(past_set_id, client),
        )

    # ── Step 4: Select specific version from each history ─────────────────────

    # Present: absolute latest version in the modern lineage
    present_version = present_versions[-1]

    # Past: latest version in the historical lineage published on or before rx_date
    past_version: Optional[SPLVersion] = None
    for v in past_versions:  # ascending order — last match wins
        if _parse_dailymed_date(v.effective_time) <= rx_date:
            past_version = v

    if past_version is None:
        # Prescription predates all known DailyMed records — use earliest available
        past_version = past_versions[0]
        logger.warning(
            f"Prescription date {prescription_date} predates all DailyMed records "
            f"for '{drug_name}' in the historical lineage "
            f"(earliest: {past_version.effective_time}). "
            f"Using earliest available version as past label."
        )

    # ── Step 5: Fetch both XML labels concurrently ────────────────────────────
    past_label, present_label = await asyncio.gather(
        get_label_content(
            past_version.spl_set_id,
            past_version.spl_version,
            past_version.effective_time,
            client,
        ),
        get_label_content(
            present_version.spl_set_id,
            present_version.spl_version,
            present_version.effective_time,
            client,
        ),
    )

    logger.info(
        f"Past:    {past_label.spl_id} ({past_label.effective_time}) "
        f"[lineage: {past_version.spl_set_id}]\n"
        f"Present: {present_label.spl_id} ({present_label.effective_time}) "
        f"[lineage: {present_version.spl_set_id}]"
    )
    return past_label, present_label