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


def _extract_section_text(xml_tree: etree._Element, loinc_code: str) -> Optional[str]:
    """
    Extract all paragraph text from a DailyMed SPL XML section identified by LOINC code.
    Returns all paragraph text joined as a single string, or None if section absent.
    """
    ns = {"hl7": SPL_XML_NAMESPACE}

    # Find the <section> whose <code> attribute matches the LOINC code
    sections = xml_tree.xpath(
        f".//hl7:section[hl7:code[@code='{loinc_code}']]",
        namespaces=ns,
    )

    if not sections:
        return None
    
    chunks = []
    for section in sections:
        text_node = section.find("hl7:text", namespaces=ns)
        if text_node is None:
            continue

        for child in text_node:
            # Strip namespace for tag comparison
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
                # Flatten table rows — preserve cell relationships on one line
                for row in child.findall(".//hl7:tr", namespaces=ns):
                    cells = row.findall(".//*")
                    cell_texts = []
                    for cell in row:
                        cell_text = "".join(cell.itertext()).strip()
                        if cell_text:
                            cell_texts.append(cell_text)
                    if cell_texts:
                        chunks.append(" | ".join(cell_texts))

    return "\n\n".join(chunks) if chunks else None


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


def _select_canonical_set_id(candidates: list[dict]) -> str:
    """
    Priority waterfall for selecting the canonical spl_set_id.
    Priority:
      1. Most versions overall (actively maintained lineage proxy)
      2. Most recently published (tie-breaker)
    """
    if not candidates:
        raise ValueError("No candidates provided")


    # Sort by: version count (desc), then published date (desc)
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
    return sorted_candidates[0]["setid"]


# ─── Public API Functions ──────────────────────────────────────────────────────


async def get_spl_set_id(drug_name: str, client: httpx.AsyncClient) -> str:
    """
    Resolve a drug name to its canonical spl_set_id via DailyMed /spls search.

    Applies the canonical selection waterfall when multiple manufacturers exist.
    Uses human/rxonly filter to exclude animal and OTC labels from the candidate pool.
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
            logger.info(f"Resolved '{drug_name}' with params: {params}")
            return _select_canonical_set_id(candidates)

    raise ValueError(f"No SPL found in DailyMed for drug: '{drug_name}'")

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

    Both labels are guaranteed to come from the same spl_set_id lineage,
    making the temporal diff in Agent 4 semantically valid.

    Version selection logic:
      - past    = latest version whose published_date <= prescription_date
      - present = latest version overall (highest spl_version)

    If the prescription_date is before the earliest known version, we use
    version 1 as the past label and log a warning. This handles edge cases
    where the drug existed before DailyMed records began.

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

    spl_set_id = await get_spl_set_id(drug_name, client)
    logger.info(f"Resolved '{drug_name}' → spl_set_id: {spl_set_id}")

    versions = await get_version_history(spl_set_id, client)

    # Select past version — latest version published on or before rx_date
    past_version: Optional[SPLVersion] = None
    for v in versions:  # ascending order, so last match wins
        if _parse_dailymed_date(v.effective_time) <= rx_date:
            past_version = v

    if past_version is None:
        # Prescription predates all known DailyMed records — use earliest available
        past_version = versions[0]
        logger.warning(
            f"Prescription date {prescription_date} predates all DailyMed records "
            f"for {drug_name} (earliest: {past_version.effective_time}). "
            f"Using earliest available version."
        )

    # Present version is always the last element (sorted ascending)
    present_version = versions[-1]

    # Fetch both label XMLs in parallel — no reason to wait for one before the other
    past_label, present_label = await asyncio.gather(
        get_label_content(
            spl_set_id, past_version.spl_version, past_version.effective_time, client
        ),
        get_label_content(
            spl_set_id,
            present_version.spl_version,
            present_version.effective_time,
            client,
        ),
    )

    logger.info(f"Past: {past_label.spl_id} ({past_label.effective_time}) | Present: {present_label.spl_id} ({present_label.effective_time})")
    return past_label, present_label