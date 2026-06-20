"""
rxnorm_client.py
────────────────
Async RxNorm API client for Agent 1 (Drug Resolution).

Resolves a single INN generic name → RxCUI string via the
NLM RxNorm REST API.

API endpoint used:
    GET /rxcui.json?name={drug_name}&search=1

    search=1 — approximate match mode. Handles minor spelling
    variations in generic names extracted from the composition
    waterfall (e.g. "Amoxycillin" vs "Amoxicillin").

Design contract (matches resolution.py expectations):
    - Returns the first RxCUI string on success.
    - Returns None on a clean miss (no CUI found) — caller decides
      whether to warn or error. Never raises on a miss.
    - Raises httpx.HTTPStatusError on non-2xx API responses.
    - Raises httpx.TimeoutException on network timeout.
    - Never instantiates its own client — caller injects shared instance.
"""

import logging
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

# RxNorm REST API base URL.
# Override via config if your deployment uses a local NLM mirror.
RXNORM_BASE_URL = "https://rxnav.nlm.nih.gov/REST"


async def get_rxcui(
    drug_name: str,
    http_client: httpx.AsyncClient,
) -> Optional[str]:
    """
    Resolves a single INN generic name to its primary RxCUI.

    Args:
        drug_name:   Cleaned INN generic name from the composition waterfall.
                     Must be dosage-stripped before calling — RxNorm cannot
                     resolve strings containing numbers and units.
                     e.g. "Ambroxol" ✅  |  "Ambroxol 30mg/5ml" ❌
        http_client: Shared httpx.AsyncClient injected by the caller
                     (resolution.py). Lifecycle owned by the caller —
                     never created or closed here.

    Returns:
        First RxCUI string if resolved, e.g. "41493".
        None if the API returned a clean empty response (no match found).

    Raises:
        httpx.HTTPStatusError:   Non-2xx response from the RxNorm API.
        httpx.TimeoutException:  Request exceeded the client timeout.

    Note:
        The caller (_resolve_rxcui_list in resolution.py) handles the None
        case with a warning and only raises ResolutionError if ALL generics
        in a drug return None. Do not raise here on a miss.
    """
    url = f"{RXNORM_BASE_URL}/rxcui.json"
    params = {
        "name":   drug_name,
        "search": 1,            # Approximate match — handles spelling variants
    }

    logger.debug("RxNorm request: GET %s | params=%s", url, params)

    response = await http_client.get(url, params=params)
    response.raise_for_status()

    data = response.json()
    cuis: list[str] = data.get("idGroup", {}).get("rxnormId", [])

    if not cuis:
        logger.debug("RxNorm: no CUI found for '%s'.", drug_name)
        return None

    logger.debug("RxNorm: resolved '%s' → '%s'.", drug_name, cuis[0])
    return cuis[0]


async def get_drug_classes(drug_name: str) -> list[str]:
    """
    Asynchronously calls the NLM RxClass API to retrieve drug classes
    associated with a specified drug name.

    Args:
        drug_name: The name of the generic or branded drug.

    Returns:
        A flat, de-duplicated list of class names (strings) associated with the drug.
        Returns an empty list [] on failure, missing data, or timeout.
    """
    url = f"{RXNORM_BASE_URL}/rxclass/class/byDrugName.json"
    params = {"drugName": drug_name}

    logger.debug("RxClass request: GET %s | params=%s", url, params)

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(url, params=params)
            response.raise_for_status()
            data = response.json()

        drug_info_list = data.get("rxclassDrugInfoList", {}).get("rxclassDrugInfo", [])
        
        classes = []
        for item in drug_info_list:
            class_item = item.get("rxclassMinConceptItem", {})
            class_name = class_item.get("className")
            if class_name:
                classes.append(class_name)

        # De-duplicate while preserving order
        seen = set()
        unique_classes = []
        for c in classes:
            if c not in seen:
                seen.add(c)
                unique_classes.append(c)

        logger.debug("RxClass: resolved '%s' to classes: %s", drug_name, unique_classes)
        return unique_classes

    except Exception as e:
        logger.warning("RxClass lookup failed for '%s': %s", drug_name, e)
        return []

