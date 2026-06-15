import requests
from config import RXNORM_BASE_URL

def get_rxcui(drug_name: str) -> str:
    url = f"{RXNORM_BASE_URL}/rxcui.json"
    params = {"name": drug_name, "search": 1}
    
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    
    data = resp.json()
    cuis = data.get("idGroup", {}).get("rxnormId", [])
    
    if not cuis:
        raise ValueError(f"No RxCUI found for: {drug_name}")
    
    return cuis[0]

