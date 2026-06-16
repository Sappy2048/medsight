import pandas as pd
from rapidfuzz import process, fuzz
from functools import lru_cache
from pathlib import Path
import re

_DATASET_PATH = Path("data/updated_indian_medicine_data.csv")

def _load_dataset() -> dict[str, dict]:
    """
    Returns a flat dict: { normalized_brand_name: {row_data} }
    Normalized = lowercase, stripped whitespace.
    """
    df = pd.read_csv(_DATASET_PATH)

    lookup = {}
    for _, row in df.iterrows():
        # Guard against empty/NaN rows in the 'name' column
        if pd.isna(row["name"]):
            continue
            
        key = str(row["name"]).strip().lower()
        lookup[key] = row.to_dict()
        
    return lookup

# Module-level singleton
_INDIAN_DRUG_INDEX: dict[str, dict] = _load_dataset()
_BRAND_NAMES: list[str] = list(_INDIAN_DRUG_INDEX.keys())

