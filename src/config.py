import os

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY      = os.getenv("GROQ_API_KEY")
SEVERITY_ONTOLOGY = {
    "contraindicated": 5,
    "avoid":           4,
    "use caution":     3,
    "monitor closely": 2,
    "monitor":         1,
    "no clinically significant interaction": 0,
}

MVP_DRUGS = ["Warfarin", "Azithromycin", "Metformin", "Ibuprofen", "Lisinopril"]

GROQ_MODEL        = "llama-3.3-70b-versatile"

QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = "icmr_guidelines"
EMBEDDING_MODEL   = "all-MiniLM-L6-v2"
EMBEDDING_DIM     = 384
CHUNK_SIZE        = 500
CHUNK_OVERLAP     = 50

OPENFDA_BASE_URL  = "https://api.fda.gov/drug/label.json"
RXNORM_BASE_URL   = "https://rxnav.nlm.nih.gov/REST"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

LOINC_SECTIONS = {
    "boxed_warning"    : "34066-1",
    "contraindications": "34070-3",
    "warnings"         : "43685-7",
    "drug_interactions": "34073-7",
}
