import os
from dotenv import load_dotenv

load_dotenv()

# Set FastEmbed cache directory
# Priority: 1. Existing env var, 2. /app/data for containers, 3. Local project directory
if "FASTEMBED_CACHE_DIR" not in os.environ:
    # Check if we're in a Docker container (/app exists)
    if os.path.isdir("/app/data"):
        os.environ["FASTEMBED_CACHE_DIR"] = "/app/data/fastembed_cache"
    else:
        # Local development path
        os.environ["FASTEMBED_CACHE_DIR"] = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "data",
            "fastembed_cache"
        )

# Ensure the cache directory exists
cache_dir = os.environ["FASTEMBED_CACHE_DIR"]
os.makedirs(cache_dir, exist_ok=True)

# ─── LLM Configuration (Local Ollama via OpenAI Client) ──────────────────────
LLM_MODEL         = "Qwen/Qwen2.5-7B-Instruct-Turbo"
TOGETHER_BASE_URL = "https://api.together.xyz/v1"
TOGETHER_API_KEY  = os.getenv("TOGETHER_API_KEY")

SEVERITY_ONTOLOGY = {
    "contraindicated": 5,
    "avoid":           4,
    "use caution":     3,
    "monitor closely": 2,
    "monitor":         1,
    "no clinically significant interaction": 0,
}

MVP_DRUGS = ["Warfarin", "Azithromycin", "Metformin", "Ibuprofen", "Lisinopril"]

# Qdrant configuration with validation
QDRANT_URL        = os.getenv("QDRANT_URL")
QDRANT_API_KEY    = os.getenv("QDRANT_API_KEY")

# Validate Qdrant configuration at module level (fail fast with helpful message)
if not QDRANT_URL:
    print("WARNING: QDRANT_URL environment variable is not set.")
    print("The application will fail when attempting vector search operations.")
    print("Please set QDRANT_URL to your Qdrant Cloud cluster URL (e.g., https://xxx.cloud.qdrant.io:6333)")
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
