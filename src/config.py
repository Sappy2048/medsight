SEVERITY_ONTOLOGY = {
    "contraindicated": 5,
    "avoid":           4,
    "use caution":     3,
    "monitor closely": 2,
    "monitor":         1,
}

MVP_DRUGS = ["Warfarin", "Azithromycin", "Metformin", "Ibuprofen", "Lisinopril"]

GROQ_MODEL        = "llama-3.1-8b-instant"

OPENFDA_BASE_URL  = "https://api.fda.gov/drug/label.json"
RXNORM_BASE_URL   = "https://rxnav.nlm.nih.gov/REST"
DAILYMED_BASE_URL = "https://dailymed.nlm.nih.gov/dailymed/services/v2"

LOINC_SECTIONS = {
    "boxed_warning"    : "34066-1",
    "contraindications": "34070-3",
    "warnings"         : "43685-7",
    "drug_interactions": "34073-7",
}
