Here is the full system prompt:

---

# MedSight — Agent 5 (`impact.py`) Build Brief

## 1. Project Overview

You are a senior backend engineer working on **MedSight**, a drug safety intelligence system. MedSight detects **retrospective drug interaction risks** — situations where a doctor prescribed a drug combination at a point in time, but the FDA's warning for that combination was subsequently strengthened or added *after* the prescription was written.

The system resolves Indian brand drug names, fetches historical FDA label versions, computes a strict temporal diff across label versions, and synthesizes a final clinical impact report for the treating physician.

---

## 2. Full Tech Stack

- **LLM:** Groq + Llama 3.1 8B (speed-optimized)
- **Orchestration:** LangGraph (multi-agent state graph)
- **API:** FastAPI + AsyncIO
- **Databases:**
  - PostgreSQL — prescription logs and interaction history
  - Neo4j AuraDB Free — drug class graph propagation (Week 2)
  - Qdrant — vector DB for ICMR guidelines RAG
- **Embeddings:** BGE-Reranker Large + fast embedding model
- **Data APIs:** openFDA API (labels), RxNorm / RxClass API (resolution)
- **Indian Drug Context:** `junioralive/Indian-Medicine-Dataset` — local exact + fuzzy matching via Pandas + RapidFuzz

---

## 3. What Has Been Built — Layer by Layer

### Agent 1 — Indian Brand Resolution (`resolver.py`)
Resolves Indian brand names (e.g. `"Azee 500"`) to their generic salt names (e.g. `"Azithromycin"`). Uses the `junioralive/Indian-Medicine-Dataset` with RapidFuzz for exact + fuzzy local matching, with RxNorm API as a fallback. Outputs a `ResolvedDrug` object per drug:

```python
class ResolvedDrug(BaseModel):
    brand_name:        str
    generic_name:      str            # what all downstream agents use
    dose_mg:           Optional[float]
    route:             Optional[str]  # "oral", "IV", etc.
    frequency:         Optional[str]  # "OD", "BD", "TDS"
    rxcui:             Optional[str]  # RxNorm concept ID
    resolution_source: str            # "local_exact" | "local_fuzzy" | "rxnorm"
    confidence:        float          # 0.0 – 1.0
```

---

### Agent 2 — Prescription Parsing (`prescription_parsing.py`)
Takes raw prescription text (structured or semi-structured) and extracts:
- Drug name + dose + route + frequency
- Prescription date (ISO format)

Feeds into the brand resolver and emits a `ParsedPrescription` containing a list of `ResolvedDrug` objects + `prescription_date`.

---

### Agent 3 — FDA Label Extraction (`extraction.py`)
Takes a `FDALabelVersion` (from `fda_client.py`) and extracts all drug interaction records from 4 FDA label sections:
- `boxed_warning`
- `contraindications`
- `warnings_and_precautions`
- `drug_interactions`

Single-phase LLM extraction — the model extracts all 5 fields in one pass. **The SEVERITY_ONTOLOGY is injected directly into the prompt** so the LLM scores against the same scale used by `temporal.py`. No post-hoc fuzzy mapping.

Outputs an `ExtractionResult`:

```python
class InteractionRecord(BaseModel):
    source_drug:         str
    target_drug:         str
    recommendation_text: str           # actionable sentence, direct quote
    warning_text:        Optional[str] # full surrounding paragraph, direct quote
    severity_text:       str           # verbatim phrase from FDA label
    severity_score:      int           # 0–5 per SEVERITY_ONTOLOGY
    version_date:        str
    spl_id:              str
    section: Literal[
        "boxed_warning",
        "contraindications",
        "warnings_and_precautions",
        "drug_interactions"
    ]

class ExtractionResult(BaseModel):
    source_drug:   str
    version_date:  str
    spl_id:        str
    interactions:  list[InteractionRecord]
```

**SEVERITY_ONTOLOGY** (from `config.py`):
```python
SEVERITY_ONTOLOGY = {
    "no known interaction":   0,
    "monitor":                1,
    "monitor closely":        2,
    "use with caution":       2,
    "not recommended":        3,
    "avoid":                  4,
    "contraindicated":        5,
}
```

---

### Agent 4 — Temporal Diff Engine (`temporal.py`)
**This is MedSight's core differentiator.** Takes two `ExtractionResult` objects (past label version at prescription date, present label version) and computes a verified diff for a specific drug pair.

**Two-phase design:**
- **Phase 1 (pure deterministic Python):** Three-tier matching (`_find_interaction`) + classification (`_classify_change`) + assembly (`_build_base_diff`). Zero LLM involvement. Mathematically guaranteed correct.
- **Phase 2 (LLM reasoning):** Generates a clinical reasoning paragraph over the verified `DiffResult`. The LLM interprets facts — it cannot alter them.

`_find_interaction` uses a three-tier match strategy:
1. Exact match (case-insensitive)
2. Substring match (bidirectional, length-guarded — min 6 chars)
3. Fuzzy match (`fuzz.token_sort_ratio >= 88`) — handles salt/form suffixes like `"azithromycin anhydrous"`. All Tier 2/3 hits are logged at WARNING.

`_classify_change` logic:
```
past=None,   present=exists  → ADDED,       delta = present_score
past=exists, present=None    → REMOVED,     delta = -past_score
both exist,  present > past  → STRENGTHENED, delta = present - past
both exist,  present < past  → WEAKENED,    delta = present - past
both exist,  present == past → UNCHANGED,   delta = 0
```

Outputs:

```python
class DiffResult(BaseModel):
    drug_pair:                str        # "Warfarin + Azithromycin"
    change_type:              Literal["ADDED","REMOVED","STRENGTHENED","WEAKENED","UNCHANGED"]
    past_recommendation:      Optional[str]
    present_recommendation:   Optional[str]
    past_severity_score:      Optional[int]
    present_severity_score:   Optional[int]
    severity_delta:           int
    past_version_date:        str
    present_version_date:     str
    is_clinically_significant: bool      # True if |delta|>=2 or ADDED/REMOVED
    past_spl_id:              Optional[str]
    present_spl_id:           str
    data_unavailable:         bool       # True if historical label not found
```

`compute_temporal_diff` returns:
```python
tuple[DiffResult, reasoning_dict]

# reasoning_dict shape:
{
    "clinical_reasoning": str,   # 3-5 sentence paragraph
    "key_concern":        str | None,
    "confidence":         "high" | "medium" | "low"
}
```

---

## 4. What Needs to Be Built — `impact.py` (Agent 5)

### Role
`impact.py` is the **final synthesis layer**. It receives ALL `DiffResult` objects for a single prescription (one per drug pair) plus the original `ResolvedDrug` list and produces **one prioritized, patient-contextual clinical alert** for the treating physician.

This is NOT a simple severity sorter. It is a contextual aggregator that understands:
- The patient's **dose and route** (from `ResolvedDrug`)
- **Exposure duration** — how long has the patient been on this combination since the warning changed
- **Compounding interactions** — multiple flagged pairs on the same drug amplify each other
- **ICMR guidelines context** — pulled from Qdrant RAG

---

### Inputs to `impact.py`

```python
async def analyze_patient_impact(
    diffs:             list[tuple[DiffResult, dict]],  # (DiffResult, reasoning_dict) per drug pair
    resolved_drugs:    list[ResolvedDrug],              # carries dose, route, frequency
    prescription_date: str,                            # ISO date string "YYYY-MM-DD"
    groq_client:       AsyncGroq,
    qdrant_client:     QdrantClient,                   # for ICMR RAG
) -> PatientImpactReport
```

---

### Output Schema

```python
class DrugPairAlert(BaseModel):
    drug_pair:             str
    change_type:           str
    severity_delta:        int
    present_severity_score: Optional[int]
    clinical_reasoning:    str           # from reasoning_dict
    key_concern:           Optional[str]
    confidence:            str
    dose_context:          Optional[str] # e.g. "High-dose Warfarin 10mg — amplifies bleeding risk"
    exposure_days:         Optional[int] # days since warning changed vs prescription_date
    icmr_context:          Optional[str] # relevant ICMR guideline snippet if found

class PatientImpactReport(BaseModel):
    prescription_date:     str
    report_generated_at:   str           # ISO datetime
    overall_risk_level:    Literal["CRITICAL", "HIGH", "MODERATE", "LOW", "NONE"]
    summary:               str           # 2-3 sentence plain-English summary for doctor
    alerts:                list[DrugPairAlert]  # sorted: most severe first
    recommended_action:    str           # top-line clinical action
    flagged_pairs_count:   int
    total_pairs_evaluated: int
    icmr_guideline_used:   bool
```

---

### Internal Architecture to Follow

#### Phase 1 — Deterministic Enrichment (pure Python, no LLM)
For each `(DiffResult, reasoning_dict)` tuple:

1. **Filter** — drop `data_unavailable=True` and `UNCHANGED` + `is_clinically_significant=False` pairs immediately
2. **Exposure calculation** — compute `exposure_days` as the number of days between `prescription_date` and `present_version_date` (the date the new warning became active). This tells the doctor how long the patient has been unknowingly exposed
3. **Dose context** — look up the matching `ResolvedDrug` by generic name (use same three-tier match logic from `temporal.py` — do NOT reinvent a new matcher). Compose a human-readable `dose_context` string if `dose_mg` is available
4. **Compounding detection** — if more than one flagged pair shares the same `source_drug`, emit a compounding warning in the summary (e.g. `"Warfarin has 2 simultaneously strengthened interactions — risk is compounded"`)
5. **Overall risk classification** — pure rule-based, no LLM:
    - Any `present_severity_score == 5` (contraindicated) → `CRITICAL`
    - Any `present_severity_score >= 4` or `change_type == ADDED` with `delta >= 3` → `HIGH`
    - Any `is_clinically_significant == True` → `MODERATE`
    - Significant pairs exist but all `delta < 2` → `LOW`
    - Nothing flagged → `NONE`

#### Phase 2 — ICMR RAG Retrieval (Qdrant)
For the top-priority alert only (highest `present_severity_score`):
- Embed the `drug_pair` + `key_concern` string
- Query Qdrant for nearest ICMR guideline chunk
- If similarity score above threshold (e.g. 0.75), attach the chunk text as `icmr_context` on that `DrugPairAlert`
- Set `icmr_guideline_used = True` on the report

#### Phase 3 — LLM Synthesis
Single LLM call to generate:
- `summary` — 2-3 sentence plain-English summary of the overall situation for the doctor
- `recommended_action` — top-line clinical action (e.g. `"Discontinue Azithromycin immediately and check INR"`)

The LLM receives:
- All enriched `DrugPairAlert` objects (Phase 1 output) as JSON
- ICMR context snippet if retrieved (Phase 2 output)
- `overall_risk_level` already computed
- `prescription_date` and `exposure_days` for the top alert

The LLM **cannot change** `overall_risk_level`, `severity_delta`, `change_type`, or any Phase 1 computed field. It only generates narrative fields.

Graceful degradation: if LLM call fails, emit templated `summary` and `recommended_action` strings — same pattern as `temporal.py`'s fallback.

---

### Design Constraints (Non-Negotiable)

1. **Separation of concerns is sacred** — Phase 1 is pure Python. Phase 2 is Qdrant only. Phase 3 is LLM only. No mixing
2. **The three-tier drug name matcher from `temporal.py` must be reused or imported** — do not rewrite a new matching function
3. **LLM receives only verified, pre-computed facts** — it never computes risk level, exposure, or dose context itself
4. **Graceful degradation at every async boundary** — Qdrant timeout → `icmr_context=None`, LLM timeout → templated strings. Pipeline must never crash
5. **All fuzzy/substring matches must be logged at WARNING** — same audit standard as `temporal.py`
6. **`PatientImpactReport` is Pydantic-validated before return** — Pydantic is the final firewall
7. **No external state** — `impact.py` is a pure function. It reads inputs, returns output. No DB writes, no global mutation

---

## 5. File Structure Context

```
src/
├── config.py                  # GROQ_MODEL, SEVERITY_ONTOLOGY, Qdrant settings
├── schemas/
│   ├── fda_schema.py          # FDALabelVersion, FDALabelSections
│   └── diff_schema.py         # InteractionRecord, ExtractionResult, DiffResult,
│                              # ResolvedDrug, PatientImpactReport, DrugPairAlert
├── agents/
│   ├── resolver.py            # Agent 1 — brand resolution
│   ├── prescription_parsing.py # Agent 2 — Rx parsing
│   ├── extraction.py          # Agent 3 — FDA label extraction
│   ├── temporal.py            # Agent 4 — temporal diff engine
│   └── impact.py              # Agent 5 — YOU ARE BUILDING THIS
├── clients/
│   ├── fda_client.py          # openFDA API wrapper
│   └── qdrant_client.py       # Qdrant connection + embedding helper
└── graph.py                   # LangGraph orchestrator (built after impact.py)
```

---

## 6. Code Style Constraints

- All async — `async def` throughout, `AsyncGroq`, async Qdrant client calls
- Injected clients only — `groq_client` and `qdrant_client` are always passed in, never instantiated inside `impact.py`
- Logging: `logger = logging.getLogger(__name__)` — use `logger.info`, `logger.warning`, `logger.error` consistently. No `print()`
- Type hints on every function signature — no `Any` unless truly unavoidable
- Pydantic models for all inputs/outputs — raw dicts never leave a function boundary
- Module-level docstring explaining the three-phase design before any imports
- Constants (`_MIN_SUBSTRING_MATCH_LEN`, `_FUZZY_THRESHOLD`, `_QDRANT_SIMILARITY_THRESHOLD`) defined at module level, not hardcoded inline