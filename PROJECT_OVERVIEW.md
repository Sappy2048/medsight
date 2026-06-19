# MedSight — Project Overview

MedSight is a **Temporal Drug Safety Intelligence System** designed to detect retrospective drug interaction risks. It identifies if a prescription written in the past has become dangerous due to updated FDA safety warnings.

---

## 1. Core Architecture: The 6-Agent Workflow

MedSight uses a **LangGraph** orchestration engine to route data through six specialized agents. Each agent handles a specific layer of the transformation from raw prescription text to a verified clinical alert.

### Agent 1: Copilot Agent (Entry Gate)
*   **Role:** Extracts structured prescription data from raw natural language.
*   **Key Function:** `preflight_validate(raw_input, llm_client)`
*   **Output:** `ParsedPrescription` Pydantic model.

### Agent 2: Drug Resolution Agent
*   **Role:** Translates Indian brand names (e.g., "Dolo 650") to canonical RxNorm IDs.
*   **Key Function:** `resolve_prescription(parsed_prescription, http_client)`
*   **Process:** Checks `Indian-Medicine-Dataset` (via `extract_drug_data`), resolves generics to RxCUIs using the RxNorm API.

### Agent 3: Extraction Agent (Layer 1)
*   **Role:** Fetches historical and current FDA labels and extracts structured interaction data.
*   **Key Functions:** 
    *   `get_past_and_present_labels(drug_name, prescription_date, client)` (Service)
    *   `extract_interactions(label_version, source_generic, llm_client)`
*   **Process:** Uses `spl_set_id` to trace label history on DailyMed; LLM extracts interactions into `ExtractionResult`.

### Agent 4: Temporal Diff Agent (Layer 2)
*   **Role:** The core differentiator. Compares "Past" vs "Present" JSON versions of FDA warnings.
*   **Key Function:** `compute_temporal_diff(past_ext, present_ext, target_drug, llm_client)`
*   **Process:** Deterministic Python logic classifies changes (`ADDED`, `STRENGTHENED`, etc.) and calculates a severity delta.

### Agent 5: Patient Impact Agent (Layer 3)
*   **Role:** Evaluates the diff against the specific patient and ICMR guidelines.
*   **Key Function:** `analyze_patient_impact(diffs, resolved_drugs, prescription_date, llm_client, qdrant_client)`
*   **Process:** RAG retrieval from Qdrant for ICMR context; rule-based risk classification (`CRITICAL` to `NONE`).

### Agent 6: Synthesis + Reflection Agent
*   **Role:** Final safety and verification layer.
*   **Key Function:** `synthesize_final_report(report, diff_results, llm_client)`
*   **Process:** Cross-checks clinical claims against deterministic "Anchor Facts" to ensure zero hallucinations.

---

## 2. Key Endpoint Functions

| Function | File | Description |
| :--- | :--- | :--- |
| `run_medsight()` | `src/agents/graph.py` | **Main Entry Point.** Orchestrates the full 6-agent pipeline. |
| `run_copilot_qa()` | `src/agents/graph.py` | Interactive Q&A grounded strictly on the generated report. |
| `preflight_validate()` | `src/agents/copilot.py` | Normalizes raw user input into structured Pydantic models. |
| `resolve_prescription()` | `src/agents/resolution.py` | Maps local brand names to international RxNorm CUIs. |
| `compute_temporal_diff()`| `src/agents/temporal.py` | Performs deterministic comparison of two FDA label versions. |

---

## 3. Core Data Schemas (Pydantic)

The system enforces strict data integrity using Pydantic models. No unstructured text is passed between agents.

### Prescription Schemas (`prescription_schema.py`)
*   **`ParsedDrug`**: Individual medicine from a prescription (brand, dose, route, frequency).
*   **`ParsedPrescription`**: Collection of drugs, prescription date, and patient metadata.

### Resolution Schemas (`resolution_schema.py`)
*   **`ResolvedDrug`**: Fully resolved drug with RxNorm IDs and Indian dataset metadata.

### FDA & Extraction Schemas (`fda_schema.py`, `diff_schema.py`)
*   **`FDALabelVersion`**: Structured FDA label sections (Boxed Warning, Contraindications, etc.).
*   **`InteractionRecord`**: A single interaction extracted from an FDA label.
*   **`ExtractionResult`**: All interactions found in one specific label version.
*   **`DiffResult`**: The delta between two label versions (Severity scores, change type).

### Impact & Synthesis Schemas (`impact_schema.py`, `synthesizer_schema.py`)
*   **`DrugPairAlert`**: A patient-specific alert including clinical reasoning and ICMR context.
*   **`PatientImpactReport`**: The overall clinical evaluation (Summary, Action, Risk Level).
*   **`MedSightFinalReport`**: The final verified payload with integrity scores and hallucination checks.

---

## 4. Current State & Built Components

*   **LangGraph Orchestration**: Fully implemented DAG with conditional re-entry for quality control.
*   **FDA Temporal Logic**: `fda_client.py` successfully traverses `spl_set_id` history.
*   **Indian Drug Context**: Local dataset integration with fuzzy matching via RapidFuzz.
*   **Deterministic Diffing**: Layer 2 logic ensures severity changes are calculated arithmetically, not narratively.
*   **Grounding Verification**: Agent 6 uses a deterministic "Anchor Fact" check to prevent LLM drift.
