# MedSight — Drug Safety Intelligence System

MedSight is a **retrospective drug-drug interaction detection system** that analyzes historical prescriptions against evolving FDA label warnings. It uses a multi-agent LangGraph pipeline to parse prescriptions, resolve drugs, fetch FDA labels, compute temporal diffs, and synthesize patient safety reports.

## Architecture Overview

The system consists of:

- **Backend API** (`src/main.py`): FastAPI server exposing REST endpoints
- **LangGraph Pipeline** (`src/agents/`): Multi-agent workflow for prescription analysis
- **ETL Pipeline** (`src/services/etl_pipeline.py`): ICMR PDF → structured JSON extraction
- **RAG Engine** (`src/services/rag_engine.py`): Qdrant-based vector search for Indian clinical guidelines
- **Frontend** (`frontend/index.html`): Alpine.js-based clinical UI

## Quick Start

### 1. Clone and Setup Environment

```bash
git clone https://github.com/Sappy2048/medsight.git
cd medsight

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Python dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Create a `.env` file in the project root. You can either use a cloud provider like Together AI, or run the entire pipeline completely locally using a local LLM server.

**Option A: Cloud LLM (Together AI)**

```bash
# LLM Configuration (Together AI)
TOGETHER_API_KEY=your_together_api_key_here
TOGETHER_BASE_URL=https://api.together.xyz/v1
LLM_MODEL=Qwen/Qwen2.5-7B-Instruct-Turbo

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key  # Optional for local development
QDRANT_COLLECTION=icmr_guidelines

# Optional: RxNorm API (uses NLM public API by default)
RXNORM_BASE_URL=https://rxnav.nlm.nih.gov/REST

```

**Option B: Local LLM (Privacy-Preserving)**
If you are running a local model via an OpenAI-compatible server (like **Ollama**, **LM Studio**, or **vLLM**), you can run the entire MedSight pipeline locally without any external API calls. Simply point the base URL to your localhost port and use a dummy API key:

```bash
# Local LLM Configuration (Example for Ollama/LM Studio)
TOGETHER_API_KEY=dummy_key_not_used
TOGETHER_BASE_URL=http://localhost:11434/v1  # Port 11434 for Ollama, 1234 for LM Studio
LLM_MODEL=qwen2.5:7b-instruct                # Replace with your downloaded model name

# Qdrant Vector Database
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=local_dummy_key
QDRANT_COLLECTION=icmr_guidelines

# Optional: RxNorm API
RXNORM_BASE_URL=https://rxnav.nlm.nih.gov/REST

```

*(Note: Running the full LangGraph pipeline locally requires sufficient RAM/VRAM depending on the model size you choose).*

### 3. Start Qdrant (Docker)

```bash
# Using docker-compose (recommended)
docker-compose up -d qdrant

# Or manually with Docker
docker run -d \
  -p 6333:6333 \
  -v $(pwd)/data/qdrant_storage:/qdrant/storage \
  qdrant/qdrant:latest
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/healthz
```

### 4. Run the ETL Pipeline

**Important:** All Python scripts must be run with `PYTHONPATH=.` to ensure proper module imports.

The ETL pipeline extracts structured clinical data from ICMR PDF guidelines:

```bash
# Place ICMR PDF files in data/raw_pdfs/
mkdir -p data/raw_pdfs
# Copy your ICMR guideline PDFs to this directory

# Run the ETL pipeline
PYTHONPATH=. python -m src.services.etl_pipeline
```

This will:
- Extract text from PDFs using `pymupdf4llm`
- Process with LLM (Qwen 2.5 via Together AI)
- Generate `data/icmr_chunks.json` with structured clinical chunks


> ⚠️ **Important Note on Clinical Accuracy & Local Models**
> Document parsing and chunk extraction require high-reasoning models. If you are using a smaller or weaker local LLM (e.g., 7B parameters or smaller), **hallucinations and false extractions are common**, which can compromise clinical safety logic.
> **Alternative (Recommended for Local Deployments):**
> * **Skip the ETL step entirely:** You can leave the `data/raw_pdfs/` folder completely unpopulated. MedSight includes a pre-populated, verified `data/icmr_chunks.json` file right out of the box.
> * **Direct Ingestion:** You can skip directly to **Step 5** to ingest this pre-baked file into Qdrant.
> * **Manual Verification:** For 100% clinical accuracy, you can manually edit or populate the `data/icmr_chunks.json` file by hand. This completely bypasses the risk of LLM extraction errors and ensures your vector database holds perfect ground-truth data.
 


### 5. Ingest into Qdrant (RAG Engine)

```bash
# Ingest the processed chunks into Qdrant
PYTHONPATH=. python -m src.services.rag_engine ingest
```

Verify ingestion:
```bash
# Search for a specific drug
PYTHONPATH=. python -m src.services.rag_engine search Warfarin "drug interaction"
```

### 6. Start the Main Server

```bash
# Development mode with auto-reload
PYTHONPATH=. python src/main.py

# Or with uvicorn directly
PYTHONPATH=. uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`

### 7. Access the Frontend

Open your browser and navigate to:
```
http://localhost:8000/
```

The frontend provides:
- Prescription input form
- Real-time pipeline progress tracking
- Clinical safety reports with severity badges
- Temporal diff visualization (FDA label changes over time)
- Grounded clinician copilot for follow-up questions

## API Endpoints

### Health Check
```bash
curl http://localhost:8000/health
```

### Evaluate Prescription
```bash
curl -X POST http://localhost:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Patient (62M) prescribed Warfarin 5mg OD and Azithromycin 500mg OD for 5 days on 2015-05-12",
    "prescription_date": "2015-05-12",
    "patient_age": 62
  }'
```

### Streaming Evaluation
```bash
curl -X POST http://localhost:8000/evaluate/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Tab Augmentin 625 BD + Dolo 650 TDS for 5 days", "prescription_date": "2021-03-15", "patient_age": 45}'
```

### Clinician Copilot Q&A
```bash
curl -X POST http://localhost:8000/copilot \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the alternative anti-infectives if Azithromycin is contraindicated?",
    "report": {...},
    "history": []
  }'
```

## Testing

### Integration Test Pipeline

**Important:** All test commands require `PYTHONPATH=.` prefix for proper module resolution.

Run the full end-to-end integration test:

```bash
# Run with default test cases
PYTHONPATH=. python tests/integration_test_pipeline.py

# Run with custom prescription
PYTHONPATH=. python tests/integration_test_pipeline.py "Patient prescribed Metformin 500mg and Ibuprofen 400mg on 2020-01-15"
```

This tests all pipeline stages:
1. Copilot Preflight (prescription parsing)
2. Drug Resolution (Indian Medicine Dataset + RxNorm)
3. FDA Label Fetching (historical + current)
4. Interaction Extraction (LLM structured JSON)
5. Temporal Diff (past vs present warnings)
6. Patient Impact (risk synthesis)
7. Final Report (integrity verification)

### Other Tests

```bash
# Test resolution pipeline
PYTHONPATH=. python tests/test_resolution_pipeline.py

# Test prescription parser
PYTHONPATH=. python tests/test_prescription_parser.py

# Test graph workflow
PYTHONPATH=. python tests/test_graph.py

# Test temporal agent
PYTHONPATH=. python tests/test_temporal_agent.py

# Test FDA pipeline
PYTHONPATH=. python tests/test_fda_pipeline.py

# Test popular Indian drugs resolution
PYTHONPATH=. python tests/test_popular_indian_drugs_resolution.py
```

## Project Structure

```
medsight/
├── src/
│   ├── main.py                 # FastAPI entry point
│   ├── config.py               # Configuration settings
│   ├── agents/                 # LangGraph agents
│   │   ├── graph.py            # Main orchestration graph
│   │   ├── copilot.py          # Clinical copilot agent
│   │   ├── extraction.py       # FDA label extraction
│   │   ├── impact.py           # Patient impact analysis
│   │   ├── prescription_parser.py
│   │   ├── resolution.py       # Drug name resolution
│   │   ├── synthesis.py        # Report synthesis
│   │   └── temporal.py         # Temporal diff computation
│   ├── schemas/                # Pydantic models
│   ├── services/               # External service clients
│   │   ├── etl_pipeline.py     # ICMR PDF processing
│   │   ├── rag_engine.py       # Qdrant RAG
│   │   ├── fda_client.py       # DailyMed API client
│   │   ├── rxnorm_client.py    # RxNorm API client
│   │   └── indian_drug_loader.py
│   └── database.py             # Database models (optional)
├── frontend/
│   └── index.html              # Alpine.js frontend
├── data/
│   ├── raw_pdfs/               # Input ICMR PDFs
│   ├── icmr_chunks.json        # Processed guidelines
│   └── qdrant_storage/         # Qdrant persistent data
├── tests/                      # Integration and unit tests
├── docker-compose.yml          # Docker services
└── requirements.txt            # Python dependencies
```

## Data Flow

1. **Prescription Input** → Raw text with date and patient age
2. **Preflight** → LLM parses into structured `ParsedPrescription`
3. **Resolution** → Resolve brand names to generics via Indian Medicine Dataset + RxNorm
4. **Label Fetching** → Fetch historical + current FDA labels from DailyMed
5. **Extraction** → LLM extracts interaction warnings from label sections
6. **Temporal Diff** → Compare past vs present warnings for each drug pair
7. **Patient Impact** → Synthesize patient-specific risks using ICMR guidelines
8. **Final Report** → Generate verified clinical summary with integrity scoring

## Important Note on PYTHONPATH

All Python scripts in this project use absolute imports (e.g., `from src.config import ...`). **You must set `PYTHONPATH=.` before running any script** to ensure Python can resolve the `src` module correctly.

**Correct:**
```bash
PYTHONPATH=. python -m src.services.etl_pipeline
PYTHONPATH=. python tests/integration_test_pipeline.py
```

**Incorrect (will fail with ModuleNotFoundError):**
```bash
python -m src.services.etl_pipeline
python tests/integration_test_pipeline.py
```

## Troubleshooting

### Qdrant Connection Issues
```bash
# Check if Qdrant is running
docker ps | grep qdrant

# View Qdrant logs
docker logs <qdrant-container-id>

# Restart Qdrant
docker-compose restart qdrant
```

### LLM API Errors
- Verify `TOGETHER_API_KEY` is set correctly in `.env`
- Check Together AI dashboard for rate limits
- The pipeline has retry logic (3 attempts with exponential backoff)

### Missing Drug Resolutions
- Ensure Indian Medicine Dataset CSV is present (auto-downloaded on first run)
- Check RxNorm API connectivity: `curl https://rxnav.nlm.nih.gov/REST/rxcui.json?name=metformin&search=1`

### ETL Pipeline Failures
- Ensure PDFs are valid and readable
- Check Together AI API key and rate limits
- Review `data/icmr_chunks.json` output for structure

### Module Import Errors
If you see `ModuleNotFoundError: No module named 'src'`:
- Make sure you're running from the project root directory
- Always prefix commands with `PYTHONPATH=.`

## License

MIT License — See LICENSE file for details.

## Acknowledgments

- [Together AI](https://www.together.ai/) for LLM inference
- [NLM RxNorm](https://www.nlm.nih.gov/research/umls/rxnorm/index.html) for drug standardization
- [DailyMed](https://dailymed.nlm.nih.gov/) for FDA label data
- [ICMR](https://www.icmr.gov.in/) for Indian clinical guidelines
- [Qdrant](https://qdrant.tech/) for vector search