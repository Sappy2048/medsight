import os
import logging
from typing import Optional, Union, Literal
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from qdrant_client import QdrantClient

from src.config import (
    TOGETHER_BASE_URL,
    TOGETHER_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
)
from src.agents.graph import run_medsight
from src.schemas.synthesizer_schema import MedSightFinalReport
from src.schemas.prescription_schema import ParsedPrescription
from src.services.rag_engine import FASTEMBED_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medsight.api")

app = FastAPI(
    title="MedSight Drug Safety Intelligence System",
    description="REST API for retrospective drug-drug interaction detection using LangGraph.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize global clients
if not TOGETHER_API_KEY:
    logger.warning("TOGETHER_API_KEY is not set in environment. LLM calls may fail.")

llm_client = AsyncOpenAI(
    base_url=TOGETHER_BASE_URL,
    api_key=TOGETHER_API_KEY or "missing-key",
)

# Initialize Qdrant Client
qdrant_client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
try:
    qdrant_client.set_model(FASTEMBED_MODEL)
    logger.info(f"Qdrant client initialized with FastEmbed model: {FASTEMBED_MODEL}")
except Exception as e:
    logger.error(f"Failed to initialize Qdrant FastEmbed model: {e}")

class EvaluateRequest(BaseModel):
    query: str = Field(
        ...,
        description="The raw prescription or clinical query string to evaluate.",
        examples=["Patient prescribed Warfarin and Azithromycin in May 2015"],
    )
    prescription_date: Optional[str] = Field(
        default=None,
        description="Optional ISO date string (YYYY-MM-DD) if not included in the query text.",
        examples=["2015-05-15"],
    )
    patient_age: Optional[int] = Field(
        default=None,
        description="Optional patient age in years if not included in the query text.",
        examples=[45],
    )

class MedSightSuccessResponse(BaseModel):
    status: Literal["success"]
    report: MedSightFinalReport
    resolved_drugs: Optional[list] = None
    diffs: Optional[list] = None
    extraction_results: Optional[dict] = None

class MedSightClarificationResponse(BaseModel):
    status: Literal["clarification_required"]
    message: str
    partial_prescription: Optional[ParsedPrescription] = None

MedSightEvaluateResponse = Union[MedSightSuccessResponse, MedSightClarificationResponse]

class HealthResponse(BaseModel):
    status: str
    qdrant_connected: bool

@app.get("/health", response_model=HealthResponse, tags=["Diagnostics"])
async def health_check():
    """Verify that the API server and key integrations are healthy."""
    qdrant_connected = False
    try:
        # Check Qdrant connection by listing collections
        qdrant_client.get_collections()
        qdrant_connected = True
    except Exception as e:
        logger.error(f"Health check: Qdrant connection failed: {e}")
        
    return HealthResponse(
        status="healthy",
        qdrant_connected=qdrant_connected,
    )

@app.post(
    "/evaluate",
    response_model=MedSightEvaluateResponse,
    status_code=status.HTTP_200_OK,
    summary="Evaluate prescription drug safety",
    description="Parses prescription, resolves drugs, fetches FDA labels, computes temporal diffs, and synthesizes patient safety reports.",
    tags=["Core"],
)
async def evaluate(request: EvaluateRequest):
    """
    Accepts a raw clinical query, runs the 6-agent LangGraph workflow,
    and returns a verified clinical safety report.
    """
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    logger.info(f"Received safety evaluation request: '{request.query}'")
    try:
        report = await run_medsight(
            raw_input=request.query,
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            db_pool=None, # PostgreSQL pool not required for MVP evaluations
        )
        logger.info("Successfully generated safety report.")
        return report
    except Exception as e:
        logger.error(f"Error executing LangGraph pipeline: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"An error occurred while evaluating the prescription: {str(e)}",
        )

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the MedSight frontend HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    # Allow running directly via python src/main.py
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)