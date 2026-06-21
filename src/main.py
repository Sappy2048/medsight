import os
import logging
from typing import Optional, Union, Literal, List, Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
import json

from src.config import (
    TOGETHER_BASE_URL,
    TOGETHER_API_KEY,
    QDRANT_URL,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
)
from src.agents.graph import run_medsight, run_copilot_qa, build_medsight_graph, MedSightState
from src.schemas.synthesizer_schema import MedSightFinalReport
from src.schemas.prescription_schema import ParsedPrescription
from src.services.rag_engine import FASTEMBED_MODEL

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medsight.api")

# Startup validation logger
startup_logger = logging.getLogger("medsight.startup")

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

@app.on_event("startup")
async def startup_event():
    """Run startup validations and log configuration status."""
    startup_logger.info("=" * 60)
    startup_logger.info("MedSight API Starting Up")
    startup_logger.info("=" * 60)
    
    # Log configuration status
    startup_logger.info(f"FastEmbed cache directory: {os.environ.get('FASTEMBED_CACHE_DIR', 'Not set')}")
    startup_logger.info(f"Cache directory exists: {os.path.isdir(os.environ.get('FASTEMBED_CACHE_DIR', ''))}")
    
    # Check Qdrant configuration
    if QDRANT_URL:
        startup_logger.info(f"Qdrant URL configured: {QDRANT_URL[:30]}..." if len(QDRANT_URL) > 30 else f"Qdrant URL configured: {QDRANT_URL}")
    else:
        startup_logger.error("CRITICAL: QDRANT_URL is not set. Vector search will fail.")
    
    if TOGETHER_API_KEY:
        startup_logger.info("Together AI API key is configured")
    else:
        startup_logger.error("CRITICAL: TOGETHER_API_KEY is not set. LLM calls will fail.")
    
    # Verify frontend file exists
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    if os.path.exists(frontend_path):
        startup_logger.info(f"Frontend file found: {frontend_path}")
    else:
        startup_logger.error(f"Frontend file NOT found at: {frontend_path}")
    
    # Test Qdrant connectivity
    try:
        collections = qdrant_client.get_collections()
        collection_names = [c.name for c in collections.collections]
        startup_logger.info(f"Qdrant connected successfully. Collections: {collection_names}")
        
        # Check for required collection
        if QDRANT_COLLECTION in collection_names:
            startup_logger.info(f"Required collection '{QDRANT_COLLECTION}' exists")
        else:
            startup_logger.warning(f"Required collection '{QDRANT_COLLECTION}' NOT found. Run ETL pipeline.")
    except Exception as e:
        startup_logger.error(f"Qdrant connection failed: {e}")
        startup_logger.error("Vector search functionality will not work.")
    
    startup_logger.info("=" * 60)

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

def serialize_val(val):
    if hasattr(val, "model_dump"):
        return val.model_dump()
    elif isinstance(val, list):
        return [serialize_val(x) for x in val]
    elif isinstance(val, dict):
        return {k: serialize_val(v) for k, v in val.items()}
    elif hasattr(val, "__dict__"):
        return {k: serialize_val(v) for k, v in val.__dict__.items() if not k.startswith('_')}
    return val

class CopilotRequest(BaseModel):
    question: str
    report: MedSightFinalReport
    history: List[Dict[str, str]] = Field(default_factory=list)

@app.post(
    "/copilot",
    status_code=status.HTTP_200_OK,
    summary="Clinician Grounded Q&A Copilot",
    description="Allows follow-up questions about the safety report using clinical guidelines and FDA labels.",
    tags=["Core"],
)
async def copilot_qa(request: CopilotRequest):
    try:
        answer = await run_copilot_qa(
            question=request.question,
            report=request.report,
            history=request.history,
            llm_client=llm_client
        )
        return {"answer": answer}
    except Exception as e:
        logger.error(f"Error in copilot QA: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process copilot query: {str(e)}",
        )

@app.post(
    "/evaluate/stream",
    summary="Stream prescription drug safety analysis",
    description="Streams graph traversal events as SSE followed by the final safety report.",
    tags=["Core"],
)
async def evaluate_stream(request: EvaluateRequest):
    if not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query string cannot be empty.",
        )

    logger.info(f"Received safety evaluation stream request: '{request.query}'")

    async def event_generator():
        app_graph = build_medsight_graph(llm_client, qdrant_client, None)
        
        # Format the query with metadata if provided
        formatted_query = request.query
        if request.prescription_date or request.patient_age:
            parts = [request.query]
            if request.prescription_date:
                parts.append(f"Prescription Date: {request.prescription_date}")
            if request.patient_age:
                parts.append(f"Patient Age: {request.patient_age}")
            formatted_query = "\n".join(parts)

        initial_state = MedSightState(
            raw_input=formatted_query,
            prescription=None,
            resolved_drugs=[],
            label_history={},
            extraction_results={},
            diffs=[],
            reasoning=[],
            impact_report=None,
            final_report=None,
            copilot_session=[],
            loop_count=0,
            awaiting_input=False,
            should_rerun=False,
            clarification_message=None,
            errors=[]
        )

        accumulated_state = dict(initial_state)

        try:
            async for event in app_graph.astream(initial_state):
                if not event:
                    continue
                node_name = list(event.keys())[0]
                state_diff = event[node_name]
                
                # Merge diff into accumulated state
                for k, v in state_diff.items():
                    accumulated_state[k] = v
                
                # Yield a progress message to the client
                yield f"data: {json.dumps({'event': 'progress', 'node': node_name})}\n\n"
            
            # Post-execution outcome handling
            if accumulated_state.get("awaiting_input"):
                clarif_data = {
                    'event': 'clarification_required',
                    'message': accumulated_state['clarification_message'],
                    'partial_prescription': serialize_val(accumulated_state['prescription'])
                }
                yield f"data: {json.dumps(clarif_data)}\n\n"
            elif accumulated_state.get("final_report") is not None:
                success_data = {
                    "status": "success",
                    "report": serialize_val(accumulated_state["final_report"]),
                    "resolved_drugs": serialize_val(accumulated_state.get("resolved_drugs")),
                    "diffs": serialize_val(accumulated_state.get("diffs")),
                    "extraction_results": serialize_val(accumulated_state.get("extraction_results"))
                }
                payload = {
                    'event': 'success',
                    'data': success_data
                }
                yield f"data: {json.dumps(payload)}\n\n"
            else:
                err_data = {
                    'event': 'error',
                    'message': 'MedSight pipeline completed without generating a final report.'
                }
                yield f"data: {json.dumps(err_data)}\n\n"
        except Exception as e:
            logger.error(f"Error in streaming LangGraph pipeline: {e}", exc_info=True)
            err_data = {
                'event': 'error',
                'message': str(e)
            }
            yield f"data: {json.dumps(err_data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the MedSight frontend HTML file."""
    frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend", "index.html")
    return FileResponse(frontend_path, media_type="text/html")

if __name__ == "__main__":
    import uvicorn
    # Allow running directly via python src/main.py
    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=True)