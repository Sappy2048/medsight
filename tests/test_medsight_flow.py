import asyncio
import os
import logging
from openai import AsyncOpenAI
from qdrant_client import QdrantClient
from dotenv import load_dotenv

from src.agents.graph import run_medsight, run_copilot_qa
from src.schemas.synthesizer_schema import MedSightFinalReport
from src.config import TOGETHER_BASE_URL, TOGETHER_API_KEY

async def test_full_agentic_flow():
    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("test_flow")

    # 1. Setup Clients
    llm_client = AsyncOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=TOGETHER_API_KEY
    )
    
    # Setup Qdrant
    from src.config import QDRANT_URL, QDRANT_API_KEY
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Mock DB Pool
    db_pool = None

    # 2. Input
    # Classic interaction: Warfarin + Azithromycin
    # Using an older date to trigger temporal logic
    raw_input = "Patient prescribed Warfarin 5mg and Azee 500mg in March 2010"
    
    logger.info(f"\n--- Starting MedSight Pipeline for: {raw_input} ---")
    
    try:
        # 3. Run Pipeline
        report = await run_medsight(
            raw_input=raw_input,
            llm_client=llm_client,
            qdrant_client=qdrant_client,
            db_pool=db_pool
        )
        
        # 4. Validate Report
        logger.info("\n--- Pipeline Complete. Final Report Analysis ---")
        logger.info(f"Verified: {report.verified}")
        logger.info(f"Integrity Score: {report.integrity_score}")
        logger.info(f"Severity Badge: {report.severity_badge}")
        logger.info(f"Summary: {report.final_summary[:200]}...")
        logger.info(f"Action: {report.final_recommended_action}")
        
        if report.verification_notes:
            logger.info(f"Verification Notes: {report.verification_notes}")

        # 5. Test Copilot Q&A Mode
        logger.info("\n--- Testing Copilot Q&A Mode ---")
        question = "What is the specific risk for Warfarin in this combination?"
        history = [] # Fresh session
        
        answer = await run_copilot_qa(
            question=question,
            report=report,
            history=history,
            llm_client=llm_client
        )
        
        logger.info(f"Q: {question}")
        logger.info(f"A: {answer}")
        
    except Exception as e:
        logger.error(f"E2E Test Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_full_agentic_flow())
