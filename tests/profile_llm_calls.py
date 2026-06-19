import asyncio
import os
import time
import json
import logging
from typing import Dict, List, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.config import TOGETHER_BASE_URL, TOGETHER_API_KEY

# Import the actual unified graph entry point
from src.agents.graph import run_medsight

# Mock clients for infrastructure nodes not being profiled
class MockQdrantClient: pass

class LLMProfiler:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}

    def record(self, stage: str, duration: float):
        if stage not in self.timings:
            self.timings[stage] = []
        self.timings[stage].append(duration)

    def print_report(self):
        print("\n" + "="*60)
        print(f"{'STAGE':<35} | {'CALLS':<6} | {'AVG (s)':<8} | {'TOTAL (s)':<8}")
        print("-" * 60)
        grand_total = 0
        for stage, times in sorted(self.timings.items()):
            if not times: continue
            avg = sum(times) / len(times)
            total = sum(times)
            grand_total += total
            print(f"{stage:<35} | {len(times):<6} | {avg:<8.3f} | {total:<8.3f}")
        print("-" * 60)
        print(f"{'GRAND TOTAL LLM TIME':<35} | {' ':<6} | {' ':<8} | {grand_total:<8.3f}")
        print("="*60 + "\n")

profiler = LLMProfiler()

def patch_llm_client(client: AsyncOpenAI):
    original_create = client.chat.completions.create

    async def timed_create(*args, **kwargs):
        messages = kwargs.get("messages", [])
        system_content = next((m["content"] for m in messages if m.get("role") == "system"), "")
        user_content = next((m["content"] for m in messages if m.get("role") == "user"), "")
        
        # Dynamically classify the stage by inspecting the prompt signatures
        # This completely avoids async race condition tracking bugs!
        stage = "Unknown Agent"
        if "clinical data extraction engine" in system_content:
            # Parse out source drug name from the user text chunk block
            drug_line = next((line for line in user_content.split("\n") if line.startswith("Source drug:")), "Source drug: Unknown")
            drug_name = drug_line.split(":")[-1].strip()
            stage = f"Agent 3: Extraction ({drug_name})"
        elif "clinical reasoning assistant" in system_content:
            stage = "Agent 4: Temporal Reasoning"
        elif "Prescription Parsing" in system_content or "You are an expert medical co-pilot" in system_content:
            stage = "Agent 0: Parsing"
        elif "patient impact" in system_content.lower():
            stage = "Agent 5: Impact Synthesis"
        elif "synthesize" in system_content.lower():
            stage = "Agent 6: Final Synthesis"

        start = time.perf_counter()
        try:
            return await original_create(*args, **kwargs)
        finally:
            end = time.perf_counter()
            profiler.record(stage, end - start)

    client.chat.completions.create = timed_create

async def run_profiled_pipeline():
    load_dotenv()
    
    llm_client = AsyncOpenAI(
        base_url=TOGETHER_BASE_URL,
        api_key=TOGETHER_API_KEY
    )
    patch_llm_client(llm_client)
    
    raw_input = "Patient (62M) was prescribed Warfarin 5mg OD and Azithromycin 500mg OD for 5 days on 2015-05-12."
    
    # Instantiate mock endpoints for required graph intermediates
    qdrant_client = MockQdrantClient()  # type: ignore
    db_pool = None
    
    print("Launching unified production graph execution...")
    result = await run_medsight(
        raw_input=raw_input,
        llm_client=llm_client,
        qdrant_client=qdrant_client,
        db_pool=db_pool
    )
    print(f"Graph finished with status: {result.get('status')}")
    
    profiler.print_report()

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run_profiled_pipeline())
