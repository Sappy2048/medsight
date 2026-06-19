import asyncio
import os
import time
import json
import logging
from typing import Dict, List
from openai import AsyncOpenAI
import httpx
from dotenv import load_dotenv

# Import MedSight Agents
from src.agents.prescription_parser import PrescriptionParsingAgent
from src.agents.resolution import resolve_prescription
from src.services.fda_client import get_past_and_present_labels
from src.agents.extraction import extract_interactions
from src.agents.temporal import compute_temporal_diff
from src.agents.impact import analyze_patient_impact
from src.agents.synthesis import synthesize_final_report
from src.config import OLLAMA_BASE_URL, OLLAMA_API_KEY

# ─── Profiling Utility ────────────────────────────────────────────────────────

class LLMProfiler:
    def __init__(self):
        self.timings: Dict[str, List[float]] = {}
        self.current_stage: str = "Unknown"

    def set_stage(self, stage: str):
        self.current_stage = stage
        if stage not in self.timings:
            self.timings[stage] = []

    def record(self, duration: float):
        self.timings[self.current_stage].append(duration)

    def print_report(self):
        print("\n" + "="*60)
        print(f"{'STAGE':<30} | {'CALLS':<6} | {'AVG (s)':<8} | {'TOTAL (s)':<8}")
        print("-" * 60)
        
        grand_total = 0
        for stage, times in self.timings.items():
            if not times: continue
            avg = sum(times) / len(times)
            total = sum(times)
            grand_total += total
            print(f"{stage:<30} | {len(times):<6} | {avg:<8.3f} | {total:<8.3f}")
        
        print("-" * 60)
        print(f"{'GRAND TOTAL LLM TIME':<30} | {' ':<6} | {' ':<8} | {grand_total:<8.3f}")
        print("="*60 + "\n")

profiler = LLMProfiler()

def patch_llm_client(client: AsyncOpenAI):
    """
    Monkey-patches the OpenAI client to record timing for every completion call.
    """
    original_create = client.chat.completions.create

    async def timed_create(*args, **kwargs):
        start = time.perf_counter()
        try:
            result = await original_create(*args, **kwargs)
            return result
        finally:
            end = time.perf_counter()
            profiler.record(end - start)

    client.chat.completions.create = timed_create

# ─── Execution ───────────────────────────────────────────────────────────────

async def run_profiled_pipeline():
    load_dotenv()
    
    llm_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY
    )
    patch_llm_client(llm_client)
    
    raw_input = "Patient prescribed Warfarin and Azithromycin in March 2010"
    prescription_date = "2010-03-15"
    
    async with httpx.AsyncClient(timeout=60.0) as http_client:
        # 1. Parsing
        profiler.set_stage("Agent 0: Parsing")
        parser = PrescriptionParsingAgent(llm_client)
        parsed_rx = await parser.parse(raw_input)

        # 2. Resolution (Mostly deterministic, but timing just in case)
        profiler.set_stage("Agent 1: Resolution")
        resolved_drugs = await resolve_prescription(parsed_rx, http_client)

        # 3. Extraction & Temporal
        all_diffs = []
        all_generics = []
        for drug in resolved_drugs:
            all_generics.extend(drug.generic_names)

        for source_resolved in resolved_drugs:
            source_primary = source_resolved.generic_names[0]
            past_label, present_label = await get_past_and_present_labels(
                source_primary, prescription_date, http_client
            )

            # Extraction
            profiler.set_stage(f"Agent 3: Extraction ({source_primary})")
            past_ext, present_ext = await asyncio.gather(
                extract_interactions(past_label, source_primary, llm_client),
                extract_interactions(present_label, source_primary, llm_client)
            )

            # Temporal Diff Reasoning
            profiler.set_stage(f"Agent 4: Temporal Reasoning")
            other_generics = [g for g in all_generics if g not in source_resolved.generic_names]
            for target in other_generics:
                diff, reasoning = await compute_temporal_diff(
                    past_ext, present_ext, target, llm_client, prescription_date
                )
                all_diffs.append((diff, reasoning))

        # 4. Impact
        profiler.set_stage("Agent 5: Impact Synthesis")
        report = await analyze_patient_impact(
            all_diffs, resolved_drugs, prescription_date, llm_client
        )

        # 5. Final Synthesis
        profiler.set_stage("Agent 6: Final Synthesis")
        flat_diffs = [d for d, r in all_diffs]
        await synthesize_final_report(report, flat_diffs, llm_client)

    profiler.print_report()

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run_profiled_pipeline())
