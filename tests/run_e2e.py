import asyncio
import os
import logging
import json
from datetime import datetime

import httpx
from openai import AsyncOpenAI
from dotenv import load_dotenv

from src.agents.prescription_parser import PrescriptionParsingAgent
from src.agents.resolution import resolve_prescription
from src.services.fda_client import get_past_and_present_labels
from src.agents.extraction import extract_interactions
from src.agents.temporal import compute_temporal_diff
from src.schemas.resolution_schema import ResolvedDrug
from src.config import OLLAMA_BASE_URL, OLLAMA_API_KEY, LLM_MODEL

# ─── Formatting Helpers ────────────────────────────────────────────────────────

class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def print_step(title):
    print(f"\n{Colors.BOLD}{Colors.OKBLUE}=== {title} ==={Colors.ENDC}")

def print_success(msg):
    print(f"{Colors.OKGREEN}✔ {msg}{Colors.ENDC}")

def print_info(msg):
    print(f"{Colors.OKCYAN}ℹ {msg}{Colors.ENDC}")

def print_warning(msg):
    print(f"{Colors.WARNING}⚠ {msg}{Colors.ENDC}")

def print_error(msg):
    print(f"{Colors.FAIL}✘ {msg}{Colors.ENDC}")

# ─── Core E2E Logic ───────────────────────────────────────────────────────────

async def run_e2e_pipeline():
    load_dotenv()
    
    llm_client = AsyncOpenAI(
        base_url=OLLAMA_BASE_URL,
        api_key=OLLAMA_API_KEY
    )
    
    # Standard configuration for the demo
    # Warfarin + Azithromycin (Classic Interaction)
    raw_input = "Patient prescribed Warfarin and Azithromycin in May 2015"
    prescription_date = "2015-05-15"
    
    print_step(f"INPUT: '{raw_input}'")
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            # ── Agent 0: Parsing ──────────────────────────────────────────────
            print_step("Agent 0: Prescription Parsing")
            parser = PrescriptionParsingAgent(llm_client)
            parsed_prescription = await parser.parse(raw_input)
            print_success(f"Extracted {len(parsed_prescription.drugs)} drugs.")
            for drug in parsed_prescription.drugs:
                print_info(f"Found: {drug.target_brand_name} (Dose: {drug.prescribed_dose})")

            # ── Agent 1: Resolution ───────────────────────────────────────────
            print_step("Agent 1: Drug Resolution")
            resolved_drugs = await resolve_prescription(parsed_prescription, http_client)
            print_success(f"Resolved {len(resolved_drugs)} drugs to canonical generics.")
            
            for drug in resolved_drugs:
                res_info = f"{drug.raw_prescription_input} -> {', '.join(drug.generic_names)}"
                if drug.formulated_strength:
                    res_info += f" [{drug.formulated_strength}]"
                print_info(res_info)
                print_info(f"   RxCUIs: {', '.join(drug.rxcui_list)}")

            # ── Agent 2 & 3: Temporal Analysis ───────────────────────────────
            print_step("Agent 2 & 3: FDA Label Extraction & Temporal Diff")
            
            # Flatten generics for cross-interaction checking
            all_generics = []
            for drug in resolved_drugs:
                all_generics.extend(drug.generic_names)
            
            # We check interactions for each resolved drug against all others
            for i, source_resolved in enumerate(resolved_drugs):
                source_primary_generic = source_resolved.generic_names[0]
                
                print(f"\n{Colors.BOLD}Analyzing interactions for: {source_primary_generic}{Colors.ENDC}")
                
                # Small sleep to avoid rate limiting on consecutive label extractions
                # (Ollama is local but connection pool management is still good practice)
                await asyncio.sleep(0.5)
                
                # 1. Fetch labels (Past and Present)
                past_label, present_label = await get_past_and_present_labels(
                    source_primary_generic, 
                    prescription_date, 
                    http_client
                )
                print_success(f"Fetched historical FDA labels for {source_primary_generic}")
                
                # 2. Extract structured interactions from both
                past_ext = await extract_interactions(past_label, source_primary_generic, llm_client)
                present_ext = await extract_interactions(present_label, source_primary_generic, llm_client)
                print_success(f"Extracted interactions from {len(past_ext.interactions)} (past) and {len(present_ext.interactions)} (present) versions.")

                # 3. Compute Temporal Diffs against other generics
                other_generics = [g for g in all_generics if g not in source_resolved.generic_names]
                
                for target_generic in other_generics:
                    print_info(f"Checking diff vs {target_generic}...")
                    diff, reasoning = await compute_temporal_diff(
                        past_ext, 
                        present_ext, 
                        target_generic, 
                        llm_client, 
                        prescription_date
                    )
                    
                    if diff.is_clinically_significant:
                        print_warning(f"SIGNIFICANT CHANGE: {diff.drug_pair}")
                        print(f"   Change: {diff.change_type} (Delta: {diff.severity_delta})")
                        print(f"   Reason: {reasoning['clinical_reasoning']}")
                    else:
                        print_info(f"   No significant change for {diff.drug_pair}")

        except Exception as e:
            print_error(f"Pipeline crashed: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run_e2e_pipeline())
