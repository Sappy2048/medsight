import asyncio
import os
import logging
import json
from datetime import datetime

import httpx
from groq import AsyncGroq
from dotenv import load_dotenv

from src.agents.prescription_parser import PrescriptionParsingAgent
from src.agents.resolution import resolve_prescription
from src.services.fda_client import get_past_and_present_labels
from src.agents.extraction import extract_interactions
from src.agents.temporal import compute_temporal_diff
from src.schemas.resolution_schema import ResolvedDrug

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
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print_error("GROQ_API_KEY not found in environment. Please check your .env file.")
        return

    groq_client = AsyncGroq(api_key=api_key)
    
    # Standard configuration for the demo
    # Warfarin + Azithromycin (Classic Interaction)
    raw_input = "Patient prescribed Warfarin and Azithromycin in May 2015"
    prescription_date = "2015-05-15"
    
    print_step(f"INPUT: '{raw_input}'")
    
    async with httpx.AsyncClient(timeout=30.0) as http_client:
        try:
            # ── Agent 0: Parsing ──────────────────────────────────────────────
            print_step("Agent 0: Prescription Parsing")
            parser = PrescriptionParsingAgent(groq_client)
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
                await asyncio.sleep(2)
                
                try:
                    # Fetch past and present labels
                    past_label, present_label = await get_past_and_present_labels(
                        source_primary_generic, prescription_date, http_client
                    )
                    
                    print_info(f"FDA Label: {past_label.spl_set_id}")
                    print_info(f"  Past Version:    {past_label.effective_time}")
                    print_info(f"  Present Version: {present_label.effective_time}")

                    # Extract interactions
                    print_info("Extracting interaction records via LLM...")
                    past_extraction, present_extraction = await asyncio.gather(
                        extract_interactions(past_label, source_primary_generic, groq_client),
                        extract_interactions(present_label, source_primary_generic, groq_client)
                    )
                    
                    print_info(f"  Extracted {len(past_extraction.interactions)} records from past label.")
                    print_info(f"  Extracted {len(present_extraction.interactions)} records from present label.")
                    
                    # Check against all other generics in the prescription
                    other_generics = [g for g in all_generics if g not in source_resolved.generic_names]
                    
                    for target_generic in other_generics:
                        print(f"\n  {Colors.UNDERLINE}Diff: {source_primary_generic} + {target_generic}{Colors.ENDC}")
                        
                        diff_result, reasoning = await compute_temporal_diff(
                            past_extraction, present_extraction, target_generic, groq_client, prescription_date
                        )
                        
                        # Display Results
                        status_color = Colors.OKGREEN
                        if diff_result.is_clinically_significant:
                            status_color = Colors.FAIL
                        elif diff_result.change_type != "UNCHANGED":
                            status_color = Colors.WARNING
                            
                        print(f"  Change Type:    {status_color}{diff_result.change_type}{Colors.ENDC}")
                        print(f"  Severity Delta: {status_color}{diff_result.severity_delta}{Colors.ENDC}")
                        print(f"  Significant:    {status_color}{diff_result.is_clinically_significant}{Colors.ENDC}")
                        
                        if diff_result.past_severity_score is not None:
                            print(f"  Past Score:     {diff_result.past_severity_score} ({past_label.effective_time})")
                        if diff_result.present_severity_score is not None:
                            print(f"  Present Score:  {diff_result.present_severity_score} (Current)")
                            
                        print(f"\n  {Colors.BOLD}Clinical Reasoning:{Colors.ENDC}")
                        print(f"  {reasoning['clinical_reasoning']}")
                        if reasoning.get('key_concern'):
                            print(f"  {Colors.BOLD}Key Concern:{Colors.ENDC} {reasoning['key_concern']}")

                except Exception as e:
                    print_error(f"Failed to process {source_primary_generic}: {str(e)}")

            print_step("PIPELINE EXECUTION COMPLETE")

        except httpx.TimeoutException:
            print_error("API Request Timed Out. Please check your network connection.")
        except Exception as e:
            print_error(f"Pipeline crashed: {str(e)}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    # Ensure logs are not too noisy for the demo
    logging.basicConfig(level=logging.ERROR)
    asyncio.run(run_e2e_pipeline())
