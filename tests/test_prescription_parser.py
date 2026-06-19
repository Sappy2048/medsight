import asyncio
import json
import os
from openai import AsyncOpenAI
# Import your agent and schemas here
from src.agents.prescription_parser import PrescriptionParsingAgent
from src.schemas.prescription_schema import ParsedPrescription

# --- ANSI Colors for Terminal Output ---
class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    RESET = '\033[0m'
    BOLD = '\033[1m'

async def run_single_test(agent, case: dict, index: int):
    """Runs a single test case and prints the result clearly to the terminal."""
    print(f"\n{Colors.HEADER}{Colors.BOLD}--- Test Case {index}: {case['name']} ---{Colors.RESET}")
    print(f"{Colors.BLUE}Input:{Colors.RESET} {case['input_text']}")
    
    try:
        # Call the actual LLM
        result = await agent.parse(case['input_text'])
        
        # Convert Pydantic model to dict for easy viewing
        result_dict = result.model_dump(mode="json")
        
        print(f"{Colors.GREEN}Output:{Colors.RESET}")
        print(json.dumps(result_dict, indent=2))
        
        # Optional: Add custom assertion logic here later
        # if not check_expectations(result_dict, case['expected']):
        #     print(f"{Colors.WARNING}Status: NEEDS REVIEW (Did not match expected exactly){Colors.RESET}")
        # else:
        #     print(f"{Colors.GREEN}Status: PASS{Colors.RESET}")

    except Exception as e:
        print(f"{Colors.FAIL}Error:{Colors.RESET} {str(e)}")


async def main():
    # Ensure you have your API key set in your environment
    client = AsyncOpenAI(api_key=os.getenv("TOGETHER_API_KEY"), base_url="https://api.together.xyz/v1")
    agent = PrescriptionParsingAgent(llm_client=client)

    # We will populate these 10 edge cases based on our discussion
    test_cases = [
        {
        "name": "Relative Dates",
        "input_text": "Rx: Dolo 650 TDS. Prescribed today for a 45yo male."
    },
    {
        "name": "Missing Date Components",
        "input_text": "Start Augmentin 625 BD. Written in Oct 2023."
    },
    {
        "name": "Non-Standard Date Formats",
        "input_text": "Consulted on 12/11/10. Take Azithromycin 500mg OD x 3d."
    },
    {
        "name": "Numbers Native to Brand Name",
        "input_text": "Take Vitamin B12 and Omega 3 capsules OD for a month."
    },
    {
        "name": "Compound Dosages",
        "input_text": "Tab Augmentin 875/125 twice daily for 10 days."
    },
    {
        "name": "Extreme Typos & Shorthand",
        "input_text": "Azythromaecin 500 od 3 days + PCM 650 1-0-1 x 5d"
    },
    {
        "name": "Tapering Doses",
        "input_text": "Prednisolone 40mg OD for 3 days, then 20mg OD for 3 days, then 10mg OD for 3 days."
    },
    {
        "name": "Conditional & Conflicting Durations",
        "input_text": "Take 1 pill of Cetirizine everyday for two weeks, but stop after 5 days if rash goes away."
    },
    {
        "name": "Lifestyle Advice (No Drugs)",
        "input_text": "Patient advised complete bed rest. Drink plenty of water and review CBC reports after 14 days. Age: 32."
    },
    {
        "name": "Multi-Route Topical Chaos",
        "input_text": "Apply Volini gel locally QID and take Volini tab SOS."
    },
    {
    "name": "Non-Prescription / Clinical Query Guardrail",
    "input_text": "What are the common side effects of taking Metformin 500mg daily?"
    }
    ]

    print(f"{Colors.BOLD}Starting LLM Evaluation Run...{Colors.RESET}")
    
    # Running sequentially to respect rate limits and keep terminal output ordered
    for i, case in enumerate(test_cases, start=1):
        await run_single_test(agent, case, i)
        # Small sleep to prevent hitting API rate limits on basic tiers
        await asyncio.sleep(1) 
        
    print(f"\n{Colors.BOLD}{Colors.GREEN}Evaluation Complete!{Colors.RESET}")

if __name__ == "__main__":
    asyncio.run(main())