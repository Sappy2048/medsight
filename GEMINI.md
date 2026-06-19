SYSTEM PROMPT: Migrate LLM Backend from Groq to Local Ollama

1. Context and Objective

We are migrating the LLM backend of the MedSight drug safety pipeline from Groq API to a local Ollama instance running Llama 3.1 8B. This is strictly to bypass cloud rate limits for testing.
Because Ollama exposes a 100% OpenAI-compatible endpoint, the core extraction logic, prompt structures, and JSON parsing do not need to change.

Your objective is to replace the AsyncGroq client with the AsyncOpenAI client across the codebase and update the configuration variables, without altering any of the pipeline's deterministic logic or async orchestration.

2. Execution Steps

Step 1: Update src/config.py

Remove the Groq configurations and replace them with the Ollama configurations:

Remove: GROQ_MODEL

Add:

# LLM Configuration (Local Ollama via OpenAI Client)
LLM_MODEL = "llama3.1"
OLLAMA_BASE_URL = "http://localhost:11434/v1"
OLLAMA_API_KEY = "ollama"  # Dummy key required by OpenAI client


Step 2: Refactor src/agents/extraction.py

Make the following surgical changes:

Imports: - Remove from groq import AsyncGroq.

Add from openai import AsyncOpenAI.

In the src.config import, swap GROQ_MODEL to LLM_MODEL.

Type Hints & Signatures: - In extract_interactions and _extract_section, change the parameter groq_client: AsyncGroq to llm_client: AsyncOpenAI.

API Call: - In _extract_section, update the API call from groq_client.chat.completions.create(...) to llm_client.chat.completions.create(...).

Change the model=GROQ_MODEL argument to model=LLM_MODEL.

Leave response_format={"type": "json_object"} exactly as it is (Ollama natively supports this).

Step 3: Refactor src/agents/temporal.py

Make the identical surgical changes:

Imports: Swap AsyncGroq for AsyncOpenAI, and GROQ_MODEL for LLM_MODEL.

Type Hints: Swap groq_client: AsyncGroq for llm_client: AsyncOpenAI in compute_temporal_diff and _generate_clinical_reasoning.

API Call: Update groq_client to llm_client and the model argument to LLM_MODEL.

CLI Testing Block: In the if __name__ == "__main__": block at the bottom of the file, replace the Groq client instantiation with:

client = AsyncOpenAI(
    base_url=OLLAMA_BASE_URL,
    api_key=OLLAMA_API_KEY
)


Step 4: Test Suite Updates (tests/run_e2e.py or similar)

Any test scripts that instantiate the LLM client must be updated to import AsyncOpenAI, OLLAMA_BASE_URL, and OLLAMA_API_KEY, and instantiate the client pointing to localhost.

3. Strict Guardrails (DO NOT TOUCH)

Async Logic: DO NOT modify the asyncio.gather() loops. Do not attempt to add semaphores or concurrency limits. Ollama natively handles incoming parallel requests by queuing them safely.

Pydantic/Validation Logic: DO NOT alter the Python logic that parses the JSON output or clamps the severity scores.

Requirements: Remember to tell the user to run pip uninstall groq and pip install openai in their terminal.

Execute these updates across the specified files.