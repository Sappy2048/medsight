import requests
import os
from src.config import OLLAMA_BASE_URL, LLM_MODEL

url = f"{OLLAMA_BASE_URL}/models"

try:
    response = requests.get(url)
    response.raise_for_status()
    print(f"✅ Local Ollama is reachable at {OLLAMA_BASE_URL}")
    print(f"Models available: {[m['id'] for m in response.json().get('data', [])]}")
    print(f"Active project model: {LLM_MODEL}")
except Exception as e:
    print(f"❌ Failed to reach Ollama: {e}")
    print(f"Ensure Ollama is running and Llama 3.1 is pulled: 'ollama pull {LLM_MODEL}'")
