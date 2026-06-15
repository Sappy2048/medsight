import httpx
import asyncio
import logging
from services.fda_client import get_past_and_present_labels
from logging import getLogger

logger = getLogger(__name__)

# ─── Main Driver ──────────────────────────────────────────────────────────────

async def main():
    # Example: Let's fetch labels for Lisinopril for a prescription written in 2018
    drug_name = "lisinopril"
    prescription_date = "2018-06-15"

    print(f"🔍 Searching DailyMed for '{drug_name}' (Target Date: {prescription_date})...")
    
    # Use a single client session for connection pooling
    async with httpx.AsyncClient() as client:
        try:
            past_label, present_label = await get_past_and_present_labels(
                drug_name=drug_name,
                prescription_date=prescription_date,
                client=client
            )

            print("\n" + "="*50)
            print("🕰️  PAST LABEL (Active on Prescription Date)")
            print("="*50)
            print(f"SPL ID:         {past_label.spl_id}")
            print(f"Set ID:         {past_label.spl_set_id}")
            print(f"Published Date: {past_label.effective_time}")
            
            # Print a snippet of the Boxed Warning (or whichever section you configured)
            boxed_warning = past_label.sections.dict().get('boxed_warning', 'No boxed warning found.')
            snippet = boxed_warning[:200].replace('\n', ' ') if boxed_warning else "None"
            print(f"Snippet:        {snippet}...")

            print("\n" + "="*50)
            print("🟢 PRESENT LABEL (Current Latest Version)")
            print("="*50)
            print(f"SPL ID:         {present_label.spl_id}")
            print(f"Set ID:         {present_label.spl_set_id}")
            print(f"Published Date: {present_label.effective_time}")

        except Exception as e:
            logger.error(f"Failed to fetch labels: {e}")
            raise

if __name__ == "__main__":
    # Configure basic logging so we can see warnings
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())