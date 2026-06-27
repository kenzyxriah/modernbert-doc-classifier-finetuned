import asyncio
import pandas as pd
from pydantic import BaseModel, Field
import random
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lib.chat import structured_output
from lib.file_utils import cap_doc_text
from datasets import load_dataset

class GeneratedDocument(BaseModel):
    document_text: str = Field(
        ..., 
        min_length=700, 
        max_length=1500
    )

async def generate_single_doc(system_prompt:str ,category: str) -> str:
    prompts = {
        "Public": "Generate a cohesive public document. Press release, external blog post, or public company announcement. STRICT LENGTH: exactly between 700 and 1500 characters.",
        "Internal": "Generate a cohesive internal company document. HR manual, company-wide policy, meeting notes. STRICT LENGTH: exactly between 700 and 1500 characters.",
        "Confidential": "Generate a cohesive confidential corporate document. Q3 financial report, M&A target assessment. Include realistic fake metrics. STRICT LENGTH: exactly between 700 and 1500 characters.",
        "Restricted": "Generate a cohesive highly restricted IT or security document. Disaster recovery plan, incident response playbook. STRICT LENGTH: exactly between 700 and 1500 characters."
    }
    
    try:
        result: dict = await structured_output(
            system_prompt=system_prompt,
            user_msg=prompts[category],
            response_format=GeneratedDocument,
            temperature=0.8,
            max_tokens=3000
        )
        text = result.get("document_text", "")
        if not text:
            return ""
        
        capped_text = await cap_doc_text(text, max_tokens=1500)
        return capped_text
    except Exception as e:
        print(f"Error generating doc for {category}: {e}")
        return ""

async def generate_llm_data(total_per_category=200, batch_size=49):
    categories = ["Public", "Internal", "Confidential", "Restricted"]
    system_prompt="You are an expert technical writer and corporate document generator."
    
    all_rows = []
    
    for category in categories:
        print(f"\n--- Generating LLM data for {category} ---")
        generated_count = 0
        while generated_count < total_per_category:
            current_batch = min(batch_size, total_per_category - generated_count)
            print(f"Executing batch of {current_batch} parallel requests...")
            
            tasks = [generate_single_doc(system_prompt, category) for _ in range(current_batch)]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            success_count = 0
            for res in results:
                if isinstance(res, str) and len(res) > 100:
                    all_rows.append((res, category))
                    generated_count += 1
                    success_count += 1
            
            print(f"Batch completed: {success_count} successful. Total for {category}: {generated_count}/{total_per_category}.")
            
            if generated_count < total_per_category:
                print("Waiting 60 seconds to respect rate limits before next batch...")
                await asyncio.sleep(60)
                
    return all_rows

async def fetch_public_datasets():
    print("\n--- Fetching supplemental public datasets from HuggingFace ---")
    rows = []
    
    try:
        dataset = load_dataset("wikitext", "wikitext-2-v1", split="train")
        long_articles = [row['text'] for row in dataset if len(row['text']) > 1500]
        random.shuffle(long_articles)
        for text in long_articles[:200]:
            capped = await cap_doc_text(text, max_tokens=1000)
            rows.append((capped, "Public"))
        print(f"Successfully fetched {len(rows)} Public dataset documents.")
    except Exception as e:
        print(f"Error fetching wikitext: {e}")

    try:
        dataset = load_dataset("aeslc", split="train")
        long_emails = [row['email_body'] for row in dataset if len(row['email_body']) > 700]
        random.shuffle(long_emails)
        fetched = 0
        for text in long_emails[:200]:
            capped = await cap_doc_text(text, max_tokens=1000)
            rows.append((capped, "Internal"))
            fetched += 1
        print(f"Successfully fetched {fetched} Internal dataset documents.")
    except Exception as e:
        print(f"Error fetching aeslc: {e}")


    return rows
    
async def main():
    csv_file = "scratch/training_data.csv"
    
    llm_rows = await generate_llm_data(total_per_category=300, batch_size=49)
    hf_rows = await fetch_public_datasets()
    
    all_data = llm_rows + hf_rows
    random.shuffle(all_data)
    
    if all_data:
        df = pd.DataFrame(all_data, columns=['text', 'label'])
        df.to_csv(csv_file, mode='a', index=False, header=False)
        print(f"\nSUCCESS! Appended {len(all_data)} documents to {csv_file}")
    else:
        print("\nNo data was generated.")

if __name__ == "__main__":
    asyncio.run(main())
