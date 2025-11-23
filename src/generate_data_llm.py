import os
import json
import random
import time
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
import google.generativeai as genai
from google.api_core import exceptions as google_exceptions
from threading import Lock

# --- CONFIGURATION ---

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    print("CRITICAL ERROR: GOOGLE_API_KEY environment variable not set.")
    exit(1)

genai.configure(api_key=GOOGLE_API_KEY)

# UPDATED: Using actual available public models to ensure generation works
MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
]

TARGET_COUNTS = {
    "train": 2500,
    "dev": 250,
    "test": 250
}

BATCH_SIZE = 20    
MAX_WORKERS = 5    # Parallel threads
OUTPUT_DIR = "data_llm"

# Global lock to ensure threads write to file safely without corruption
file_lock = Lock()

# --- PROMPTS ---

SYSTEM_PROMPT = """
You are a data generator for a Named Entity Recognition (NER) model. 
Your task is to generate realistic, noisy Speech-to-Text (STT) transcripts containing Personal Identifiable Information (PII).
"""

PROMPT_VARIATIONS = [
    """
    Generate {n} distinct sentences typical of a banking or payment call.
    Annotate: <CREDIT_CARD>, <PHONE>, <EMAIL>, <PERSON_NAME>, <DATE>, <CITY>, <LOCATION>.
    STT NOISE RULES: Lowercase, no punctuation. Mix digits/words. Decoys: Do NOT tag amounts.
    EXAMPLES:
    pay with card <CREDIT_CARD>4111 2222</CREDIT_CARD> thanks
    charge 500 dollars to visa <CREDIT_CARD>1234</CREDIT_CARD>
    Output only raw lines.
    """,
    """
    Generate {n} distinct sentences regarding travel bookings.
    Annotate: <CREDIT_CARD>, <PHONE>, <EMAIL>, <PERSON_NAME>, <DATE>, <CITY>, <LOCATION>.
    STT NOISE RULES: Lowercase. Spoken dates. Decoys: Do NOT tag flight numbers.
    EXAMPLES:
    flying to <CITY>paris</CITY> on <DATE>march 10th</DATE> flight 99
    pick me up at <LOCATION>heathrow</LOCATION> terminal 5
    Output only raw lines.
    """,
    """
    Generate {n} distinct sentences for identity verification.
    Annotate: <CREDIT_CARD>, <PHONE>, <EMAIL>, <PERSON_NAME>, <DATE>, <CITY>, <LOCATION>.
    STT NOISE RULES: Complex emails ("dot", "at"). Spaced digits. Dysfluencies ("um").
    EXAMPLES:
    email is <EMAIL>john dot doe at gmail dot com</EMAIL>
    call <PHONE>07700 900 123</PHONE>
    Output only raw lines.
    """,
    """
    Generate {n} sentences focusing on names that sound like places or things.
    Annotate: <CREDIT_CARD>, <PHONE>, <EMAIL>, <PERSON_NAME>, <DATE>, <CITY>, <LOCATION>.
    FOCUS: Names vs Cities ("austin" vs "austin"). Decoy Numbers ("room 101").
    EXAMPLES:
    is <PERSON_NAME>austin</PERSON_NAME> in <CITY>austin</CITY>
    ticket 405 for <PERSON_NAME>sydney</PERSON_NAME> not <CITY>sydney</CITY>
    Output only raw lines.
    """,
    """
    Generate {n} complex sentences where the user corrects themselves.
    Annotate: <CREDIT_CARD>, <PHONE>, <EMAIL>, <PERSON_NAME>, <DATE>, <CITY>, <LOCATION>.
    STT NOISE RULES: Corrections ("no wait"). Lists.
    EXAMPLES:
    change <PERSON_NAME>bob</PERSON_NAME> to <PERSON_NAME>robert</PERSON_NAME>
    moved from <CITY>london</CITY> to <CITY>ny</CITY> on <DATE>may 1st</DATE>
    Output only raw lines.
    """
]

TAG_RE = re.compile(r"<(CREDIT_CARD|PHONE|EMAIL|PERSON_NAME|DATE|CITY|LOCATION)>(.*?)</\1>")

def parse_tagged_line(line_raw):
    line = line_raw.strip()
    if not line or "<" not in line or ">" not in line: return None 

    entities = []
    clean_text = ""
    last_pos = 0

    for match in TAG_RE.finditer(line):
        pre_text = line[last_pos:match.start()]
        start_idx = len(clean_text) + len(pre_text)
        label = match.group(1)
        content = match.group(2)
        clean_text += pre_text + content
        end_idx = len(clean_text)
        entities.append({"start": start_idx, "end": end_idx, "label": label})
        last_pos = match.end()

    clean_text += line[last_pos:]
    if "<" in clean_text or ">" in clean_text: return None 
    return clean_text, entities

def fetch_batch_gemini(count):
    """
    Fetches a batch of data. Returns list of strings.
    """
    if not MODELS: return []
    model_name = random.choice(MODELS)
    prompt = random.choice(PROMPT_VARIATIONS).format(n=count)
    
    try:
        model = genai.GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT)
        # Set a timeout to prevent hanging indefinitely
        response = model.generate_content(prompt, request_options={"timeout": 30})
        return response.text.strip().split("\n")
    except google_exceptions.ResourceExhausted:
        time.sleep(10) # Backoff
        return []
    except Exception:
        # Swallow other errors to keep thread alive, return empty
        return []

def generate_split_parallel(split_name, target_count, global_id_counter):
    filepath = f"{OUTPUT_DIR}/{split_name}.jsonl"
    
    # 1. Resume Logic: Count lines in existing file
    current_count = 0
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            current_count = sum(1 for _ in f)
    
    # Adjust ID counter
    local_id_counter = global_id_counter + current_count
    print(f"[{split_name}] Resuming from {current_count}/{target_count}")
    
    pbar = tqdm(total=target_count, initial=current_count, desc=f"Generating {split_name}")
    
    # 2. Open file in APPEND mode
    with open(filepath, "a") as f_out:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            
            futures = []
            
            # Fill the queue
            while current_count < target_count:
                # Clean up finished futures
                done_indices = []
                for i, f in enumerate(futures):
                    if f.done():
                        done_indices.append(i)
                        try:
                            raw_lines = f.result()
                            for line in raw_lines:
                                if current_count >= target_count: break
                                
                                parsed = parse_tagged_line(line)
                                if parsed:
                                    text, entities = parsed
                                    entry = {
                                        "id": f"utt_llm_{split_name}_{local_id_counter}",
                                        "text": text,
                                        "entities": entities
                                    }
                                    if split_name == "test": del entry["entities"]
                                    
                                    # 3. INCREMENTAL SAVE
                                    with file_lock:
                                        f_out.write(json.dumps(entry) + "\n")
                                        f_out.flush() # Ensure it hits disk
                                        
                                    local_id_counter += 1
                                    current_count += 1
                                    pbar.update(1)
                        except Exception:
                            pass # Ignore failed futures
                
                # Remove finished futures from list (reverse order to avoid index shift)
                for i in sorted(done_indices, reverse=True):
                    futures.pop(i)

                # Add new tasks if we need more data and have capacity
                needed = target_count - current_count
                active_tasks = len(futures)
                if active_tasks < MAX_WORKERS and needed > 0:
                    futures.append(executor.submit(fetch_batch_gemini, BATCH_SIZE))
                else:
                    # Prevent busy loop if waiting for tasks
                    time.sleep(0.5)
            
            # Wait for remaining valid tasks
            wait(futures, timeout=10)

    pbar.close()
    return local_id_counter

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    global_cnt = 0
    
    print(f"Starting Robust Generation using: {MODELS}")
    
    # Train
    global_cnt = generate_split_parallel("train", TARGET_COUNTS["train"], global_cnt)
    # Dev
    global_cnt = generate_split_parallel("dev", TARGET_COUNTS["dev"], global_cnt)
    # Test
    global_cnt = generate_split_parallel("test", TARGET_COUNTS["test"], global_cnt)

    print("\n--- Generation Complete ---")

if __name__ == "__main__":
    main()
