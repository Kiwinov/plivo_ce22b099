import json
import random
import os
from collections import defaultdict
import glob

# --- CONFIGURATION ---
SOURCES = ["data", "data_llm"]
OUTPUT_DIR = "data_combined"
LABELS = ["CREDIT_CARD", "PHONE", "EMAIL", "PERSON_NAME", "DATE", "CITY", "LOCATION"]

# We want roughly this many entities per class in the final set
# 1000 lines * ~1.5 entities/line = 1500 entities total -> ~200 per class
TARGET_COUNTS = {
    "train": {l: 200 for l in LABELS},
    "dev":   {l: 50 for l in LABELS},
    "test":  {l: 50 for l in LABELS}
}

# Max lines allowed per file
MAX_LINES = {
    "train": 1000,
    "dev": 200,
    "test": 200
}

def load_all_source_data(split_name):
    """Loads and shuffles data from all sources for a specific split (train/dev/test)"""
    pool = []
    for d in SOURCES:
        path = os.path.join(d, f"{split_name}.jsonl")
        if os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    if line.strip():
                        try:
                            pool.append(json.loads(line))
                        except: pass
    random.shuffle(pool)
    return pool

def count_entities(entry):
    """Returns a set of labels present in this sentence"""
    return set(e['label'] for e in entry.get('entities', []))

def select_stratified(pool, target_counts, max_lines):
    selected = []
    # Current counts of entities in our selection
    current_counts = defaultdict(int)
    
    # Sort pool to prioritize "rare" classes? 
    # Actually, greedy approach works well with this much data.
    # We iterate through pool. If a sentence helps us fill a bucket that isn't full, we take it.
    
    # 1. First Pass: Hunt for needed classes
    for entry in pool:
        if len(selected) >= max_lines:
            break
            
        labels_in_line = count_entities(entry)
        if not labels_in_line: continue # Skip empty lines
        
        # Check if this line provides a label we still need
        is_useful = False
        for label in labels_in_line:
            if current_counts[label] < target_counts.get(label, 100):
                is_useful = True
                break
        
        if is_useful:
            selected.append(entry)
            for label in labels_in_line:
                current_counts[label] += 1
                
    # 2. Second Pass (If needed): Fill up to max_lines with whatever (random)
    # just to ensure we hit the 1000 line requirement, even if buckets are full.
    if len(selected) < max_lines:
        remaining_needed = max_lines - len(selected)
        # Filter out already selected
        selected_ids = set(x['id'] for x in selected)
        
        extras = [x for x in pool if x['id'] not in selected_ids]
        selected.extend(extras[:remaining_needed])
        
        # Update counts just for reporting
        for entry in extras[:remaining_needed]:
            for label in count_entities(entry):
                current_counts[label] += 1

    return selected, current_counts

def main():
    random.seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"--- Creating Stratified Dataset in {OUTPUT_DIR} ---")
    
    for split in ["train", "dev", "test"]:
        print(f"\nProcessing '{split}'...")
        
        # 1. Load massive pool
        pool = load_all_source_data(split)
        print(f"  - Source Pool: {len(pool)} lines available")
        
        # 2. Select best examples
        targets = TARGET_COUNTS[split]
        limit = MAX_LINES[split]
        
        final_data, final_counts = select_stratified(pool, targets, limit)
        
        # 3. Write
        out_path = os.path.join(OUTPUT_DIR, f"{split}.jsonl")
        with open(out_path, 'w') as f:
            for entry in final_data:
                f.write(json.dumps(entry) + "\n")
        
        # 4. Report Stats
        print(f"  - Selected: {len(final_data)} lines")
        print("  - Class Distribution:")
        for label in LABELS:
            count = final_counts[label]
            goal = targets[label]
            status = "OK" if count >= goal else "LOW"
            print(f"    {label:<15}: {count:>3} (Goal: {goal}) [{status}]")

if __name__ == "__main__":
    main()
