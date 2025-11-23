import json
import argparse
import torch
import re
from transformers import AutoTokenizer, AutoModelForTokenClassification
from labels import ID2LABEL, label_is_pii
import os


def validate_span(text, label):
    """
    Returns False if the span is likely a False Positive based on rules.
    This acts as a 'Safety Net' to boost Precision.
    """
    clean_text = text.lower().strip()
    
    # 1. EMAIL: Must contain email-like indicators
    if label == "EMAIL":
        # Check for @, 'at', or 'dot'
        if not any(x in clean_text for x in ["@", "at ", "dot"]):
            return False
        # Too short to be a real email
        if len(clean_text) < 5: 
            return False

    # 2. CREDIT_CARD: Filter out Room numbers, Flight numbers, etc.
    if label == "CREDIT_CARD":
        # Count actual digits
        digits = sum(c.isdigit() for c in clean_text)
        # If it's mostly text ("my visa card") without numbers, it's risky.
        # If it's too short (e.g. "402"), it's likely a room/order number.
        if len(clean_text) < 4:
            return False

    # 3. PHONE: Must be long enough
    if label == "PHONE":
        # "911" is valid but rare in this context. "Room 505" is common noise.
        if len(clean_text) < 4: 
            return False

    # 4. PERSON_NAME: Filter out single characters or stopwords
    if label == "PERSON_NAME":
        if len(clean_text) < 2: 
            return False
        if clean_text in ["and", "the", "was", "is", "a", "i"]:
            return False

    return True


def bio_to_spans(text, offsets, label_ids):
    spans = []
    current_label = None
    current_start = None
    current_end = None

    for (start, end), lid in zip(offsets, label_ids):
        # Ignore special tokens (CLS/SEP/PAD) which map to (0,0)
        if start == 0 and end == 0:
            continue
            
        label = ID2LABEL.get(int(lid), "O")
        
        if label == "O":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
                current_label = None
            continue

        prefix, ent_type = label.split("-", 1)
        if prefix == "B":
            if current_label is not None:
                spans.append((current_start, current_end, current_label))
            current_label = ent_type
            current_start = start
            current_end = end
        elif prefix == "I":
            if current_label == ent_type:
                current_end = end
            else:
                # Logic error in sequence (I without B), treat as new B
                if current_label is not None:
                    spans.append((current_start, current_end, current_label))
                current_label = ent_type
                current_start = start
                current_end = end

    if current_label is not None:
        spans.append((current_start, current_end, current_label))

    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", default="out_hypertuned_reg")
    ap.add_argument("--model_name", default=None)
    ap.add_argument("--input", default="data_combined/dev.jsonl")
    ap.add_argument("--output", default="out/dev_pred.json")
    ap.add_argument("--max_length", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32, help="Overlap for sliding window")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    # Load Model & Tokenizer
    model_path = args.model_dir if args.model_name is None else args.model_name
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    model.to(args.device)
    model.eval()

    results = {}

    print(f"Running inference on {args.input}...")
    
    with open(args.input, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            text = obj["text"]
            uid = obj["id"]

            # 1. Sliding Window Tokenization
            # Handles texts longer than max_length by creating overlapping chunks
            encodings = tokenizer(
                text,
                return_offsets_mapping=True,
                truncation=True,
                max_length=args.max_length,
                return_tensors="pt",
                stride=args.stride,
                return_overflowing_tokens=True
            )
            
            # Extract mappings
            offset_mappings = encodings.pop("offset_mapping")
            # We don't need overflow mapping here since we process one line at a time
            _ = encodings.pop("overflow_to_sample_mapping") 
            
            all_spans = []

            # 2. Process each chunk
            for i, input_ids in enumerate(encodings["input_ids"]):
                attention_mask = encodings["attention_mask"][i].unsqueeze(0).to(args.device)
                input_ids = input_ids.unsqueeze(0).to(args.device)
                
                with torch.no_grad():
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
                
                pred_ids = logits.argmax(dim=-1)[0].cpu().tolist()
                offsets = offset_mappings[i].tolist()
                
                # Convert this chunk's BIO tags to spans
                chunk_spans = bio_to_spans(text, offsets, pred_ids)
                all_spans.extend(chunk_spans)

            # 3. Deduplicate and Validate
            # Sliding window might detect the same entity in two chunks.
            # Use a dictionary keyed by (start, end) to keep unique ones.
            unique_spans = {}
            
            for s, e, lab in all_spans:
                # Check Bounds (sanity check)
                if s >= e: continue
                
                # Check Heuristics (Precision Boost)
                span_text = text[s:e]
                if not validate_span(span_text, lab):
                    continue
                
                # Store (overwriting duplicates is fine)
                unique_spans[(s, e)] = lab

            # 4. Format Output
            ents = []
            for (s, e), lab in unique_spans.items():
                ents.append({
                    "start": int(s),
                    "end": int(e),
                    "label": lab,
                    "pii": bool(label_is_pii(lab)),
                })
            
            # Sort by start position for cleanliness
            ents.sort(key=lambda x: x["start"])
            
            results[uid] = ents

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Wrote predictions for {len(results)} utterances to {args.output}")


if __name__ == "__main__":
    main()
