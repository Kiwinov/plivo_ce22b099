import json
import random
import os

# --- Configuration ---
OUTPUT_DIR = "data"
NUM_TRAIN = 2500
NUM_DEV = 250
NUM_TEST = 250 

# Probability of "Clean" (perfect spelling) vs "Noisy" (STT)
PROB_CLEAN = 0.3 

# --- Vocabularies ---
# Names that overlap with Cities/Locations (To force context learning)
AMBIGUOUS_NAMES = ["austin", "sydney", "paris", "london", "brooklyn", "jordan", "alexandria", "victoria", "shannon", "chelsea", "savannah"]

FIRST_NAMES = ["james", "mary", "john", "patricia", "robert", "jennifer", "michael", "linda", "william", "elizabeth", "david", "barbara", "richard", "susan", "joseph", "jessica", "thomas", "sarah", "charles", "karen", "priya", "rahul", "wei", "akira", "mohammed", "fatima", "carlos", "maria", "yuki", "sven", "rohan", "arjun", "mei", "hiro", "sofia", "luca", "oliver", "emma", "liam", "noah", "ava", "elijah", "mateo"] + AMBIGUOUS_NAMES
LAST_NAMES = ["smith", "johnson", "williams", "brown", "jones", "garcia", "miller", "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez", "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin", "mehta", "patel", "kim", "lee", "singh", "chen", "wong", "sato", "tanaka", "gupta", "sharma", "khan", "wu", "zhao"]
CITIES = ["new york", "los angeles", "chicago", "houston", "phoenix", "philadelphia", "san antonio", "san diego", "dallas", "san jose", "london", "paris", "tokyo", "mumbai", "delhi", "shanghai", "singapore", "dubai", "toronto", "sydney", "berlin", "bangalore", "beijing", "seoul", "madrid", "rome", "austin", "miami", "denver", "boston", "seattle"] + AMBIGUOUS_NAMES
LOCATIONS = ["central park", "heathrow airport", "grand central", "eiffel tower", "times square", "golden gate bridge", "hyde park", "marina bay", "burj khalifa", "opera house", "wall street", "broadway", "terminal 4", "main street", "fifth avenue", "hotel california", "empire state building", "buckingham palace", "shinjuku station"]
DOMAINS = ["gmail", "yahoo", "hotmail", "outlook", "icloud", "company", "corporate", "live", "protonmail", "aol", "zoho"]
TLDS = ["com", "net", "org", "co", "uk", "io", "gov", "edu", "co.in", "jp"]
MONTHS = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
FILLERS = ["um", "uh", "like", "actually", "you know", "i mean", "sort of", "hang on", "basically", "literally"]
DIGIT_MAP = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine"}

# --- Noise & STT Simulation ---

STT_HOMOPHONES = {
    "for": "four", "four": "for",
    "to": "two", "two": "to", "too": "two",
    "one": "won",
    "eight": "ate",
    "gmail": ["g mail", "gee mail", "gmale"],
    "yahoo": ["yahu", "ya who"],
    "dot": ["dat", "daut"],
    "at": ["et", "hat"],
    "credit": "crdit",
    "card": "cart",
    "phone": "fone",
    "number": "num",
    "zero": ["oh", "nil"]
}

def introduce_typos(text, error_rate=0.05):
    if random.random() > error_rate: return text
    chars = list(text)
    if len(chars) < 4: return text 
    idx = random.randint(0, len(chars) - 2)
    type_err = random.choice(["swap", "drop", "replace"])
    if type_err == "swap": chars[idx], chars[idx+1] = chars[idx+1], chars[idx]
    elif type_err == "drop": chars.pop(idx)
    elif type_err == "replace": chars[idx] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return "".join(chars)

def simulate_stt_errors(text, clean_mode=False, noise_prob=0.2):
    if clean_mode: return text
    words = text.split()
    new_words = []
    for w in words:
        lower_w = w.lower()
        if lower_w in STT_HOMOPHONES and random.random() < 0.4:
            rep = STT_HOMOPHONES[lower_w]
            choice = random.choice(rep) if isinstance(rep, list) else rep
            new_words.append(choice)
        elif random.random() < noise_prob:
            new_words.append(introduce_typos(w, error_rate=1.0))
        else:
            new_words.append(w)
    return " ".join(new_words)

def get_filler(clean_mode=False):
    if clean_mode: return ""
    return f"{random.choice(FILLERS)}" if random.random() < 0.25 else ""

# --- Entity Generators ---

def digit_to_speech(digit_str, clean_mode=False, complexity=0.5):
    if clean_mode: return digit_str
    output = []
    for char in digit_str:
        if random.random() < complexity:
            val = DIGIT_MAP.get(char, char)
            if char == "0" and random.random() < 0.5: val = "oh"
            output.append(val)
        else:
            output.append(char)
    return " ".join(output)

# --- DECOY Generator (Crucial for Precision) ---
def gen_decoy_number(clean_mode):
    # Generates things like "Room 102", "Flight 99", "Order 5555"
    # These look like numbers but are NOT PII.
    # We return (text, None) meaning no label
    prefix = random.choice(["room", "flight", "order", "ticket", "gate", "channel", "page", "chapter", "seat"])
    nums = str(random.randint(10, 9999))
    
    if clean_mode:
        return f"{prefix} {nums}", None 
    
    # Noisy version
    num_speech = digit_to_speech(nums, False, 0.3)
    return f"{prefix} {num_speech}", None

def gen_credit_card(clean_mode):
    groups = ["".join([str(random.randint(0,9)) for _ in range(4)]) for _ in range(4)]
    if clean_mode: return " ".join(groups), "CREDIT_CARD"
    final_parts = [digit_to_speech(g, False, 0.7) for g in groups]
    return " ".join(final_parts), "CREDIT_CARD"

def gen_phone(clean_mode):
    p1 = "".join([str(random.randint(0,9)) for _ in range(3)])
    p2 = "".join([str(random.randint(0,9)) for _ in range(3)])
    p3 = "".join([str(random.randint(0,9)) for _ in range(4)])
    full_str = p1 + p2 + p3
    if clean_mode:
        if random.random() < 0.5: return f"{p1}-{p2}-{p3}", "PHONE"
        else: return full_str, "PHONE"
    text = digit_to_speech(full_str, False, 0.5)
    return text, "PHONE"

def gen_email(clean_mode):
    name = random.choice(FIRST_NAMES)
    lname = random.choice(LAST_NAMES)
    domain = random.choice(DOMAINS)
    tld = random.choice(TLDS)
    if clean_mode: return f"{name}.{lname}@{domain}.{tld}".lower(), "EMAIL"
    sep_word = random.choice(["dot", "point", "underscore", ""])
    user = f"{name} {sep_word} {lname}".strip()
    at = random.choice(["at", "@"])
    dom = f"{domain} {random.choice(['dot', '.'])} {tld}"
    full_email = f"{user} {at} {dom}"
    return simulate_stt_errors(full_email, False, 0.1), "EMAIL"

def gen_date(clean_mode):
    month = random.choice(MONTHS)
    day = random.randint(1, 31)
    year = random.randint(1980, 2030)
    if clean_mode: return f"{month} {day}, {year}", "DATE"
    style = random.choice(["standard", "ordinal", "short"])
    if style == "standard":
        d_str = digit_to_speech(str(day), False, 0.4)
        text = f"{month} {d_str} {year}"
    elif style == "ordinal":
        suffix = "th"
        d_str = f"{day}{suffix}" if random.random() < 0.5 else str(day)
        text = f"the {d_str} of {month}"
    else:
        text = f"{month} {day}"
    return simulate_stt_errors(text, False, 0.05), "DATE"

def gen_person(clean_mode):
    name = f"{random.choice(FIRST_NAMES)} {random.choice(LAST_NAMES)}"
    if not clean_mode and random.random() < 0.1:
        name = introduce_typos(name, error_rate=1.0)
    return name.lower(), "PERSON_NAME"

def gen_city(clean_mode):
    city = random.choice(CITIES)
    if not clean_mode and random.random() < 0.05: 
        city = introduce_typos(city, error_rate=1.0)
    return city.lower(), "CITY"

def gen_location(clean_mode):
    loc = random.choice(LOCATIONS)
    if not clean_mode and random.random() < 0.05: 
        loc = introduce_typos(loc, error_rate=1.0)
    return loc.lower(), "LOCATION"

# --- Main Generator ---

def generate_utterance(idx, id_prefix):
    is_clean = random.random() < PROB_CLEAN
    
    # Up to 5 entities per utterance for density
    num_ents = random.choices([1, 2, 3, 4, 5], weights=[0.2, 0.3, 0.3, 0.1, 0.1])[0]
    
    # Define generators and WEIGHTS to prioritize PII over Locations/Cities
    # Also includes DECOY numbers to improve precision
    generators = [gen_person, gen_city, gen_location, gen_email, gen_phone, gen_credit_card, gen_date, gen_decoy_number]
    # Weights:    Name,       City,     Loc,          Email,     Phone,     CC,              Date,     Decoy
    weights    = [0.18,       0.05,     0.05,         0.15,      0.15,      0.15,            0.15,     0.12]
    
    full_text_parts = [] 
    entities_list = []
    
    # Robust Text Appender
    def add_part(txt, is_entity=False, label=None):
        if not txt: return

        # Apply noise if it's NOT an entity 
        # (Entities/Decoys handle their own noise internally)
        final_txt = txt
        if not is_entity:
            final_txt = simulate_stt_errors(txt, clean_mode=is_clean, noise_prob=0.1)
            
        # Determine Separator
        sep = ""
        if len(full_text_parts) > 0:
            sep = " "
        if final_txt in [".", ",", "?"]: 
            sep = "" 
            
        # Calculate Start/End
        current_full_str = "".join(full_text_parts)
        start_idx = len(current_full_str) + len(sep)
        end_idx = start_idx + len(final_txt)
        
        # Append
        full_text_parts.append(sep + final_txt)
        
        # Only record valid entities (Label != None)
        # Decoys return Label=None, so they get added to text but NOT to entities list
        if is_entity and label is not None:
            entities_list.append({
                "start": start_idx,
                "end": end_idx,
                "label": label
            })

    # --- Build Sentence ---
    
    if random.random() < 0.4:
        add_part(random.choice(["hi", "hello", "ok", "yeah", "hey"]))
    if random.random() < 0.4:
        add_part(random.choice(["my details are", "please update", "record this", "i am"]))

    for i in range(num_ents):
        # Select weighted generator
        gen = random.choices(generators, weights=weights, k=1)[0]
        ent_text, ent_label = gen(is_clean)
        
        # Connector
        if i > 0:
            conn = random.choice(["and", "also", "then", "plus", "wait"])
            if is_clean: conn += "," 
            
            # Context bridging based on label
            if ent_label == "PHONE": conn += " phone is"
            elif ent_label == "EMAIL": conn += " email"
            elif ent_label == "CREDIT_CARD": conn += " card number"
            elif ent_label == "DATE": conn += " date"
            elif ent_label == "CITY": conn += " in"
            # Decoy context
            elif ent_label is None: conn += " check"
            
            add_part(conn)
        elif i == 0:
            # First entity context
            if ent_label == "PHONE": add_part("my number is")
            elif ent_label == "EMAIL": add_part("email address")
            elif ent_label == "PERSON_NAME": add_part("my name is")
            elif ent_label == "CITY": add_part("living in")

        # Filler
        filler = get_filler(is_clean)
        if filler: add_part(filler.strip())
        
        # The Entity (or Decoy)
        add_part(ent_text, is_entity=True, label=ent_label)
        
    if random.random() < 0.3:
        add_part(random.choice(["thanks", "bye", "is that correct", "please"]))

    full_string = "".join(full_text_parts)
    
    return {
        "id": f"{id_prefix}_{idx:04d}",
        "text": full_string,
        "entities": entities_list
    }

def write_jsonl(data, filepath):
    with open(filepath, 'w') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")
    print(f"Saved {len(data)} lines to {filepath}")

# --- Execution ---
if __name__ == "__main__":
    random.seed(42) 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("Generating Training Data...")
    train_data = [generate_utterance(i, "utt_train") for i in range(NUM_TRAIN)]
    write_jsonl(train_data, os.path.join(OUTPUT_DIR, "train.jsonl"))
    
    print("Generating Dev Data...")
    dev_data = [generate_utterance(i, "utt_dev") for i in range(NUM_DEV)]
    write_jsonl(dev_data, os.path.join(OUTPUT_DIR, "dev.jsonl"))
    
    print("Generating Test Data...")
    test_data = [generate_utterance(i, "utt_test") for i in range(NUM_TEST)]
    for t in test_data:
        del t["entities"]
    write_jsonl(test_data, os.path.join(OUTPUT_DIR, "test.jsonl"))
