import os
import re
import time
import urllib.request
import json
import random
from concurrent.futures import ThreadPoolExecutor

INPUT_DIR = r"c:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\claude_judge_pkg3"
OUTPUT_DIR = os.path.join(INPUT_DIR, "outputs")
KEY_FILE = r"c:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\openai_key.txt"

SYS_PROMPT = (
    "Sen Trendyol arama-alaka uzmanısın. Her (sorgu | ürün) çifti için ürünün "
    "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURALLAR: Sorgu marka "
    "ise o markanın her ürünü 1. Sorgu kategori ise o kategorideki her ürün 1. "
    "Ürün tipi aynı veya yakın kullanım amaçlıysa 1 — renk/beden/model/marka "
    "farkı ÖNEMSİZ, ikame ürünler de 1. SADECE bambaşka bir ihtiyaca yönelik "
    "ürün 0. Kararsız kalırsan 1 ver. Sadece 'numara. 0' veya 'numara. 1', "
    "her çift TEK satır, başka hiçbir şey yazma."
)

def get_key():
    return open(KEY_FILE, encoding="utf-8").read().strip()

def call_openai_gpt_4o_mini(key, query, product_text):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {key}"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": SYS_PROMPT},
            {"role": "user", "content": f"Çiftler:\n1. {query} | {product_text}"}
        ],
        "temperature": 0.0,
        "max_tokens": 6
    }
    
    body = json.dumps(data).encode("utf-8")
    
    # Retry loop with exponential backoff and rate limit handling
    max_retries = 10
    backoff = 2
    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(url, data=body, headers=headers)
            with urllib.request.urlopen(req, timeout=20) as resp:
                res = json.loads(resp.read().decode("utf-8"))
                txt = res["choices"][0]["message"]["content"].strip()
                
                # Extract label
                digits = re.findall(r'[0-9]+', txt)
                if digits:
                    val = int(digits[-1])
                    if val in (0, 1):
                        return val
                matches = re.findall(r'[01]', txt)
                if matches:
                    return int(matches[-1])
                return 1
        except urllib.error.HTTPError as e:
            if e.code == 429:
                # Rate limit hit
                retry_after = e.headers.get("Retry-After")
                sleep_time = float(retry_after) if retry_after else (backoff + random.random() * 2)
                print(f"Rate limit hit for '{query[:30]}'. Sleeping for {sleep_time:.2f} seconds... (Attempt {attempt+1}/{max_retries})")
                time.sleep(sleep_time)
                backoff *= 2
            else:
                print(f"HTTP Error {e.code} for '{query[:30]}': {e}. Retrying... (Attempt {attempt+1}/{max_retries})")
                time.sleep(backoff + random.random() * 2)
                backoff *= 2
        except Exception as e:
            print(f"Request error for '{query[:30]}': {e}. Retrying... (Attempt {attempt+1}/{max_retries})")
            time.sleep(backoff + random.random() * 2)
            backoff *= 2
            
    # If all failed, return a fallback of 1
    print(f"Failed to get label for '{query}' | '{product_text[:40]}' after {max_retries} attempts.")
    return 1

def load_existing_progress(output_path):
    if not os.path.exists(output_path):
        return {}
    results = {}
    try:
        with open(output_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            if len(lines) > 1:
                # check if header is correct
                if lines[0].strip() == "id,label":
                    for line in lines[1:]:
                        if ',' in line:
                            parts = line.strip().split(',')
                            if len(parts) == 2:
                                results[parts[0]] = int(parts[1])
    except Exception as e:
        print(f"Warning: could not read existing progress from {output_path}: {e}")
    return results

def process_file(file_num, key):
    filename = f"input_{file_num:03d}.txt"
    input_path = os.path.join(INPUT_DIR, filename)
    output_filename = f"input_{file_num:03d}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    print(f"\n==================================================")
    print(f"PROCESSING FILE: {filename}")
    print(f"==================================================")
    
    # Read all lines from input
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    print(f"Read {len(lines)} lines from {filename}")
    
    # Parse pairs
    pairs = []
    for line in lines:
        parts = [p.strip() for p in line.split("|")]
        if len(parts) >= 3:
            item_id = parts[0]
            query = parts[1]
            product_text = "|".join(parts[2:])
            pairs.append((item_id, query, product_text))
            
    # Load any already evaluated pairs from previous runs
    results = load_existing_progress(output_path)
    if results:
        print(f"Loaded {len(results)} existing labels from {output_filename}")
        
    # Pairs that still need to be evaluated
    todo_pairs = [p for p in pairs if p[0] not in results]
    print(f"Remaining pairs to evaluate: {len(todo_pairs)}")
    
    if todo_pairs:
        # Process remaining with a smaller thread pool to respect rate limits
        def process_pair(pair):
            item_id, query, product_text = pair
            label = call_openai_gpt_4o_mini(key, query, product_text)
            # Sleep slightly to avoid sending requests too fast
            time.sleep(0.1 + random.random() * 0.1)
            return item_id, label

        # Use 3 workers to prevent hitting rate limit too often
        with ThreadPoolExecutor(max_workers=3) as executor:
            for idx, (item_id, label) in enumerate(executor.map(process_pair, todo_pairs)):
                results[item_id] = label
                # Flush to file periodically in case of interruption
                if (idx + 1) % 10 == 0 or (idx + 1) == len(todo_pairs):
                    temp_output_path = output_path + ".tmp"
                    with open(temp_output_path, "w", encoding="utf-8") as f:
                        f.write("id,label\n")
                        # Write what we have in the original order
                        for original_id, _, _ in pairs:
                            if original_id in results:
                                f.write(f"{original_id},{results[original_id]}\n")
                    os.replace(temp_output_path, output_path)
                    print(f"Progress: {len(results)}/{len(pairs)} written to {output_filename}")
                    
    # Write final verified output to ensure order and completeness
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for item_id, _, _ in pairs:
            f.write(f"{item_id},{results[item_id]}\n")
            
    # Verify the output
    print(f"Verifying {output_filename}...")
    with open(output_path, "r", encoding="utf-8") as f:
        out_lines = [l.strip() for l in f if l.strip()]
        
    expected_lines = len(lines) + 1
    actual_lines = len(out_lines)
    if expected_lines != actual_lines:
        raise ValueError(f"Line count verification failed for {output_filename}. Expected {expected_lines}, got {actual_lines}")
        
    if out_lines[0] != "id,label":
        raise ValueError(f"Header verification failed for {output_filename}. Got: {out_lines[0]}")
        
    for i, line in enumerate(lines):
        parts = [p.strip() for p in line.split("|")]
        expected_id = parts[0]
        out_parts = out_lines[i + 1].split(",")
        actual_id = out_parts[0]
        if expected_id != actual_id:
            raise ValueError(f"ID matching failed at line {i+2}. Expected {expected_id}, got {actual_id}")
            
    print(f"Verification successful for {output_filename}! Verified {len(lines)} records.")

def main():
    key = get_key()
    for file_num in range(52, 57):
        process_file(file_num, key)
    print("\nAll files processed and verified successfully!")

if __name__ == "__main__":
    main()
