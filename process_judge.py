import os
import re
import time
import sys
import concurrent.futures
from openai import OpenAI

# Ensure stdout uses UTF-8 to prevent encoding errors on Windows console
sys.stdout.reconfigure(encoding='utf-8')

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\claude_judge_pkg3"
OUTPUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load OpenAI Key
with open(r"C:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\openai_key.txt", "r", encoding="utf-8") as f:
    api_key = f.read().strip()

client = OpenAI(api_key=api_key)

SYS_PROMPT = """Sen Trendyol arama-alaka uzmanısın. Her (sorgu | ürün) çifti için ürünün aramanın MAKUL bir sonucu olup olmadığına karar ver.
Karar kuralları:
- Sorgu marka ise o markanın her ürünü 1.
- Sorgu kategori ise o kategorideki her ürün 1.
- Ürün tipi aynı veya yakın kullanım amaçlıysa 1 - renk/beden/model/marka farkı ÖNEMSİZ, ikame ürünler de 1.
- SADECE bambaşka bir ihtiyaca yönelik ürün 0.
- Kararsız kalırsan 1 ver.

Sadece '0' veya '1' yaz. Başka hiçbir şey yazma, açıklama ekleme."""

def evaluate_pair(query, product, retries=5):
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
                max_tokens=2,
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": f"Sorgu: {query} | Ürün: {product}"}
                ]
            )
            ans = response.choices[0].message.content.strip()
            m = re.search(r"[01]", ans)
            if m:
                return int(m.group(0))
            return 1  # Fallback to 1 on parse fail
        except Exception as e:
            # Check for rate limit or similar
            sleep_time = (attempt + 1) * 2
            if "rate_limit" in str(e).lower() or "429" in str(e):
                sleep_time = (attempt + 1) * 4
            
            # Print error safely
            try:
                print(f"Error (attempt {attempt+1}/{retries}): {e}")
            except Exception:
                print(f"Error (attempt {attempt+1}/{retries}) on API call.")
                
            if attempt < retries - 1:
                time.sleep(sleep_time)
            else:
                return 1  # Fallback to 1 on final failure

def process_file(file_name):
    input_path = os.path.join(DATA_DIR, file_name)
    output_name = file_name.replace(".txt", ".csv")
    output_path = os.path.join(OUTPUT_DIR, output_name)
    
    print(f"\nProcessing {file_name}...")
    
    with open(input_path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    pairs = []
    ids = []
    for line in lines:
        parts = line.split(" | ", 2)
        if len(parts) == 3:
            ids.append(parts[0])
            pairs.append((parts[1], parts[2]))
        else:
            print(f"Warning: line formatted incorrectly.")
            
    print(f"Found {len(pairs)} pairs to evaluate.")
    
    labels = [None] * len(pairs)
    # Using 10 workers and sleeping a tiny bit to respect rate limit of 500 RPM
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {}
        for idx, (q, p) in enumerate(pairs):
            futures[executor.submit(evaluate_pair, q, p)] = idx
            # Add a small delay between submissions to smooth out request bursts
            time.sleep(0.08)
            
        for future in concurrent.futures.as_completed(futures):
            idx = futures[future]
            labels[idx] = future.result()
            
    # Write to CSV
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("id,label\n")
        for iid, label in zip(ids, labels):
            f.write(f"{iid},{label}\n")
            
    print(f"Wrote to {output_path}")
    
    # Verification
    with open(output_path, "r", encoding="utf-8") as f:
        out_lines = [l.strip() for l in f if l.strip()]
    
    out_data_lines = out_lines[1:]
    
    assert len(out_data_lines) == len(lines), f"Line count mismatch! Input has {len(lines)} lines, Output has {len(out_data_lines)} lines."
    
    out_ids = [l.split(",")[0] for l in out_data_lines]
    assert set(out_ids) == set(ids), "IDs do not match exactly!"
    assert len(out_ids) == len(set(out_ids)), "Duplicate IDs found in output!"
    
    print(f"Verification successful for {file_name}! Row count: {len(out_data_lines)}")

def main():
    files_to_process = ["input_027.txt", "input_028.txt", "input_029.txt", "input_030.txt", "input_031.txt"]
    for fname in files_to_process:
        process_file(fname)
    print("\nAll files processed successfully!")

if __name__ == "__main__":
    main()
