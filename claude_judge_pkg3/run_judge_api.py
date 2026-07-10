import os
import re
import sys
import time
from openai import OpenAI

DATA_DIR = r"c:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\claude_judge_pkg3"
OUT_DIR = os.path.join(DATA_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

def get_key():
    kf = r"c:\Users\ASUS\Desktop\trendyol\trendyol-e-ticaret-yarismasi-2026-kaggle\openai_key.txt"
    if os.path.exists(kf):
        return open(kf, encoding="utf-8").read().strip()
    sys.exit("OpenAI API key file not found.")

client = OpenAI(api_key=get_key())

SYS_PROMPT = (
    "Sen Trendyol arama-alaka uzmanısın. Ürün, sorgunun MAKUL bir sonucu mu? "
    "KURALLAR:\n"
    "- Sorgu marka ise o markanın her ürünü 1.\n"
    "- Sorgu kategori ise o kategorideki her ürün 1.\n"
    "- Ürün tipi aynı veya yakın kullanım amaçlıysa 1 - renk/beden/model/marka farkı ÖNEMSİZ, ikame ürünler de 1.\n"
    "- SADECE bambaşka bir ihtiyaca yönelik ürün 0.\n"
    "- Kararsız kalırsan 1 ver.\n\n"
    "Cevap olarak SADECE 0 veya 1 yaz, başka hiçbir şey yazma, açıklama ekleme."
)

def evaluate_pair(query, product_text):
    prompt = f"Sorgu: {query}\nÜrün: {product_text}"
    for attempt in range(5):
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": SYS_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,
                max_tokens=2
            )
            content = response.choices[0].message.content.strip()
            # Find the first 0 or 1 in the response
            m = re.search(r"[01]", content)
            if m:
                return int(m.group(0))
            else:
                print(f"Unexpected response format: '{content}'. Retrying...")
        except Exception as e:
            print(f"API Error on attempt {attempt+1}: {e}. Retrying in 5 seconds...")
            time.sleep(5)
    print("Failed to get response after 5 attempts. Defaulting to 1.")
    return 1

def process_file(file_num):
    input_file = os.path.join(DATA_DIR, f"input_{file_num:03d}.txt")
    output_file = os.path.join(OUT_DIR, f"input_{file_num:03d}.csv")
    
    if not os.path.exists(input_file):
        print(f"File {input_file} does not exist. Skipping.")
        return
        
    print(f"Processing {input_file} -> {output_file}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
        
    results = []
    total = len(lines)
    for idx, line in enumerate(lines):
        parts = line.split(" | ")
        if len(parts) < 3:
            print(f"Skipping malformed line: {line}")
            continue
        pid = parts[0].strip()
        query = parts[1].strip()
        prod = parts[2].strip()
        
        label = evaluate_pair(query, prod)
        results.append((pid, label))
        
        if (idx + 1) % 10 == 0 or (idx + 1) == total:
            print(f"  Processed {idx + 1}/{total} pairs...")
            
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        f.write("id,label\n")
        for pid, label in results:
            f.write(f"{pid},{label}\n")
            
    # Verify line count and ID matching
    with open(output_file, "r", encoding="utf-8") as f:
        out_lines = [l.strip() for l in f if l.strip()]
    csv_pairs = [l.split(",") for l in out_lines[1:]]
    csv_ids = [p[0] for p in csv_pairs]
    
    input_ids = [line.split(" | ")[0].strip() for line in lines]
    
    if len(csv_ids) != len(input_ids):
        print(f"WARNING: Line count mismatch! Input has {len(input_ids)} lines, output has {len(csv_ids)} lines.")
    elif csv_ids != input_ids:
        print("WARNING: ID mismatch or ordering mismatch!")
    else:
        print(f"SUCCESS: Verified {output_file} matches input line count ({len(csv_ids)}) and exact ID order.")

if __name__ == "__main__":
    for i in range(17, 22):
        process_file(i)
