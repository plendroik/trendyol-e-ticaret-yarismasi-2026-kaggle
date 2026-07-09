import os
import argparse
import random
import numpy as np
import pandas as pd
from judge_local import prod_text

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_pairs", type=int, default=10000)
    args = ap.parse_args()
    
    # Paths configured for the current local workspace
    DATA = "."
    ART = "artifacts"
    PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_judge_friend")
    
    os.makedirs(os.path.join(PKG, "outputs"), exist_ok=True)
    
    print("Loading terms, items and training pairs...", flush=True)
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
            
    # Load available predictions
    print("Loading model predictions...", flush=True)
    ce = np.load(os.path.join(ART, "test_ensemble_probs.npy"))
    savg = (np.load(os.path.join(ART, "ce_test_distill.npy")) +
            np.load(os.path.join(ART, "ce_test_xlmr_distill.npy"))) / 2
            
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    ids = sub["id"].astype(str).to_numpy()
    tids = sub["term_id"].to_numpy()
    iids = sub["item_id"].to_numpy()
    
    print("Loading test judge labels...", flush=True)
    lab = pd.read_csv(os.path.join("etiketliveri", "test_judge_labels.csv"))
    id2 = dict(zip(lab["id"].astype(str), lab["label"].astype(int)))
    
    # Filter for uncertain band of the base model [0.03, 0.97]
    band = (ce >= 0.03) & (ce <= 0.97)
    idx = np.where(band)[0]
    
    # Match indices to existing judge labels
    jl = np.array([id2.get(ids[i], -1) for i in idx])
    m = jl >= 0
    
    # Compute disagreement conf = |label - savg|
    conf = np.abs(jl[m] - savg[idx[m]])
    
    # Disputed tier [0.30, 0.50)
    sel = idx[m][(conf >= 0.30) & (conf < 0.50)]
    order = np.argsort(-np.abs(jl[m][(conf >= 0.30) & (conf < 0.50)] - savg[sel]))
    sel = sel[order][:args.max_pairs]
    
    print(f"Disputed tier [0.30, 0.50): Selected {len(sel):,} pairs out of {len(idx):,} candidate band pairs.")
    
    if len(sel) == 0:
        print("ERROR: No pairs match the criteria!")
        return
        
    B = 400
    for b in range(0, len(sel), B):
        fn = os.path.join(PKG, f"input_{b//B + 1:03d}.txt")
        with open(fn, "w", encoding="utf-8") as f:
            for i in sel[b:b + B]:
                q = term2q.get(tids[i], "")
                p = itxt.get(iids[i], "")
                f.write(f"{ids[i]} | {q} | {p}\n")
                
    num_files = (len(sel) - 1) // B + 1
    print(f"Successfully wrote {num_files} input files to {PKG}")
    
    # Generate INSTRUCTIONS.md
    INSTR = """# LLM Hakem Gorevi (VS Code ajani icin)

Sen Trendyol arama-alaka uzmanisin. Bu klasordeki `input_NNN.txt` (ve once
`exam2_input.txt`) dosyalarini isleyeceksin. Her satir bir (sorgu, urun) cifti:

    <id> | <sorgu> | <urun metni>

## Karar kurali (AYNEN uygula)
Urun, sorgunun MAKUL bir sonucu mu? KURALLAR:
- Sorgu marka ise o markanin her urunu 1.
- Sorgu kategori ise o kategorideki her urun 1.
- Urun tipi ayni veya yakin kullanim amacliysa 1 - renk/beden/model/marka farki
  ONEMSIZ, ikame urunler de 1.
- SADECE bambaska bir ihtiyaca yonelik urun 0.
- Kararsiz kalirsan 1 ver.

## Cikti kurali
Her input dosyasi icin `outputs/` klasorune ayni adla `.csv` yaz
(orn. `input_001.txt` -> `outputs/input_001.csv`; sinav: `outputs/exam2.csv`).
Format TAM OLARAK soyle (baslik dahil):

    id,label
    TST_xxx,1
    TST_yyy,0

- Girdideki HER id ciktida TAM BIR KEZ olmali, sira serbest.
- label sadece 0 veya 1. Baska hicbir sey yazma, aciklama ekleme.
- Dosyalari kod calistirarak degil, kendi yargi yetenegin ile etiketle
  (satirlari oku, karar ver, csv yaz). Gerekirse dosyayi parcalar halinde isle.

## Islem sirasi
1. ONCE `exam2_input.txt` -> `outputs/exam2.csv` (sinav; sonucu insan kontrol edecek)
2. Onay gelince `input_001.txt`den itibaren sirayla.
"""
    with open(os.path.join(PKG, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
        f.write(INSTR)
    print("INSTRUCTIONS.md written successfully.")

if __name__ == "__main__":
    main()
