"""
Build the claude_judge/ package: work files for a VS Code chat agent
(Sonnet/Opus/Gemini) to act as a free relevance judge.

Contents:
  claude_judge/INSTRUCTIONS.md   judging prompt + I/O contract
  claude_judge/exam_input.txt    200 exam pairs (GT hidden in artifacts/exam_gt.csv)
  claude_judge/input_NNN.txt     disputed pairs, conf in [0.30, 0.50), 400/file
  claude_judge/outputs/          agent writes output CSVs here
Scoring/merging: score_agent_exam.py judges the agent; merge happens in build step.

Usage: python gen_agent_package.py --max_pairs 10000
"""
import os, argparse, random
import numpy as np, pandas as pd
from judge_local import prod_text

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
PKG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "claude_judge")

ap = argparse.ArgumentParser()
ap.add_argument("--max_pairs", type=int, default=10000)
args = ap.parse_args()
os.makedirs(os.path.join(PKG, "outputs"), exist_ok=True)

terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
items = pd.read_csv(os.path.join(DATA, "items.csv"))
train = pd.read_csv(os.path.join(DATA, "training_pairs.csv"))
term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
        for r in items.itertuples(index=False)}

# ---- exam (GT known, hidden from the agent) --------------------------------
rng = random.Random(42)
pos = train.sample(100, random_state=42)
exam = [(f"EX_P{k:03d}", term2q[t], itxt[i], 1)
        for k, (t, i) in enumerate(zip(pos.term_id, pos.item_id))]
tl = list(train.term_id.unique()); ai = items["item_id"].tolist()
exam += [(f"EX_N{k:03d}", term2q[rng.choice(tl)], itxt[rng.choice(ai)], 0)
         for k in range(100)]
rng.shuffle(exam)
with open(os.path.join(PKG, "exam_input.txt"), "w", encoding="utf-8") as f:
    for eid, q, p, _ in exam:
        f.write(f"{eid} | {q} | {p}\n")
pd.DataFrame([(e[0], e[3]) for e in exam], columns=["id", "label"]).to_csv(
    os.path.join(ART, "exam_gt.csv"), index=False)
print(f"exam: 200 cift -> exam_input.txt (GT: artifacts/exam_gt.csv)")

# ---- disputed tier [0.30, 0.50) --------------------------------------------
ce = np.load(os.path.join(ART, "ce_test_trendyol_ce.npy"))
savg = (np.load(os.path.join(ART, "ce_test_distill2.npy")) +
        np.load(os.path.join(ART, "ce_test_tybert_distill2.npy")) +
        np.load(os.path.join(ART, "ce_test_xlmr_distill.npy"))) / 3
sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
ids = sub["id"].astype(str).to_numpy()
tids = sub["term_id"].to_numpy(); iids = sub["item_id"].to_numpy()
lab = pd.read_csv(os.path.join(DATA, "test_judge_labels.csv"))
id2 = dict(zip(lab["id"].astype(str), lab["label"].astype(int)))

band = (ce >= 0.03) & (ce <= 0.97)
idx = np.where(band)[0]
jl = np.array([id2.get(ids[i], -1) for i in idx])
m = jl >= 0
conf = np.abs(jl[m] - savg[idx[m]])
sel = idx[m][(conf >= 0.30) & (conf < 0.50)]
order = np.argsort(-np.abs(jl[m][(conf >= 0.30) & (conf < 0.50)] -
                            savg[sel]))
sel = sel[order][:args.max_pairs]
print(f"ihtilaf katmani [0.30,0.50): secilen {len(sel):,} cift")

B = 400
for b in range(0, len(sel), B):
    fn = os.path.join(PKG, f"input_{b//B + 1:03d}.txt")
    with open(fn, "w", encoding="utf-8") as f:
        for i in sel[b:b + B]:
            q = term2q.get(tids[i], ""); p = itxt.get(iids[i], "")
            f.write(f"{ids[i]} | {q} | {p}\n")
print(f"{(len(sel)-1)//B + 1} input dosyasi yazildi -> {PKG}")

INSTR = """# LLM Hakem Gorevi (VS Code ajani icin)

Sen Trendyol arama-alaka uzmanisin. Bu klasordeki `input_NNN.txt` (ve once
`exam_input.txt`) dosyalarini isleyeceksin. Her satir bir (sorgu, urun) cifti:

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
(orn. `input_001.txt` -> `outputs/input_001.csv`; sinav: `outputs/exam.csv`).
Format TAM OLARAK soyle (baslik dahil):

    id,label
    TST_xxx,1
    TST_yyy,0

- Girdideki HER id ciktida TAM BIR KEZ olmali, sira serbest.
- label sadece 0 veya 1. Baska hicbir sey yazma, aciklama ekleme.
- Dosyalari kod calistirarak degil, kendi yargi yetenegin ile etiketle
  (satirlari oku, karar ver, csv yaz). Gerekirse dosyayi parcalar halinde isle.

## Islem sirasi
1. ONCE `exam_input.txt` -> `outputs/exam.csv` (sinav; sonucu insan kontrol edecek)
2. Onay gelince `input_001.txt`den itibaren sirayla.
"""
with open(os.path.join(PKG, "INSTRUCTIONS.md"), "w", encoding="utf-8") as f:
    f.write(INSTR)
print("INSTRUCTIONS.md yazildi")
