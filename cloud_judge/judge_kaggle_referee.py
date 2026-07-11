# =============================================================================
# ACIK-KAYNAK HAKEM — Kaggle Notebook scripti (2xT4, vLLM + Qwen3-14B-AWQ)
# Kurallara uygun: acik model, self-host, veri Kaggle disina cikmiyor.
#
# KULLANIM (Kaggle Notebook):
#   1. Yeni notebook ac -> Add Data -> yarisma verisi (otomatik ekli olabilir)
#      + 'cloud-judge-inputs' adiyla yukledigin private dataset (band_ids_partN.csv)
#   2. Accelerator: GPU T4 x2   |  Internet: ON (model indirmek icin)
#   3. Bu dosyanin icerigini tek hucreye yapistir, PART degiskenini ayarla, calistir.
#   4. Bitince /kaggle/working/labels_partN.csv ve exam_result.txt'yi indir.
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)

PART = 1                       # <<< HANGI PARCA (1..4) - hesap basina bir parca
MODEL = "Qwen/Qwen3-14B"  # TEMYIZ hakemi (fp16 tp2 2xT4'e kil payi sigar; sigmazsa google/gemma-3-12b-it)

import os, re, glob, random
import numpy as np, pandas as pd
from vllm import LLM, SamplingParams

# yarisma verisi ve id dosyalarini derinlemesine ara (mount yollari degisebiliyor)
COMP = IDS_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if COMP is None and "items.csv" in files and "submission_pairs.csv" in files:
        COMP = root
    if IDS_DIR is None and any(f.startswith("band_ids_part") for f in files):
        IDS_DIR = root
    if COMP and IDS_DIR:
        break
print("COMP =", COMP); print("IDS_DIR =", IDS_DIR)
assert COMP, "yarisma verisi bulunamadi - Add Input'tan yarisma datasetini ekle"
assert IDS_DIR, "band_ids dosyalari bulunamadi - cloud-judge-inputs datasetini ekle"

SYS = ("Sen Trendyol arama-alaka uzmanısın. Verilen (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURALLAR: Sorgu marka "
       "ise o markanın her ürünü 1. Sorgu kategori ise o kategorideki her ürün 1. "
       "Ürün tipi aynı veya yakın kullanım amaçlıysa 1 — renk/beden/model/marka "
       "farkı ÖNEMSİZ, ikame ürünler de 1. Ürün sorgudakinden FARKLI bir ürün "
       "tipine aitse ve sorgudaki ihtiyacı karşılamıyorsa 0. Önce ürün tipini "
       "karşılaştır, sonra karar ver. SADECE tek karakter yaz: 0 ya da 1.")

FEWSHOT = [
    ("kırmızı kadın elbise | siyah uzun kollu elbise [elbise] marka:koton kadın", "1"),
    ("puma bayan ayakkabı | zenit çift kişilik yatak örtüsü [yatak örtüsü] marka:zenit", "0"),
    ("stanley termos | 0.89l pipetli termos bardak [termos] marka:stanley", "1"),
    ("laptop çantası | paslanmaz çelik tencere seti [tencere] marka:karaca", "0"),
]

_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")
def prod_text(title, cat, attrs, brand=None, gender=None):
    leaf = cat.split("/")[-1] if isinstance(cat, str) and cat else ""
    al = attrs.lower() if isinstance(attrs, str) else ""
    cm = _C.search(al); mm = _M.search(al)
    parts = [(title or "")[:90], f"[{leaf}]"]
    if isinstance(brand, str) and brand and brand.lower() != "unknown": parts.append(f"marka:{brand}")
    if isinstance(gender, str) and gender and gender.lower() != "unknown": parts.append(gender)
    if cm: parts.append(f"renk:{cm.group(1).strip()}")
    if mm: parts.append(f"materyal:{mm.group(1).strip()}")
    return " ".join(parts).strip()

print("Veri yukleniyor...")
items = pd.read_csv(os.path.join(COMP, "items.csv"))
terms = pd.read_csv(os.path.join(COMP, "terms.csv"))
sub = pd.read_csv(os.path.join(COMP, "submission_pairs.csv"))
train = pd.read_csv(os.path.join(COMP, "training_pairs.csv"))
term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
        for r in items.itertuples(index=False)}
sub_idx = sub.set_index("id")

print("Model yukleniyor (5-10 dk)...")
llm = LLM(model=MODEL, tensor_parallel_size=2, max_model_len=512,
          gpu_memory_utilization=0.95, dtype="float16", enforce_eager=True,
          max_num_seqs=64)
sp = SamplingParams(temperature=0.0, max_tokens=4)

def judge(pairs):
    """pairs: [(q, ptext)] -> [0/1] (vLLM continuous batching, tek-cift prompt)."""
    shots = []
    for fq, fl in FEWSHOT:
        shots += [{"role": "user", "content": fq}, {"role": "assistant", "content": fl}]
    msgs = [[{"role": "system", "content": SYS}] + shots +
            [{"role": "user", "content": f"{q} | {p}"}] for q, p in pairs]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    res = []
    for o in outs:
        m = re.search(r"[01]", o.outputs[0].text)
        res.append(int(m.group(0)) if m else 1)
    return res

# ---- 1) SINAV (200 bilinen cift; hedef RECALL>=0.95, FPR<=0.05) -------------
rng = random.Random(42)
pos = train.sample(100, random_state=4242)
exam = [(term2q[t], itxt[i], 1) for t, i in zip(pos.term_id, pos.item_id)]
tl = list(train.term_id.unique()); ai = items["item_id"].tolist()
exam += [(term2q[rng.choice(tl)], itxt[rng.choice(ai)], 0) for _ in range(100)]
lab = judge([(q, p) for q, p, _ in exam])
gt = np.array([e[2] for e in exam]); pr = np.array(lab)
rec = pr[gt == 1].mean(); fpr = pr[gt == 0].mean()
msg = f"SINAV: RECALL={rec:.3f} FPR={fpr:.3f}  (hedef >=0.95 / <=0.05)"
print(msg); open("/kaggle/working/exam_result.txt", "w").write(msg + "\n")
if rec < 0.93 or fpr > 0.07:
    raise SystemExit("SINAV BASARISIZ - devam etme, modeli/promptu degistir.")

# ---- 2) ETIKETLEME ----------------------------------------------------------
idsf = sorted(glob.glob(os.path.join(IDS_DIR, f"referee_ids_part{PART}.csv*")))[0]
ids = pd.read_csv(idsf)["id"].astype(str).tolist()
print(f"part {PART}: {len(ids):,} cift")
OUT = f"/kaggle/working/labels_referee{PART}.csv"
done = set()
if os.path.exists(OUT):
    done = set(pd.read_csv(OUT)["id"].astype(str))
todo = [i for i in ids if i not in done]
f = open(OUT, "a", encoding="utf-8")
if not done: f.write("id,label\n")
B = 2000
import time; t0 = time.time()
for b in range(0, len(todo), B):
    chunk = todo[b:b + B]
    rows = sub_idx.loc[chunk]
    pairs = [(term2q.get(t, ""), itxt.get(i, "")) for t, i in zip(rows["term_id"], rows["item_id"])]
    for cid, l in zip(chunk, judge(pairs)):
        f.write(f"{cid},{l}\n")
    f.flush()
    el = time.time() - t0
    print(f"{b+len(chunk)}/{len(todo)}  {(b+len(chunk))/el:.1f} cift/sn  ETA {(len(todo)-b-len(chunk))/max((b+len(chunk))/el,1e-9)/3600:.1f} sa", flush=True)
print("BITTI ->", OUT)
