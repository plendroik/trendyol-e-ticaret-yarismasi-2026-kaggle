# =============================================================================
# ARKADAS icin KAGGLE HAKEM notebook'u — tum bandi (577k) yeniden etiketler.
# Acik-kaynak Qwen3-8B, self-host, kurallara uygun. Yuksek-recall prompt.
# Hizli mod (dusunme KAPALI) -> 2xT4'te ~20-30 cift/sn -> tum band ~6-8 saat.
#
# ADIMLAR (notebook ustunde ayrica anlatildi):
#   1. cloud-judge-inputs private dataseti (band_ids_all.csv.gz) + yarisma verisi ekli
#   2. Accelerator GPU T4 x2, Internet ON, Persistence Files
#   3. Bu hucreyi yapistir, Save Version -> Save & Run All (Commit)
#   4. Bitince /kaggle/working/labels_friend.csv indir.
# =============================================================================
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=True)
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "-q", "torchcodec"])  # CUDA13 cakismasini engelle

import os, re, glob, random, time
import numpy as np, pandas as pd
from vllm import LLM, SamplingParams

COMP = IDS_DIR = None
for root, dirs, files in os.walk("/kaggle/input"):
    if COMP is None and "items.csv" in files and "submission_pairs.csv" in files:
        COMP = root
    if IDS_DIR is None and any(f.startswith("band_ids_all") for f in files):
        IDS_DIR = root
    if COMP and IDS_DIR: break
print("COMP =", COMP, "| IDS_DIR =", IDS_DIR)
assert COMP and IDS_DIR, "yarisma verisi + band_ids_all dataseti ekli olmali"

# YUKSEK-RECALL prompt: ikame/kategori/marka pozitiflerini kacirma
SYS = ("Sen Trendyol arama-alaka uzmanısın. Verilen (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURALLAR: Sorgu marka "
       "ise (yazım hatalı olsa bile) o markanın her ürünü 1. Sorgu kategori ise o "
       "kategorideki her ürün 1. Ürün tipi aynı veya YAKIN kullanım amaçlıysa 1 — "
       "renk/beden/model/marka/cinsiyet/malzeme farkı ÖNEMSİZ, ikame ürünler ve aynı "
       "kategorinin alt türleri de 1. Müşteri bu ürünü arama sonucunda görünce "
       "'evet bunu arıyordum' veya 'olabilir' derse 1. SADECE tamamen alakasız, "
       "bambaşka bir ihtiyaca yönelik ürün 0. Kararsızsan 1 ver. Cevabın son "
       "karakteri 0 ya da 1 olsun.")
FEWSHOT = [
    ("kırmızı kadın elbise | siyah uzun kollu elbise [elbise] marka:koton kadın", "1"),
    ("puma bayan ayakkabı | zenit çift kişilik yatak örtüsü [yatak örtüsü] marka:zenit", "0"),
    ("stanley termos | 0.89l pipetli termos bardak [termos] marka:stanley", "1"),
    ("laptop çantası | paslanmaz çelik tencere seti [tencere] marka:karaca", "0"),
    ("yolluk | kaymaz taban halı yolluk mutfak [halı] marka:else", "1"),
    ("acar yemek takımı | athen 57 parça porselen yemek takımı [yemek takımı] marka:athen", "1"),
    ("bebek biberonu | avent natural yenidoğan biberon seti [biberon] marka:philips avent", "1"),
    ("erkek spor ayakkabı | kadın topuklu abiye ayakkabı [topuklu] marka:elle kadın", "0"),
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
llm = LLM(model="Qwen/Qwen3-8B", tensor_parallel_size=2, max_model_len=1024,
          gpu_memory_utilization=0.90, dtype="float16", enforce_eager=True)
sp = SamplingParams(temperature=0.0, max_tokens=4)   # dusunme KAPALI -> hizli

def judge(pairs):
    shots = []
    for fq, fl in FEWSHOT:
        shots += [{"role": "user", "content": fq}, {"role": "assistant", "content": fl}]
    msgs = [[{"role": "system", "content": SYS}] + shots +
            [{"role": "user", "content": f"{q} | {p}"}] for q, p in pairs]
    outs = llm.chat(msgs, sp, chat_template_kwargs={"enable_thinking": False})
    return [int(m.group(0)) if (m := re.search(r"[01]", o.outputs[0].text)) else 1 for o in outs]

# ---- SINAV (kapi: recall>=0.93, fpr<=0.08) ----
rng = random.Random(42)
pos = train.sample(100, random_state=4242)
exam = [(term2q[t], itxt[i], 1) for t, i in zip(pos.term_id, pos.item_id)]
tl = list(train.term_id.unique()); ai = items["item_id"].tolist()
exam += [(term2q[rng.choice(tl)], itxt[rng.choice(ai)], 0) for _ in range(100)]
lab = judge([(q, p) for q, p, _ in exam])
gt = np.array([e[2] for e in exam]); pr = np.array(lab)
rec = pr[gt == 1].mean(); fpr = pr[gt == 0].mean()
msg = f"SINAV: RECALL={rec:.3f} FPR={fpr:.3f} (kapi >=0.93 / <=0.08)"
print(msg); open("/kaggle/working/exam_result.txt", "w").write(msg + "\n")
if rec < 0.93 or fpr > 0.08:
    raise SystemExit("SINAV BASARISIZ - promptu ayarlayacagiz, bize sonucu ilet.")

# ---- ETIKETLEME (tum band, resume-safe) ----
idsf = sorted(glob.glob(os.path.join(IDS_DIR, "band_ids_all.csv*")))[0]
ids = pd.read_csv(idsf)["id"].astype(str).tolist()
OUT = "/kaggle/working/labels_friend.csv"
done = set(pd.read_csv(OUT)["id"].astype(str)) if os.path.exists(OUT) else set()
todo = [i for i in ids if i not in done]
print(f"band: {len(ids):,}  kalan: {len(todo):,}")
f = open(OUT, "a", encoding="utf-8")
if not done: f.write("id,label\n")
B = 3000; t0 = time.time()
for b in range(0, len(todo), B):
    chunk = todo[b:b + B]
    rows = sub_idx.loc[chunk]
    pairs = [(term2q.get(t, ""), itxt.get(i, "")) for t, i in zip(rows["term_id"], rows["item_id"])]
    for cid, l in zip(chunk, judge(pairs)):
        f.write(f"{cid},{l}\n")
    f.flush()
    d = b + len(chunk); el = time.time() - t0
    print(f"{d}/{len(todo)}  {d/el:.1f} cift/sn  ETA {(len(todo)-d)/max(d/el,1e-9)/3600:.1f} sa", flush=True)
print("BITTI ->", OUT)
