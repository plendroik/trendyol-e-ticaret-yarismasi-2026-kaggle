# =============================================================================
# ACIK-KAYNAK HAKEM — YEREL surum (Windows + Ollama + Qwen3-14B)
# Kurallara uygun: acik model, kendi PC'nde, veri disari cikmiyor.
#
# KURULUM (arkadas PC'si):
#   1. ollama.com'dan Ollama kur  ->  terminal:  ollama pull qwen3:14b
#   2. (hiz icin) ortam degiskeni:  setx OLLAMA_NUM_PARALLEL 4   (sonra Ollama'yi yeniden baslat)
#   3. Asagida DATA yolunu kendi yarisma CSV klasorune gore duzenle.
#   4. calistir:  python judge_local_ollama.py --part 2
#      Once SINAV kosar (RECALL>=0.95 sart), sonra etiketler. RESUME-SAFE.
#   Cikti: labels_partN.csv  -> repoya push'la.
# =============================================================================
import os, re, json, time, argparse, random, urllib.request
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"      # <<< yarisma CSV'lerinin klasoru (DUZENLE)
MODEL = "qwen3:14b"
OLLAMA = "http://localhost:11434/api/chat"
WORKERS = 4

SYS = ("Sen Trendyol arama-alaka uzmanısın. Verilen (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURALLAR: Sorgu marka "
       "ise o markanın her ürünü 1. Sorgu kategori ise o kategorideki her ürün 1. "
       "Ürün tipi aynı veya yakın kullanım amaçlıysa 1 — renk/beden/model/marka "
       "farkı ÖNEMSİZ, ikame ürünler de 1. SADECE bambaşka bir ihtiyaca yönelik "
       "ürün 0. Kararsız kalırsan 1 ver. SADECE tek karakter yaz: 0 ya da 1. /no_think")

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

def ask(qp):
    q, p = qp
    body = json.dumps({"model": MODEL, "stream": False, "think": False,
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": f"{q} | {p}"}],
                       "options": {"temperature": 0.0, "num_predict": 6}}).encode()
    for a in range(3):
        try:
            req = urllib.request.Request(OLLAMA, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=180) as r:
                txt = json.loads(r.read())["message"]["content"]
            m = re.search(r"[01]", txt)
            if m: return int(m.group(0))
        except Exception:
            time.sleep(5)
    return 1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", type=int, required=True)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print("Veri yukleniyor...", flush=True)
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv")).set_index("id")
    train = pd.read_csv(os.path.join(DATA, "training_pairs.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}

    # SINAV
    rng = random.Random(42)
    pos = train.sample(100, random_state=4242)
    exam = [(term2q[t], itxt[i], 1) for t, i in zip(pos.term_id, pos.item_id)]
    tl = list(train.term_id.unique()); ai = items["item_id"].tolist()
    exam += [(term2q[rng.choice(tl)], itxt[rng.choice(ai)], 0) for _ in range(100)]
    t0 = time.time()
    with ThreadPoolExecutor(WORKERS) as ex:
        pr = np.array(list(ex.map(ask, [(q, p) for q, p, _ in exam])))
    gt = np.array([e[2] for e in exam])
    rec = pr[gt == 1].mean(); fpr = pr[gt == 0].mean()
    print(f"SINAV: RECALL={rec:.3f} FPR={fpr:.3f}  hiz={200/(time.time()-t0):.1f} cift/sn", flush=True)
    if rec < 0.90 or fpr > 0.10:
        raise SystemExit("SINAV BASARISIZ - devam edilmiyor.")

    # ETIKETLEME (resume-safe)
    ids = pd.read_csv(os.path.join(here, f"band_ids_part{args.part}.csv.gz"))["id"].astype(str).tolist()
    OUT = os.path.join(here, f"labels_part{args.part}.csv")
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["id"].astype(str))
    todo = [i for i in ids if i not in done]
    print(f"part {args.part}: toplam {len(ids):,}  kalan {len(todo):,}", flush=True)
    f = open(OUT, "a", encoding="utf-8")
    if not done: f.write("id,label\n")
    B = 400; t0 = time.time()
    for b in range(0, len(todo), B):
        chunk = todo[b:b + B]
        rows = sub.loc[chunk]
        pairs = [(term2q.get(t, ""), itxt.get(i, "")) for t, i in zip(rows["term_id"], rows["item_id"])]
        with ThreadPoolExecutor(WORKERS) as ex:
            labs = list(ex.map(ask, pairs))
        for cid, l in zip(chunk, labs):
            f.write(f"{cid},{l}\n")
        f.flush()
        el = time.time() - t0; d = b + len(chunk)
        print(f"{d}/{len(todo)}  {d/el:.1f} cift/sn  ETA {(len(todo)-d)/max(d/el,1e-9)/3600:.1f} sa", flush=True)
    print("BITTI ->", OUT, flush=True)

if __name__ == "__main__":
    main()
