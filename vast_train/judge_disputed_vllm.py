# =============================================================================
# VAST.AI (RTX 3090/4090/A100) UZERINDE PURE OPEN-SOURCE ACTIVE LEARNING (vLLM)
#
# Bu script, modellerin en cok kararsiz kaldigi (tahmin ortalamasi 0.25 - 0.75 arasi olan)
# 40k-50k disputed çifti tespit eder. Ardından vLLM ve Qwen-72B kullanarak
# bu çiftleri saniyeler icinde etiketler. Elde edilen yeni temiz etiketler,
# stacker'in egitim kumesine eklenerek skoru dogrudan artirir.
#
# GEREKLI: pip install vllm
# =============================================================================
import os, re, json, time, argparse
import numpy as np, pandas as pd
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
# Modellerin tahminlerinin bulundugu artifacts klasörü (Vast'a kopyalanacak)
ART_DIR = os.path.join(os.path.dirname(HERE), "artifacts")
DATA_DIR = os.path.dirname(HERE)

# Kullanılacak acik kaynakli büyük model (RTX 3090/4090'a sigmasi icin AWQ 4-bit)
LLM_MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"

SYS = ("Sen Trendyol arama-alaka hakemisin. Ürün, sorgunun MAKUL bir sonucu mu? "
       "Kurallar: Sorgu marka ise o markanın her ürünü 1. Sorgu kategori ise o "
       "kategorideki her ürün 1. Ürün tipi aynı veya yakın kullanım amaçlıysa 1 "
       "(renk/beden/model/marka farkı önemsiz). SADECE ürün bambaşka bir ihtiyaca "
       "yönelikse 0. SADECE tek karakter yaz: 0 ya da 1.")

_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

def prod_text(title, cat, attrs, brand=None, gender=None):
    leaf = cat.split("/")[-1] if isinstance(cat, str) and cat else ""
    al = attrs.lower() if isinstance(attrs, str) else ""
    cm = _C.search(al); mm = _M.search(al)
    t = (title or "")[:90]
    parts = [t, f"[{leaf}]"]
    if isinstance(brand, str) and brand and brand.lower() != "unknown":
        parts.append(f"marka:{brand}")
    if isinstance(gender, str) and gender and gender.lower() != "unknown":
        parts.append(gender)
    if cm:
        parts.append(f"renk:{cm.group(1).strip()}")
    if mm:
        parts.append(f"materyal:{mm.group(1).strip()}")
    return " ".join(parts).strip()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=45000, help="Etiketlenecek disputed çift sayısı")
    args = ap.parse_args()

    OUT = os.path.join(HERE, f"clean_labels_72b_append.csv")
    
    log("Veriler yukleniyor...")
    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    terms = pd.read_csv(os.path.join(DATA_DIR, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
            
    ids = sub["id"].astype(str).to_numpy()
    tids = sub["term_id"].to_numpy()
    iids = sub["item_id"].to_numpy()

    # Mevcut etiketlenmis çiftleri yukle (tekrar etiketlememek icin)
    already_labeled = set()
    labels_file = os.path.join(DATA_DIR, "clean_labels_72b.csv")
    if os.path.exists(labels_file):
        already_labeled = set(pd.read_csv(labels_file)["id"].astype(str))
        log(f"Mevcut etiketli çift sayısı: {len(already_labeled):,}")

    log("Model tahminleri yukleniyor...")
    # Tahminlerin ortalamasını alarak belirsizlik bandını bulacagiz
    predictions = []
    for name in ["ce_test_distill2.npy", "ce_test_tybert_distill2.npy", "ce_test_xlmr_distill.npy"]:
        p_path = os.path.join(ART_DIR, name)
        if os.path.exists(p_path):
            predictions.append(np.load(p_path))
            log(f"  + {name} yuklendi")
            
    assert len(predictions) > 0, "Tahmin npy dosyalari bulunamadi. Lutfen artifacts klasorunu kopyalayin."
    savg = np.mean(predictions, axis=0)

    # Modellerin en belirsiz oldugu (0.5'e en yakin oldugu) çiftleri bul
    uncertainty = np.abs(savg - 0.5)
    candidates = np.argsort(uncertainty) # 0.5'e en yakın olandan en uzak olana dogru sirala

    # Halihazırda etiketlenmemis olan belirsiz çiftleri sec
    todo_indices = []
    for idx in candidates:
        if ids[idx] not in already_labeled:
            todo_indices.append(idx)
        if len(todo_indices) >= args.topk:
            break
            
    log(f"Etiketlenecek yeni disputed çift sayısı: {len(todo_indices):,}")
    
    if len(todo_indices) == 0:
        log("Etiketlenecek yeni çift bulunamadı. Bitti.")
        return

    # Prompts hazirlama
    log("Prompts hazirlaniyor...")
    formatted_prompts = []
    todo_ids = []
    
    for idx in todo_indices:
        q = term2q.get(tids[idx], "")
        p = itxt.get(iids[idx], "")
        
        # Qwen-Instruct chat formatina uygun sekilde prompt hazirla
        prompt_text = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nSorgu: {q} | Ürün: {p}<|im_end|>\n<|im_start|>assistant\n"
        formatted_prompts.append(prompt_text)
        todo_ids.append(ids[idx])

    # vLLM baslatma
    log(f"vLLM yukleniyor: {LLM_MODEL}...")
    llm = LLM(model=LLM_MODEL, quantization="awq", tensor_parallel_size=1, max_model_len=1024)
    sampling_params = SamplingParams(temperature=0.0, max_tokens=3)

    log("Batch tahminleme basliyor (vLLM)...")
    t0 = time.time()
    outputs = llm.generate(formatted_prompts, sampling_params)
    log(f"Tahminleme tamamlandi. Sure: {time.time()-t0:.1f}s (Hiz: {len(todo_indices)/(time.time()-t0):.1f} cift/sn)")

    # Sonuclari kaydetme
    new_labels = []
    for cid, out in zip(todo_ids, outputs):
        txt = out.outputs[0].text.strip()
        # En sondaki 0 veya 1'i bul
        m = re.search(r"([01])\b", txt[::-1])
        label = int(m.group(1)) if m else 1
        new_labels.append({"id": cid, "label": label})

    df_out = pd.DataFrame(new_labels)
    df_out.to_csv(OUT, index=False)
    log(f"Etiketler basariyla kaydedildi: {OUT}")
    log("Islem tamamlandi!")

if __name__ == "__main__":
    main()
