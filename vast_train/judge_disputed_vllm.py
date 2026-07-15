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
    te_dfs = []
    for k in range(4):
        p_path = os.path.join(HERE, f"test_text_part{k}.parquet")
        te_dfs.append(pd.read_parquet(p_path, columns=["id", "q", "d"]))
    te_df = pd.concat(te_dfs, ignore_index=True)
    
    ids = te_df["id"].astype(str).to_numpy()
    queries = te_df["q"].fillna("").astype(str).to_numpy()
    docs = te_df["d"].fillna("").astype(str).to_numpy()

    # Mevcut etiketlenmis çiftleri yukle (tekrar etiketlememek icin)
    already_labeled = set()
    labels_file = os.path.join(DATA_DIR, "etiketliveri", "clean_labels_72b.csv")
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
        q = queries[idx]
        p = docs[idx]
        
        # Qwen-Instruct chat formatina uygun sekilde prompt hazirla
        prompt_text = f"<|im_start|>system\n{SYS}<|im_end|>\n<|im_start|>user\nSorgu: {q} | Ürün: {p}<|im_end|>\n<|im_start|>assistant\n"
        formatted_prompts.append(prompt_text)
        todo_ids.append(ids[idx])

    # Vast.ai imajinda onceden calisan vLLM sunucusunu durdur (VRAM bosaltmak icin)
    import subprocess as _sp
    log("Onceden calisan vLLM/GPU islemlerini durduruluyor (VRAM bosaltiliyor)...")
    _sp.run("pkill -9 -f vllm || true", shell=True)
    _sp.run("pkill -9 -f 'python.*serve' || true", shell=True)
    time.sleep(5)
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except: pass
    log("VRAM bosaltildi. Model yukleniyor...")

    # vLLM baslatma
    log(f"vLLM yukleniyor: {LLM_MODEL}...")
    llm = LLM(model=LLM_MODEL, quantization="awq", tensor_parallel_size=1,
              max_model_len=1024, gpu_memory_utilization=0.90)
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
    
    # tmpfiles.org sitesine otomatik yukleyip indirme linkini ekrana yazdir
    import subprocess
    try:
        log("Cikti dosyasi tmpfiles.org sitesine yukleniyor...")
        cmd = f'curl -s -F "file=@{OUT}" https://tmpfiles.org/api/v1/upload'
        res = subprocess.check_output(cmd, shell=True).decode("utf-8")
        data = json.loads(res)
        if data.get("status") == "success":
            url = data["data"]["url"]
            dl_url = url.replace("https://tmpfiles.org/", "https://tmpfiles.org/dl/")
            log(f"*** INDIRME LINKINIZ HAZIR (Direkt tiklayip indirin): {dl_url} ***")
        else:
            log(f"Yükleme basarisiz oldu: {res}")
    except Exception as e:
        log(f"Otomatik yukleme hatasi: {e}")
        log(f"Yerel olarak indirmek isterseniz: curl -F \"file=@{OUT}\" https://tmpfiles.org/api/v1/upload")
        
    log("Islem tamamlandi!")

if __name__ == "__main__":
    main()
