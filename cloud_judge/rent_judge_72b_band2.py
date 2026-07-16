# =============================================================================
# KIRALIK A100/H100 icin 72B HAKEM — tum bandi (577k) etiketler.
# Acik-kaynak Qwen2.5-72B-Instruct-AWQ (self-host, kurallara uygun).
# GEREKLI: band2_input.parquet, exam_input.parquet (ayni klasorde)
# CIKTI: labels_72b_band2.csv  (resume-safe)
#   pip install -q vllm pandas pyarrow  &&  python rent_judge_72b.py
# =============================================================================
import os, re, time
import numpy as np, pandas as pd
from vllm import LLM, SamplingParams

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL = "Qwen/Qwen2.5-72B-Instruct-AWQ"   # daha hizli: Qwen/Qwen2.5-32B-Instruct-AWQ

SYS = ("Sen Trendyol arama-alaka uzmanısın. Verilen (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURALLAR: Sorgu marka "
       "ise (yazım hatalı olsa bile) o markanın her ürünü 1. Sorgu kategori ise o "
       "kategorideki her ürün 1. Ürün tipi aynı veya yakın kullanım amaçlıysa 1 — "
       "renk/beden/model/marka/cinsiyet farkı ÖNEMSİZ, ikame ürünler de 1. Ürün "
       "sorgudakinden FARKLI bir ürün tipine aitse ve o ihtiyacı karşılamıyorsa 0. "
       "Önce ürün tipini karşılaştır, sonra karar ver. SADECE tek karakter yaz: 0 ya da 1.")
FEWSHOT = [
    ("kırmızı kadın elbise | siyah uzun kollu elbise [elbise] marka:koton kadın", "1"),
    ("puma bayan ayakkabı | zenit çift kişilik yatak örtüsü [yatak örtüsü] marka:zenit", "0"),
    ("stanley termos | 0.89l pipetli termos bardak [termos] marka:stanley", "1"),
    ("laptop çantası | paslanmaz çelik tencere seti [tencere] marka:karaca", "0"),
    ("yolluk | kaymaz taban halı yolluk mutfak [halı] marka:else", "1"),
    ("acar yemek takımı | athen 57 parça porselen yemek takımı [yemek takımı] marka:athen", "1"),
]


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    log(f"Model yukleniyor: {MODEL} ...")
    llm = LLM(model=MODEL, tensor_parallel_size=1, max_model_len=1024,
              gpu_memory_utilization=0.92, quantization="awq_marlin")
    sp = SamplingParams(temperature=0.0, max_tokens=4)

    shots = []
    for fq, fl in FEWSHOT:
        shots += [{"role": "user", "content": fq}, {"role": "assistant", "content": fl}]

    def judge(texts):
        msgs = [[{"role": "system", "content": SYS}] + shots + [{"role": "user", "content": t}]
                for t in texts]
        outs = llm.chat(msgs, sp)
        return [int(m.group(0)) if (m := re.search(r"[01]", o.outputs[0].text)) else 1 for o in outs]

    # ---- SINAV ----
    ex = pd.read_parquet(os.path.join(HERE, "exam_input.parquet"))
    pr = np.array(judge(ex["text"].tolist())); gt = ex["label"].to_numpy()
    rec = pr[gt == 1].mean(); fpr = pr[gt == 0].mean()
    log(f"SINAV: RECALL={rec:.3f} FPR={fpr:.3f} (hedef >=0.93 / <=0.06)")
    if rec < 0.90 or fpr > 0.08:
        raise SystemExit("SINAV zayif - promptu/modeli degistir.")

    # ---- ETIKETLEME (resume-safe) ----
    band = pd.read_parquet(os.path.join(HERE, "band2_input.parquet"))
    OUT = os.path.join(HERE, "labels_72b_band2.csv")
    done = set(pd.read_csv(OUT)["id"].astype(str)) if os.path.exists(OUT) else set()
    todo = band[~band["id"].astype(str).isin(done)].reset_index(drop=True)
    log(f"band: {len(band):,}  kalan: {len(todo):,}")
    f = open(OUT, "a", encoding="utf-8")
    if not done:
        f.write("id,label\n")
    B = 4000; t0 = time.time()
    for b in range(0, len(todo), B):
        chunk = todo.iloc[b:b + B]
        labs = judge(chunk["text"].tolist())
        for cid, l in zip(chunk["id"], labs):
            f.write(f"{cid},{l}\n")
        f.flush()
        d = b + len(chunk); el = time.time() - t0
        log(f"{d}/{len(todo)}  {d/el:.1f} cift/sn  ETA {(len(todo)-d)/max(d/el,1e-9)/3600:.1f} sa")
    log("BITTI -> labels_72b_band2.csv")


if __name__ == "__main__":
    main()
