"""
AI (LLM) query expansion via OpenAI gpt-4o-mini.

Enriches each search query with 4-6 relevant Turkish keywords (synonyms, category,
product type, brand/intent) that a general embedding/TF-IDF expansion cannot infer
(e.g. "clio 4" -> Renault car accessories; "davlunbaz" -> aspiratör). Helps the
cold-start test queries.

Cost: gpt-4o-mini, batched -> ~$1.5-2 for all 50k queries. Resume-safe (skips terms
already in the output). API key read from env OPENAI_API_KEY or file openai_key.txt.

Usage:
  python gen_ai_expansion.py --limit 40      # small test first
  python gen_ai_expansion.py                 # all queries
Output: ai_expanded_terms.csv  (term_id, query=<expanded>)
"""
import os, sys, time, argparse, csv, pandas as pd
from openai import OpenAI

DATA = r"C:\Users\ASUS\Desktop\trendyol"
OUT = os.path.join(DATA, "ai_expanded_terms.csv")
MODEL = "gpt-4o-mini"
IN_COST, OUT_COST = 0.15 / 1e6, 0.60 / 1e6      # $/token (gpt-4o-mini)

SYS = ("Sen bir Türkçe e-ticaret arama uzmanısın. Sana verilen her arama sorgusunu, "
       "ilgili ürünleri bulmaya yardımcı olacak 4-6 Türkçe anahtar kelimeyle "
       "(eş anlamlılar, kategori, ürün tipi, marka/niyet) zenginleştir. Orijinal "
       "sorguyu koru, sonuna ekle. Kısa tut, açıklama YAPMA. Yanıtı SADECE "
       "'numara. genişletilmiş sorgu' formatında, her sorgu için tek satır ver.")


def get_key():
    k = os.environ.get("OPENAI_API_KEY")
    kf = os.path.join(os.path.dirname(__file__), "openai_key.txt")
    if not k and os.path.exists(kf):
        k = open(kf, encoding="utf-8").read().strip()
    if not k:
        sys.exit("OPENAI_API_KEY yok. openai_key.txt dosyasina anahtari koy ya da env ayarla.")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--batch", type=int, default=25)
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())

    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    if args.limit:
        terms = terms.head(args.limit)

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["term_id"].astype(str))
    todo = terms[~terms["term_id"].astype(str).isin(done)].reset_index(drop=True)
    print(f"total={len(terms)} done={len(done)} todo={len(todo)}", flush=True)

    fout = open(OUT, "a", newline="", encoding="utf-8")
    w = csv.writer(fout)
    if not done:
        w.writerow(["term_id", "query"])

    tin = tout = 0
    t0 = time.time()
    for s in range(0, len(todo), args.batch):
        chunk = todo.iloc[s:s + args.batch]
        qs = chunk["query"].fillna("").astype(str).tolist()
        prompt = "Sorgular:\n" + "\n".join(f"{i+1}. {q}" for i, q in enumerate(qs))
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0.3, max_tokens=len(qs) * 40,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": prompt}])
        except Exception as e:
            print(f"  batch {s} error: {e}; retry in 10s", flush=True); time.sleep(10); continue
        tin += r.usage.prompt_tokens; tout += r.usage.completion_tokens
        lines = [l.strip() for l in r.choices[0].message.content.splitlines() if l.strip()]
        exp = {}
        for l in lines:
            if "." in l:
                num, _, txt = l.partition(".")
                if num.strip().isdigit():
                    exp[int(num.strip())] = txt.strip()
        for i, (_, row) in enumerate(chunk.iterrows()):
            e = exp.get(i + 1, qs[i]) or qs[i]
            w.writerow([row["term_id"], e])
        fout.flush()
        if s % (args.batch * 20) == 0:
            cost = tin * IN_COST + tout * OUT_COST
            print(f"  {s+len(chunk)}/{len(todo)}  cost=${cost:.3f}  ({time.time()-t0:.0f}s)", flush=True)

    cost = tin * IN_COST + tout * OUT_COST
    print(f"DONE. tokens in={tin} out={tout}  cost=${cost:.3f}  -> {OUT}", flush=True)
    fout.close()


if __name__ == "__main__":
    main()
