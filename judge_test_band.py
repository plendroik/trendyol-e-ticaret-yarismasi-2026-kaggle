"""
Label the domain-CE's UNCERTAIN BAND on the TEST set directly (the real lever).

The 0.83 model (ce_test_trendyol_ce.npy) is confident on most test pairs; only a
minority sit in the uncertain band (score in [LO, HI]) where its errors live. We
ask gpt-4o-mini — validated at RECALL 0.94 / FPR 0.06 in SINGLE-PAIR mode (batch
mode fails, see judge_api_validate.py) — to judge just those pairs. Confident CE
pairs keep their thresholded label; band pairs take the LLM label. -> hybrid submit.

Rule-gray: this sends test (query, product) text to the OpenAI API. Confirm with
organizers before selecting any resulting submission as a FINAL pick.

Parallel (threads), resume-safe (appends id,label; skips done). Key: openai_key.txt.

Usage:
  python judge_test_band.py --lo 0.2 --hi 0.85 --workers 20            # ~278k, ~$10
  python judge_test_band.py --lo 0.3 --hi 0.8  --workers 20 --limit 5000   # pilot
Output: test_judge_labels.csv  (id, label)   [in DATA dir]
"""
import os, sys, re, time, argparse, threading
import numpy as np, pandas as pd
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from judge_local import prod_text          # brand+gender enriched product text

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
OUT = os.path.join(DATA, "test_judge_labels.csv")
CE_TEST = os.path.join(ART, "ce_test_trendyol_ce.npy")   # the 0.83 model
MODEL = "gpt-4o-mini"
IN_COST, OUT_COST = 0.15 / 1e6, 0.60 / 1e6

# v1 prompt (validated best: recall 0.94, fpr 0.06). Do NOT add a "title contains
# query word -> 1" rule; it blew FPR up to 0.51.
SYS = ("Sen Trendyol arama-alaka uzmanısın. Her (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURALLAR: Sorgu marka "
       "ise o markanın her ürünü 1. Sorgu kategori ise o kategorideki her ürün 1. "
       "Ürün tipi aynı veya yakın kullanım amaçlıysa 1 — renk/beden/model/marka "
       "farkı ÖNEMSİZ, ikame ürünler de 1. SADECE bambaşka bir ihtiyaca yönelik "
       "ürün 0. Kararsız kalırsan 1 ver. Sadece 'numara. 0' veya 'numara. 1', "
       "her çift TEK satır, başka hiçbir şey yazma.")


def get_key():
    k = os.environ.get("OPENAI_API_KEY")
    kf = os.path.join(os.path.dirname(__file__), "openai_key.txt")
    if not k and os.path.exists(kf):
        k = open(kf, encoding="utf-8").read().strip()
    if not k:
        sys.exit("OPENAI_API_KEY / openai_key.txt yok.")
    return k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.20)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--limit", type=int, default=0, help="cap pairs (pilot)")
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())

    print("Loading scores + data...", flush=True)
    ce = np.load(CE_TEST)
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    assert len(ce) == len(sub)
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}

    band = (ce >= args.lo) & (ce <= args.hi)
    idx = np.where(band)[0]
    print(f"band [{args.lo},{args.hi}]: {len(idx):,} / {len(ce):,} pairs "
          f"(low->0: {(ce < args.lo).sum():,}, high->1: {(ce > args.hi).sum():,})", flush=True)

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["id"].astype(str))
    todo = [i for i in idx if sub["id"].iat[i] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"done={len(done)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do."); return

    lock = threading.Lock()
    fout = open(OUT, "a", newline="", encoding="utf-8")
    if not done:
        fout.write("id,label\n"); fout.flush()
    t0 = time.time()
    counts = {"n": 0, "tin": 0, "tout": 0}

    def judge_row(i):
        sid = sub["id"].iat[i]
        q = term2q.get(sub["term_id"].iat[i], "")
        p = itxt.get(sub["item_id"].iat[i], "")
        lab = 1
        for a in range(4):
            try:
                r = client.chat.completions.create(
                    model=MODEL, temperature=0.0, max_tokens=6,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": f"Çiftler:\n1. {q} | {p}"}])
                txt = r.choices[0].message.content.strip()
                m = re.search(r"([01])\b", txt[::-1])   # last 0/1 in the line
                lab = int(m.group(1)) if m else 1
                with lock:
                    counts["tin"] += r.usage.prompt_tokens
                    counts["tout"] += r.usage.completion_tokens
                break
            except Exception as e:
                if a == 3:
                    print(f"  give up id={sid}: {e}", flush=True)
                time.sleep(3 * (a + 1))
        with lock:
            fout.write(f"{sid},{lab}\n")
            counts["n"] += 1
            if counts["n"] % 2000 == 0:
                fout.flush()
                cost = counts["tin"] * IN_COST + counts["tout"] * OUT_COST
                el = time.time() - t0
                print(f"  {counts['n']}/{len(todo)}  ${cost:.2f}  "
                      f"({el:.0f}s, {counts['n']/el:.1f}/s, "
                      f"ETA {(len(todo)-counts['n'])/max(counts['n']/el,1e-9)/60:.0f}dk)", flush=True)

    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(judge_row, todo))
    fout.flush(); fout.close()
    cost = counts["tin"] * IN_COST + counts["tout"] * OUT_COST
    print(f"DONE. labeled={counts['n']}  cost=${cost:.2f}  -> {OUT}", flush=True)


if __name__ == "__main__":
    main()
