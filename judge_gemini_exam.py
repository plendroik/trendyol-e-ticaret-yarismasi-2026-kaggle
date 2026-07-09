"""
Gemini judge exam v2 — with PROPER error accounting (the first run silently
defaulted failures to 1 and burned the daily free quota -> fake RECALL 1.0/FPR 0.98).

Counts API errors separately; result only over successful calls. Sized to fit
free-tier limits (default 50+50 pairs, ~100 calls; gemini-2.0-flash RPM 15).

Usage:
  python judge_gemini_exam.py --model gemini-2.0-flash --n 50
  python judge_gemini_exam.py --model gemini-2.5-flash --n 50 --key gemini_key2.txt
"""
import os, json, re, time, random, argparse, urllib.request
import numpy as np, pandas as pd
from judge_local import prod_text
from judge_test_band import SYS

DATA = r"C:\Users\ASUS\Desktop\trendyol"


def call(model, key, q, p):
    body = json.dumps({
        "system_instruction": {"parts": [{"text": SYS}]},
        "contents": [{"parts": [{"text": f"Çiftler:\n1. {q} | {p}"}]}],
        "generationConfig": {"temperature": 0, "maxOutputTokens": 1000,
                             "thinkingConfig": {"thinkingBudget": 0}}}).encode()
    req = urllib.request.Request(
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
        data=body, headers={"Content-Type": "application/json", "x-goog-api-key": key})
    with urllib.request.urlopen(req, timeout=90) as r:
        d = json.loads(r.read())
    txt = d["candidates"][0]["content"]["parts"][0]["text"].strip()
    m = re.search(r"([01])\b", txt[::-1])
    if not m:
        raise ValueError(f"parse edilemedi: {txt[:60]!r}")
    return int(m.group(1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gemini-2.0-flash")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--key", default="gemini_key.txt")
    ap.add_argument("--sleep", type=float, default=4.5, help="RPM limiti icin")
    args = ap.parse_args()
    key = open(os.path.join(os.path.dirname(__file__), args.key)).read().strip()

    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    train = pd.read_csv(os.path.join(DATA, "training_pairs.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
    rng = random.Random(42)
    pos = train.sample(args.n, random_state=42)
    pos_pairs = [(term2q[t], itxt[i]) for t, i in zip(pos.term_id, pos.item_id)]
    tl = list(train.term_id.unique()); ai = items["item_id"].tolist()
    neg_pairs = [(term2q[rng.choice(tl)], itxt[rng.choice(ai)]) for _ in range(args.n)]

    def run(pairs, name):
        ok, err = [], 0
        for q, p in pairs:
            try:
                ok.append(call(args.model, key, q, p))
            except Exception as e:
                err += 1
                if err <= 3:
                    print(f"  {name} hata: {str(e)[:100]}", flush=True)
            time.sleep(args.sleep)
        print(f"  {name}: basarili={len(ok)} hata={err}", flush=True)
        return np.array(ok)

    t0 = time.time()
    p = run(pos_pairs, "poz")
    n = run(neg_pairs, "neg")
    if len(p) < args.n * 0.8 or len(n) < args.n * 0.8:
        print("UYARI: cok hata var, sonuc guvenilmez (kota?)")
    if len(p) and len(n):
        print(f"\n{args.model}: RECALL={p.mean():.3f}  FPR={n.mean():.3f}  ({time.time()-t0:.0f}s)")
        print("hedef: RECALL>=0.97 FPR<=0.03 (gpt-4o: 0.99/0.02)")


if __name__ == "__main__":
    main()
