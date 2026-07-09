"""
Validate the API judge (gpt-4o-mini / gpt-4o) on ground truth we already have —
same benchmark as judge_local.py: recall on known positives, FPR on random pairs.
Uses the brand-enriched prod_text (the local run showed brand queries fail without it).

Usage:
  python judge_api_validate.py --n 100 --model gpt-4o-mini
"""
import os, sys, re, time, argparse, random
import numpy as np, pandas as pd
from openai import OpenAI
from judge_local import prod_text   # brand-enriched product text

DATA = r"C:\Users\ASUS\Desktop\trendyol"
BATCH = 15
IN_COST, OUT_COST = 0.15 / 1e6, 0.60 / 1e6   # gpt-4o-mini

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
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--model", default="gpt-4o-mini")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())
    rng = random.Random(args.seed)

    print(f"model={args.model}  n={args.n}/side", flush=True)
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    train = pd.read_csv(os.path.join(DATA, "training_pairs.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
    all_items = items["item_id"].tolist()

    pos = train.sample(args.n, random_state=args.seed)
    pos_pairs = [(term2q[t], itxt[i]) for t, i in zip(pos.term_id, pos.item_id)]
    tlist = list(train.term_id.unique())
    neg_pairs = [(term2q[rng.choice(tlist)], itxt[rng.choice(all_items)])
                 for _ in range(args.n)]

    tin = tout = 0

    def judge_batch(batch):
        nonlocal tin, tout
        lines = "\n".join(f"{i+1}. {q} | {p}" for i, (q, p) in enumerate(batch))
        for a in range(3):
            try:
                r = client.chat.completions.create(
                    model=args.model, temperature=0.0, max_tokens=len(batch) * 6,
                    messages=[{"role": "system", "content": SYS},
                              {"role": "user", "content": "Çiftler:\n" + lines}])
                break
            except Exception as e:
                print(f"  API error: {e}; retry 10s", flush=True); time.sleep(10)
        else:
            return [1] * len(batch)
        tin += r.usage.prompt_tokens; tout += r.usage.completion_tokens
        labs = {}
        for l in r.choices[0].message.content.splitlines():
            m = re.match(r"\s*(\d+)\s*[.:\-]\s*([01])\b", l.strip())
            if m:
                labs[int(m.group(1))] = int(m.group(2))
        return [labs.get(i + 1, 1) for i in range(len(batch))]

    t0 = time.time()

    def run(pairs):
        out = []
        for b in range(0, len(pairs), BATCH):
            out += judge_batch(pairs[b:b + BATCH])
        return np.array(out)

    p = run(pos_pairs)
    n = run(neg_pairs)
    el = time.time() - t0
    cost = tin * IN_COST + tout * OUT_COST
    print(f"\n=== {args.model} ===")
    print(f"RECALL (bilinen pozitifte 1 deme): {p.mean():.3f}   hedef >= 0.93")
    print(f"FPR    (rastgele ciftte 1 deme)  : {n.mean():.3f}   hedef <= 0.05")
    print(f"hiz: {(len(p)+len(n))/el:.1f} cift/sn  sure {el:.0f}s  maliyet ${cost:.4f}")
    # kacirilan pozitifleri goster (prompt iyilestirme icin)
    misses = [(q, pr[:80]) for (q, pr), l in zip(pos_pairs, p) if l == 0][:10]
    if misses:
        print("\nKacirilan pozitif ornekleri:")
        for q, pr in misses:
            print(f"  sorgu={q!r}  urun={pr!r}")


if __name__ == "__main__":
    main()
