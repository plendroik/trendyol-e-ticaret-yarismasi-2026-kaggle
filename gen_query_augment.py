"""
Synthetic positive augmentation via query perturbation (rule-based doc-augmentation).

Creates realistic query VARIANTS of existing training terms (typos + word-drop,
brand-aware) to widen the query distribution the model sees -> better cold-start
generalization + typo robustness. Each synthetic term inherits its source term's
(item, label) pairs, so pos/neg balance is preserved. Synthetic rows go to TRAIN
folds only (holdout fold 4 stays original-only).

Usage:
  python gen_query_augment.py --base_terms expanded_terms.csv --frac 0.3
Outputs:
  augmented_terms.csv               (base terms + synthetic terms)
  artifacts/train_pairs_augmented.parquet   (easy pairs + synthetic pairs)
"""
import os, re, argparse, random, numpy as np, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
SEED = 42
random.seed(SEED)

BRANDS = {"nike", "adidas", "puma", "mavi", "defacto", "koton", "lcw", "lc", "waikiki",
          "zara", "bershka", "colins", "converse", "vans", "reebok", "hummel", "skechers",
          "polo", "us", "levis", "loft", "network", "kigili", "damat"}
_word = re.compile(r"\w+", re.UNICODE)


def typo(w):
    if len(w) < 4 or w.lower() in BRANDS:
        return w
    i = random.randrange(len(w) - 1)
    op = random.random()
    if op < 0.4:                                   # swap adjacent
        return w[:i] + w[i+1] + w[i] + w[i+2:]
    elif op < 0.7:                                 # drop a char
        return w[:i] + w[i+1:]
    else:                                          # duplicate a char
        return w[:i] + w[i] + w[i:]


def perturb(q):
    toks = q.split()
    if len(toks) < 2:
        return None
    r = random.random()
    if r < 0.45 and len(toks) > 2:                 # drop a word
        del toks[random.randrange(len(toks))]
    elif r < 0.9:                                  # typo a word
        j = random.randrange(len(toks)); toks[j] = typo(toks[j])
    else:
        del toks[random.randrange(len(toks))]; j = random.randrange(len(toks)); toks[j] = typo(toks[j])
    out = " ".join(toks).strip()
    return out if out and out != q else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_terms", default="terms.csv")
    ap.add_argument("--frac", type=float, default=0.30)
    args = ap.parse_args()

    base = pd.read_csv(os.path.join(DATA, args.base_terms))
    term2q = dict(zip(base["term_id"], base["query"].fillna("").astype(str)))
    pairs = pd.read_parquet(os.path.join(ART, "train_pairs.parquet"))

    # terms present in training (folds 0-3 only get synthetic copies)
    tr = pairs[pairs["fold"] != 4]
    terms_in_train = tr["term_id"].unique()
    chosen = set(random.sample(list(terms_in_train), int(args.frac * len(terms_in_train))))

    syn_map = {}      # orig_term -> synth query
    for t in chosen:
        pq = perturb(term2q.get(t, ""))
        if pq:
            syn_map[t] = pq

    # build synthetic pairs (inherit source term's train-fold rows)
    src = tr[tr["term_id"].isin(syn_map)].copy()
    src["term_id"] = "SYN_" + src["term_id"].astype(str)
    combined_pairs = pd.concat([pairs, src], ignore_index=True)
    combined_pairs.to_parquet(os.path.join(ART, "train_pairs_augmented.parquet"), index=False)

    # augmented terms file
    syn_terms = pd.DataFrame({"term_id": ["SYN_" + t for t in syn_map],
                              "query": list(syn_map.values())})
    aug_terms = pd.concat([base, syn_terms], ignore_index=True)
    aug_terms.to_csv(os.path.join(DATA, "augmented_terms.csv"), index=False)

    print(f"synthetic terms={len(syn_map)}  synthetic pairs={len(src)}")
    print(f"pairs: {len(pairs)} -> {len(combined_pairs)}  pos_rate={combined_pairs.label.mean():.3f}")
    print(f"example: '{term2q[list(syn_map)[0]]}' -> '{list(syn_map.values())[0]}'")
    print("Saved augmented_terms.csv + artifacts/train_pairs_augmented.parquet")


if __name__ == "__main__":
    main()
