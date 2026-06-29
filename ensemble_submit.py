"""
CE-only ensemble + prevalence-calibrated submission.

Rank-averages several cross-encoder TEST score files (robust to per-model score
scale), then thresholds by PERCENTILE to a target predicted-positive rate (PPR).
True test prevalence ~0.30 (NOT 0.695 — that was a misread all-zeros probe), so
the macro-F1 optimum PPR is ~0.28-0.34. NO GBDT (it dilutes the CE).

Usage:
  python ensemble_submit.py path1.npy path2.npy ...   (paths to ce_test_*.npy)
  (no args -> uses the default easy-CE set)
"""
import os, sys, numpy as np, pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
TARGET_PPRS = [0.30, 0.33, 0.36]


def rank_norm(x):
    """Map scores to [0,1] percentile ranks (ties broken by position)."""
    order = np.argsort(x, kind="stable")
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.arange(len(x))
    return r / (len(x) - 1)


def main():
    paths = sys.argv[1:]
    if not paths:
        paths = [
            os.path.join(ART, "_stale_easy", "ce_test_berturk.npy"),       # the 0.80 model
            os.path.join(ART, "ce_test_berturk_easy_fgm.npy"),             # new rich+FGM model
        ]
    paths = [p for p in paths if os.path.exists(p)]
    print("ensemble members:")
    ranks = []
    for p in paths:
        s = np.load(p)
        ranks.append(rank_norm(s))
        print(f"  {os.path.basename(p)}  (mean={s.mean():.4f})")
    blend = np.mean(ranks, axis=0)

    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    ids = sub["id"].to_numpy()
    for ppr in TARGET_PPRS:
        thr = np.quantile(blend, 1 - ppr)
        pred = (blend >= thr).astype(int)
        tag = f"ens{len(paths)}_p{int(ppr*100)}"
        out = os.path.join(DATA_DIR, f"submission_{tag}.csv")
        pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)
        print(f"  PPR={ppr:.2f} actual={pred.mean():.4f} -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
