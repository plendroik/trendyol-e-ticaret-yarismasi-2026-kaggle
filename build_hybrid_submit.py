"""
Hybrid submission: confident CE decisions + LLM labels on the uncertain band.

- CE score > HI  -> 1   (model is confident relevant)
- CE score < LO  -> 0   (model is confident irrelevant)
- LO <= score <= HI -> use test_judge_labels.csv (LLM judge)
Band pairs with no LLM label yet fall back to the CE threshold decision.

Reports the resulting positive-rate; aim ~0.30-0.33 (test prevalence). If off,
nudge LO/HI or post-adjust, but the band labels should self-calibrate.

Usage: python build_hybrid_submit.py --lo 0.2 --hi 0.85 --thr 0.5
Output: submission_judge.csv
"""
import os, argparse, numpy as np, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.20)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--thr", type=float, default=0.50,
                    help="CE fallback threshold for band pairs without an LLM label")
    ap.add_argument("--out", default="submission_judge.csv")
    args = ap.parse_args()

    ce = np.load(os.path.join(ART, "ce_test_trendyol_ce.npy"))
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    assert len(ce) == len(sub)

    pred = (ce >= args.thr).astype(int)          # baseline decision everywhere
    pred[ce > args.hi] = 1
    pred[ce < args.lo] = 0

    lab = pd.read_csv(os.path.join(DATA, "test_judge_labels.csv"))
    id2lab = dict(zip(lab["id"].astype(str), lab["label"].astype(int)))
    ids = sub["id"].astype(str).to_numpy()
    band = (ce >= args.lo) & (ce <= args.hi)
    n_over = 0
    for i in np.where(band)[0]:
        v = id2lab.get(ids[i])
        if v is not None:
            pred[i] = v; n_over += 1
    print(f"band pairs={band.sum():,}  LLM-labeled overrides={n_over:,}")
    print(f"final pos_rate={pred.mean():.4f}  (hedef ~0.30-0.33)")

    out = os.path.join(DATA, args.out)
    pd.DataFrame({"id": sub["id"].values, "prediction": pred}).to_csv(out, index=False)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
