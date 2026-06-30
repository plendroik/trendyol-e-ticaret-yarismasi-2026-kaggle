"""
PER-QUERY thresholding (post-processing, no retraining).

The test has ~104 candidates per query, ~30% relevant. A GLOBAL threshold mis-
calibrates across queries (a uniformly low-scored query gets ~no positives; a
high-scored one gets too many). Instead, for EACH query predict its top FRAC of
candidates as relevant -> exploits the per-query candidate structure.

Usage: python per_query_threshold.py <ce_test_scores.npy> <frac>
"""
import os, sys, numpy as np, pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")


def main():
    score_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ART, "_stale_easy", "ce_test_berturk.npy")
    fracs = [float(x) for x in sys.argv[2:]] or [0.30, 0.33, 0.36]
    s = np.load(score_path)
    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    sub["score"] = s
    # percentile rank within each term (0=highest score). top FRAC -> positive.
    sub["pct"] = sub.groupby("term_id")["score"].rank(pct=True, ascending=False)
    print(f"model={os.path.basename(score_path)}  terms={sub['term_id'].nunique()}  "
          f"avg_cand/term={len(sub)/sub['term_id'].nunique():.1f}")
    for frac in fracs:
        pred = (sub["pct"] <= frac).astype(int)
        tag = f"perq_p{int(frac*100)}"
        out = os.path.join(DATA_DIR, f"submission_{tag}.csv")
        pd.DataFrame({"id": sub["id"], "prediction": pred}).to_csv(out, index=False)
        print(f"  frac={frac:.2f}  overall_pos_rate={pred.mean():.4f} -> {os.path.basename(out)}")


if __name__ == "__main__":
    main()
