"""
Build calibrated submissions from the overnight full domain CE.
Produces: single domain-CE submissions AND domain+BERTurk ensemble submissions,
each at PPR 0.30 / 0.33 / 0.36. Run automatically by run_overnight.bat after training.
"""
import os, numpy as np, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")


def rank_norm(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / (len(x) - 1)


def write(score, prefix):
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    for ppr in [0.30, 0.33, 0.36]:
        thr = np.quantile(score, 1 - ppr)
        pred = (score >= thr).astype(int)
        out = os.path.join(DATA, f"submission_{prefix}_p{int(ppr*100)}.csv")
        pd.DataFrame({"id": sub["id"], "prediction": pred}).to_csv(out, index=False)
        print(f"  {prefix} PPR{ppr}: pos_rate={pred.mean():.4f} -> {os.path.basename(out)}")


def main():
    full = np.load(os.path.join(ART, "ce_test_trendyol_full.npy"))
    print("Single full domain CE:")
    write(full, "trendyolfull")

    berturk_p = os.path.join(ART, "_stale_easy", "ce_test_berturk.npy")
    if os.path.exists(berturk_p):
        berturk = np.load(berturk_p)
        ens = (rank_norm(full) + rank_norm(berturk)) / 2
        print("Domain + BERTurk ensemble:")
        write(ens, "ensTF")
    print("ALL SUBMISSIONS READY in", DATA)


if __name__ == "__main__":
    main()
