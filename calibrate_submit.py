"""
Quick test-level blend + prevalence-calibrated thresholding (no holdout alignment
needed). Produces submissions at target predicted positive-rates.

Best model is berturk_hard (neutral-eval 0.747, well-calibrated). GBDT adds a bit.
True test prevalence ~0.124 (from the neutral hard-eval).
"""
import os, numpy as np, pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")

ce = np.load(os.path.join(ART, "ce_test_berturk_hard.npy"))
gb = np.load(os.path.join(ART, "gbdt_test.npy"))
# extra models if present (test-level average)
extra = {}
for name in ["berturk_hard_rich"]:
    p = os.path.join(ART, f"ce_test_{name}.npy")
    if os.path.exists(p):
        extra[name] = np.load(p)

sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
ids = sub["id"].to_numpy()

# weighted blend: cross-encoder(s) dominate, GBDT supports
ce_stack = [ce] + list(extra.values())
ce_mean = np.mean(ce_stack, axis=0)
blend = 0.65 * ce_mean + 0.35 * gb
print(f"models: berturk_hard + gbdt" + (" + " + " + ".join(extra) if extra else ""))

for target in [0.42, 0.47, 0.52]:
    thr = np.quantile(blend, 1 - target)
    pred = (blend >= thr).astype(int)
    out = os.path.join(DATA_DIR, f"submission_p{int(target*100):02d}.csv")
    pd.DataFrame({"id": ids, "prediction": pred}).to_csv(out, index=False)
    print(f"  target_pos={target:.2f} thr={thr:.4f} actual_pos={pred.mean():.4f} -> {os.path.basename(out)}")

# berturk_hard alone at its well-calibrated threshold 0.5
pred = (ce >= 0.5).astype(int)
pd.DataFrame({"id": ids, "prediction": pred}).to_csv(
    os.path.join(DATA_DIR, "submission_ce_only.csv"), index=False)
print(f"  ce_only thr=0.50 pos_rate={pred.mean():.4f} -> submission_ce_only.csv")
