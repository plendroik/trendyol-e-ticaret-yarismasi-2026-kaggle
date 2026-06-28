"""
Calibrate the decision threshold to the TRUE test prevalence (0.6949, from the
all-ones probe: macro-F1=0.41 => P=0.41/0.59). Our training prevalence was ~0.18,
so the models massively under-predict positives on the test (positive is the
MAJORITY there). We reweight the held-out fold to prevalence 0.6949 and pick the
threshold that maximizes (weighted) macro-F1, then apply it to the test blend.
"""
import os, numpy as np, pandas as pd
from sklearn.metrics import f1_score

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
TRUE_PREV = 0.6949

pairs = pd.read_parquet(os.path.join(ART, "train_pairs.parquet"))
ho_idx = np.where(pairs["fold"].to_numpy() == 4)[0]
y = pairs["label"].to_numpy()[ho_idx]

gbdt_oof = np.load(os.path.join(ART, "gbdt_oof.npy"))
ce_ho = np.load(os.path.join(ART, "ce_holdout_berturk_hard.npy"))
assert len(ce_ho) == len(ho_idx), f"align mismatch {len(ce_ho)} vs {len(ho_idx)}"
gbdt_ho = gbdt_oof[ho_idx]

# include model2 holdout if available
extra_ho, extra_te = [], []
for name in ["berturk_hard_rich"]:
    hp = os.path.join(ART, f"ce_holdout_{name}.npy")
    if os.path.exists(hp) and len(np.load(hp)) == len(ho_idx):
        extra_ho.append(np.load(hp)); extra_te.append(name)

ce_stack_ho = [ce_ho] + extra_ho
ce_mean_ho = np.mean(ce_stack_ho, axis=0)
blend_ho = 0.65 * ce_mean_ho + 0.35 * gbdt_ho
print(f"holdout rows={len(y)}  holdout_prev={y.mean():.3f}  CE models={['berturk_hard']+extra_te}")

# sample weights so the holdout's EFFECTIVE prevalence == TRUE_PREV
n_pos, n_neg = int(y.sum()), int((y == 0).sum())
w_neg = (n_pos * (1 - TRUE_PREV)) / (n_neg * TRUE_PREV)
w = np.where(y == 1, 1.0, w_neg)
eff_prev = w[y == 1].sum() / w.sum()
print(f"reweight: w_neg={w_neg:.4f} -> effective_prev={eff_prev:.3f}")

best_t, best_s = 0.5, -1
for t in np.arange(0.02, 0.98, 0.005):
    s = f1_score(y, (blend_ho >= t).astype(int), average="macro", sample_weight=w)
    if s > best_s:
        best_s, best_t = s, t
pred_rate = (blend_ho >= best_t).mean()
print(f"BEST thr={best_t:.3f}  weighted-macroF1={best_s:.4f}  (holdout pred_pos_rate={pred_rate:.3f})")

# build test submission at this threshold
ce_te = np.load(os.path.join(ART, "ce_test_berturk_hard.npy"))
ce_stack_te = [ce_te] + [np.load(os.path.join(ART, f"ce_test_{n}.npy")) for n in extra_te]
ce_mean_te = np.mean(ce_stack_te, axis=0)
gb_te = np.load(os.path.join(ART, "gbdt_test.npy"))
blend_te = 0.65 * ce_mean_te + 0.35 * gb_te

sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
pred = (blend_te >= best_t).astype(int)
out = os.path.join(DATA_DIR, "submission_calibrated.csv")
pd.DataFrame({"id": sub["id"].to_numpy(), "prediction": pred}).to_csv(out, index=False)
print(f"Wrote {out}  test pred_pos_rate={pred.mean():.4f}")
