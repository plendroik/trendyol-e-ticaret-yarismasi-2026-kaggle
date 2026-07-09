"""
Phase 2: build the JUDGE-DISTILLATION training set — the correct distillation.

The 0.79 failure trained on TRAIN-query ANN candidates (wrong distribution).
These labels are on the TEST pairs themselves: same queries, same candidate
retrieval, same (LLM-ish) labeling function as the ground truth. A domain CE
trained on them generalizes the judge to the 2.88M unlabeled pairs for free.

Labels: test_judge_labels.csv (mini, all bands) + rescue/phase-1a files if
present. GroupKFold-style folds by term (fold 4 = holdout).

Output: artifacts/train_pairs_distill.parquet
"""
import os, numpy as np, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")

sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
frames = []
for f in ["test_judge_labels.csv", "rescue_labels.csv"]:
    p = os.path.join(DATA, f)
    if os.path.exists(p):
        frames.append(pd.read_csv(p))
        print(f"{f}: {len(frames[-1])} rows")
lab = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="first")

df = lab.merge(sub[["id", "term_id", "item_id"]], on="id", how="inner")
print(f"judge-labeled pairs: {len(df)}  pos_rate={df.label.mean():.3f}  terms={df.term_id.nunique()}")

# --- FULL-SPECTRUM AUGMENTATION -------------------------------------------
# Judge labels cover only the HARD bands -> a model trained on them never sees
# easy examples and learns a warped boundary (the 50k-random > 600k-hard effect).
# Add consensus-certain easy pairs: double-confident negatives and positives.
ce = np.load(os.path.join(ART, "ce_test_trendyol_ce.npy"))
ds = np.load(os.path.join(ART, "ce_test_distill.npy"))
have = set(df["id"].astype(str))
ids_all = sub["id"].astype(str).to_numpy()
rng = np.random.RandomState(42)

def sample(mask, k, label):
    idx = np.where(mask)[0]
    idx = np.array([i for i in idx if ids_all[i] not in have])
    take = rng.choice(idx, min(k, len(idx)), replace=False)
    out = sub.iloc[take][["id", "term_id", "item_id"]].copy()
    out["label"] = label
    return out

easy_neg = sample((ce < 0.03) & (ds < 0.3), 500_000, 0)
easy_pos = sample((ce > 0.97) & (ds > 0.7), 250_000, 1)
print(f"easy-neg added: {len(easy_neg)}  easy-pos added: {len(easy_pos)}")
df = pd.concat([df, easy_neg, easy_pos], ignore_index=True)
df = df.drop_duplicates("id", keep="first")
print(f"FULL set: {len(df)}  pos_rate={df.label.mean():.3f}")

# term-grouped folds (no term leaks across folds)
terms = df["term_id"].unique()
rng = np.random.RandomState(42)
rng.shuffle(terms)
fold_of = {t: k % 5 for k, t in enumerate(terms)}
df["fold"] = df["term_id"].map(fold_of).astype(np.int8)

out = df[["term_id", "item_id", "label", "fold"]]
print(out.groupby("fold").agg(n=("label", "size"), pos=("label", "mean")))
out.to_parquet(os.path.join(ART, "train_pairs_distill.parquet"), index=False)
print(f"Saved -> {os.path.join(ART, 'train_pairs_distill.parquet')}")
