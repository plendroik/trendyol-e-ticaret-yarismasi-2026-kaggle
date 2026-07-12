"""CLEAN distill trainset: open-source judge labels (Qwen3-8B, Kaggle) + easy
tails from the CLEAN 0.83 CE only. No closed-model signal anywhere."""
import os, glob, numpy as np, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")

sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
frames = [pd.read_csv(f) for f in sorted(glob.glob(os.path.join(DATA, "clean_labels_part*.csv")))]
lab = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="first")
print(f"temiz hakem etiketi: {len(lab):,}  pos={lab.label.mean():.3f}")
df = lab.merge(sub[["id", "term_id", "item_id"]], on="id", how="inner")

ce = np.load(os.path.join(ART, "ce_test_trendyol_ce.npy"))
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

easy_neg = sample(ce < 0.03, 500_000, 0)
easy_pos = sample(ce > 0.97, 250_000, 1)
df = pd.concat([df, easy_neg, easy_pos], ignore_index=True).drop_duplicates("id")
print(f"TAM temiz set: {len(df):,}  pos={df.label.mean():.3f}")

import zlib
df["fold"] = [zlib.crc32(str(t).encode()) % 5 for t in df["term_id"]]
df["fold"] = df["fold"].astype(np.int8)
df[["term_id", "item_id", "label", "fold"]].to_parquet(
    os.path.join(ART, "train_pairs_clean.parquet"), index=False)
print("Saved -> train_pairs_clean.parquet")
