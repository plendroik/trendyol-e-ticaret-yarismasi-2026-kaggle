"""
Pseudo-labeling: augment the training set with the ensemble's most-confident TEST
predictions. This injects (a) the real test QUERIES (cold-start coverage — test
terms are unseen in training) and (b) the real test negative distribution, which
the synthetic negatives could not match (the LB-0.80 ceiling).

  ensemble score = rank-avg(ce_test_berturk, ce_test_mdeberta_easy)
  top POS_FRAC of test pairs -> pseudo label 1
  bottom NEG_FRAC            -> pseudo label 0
  uncertain middle           -> dropped
Pseudo pairs go to TRAIN folds (0-3) only; the original easy holdout (fold 4) stays
clean. Original train_pairs is backed up and the combined set overwrites it.

Output: artifacts/train_pairs.parquet (original easy + pseudo)   [backs up original]
"""
import os, shutil, numpy as np, pandas as pd

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
POS_FRAC = 0.12     # top 12% by ensemble score -> pseudo positive
NEG_FRAC = 0.25     # bottom 25% -> pseudo negative
SEED = 42
rng = np.random.RandomState(SEED)


def rank_norm(x):
    o = np.argsort(x, kind="stable"); r = np.empty(len(x)); r[o] = np.arange(len(x))
    return r / (len(x) - 1)


def main():
    print("Loading ensemble test scores...")
    a = np.load(os.path.join(ART, "_stale_easy", "ce_test_berturk.npy"))
    b = np.load(os.path.join(ART, "ce_test_mdeberta_easy.npy"))
    ens = (rank_norm(a) + rank_norm(b)) / 2

    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    assert len(sub) == len(ens)
    order = np.argsort(-ens)                      # high score first
    n = len(ens)
    n_pos = int(POS_FRAC * n); n_neg = int(NEG_FRAC * n)
    pos_idx = order[:n_pos]
    neg_idx = order[-n_neg:]
    print(f"test pairs={n}  pseudo_pos={n_pos} (score>={ens[order[n_pos-1]]:.3f})  "
          f"pseudo_neg={n_neg} (score<={ens[order[-n_neg]]:.3f})")

    pseudo = pd.DataFrame({
        "term_id": np.concatenate([sub["term_id"].to_numpy()[pos_idx], sub["term_id"].to_numpy()[neg_idx]]),
        "item_id": np.concatenate([sub["item_id"].to_numpy()[pos_idx], sub["item_id"].to_numpy()[neg_idx]]),
        "label": np.concatenate([np.ones(n_pos, np.int8), np.zeros(n_neg, np.int8)]),
    })
    # pseudo rows only into TRAIN folds 0-3 (keep holdout fold 4 clean)
    pseudo["fold"] = rng.randint(0, 4, size=len(pseudo)).astype(np.int8)
    print(f"  pseudo-labeled test queries covered: {pseudo['term_id'].nunique()} / {sub['term_id'].nunique()}")

    orig_path = os.path.join(ART, "train_pairs.parquet")
    backup = os.path.join(ART, "train_pairs_easy_backup.parquet")
    if not os.path.exists(backup):
        shutil.copy(orig_path, backup); print(f"backed up original -> {backup}")
    orig = pd.read_parquet(backup)                # always augment from the clean easy base
    orig = orig.copy(); orig["src"] = "orig"; pseudo["src"] = "pseudo"

    combined = pd.concat([orig, pseudo], ignore_index=True)
    combined.to_parquet(orig_path, index=False)
    print(f"orig={len(orig)} + pseudo={len(pseudo)} = {len(combined)}  "
          f"pos_rate={combined.label.mean():.3f}")
    print(f"Saved {orig_path}")


if __name__ == "__main__":
    main()
