"""
Blend GBDT + all cross-encoder scores, calibrate the macro-F1 threshold, and
write submission candidates.

Auto-discovers every artifacts/ce_test_<suffix>.npy (+ matching ce_holdout_<suffix>.npy).
To avoid threshold/weight overfitting, the held-out fold is split by term into
A (fit) / B (report). Final weights+threshold are refit on the full holdout.

Inputs (artifacts/):
  train_pairs.parquet, gbdt_oof.npy, gbdt_test.npy,
  ce_holdout_meta.npy, ce_holdout_<suffix>.npy, ce_test_<suffix>.npy
Outputs (DATA_DIR):
  submission_blend.csv    stacked ensemble (max score)
  submission_robust.csv   best single model (shake-up resistant)
"""
import os
import sys
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_fscore_support

try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART_DIR = os.path.join(DATA_DIR, "artifacts")


def log(m):
    print(m, flush=True)


def best_threshold(y, p):
    bt, bs = 0.5, -1.0
    for t in np.arange(0.05, 0.95, 0.005):
        s = f1_score(y, (p >= t).astype(int), average="macro")
        if s > bs:
            bs, bt = s, t
    return bt, bs


def main():
    meta = np.load(os.path.join(ART_DIR, "ce_holdout_meta.npy"))
    labels = meta[:, 0].astype(int)
    ho_rows = meta[:, 1].astype(int)

    gbdt_oof = np.load(os.path.join(ART_DIR, "gbdt_oof.npy"))
    gbdt_test = np.load(os.path.join(ART_DIR, "gbdt_test.npy"))

    # collect models: name -> (holdout_scores, test_scores)
    models = {"gbdt": (gbdt_oof[ho_rows], gbdt_test)}
    for hf in sorted(glob.glob(os.path.join(ART_DIR, "ce_holdout_*.npy"))):
        suf = os.path.basename(hf)[len("ce_holdout_"):-len(".npy")]
        tf = os.path.join(ART_DIR, f"ce_test_{suf}.npy")
        if os.path.exists(tf):
            models[suf] = (np.load(hf), np.load(tf))
    names = list(models.keys())
    log(f"Models: {names}")

    # individual holdout macro-F1
    for n in names:
        t, s = best_threshold(labels, models[n][0])
        log(f"  {n:12s} holdout macro-F1={s:.5f} (thr={t:.3f})")

    # term-grouped A/B split of the holdout to fit/report honestly
    pairs = pd.read_parquet(os.path.join(ART_DIR, "train_pairs.parquet"))
    ho_terms = pairs["term_id"].to_numpy()[ho_rows]
    uniq = pd.unique(ho_terms)
    rng = np.random.RandomState(0)
    a_terms = set(rng.choice(uniq, size=len(uniq) // 2, replace=False))
    A = np.array([t in a_terms for t in ho_terms])
    B = ~A

    Hho = np.stack([models[n][0] for n in names], axis=1)  # holdout feature matrix
    yA, yB = labels[A], labels[B]

    # --- stacking meta-learner (fit on A, threshold on A, report on B)
    lr = LogisticRegression(C=1.0, max_iter=1000)
    lr.fit(Hho[A], yA)
    pA = lr.predict_proba(Hho[A])[:, 1]
    tA, _ = best_threshold(yA, pA)
    pB = lr.predict_proba(Hho[B])[:, 1]
    sB = f1_score(yB, (pB >= tA).astype(int), average="macro")
    log(f"STACK  report-on-B macro-F1={sB:.5f}  weights={dict(zip(names, np.round(lr.coef_[0],3)))}")

    # --- simple average (report on B)
    avgB = Hho[B].mean(axis=1)
    tavg, _ = best_threshold(yA, Hho[A].mean(axis=1))
    savg = f1_score(yB, (avgB >= tavg).astype(int), average="macro")
    log(f"AVG    report-on-B macro-F1={savg:.5f}")

    # --- best single (on B)
    best_single, bss = None, -1
    for n in names:
        t, _ = best_threshold(yA, models[n][0][A])
        s = f1_score(yB, (models[n][0][B] >= t).astype(int), average="macro")
        if s > bss:
            bss, best_single = s, n
    log(f"BEST SINGLE on B: {best_single} macro-F1={bss:.5f}")

    # ============ FINAL: refit stack on full holdout, pick threshold on full holdout
    lr_full = LogisticRegression(C=1.0, max_iter=1000)
    lr_full.fit(Hho, labels)
    p_full = lr_full.predict_proba(Hho)[:, 1]
    t_full, s_full = best_threshold(labels, p_full)
    log(f"FINAL stack full-holdout macro-F1={s_full:.5f} (thr={t_full:.3f})")
    pr, rc, f1, _ = precision_recall_fscore_support(
        labels, (p_full >= t_full).astype(int), labels=[0, 1], average=None)
    log(f"  class0 F1={f1[0]:.3f} R={rc[0]:.3f} | class1 F1={f1[1]:.3f} R={rc[1]:.3f}")

    Hte = np.stack([models[n][1] for n in names], axis=1)
    blend_test = lr_full.predict_proba(Hte)[:, 1]
    blend_pred = (blend_test >= t_full).astype(int)

    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    pd.DataFrame({"id": sub["id"].to_numpy(), "prediction": blend_pred}).to_csv(
        os.path.join(DATA_DIR, "submission_blend.csv"), index=False)
    log(f"Wrote submission_blend.csv  pos_rate={blend_pred.mean():.4f}")

    # robust: best single model with threshold from full holdout
    bs_ho, bs_te = models[best_single]
    t_rob, _ = best_threshold(labels, bs_ho)
    rob_pred = (bs_te >= t_rob).astype(int)
    pd.DataFrame({"id": sub["id"].to_numpy(), "prediction": rob_pred}).to_csv(
        os.path.join(DATA_DIR, "submission_robust.csv"), index=False)
    log(f"Wrote submission_robust.csv ({best_single})  pos_rate={rob_pred.mean():.4f}")


if __name__ == "__main__":
    main()
