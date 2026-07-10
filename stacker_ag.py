"""AutoGluon stacker: same feature matrix as build_stacker, AG ensemble on top.
Trains on judge-labeled rows, predicts all 3.36M -> artifacts/stacker_ag.npy"""
import os, time, numpy as np, pandas as pd
from autogluon.tabular import TabularPredictor
from sklearn.metrics import f1_score

ART = r"C:\Users\ASUS\Desktop\trendyol\artifacts"
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

z = np.load(os.path.join(ART, "stacker_features.npz"), allow_pickle=True)
X, y, fold, names = z["X"], z["y"], z["fold"], list(z["feat_names"])
mask = y >= 0
df = pd.DataFrame(X[mask], columns=names); df["label"] = y[mask]
# holdout for honest eval: fold 4 (term-grouped)
tr = df[fold[mask] != 4]; va = df[fold[mask] == 4]
log(f"train={len(tr)} val={len(va)}")
pred = TabularPredictor(label="label", eval_metric="f1_macro", verbosity=1).fit(
    tr, time_limit=3600, presets="good_quality", num_gpus=0)
p = pred.predict_proba(va.drop(columns=["label"]))[1].to_numpy()
best = max((f1_score(va["label"], (p >= t).astype(int), average="macro"), t)
           for t in np.arange(0.3, 0.71, 0.02))
log(f"AG holdout macroF1 = {best[0]:.4f} @ thr={best[1]:.2f}  (LGBM OOF: 0.9197)")
log("scoring all rows...")
allp = pred.predict_proba(pd.DataFrame(X, columns=names))[1].to_numpy().astype(np.float32)
np.save(os.path.join(ART, "stacker_ag.npy"), allp)
log("DONE AG -> stacker_ag.npy")
