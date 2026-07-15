# =============================================================================
# ADVANCED STACKER: 100% OpenAI/Closed-Source Free
#
# Blends LightGBM, CatBoost, and XGBoost using predictions from all local and
# Kaggle-trained cross-encoders. Integrates:
#   1. XGBoost Classifier as a 3rd stacking layer.
#   2. Weighted ensembling optimized via grid-search on the holdout fold.
#   3. Finer decision threshold search (step=0.002) for maximum F1 boost.
# =============================================================================
import os, re, zlib, time, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
import xgboost as xgb
from sklearn.metrics import f1_score

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
EMB = os.path.join(DATA, "emb")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("Loading model predictions from artifacts...")
S = {}
# Acik kaynakli modeller listesi (Yeni mdeberta ve domain_fgm modelleri dahil edildi)
model_names = ["trendyol_ce", "trendyol_full", "trendyol_ai", "trendyol_qaug",
               "trendyol_llm", "distill", "tybert_distill", "tybert_distill2", "distill2",
               "xlmr_distill", "72b_domain", "72b_tybert", "72b_xlmrL2",
               "mdeberta", "domain_fgm"]

for name in model_names:
    p = os.path.join(ART, f"ce_test_{name}.npy")
    if os.path.exists(p):
        S[name] = np.load(p).astype(np.float32)
        log(f"  + {name}")

sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
n = len(sub)
ids = sub["id"].astype(str).to_numpy()
tids = sub["term_id"].to_numpy()
iids = sub["item_id"].to_numpy()

log("Loading query and item embeddings for cosine features...")
q_emb = np.load(os.path.join(EMB, "query_emb.npy")).astype(np.float32)
i_emb = np.load(os.path.join(EMB, "item_emb.npy")).astype(np.float32)
q_ids = np.load(os.path.join(EMB, "query_ids.npy"), allow_pickle=True)
i_ids = np.load(os.path.join(EMB, "item_ids.npy"), allow_pickle=True)
qrow = {t: k for k, t in enumerate(q_ids)}
irow = {t: k for k, t in enumerate(i_ids)}
qi = np.array([qrow.get(t, -1) for t in tids], dtype=np.int64)
ii = np.array([irow.get(t, -1) for t in iids], dtype=np.int64)

cos = np.zeros(n, dtype=np.float32)
B = 200000
for b in range(0, n, B):
    qs = q_emb[qi[b:b+B]]; its = i_emb[ii[b:b+B]]
    cos[b:b+B] = np.einsum("ij,ij->i", qs, its)
cos[(qi < 0) | (ii < 0)] = 0.0

log("Building lexical overlap features...")
terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
items = pd.read_csv(os.path.join(DATA, "items.csv"))
tok = re.compile(r"[a-z0-9çğıöşü]+")
t2toks = {r.term_id: frozenset(tok.findall(str(r.query).lower())) for r in terms.itertuples(index=False)}
i2toks = {r.item_id: frozenset(tok.findall(str(r.title).lower())) for r in items.itertuples(index=False)}
ov = np.zeros(n, np.float32); jac = np.zeros(n, np.float32)
qn = np.zeros(n, np.float32); tn = np.zeros(n, np.float32)
for k in range(n):
    a = t2toks.get(tids[k], frozenset()); b = i2toks.get(iids[k], frozenset())
    inter = len(a & b); uni = len(a | b) or 1
    ov[k] = inter; jac[k] = inter / uni; qn[k] = len(a); tn[k] = len(b)
    if k and k % 1000000 == 0:
        log(f"  lex {k}/{n}")

log("Building ranking and statistics features...")
df_r = pd.DataFrame({"term": tids, "d2": S["distill2"], "tb2": S["tybert_distill2"], "cos": cos})
g = df_r.groupby("term")
rank_d2 = g["d2"].rank(pct=True).to_numpy(np.float32)
rank_tb2 = g["tb2"].rank(pct=True).to_numpy(np.float32)
rank_cos = g["cos"].rank(pct=True).to_numpy(np.float32)
z_d2 = ((df_r["d2"] - g["d2"].transform("mean")) / (g["d2"].transform("std") + 1e-6)).to_numpy(np.float32)
n_cand = g["d2"].transform("size").to_numpy(np.float32)

X = np.column_stack(list(S.values()) +
                    [cos, ov, jac, qn, tn,
                     rank_d2, rank_tb2, rank_cos, z_d2, n_cand]).astype(np.float32)
feat_names = list(S.keys()) + ["emb_cos", "tok_overlap", "jaccard", "q_ntok", "t_ntok",
                               "rank_d2", "rank_tb2", "rank_cos", "z_d2", "n_cand"]
log(f"Features loaded! shape={X.shape}")

log("Loading Qwen-72B / Qwen-8B Open Source Labels...")
id2lab = {}
p_labels = os.path.join(DATA, "clean_labels_72b.csv")
assert os.path.exists(p_labels), "clean_labels_72b.csv bulunamadi!"
d = pd.read_csv(p_labels)
for i_, l_ in zip(d["id"].astype(str), d["label"].astype(int)):
    id2lab[i_] = l_

y = np.full(n, -1, dtype=np.int8)
for k in range(n):
    v = id2lab.get(ids[k])
    if v is not None:
        y[k] = v
mask = y >= 0
log(f"Labeled rows (OS Only): {mask.sum():,} | pos_rate={y[mask].mean():.4f}")

fold = np.array([zlib.crc32(str(t).encode()) % 5 for t in tids], dtype=np.int8)

log("Training LGBM + CatBoost + XGBoost Stacker (5-Fold CV)...")
Xl = X[mask]; yl = y[mask].astype(np.int32); fl = fold[mask]
oof_lgb = np.zeros(len(yl))
oof_cb = np.zeros(len(yl))
oof_xgb = np.zeros(len(yl))

for f in range(5):
    tr = fl != f; va = fl == f
    
    # 1. LightGBM
    lgb_model = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                                   num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
                                   bagging_fraction=0.9, bagging_freq=1, n_jobs=-1, verbose=-1)
    lgb_model.fit(Xl[tr], yl[tr], eval_set=[(Xl[va], yl[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = lgb_model.predict_proba(Xl[va])[:, 1]
    
    # 2. CatBoost
    cb_model = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=8,
                                  loss_function="Logloss", verbose=False)
    cb_model.fit(Xl[tr], yl[tr], eval_set=(Xl[va], yl[va]), early_stopping_rounds=50, verbose=False)
    oof_cb[va] = cb_model.predict_proba(Xl[va])[:, 1]
    
    # 3. XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7,
                                  min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", early_stopping_rounds=50, n_jobs=-1)
    xgb_model.fit(Xl[tr], yl[tr], eval_set=[(Xl[va], yl[va])], verbose=False)
    oof_xgb[va] = xgb_model.predict_proba(Xl[va])[:, 1]
    
    log(f"  Fold {f+1}/5 finished successfully.")

# Ağırlık optimizasyonu (Grid Search on OOF F1 score)
log("Optimizing blending weights...")
best_f1 = -1
best_weights = (0.5, 0.5, 0.0)
best_thr = 0.5

# Olası ağırlık kombinasyonlarını tara (toplamı 1.0 olacak şekilde)
weight_options = []
for w1 in np.arange(0.0, 1.01, 0.1):
    for w2 in np.arange(0.0, 1.01 - w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 >= -1e-6:
            weight_options.append((w1, w2, max(0.0, w3)))

for w1, w2, w3 in weight_options:
    blend_oof = w1 * oof_lgb + w2 * oof_cb + w3 * oof_xgb
    # Eşik değerini hassas tara (0.01 adımlarla kabaca, en iyi kombinasyonu bulmak icin)
    for t in np.arange(0.3, 0.71, 0.01):
        score_f1 = f1_score(yl, (blend_oof >= t).astype(int), average="macro")
        if score_f1 > best_f1:
            best_f1 = score_f1
            best_weights = (w1, w2, w3)
            best_thr = t

log(f"Best Blending Weights (LGB, CB, XGB): {best_weights[0]:.2f}, {best_weights[1]:.2f}, {best_weights[2]:.2f}")
log(f"Rough OOF F1: {best_f1:.5f} @ thr={best_thr:.2f}")

# En iyi ağırlık kombinasyonu ile aşırı hassas eşik araması (0.002 adımlarla)
log("Running fine-grained threshold optimization (step=0.002)...")
best_blend_oof = best_weights[0] * oof_lgb + best_weights[1] * oof_cb + best_weights[2] * oof_xgb
best_f1_fine = -1
best_thr_fine = 0.5

for t in np.arange(0.3, 0.71, 0.002):
    score_f1 = f1_score(yl, (best_blend_oof >= t).astype(int), average="macro")
    if score_f1 > best_f1_fine:
        best_f1_fine = score_f1
        best_thr_fine = t

log(f"OPTIMAL OOF macro-F1: {best_f1_fine:.5f} @ thr={best_thr_fine:.4f}")

log("Inference and prediction blending on all 3.36M pairs...")
pred_lgb = np.zeros(n, dtype=np.float32)
pred_cb = np.zeros(n, dtype=np.float32)
pred_xgb = np.zeros(n, dtype=np.float32)

for f in range(5):
    tr = fl != f
    
    lgb_model = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                                   num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
                                   bagging_fraction=0.9, bagging_freq=1, n_jobs=-1, verbose=-1)
    lgb_model.fit(Xl[tr], yl[tr])
    pred_lgb += lgb_model.predict_proba(X)[:, 1] / 5.0
    
    cb_model = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=8,
                                  loss_function="Logloss", verbose=False)
    cb_model.fit(Xl[tr], yl[tr])
    pred_cb += cb_model.predict_proba(X)[:, 1] / 5.0

    xgb_model = xgb.XGBClassifier(n_estimators=600, learning_rate=0.05, max_depth=7,
                                  min_child_weight=10, subsample=0.9, colsample_bytree=0.9,
                                  eval_metric="logloss", n_jobs=-1)
    xgb_model.fit(Xl[tr], yl[tr], verbose=False)
    pred_xgb += xgb_model.predict_proba(X)[:, 1] / 5.0

final_pred_ens = best_weights[0] * pred_lgb + best_weights[1] * pred_cb + best_weights[2] * pred_xgb

# Karar sınırını ve etiket ezmelerini uygula
final = np.where(mask, y, (final_pred_ens >= best_thr_fine).astype(np.int8)).astype(int)
log(f"Final positive prevalence: {final.mean():.4f} (Target ~0.30)")

out_path = os.path.join(DATA, "submission_ensemble_advanced.csv")
pd.DataFrame({"id": ids, "prediction": final}).to_csv(out_path, index=False)
log(f"Saved advanced ensemble submission to: {out_path}")
log("PROCESS COMPLETED SUCCESSFULLY!")
