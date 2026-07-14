# =============================================================================
# PURE OPEN-SOURCE STACKER: 100% OpenAI/Closed-Source Free
#
# Bu script, OpenAI (GPT-4o, GPT-4o-mini) etiketlerini tamamen devre disi birakarak
# sadece acik kaynakli Qwen-72B/Qwen-8B ile uretilen 'clean_labels_72b.csv' verisini kullanır.
# Boylece hem stacker eğitimi hem de nihai submission %100 acik kaynak kurallarina
# uygun ve etik olarak temiz hale gelir.
# =============================================================================
import os, re, zlib, time, pickle
import numpy as np, pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
EMB = os.path.join(DATA, "emb")

def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)

log("Scores...")
S = {}
# Acik kaynakli cross-encoder modelleri
for name in ["trendyol_ce", "trendyol_full", "trendyol_ai", "trendyol_qaug",
             "trendyol_llm", "distill", "tybert_distill", "tybert_distill2", "distill2",
             "xlmr_distill", "72b_domain", "72b_tybert", "72b_xlmrL2"]:
    p = os.path.join(ART, f"ce_test_{name}.npy")
    if os.path.exists(p):
        S[name] = np.load(p).astype(np.float32)
        log(f"  + {name}")

sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
n = len(sub)
ids = sub["id"].astype(str).to_numpy()
tids = sub["term_id"].to_numpy()
iids = sub["item_id"].to_numpy()

log("Embedding cosine...")
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

log("Lexical features...")
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

log("Within-term rank features...")
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
log(f"X shape: {X.shape}")

# OpenAI etiketlerini tamamen devre disi birakip, sadece Qwen (Acik Kaynak) kullaniyoruz
log("Loading Qwen-72B / Qwen-8B Open Source Labels...")
id2lab = {}
p = os.path.join(DATA, "clean_labels_72b.csv")
assert os.path.exists(p), "clean_labels_72b.csv bulunamadi!"
d = pd.read_csv(p)
for i_, l_ in zip(d["id"].astype(str), d["label"].astype(int)):
    id2lab[i_] = l_

y = np.full(n, -1, dtype=np.int8)
for k in range(n):
    v = id2lab.get(ids[k])
    if v is not None:
        y[k] = v
mask = y >= 0
log(f"Labeled rows (OS Only): {mask.sum():,}  pos_rate={y[mask].mean():.4f}")

# Fold yapilandirmasi
fold = np.array([zlib.crc32(str(t).encode()) % 5 for t in tids], dtype=np.int8)

log("Training LGBM + CatBoost Stacker (OS Only)...")
Xl = X[mask]; yl = y[mask].astype(np.int32); fl = fold[mask]
oof_lgb = np.zeros(len(yl))
oof_cb = np.zeros(len(yl))

for f in range(5):
    tr = fl != f; va = fl == f
    
    # LightGBM
    lgb_model = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                                   num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
                                   bagging_fraction=0.9, bagging_freq=1, n_jobs=-1, verbose=-1)
    lgb_model.fit(Xl[tr], yl[tr], eval_set=[(Xl[va], yl[va])], callbacks=[lgb.early_stopping(50, verbose=False)])
    oof_lgb[va] = lgb_model.predict_proba(Xl[va])[:, 1]
    
    # CatBoost
    cb_model = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=8,
                                  loss_function="Logloss", verbose=False)
    cb_model.fit(Xl[tr], yl[tr], eval_set=(Xl[va], yl[va]), early_stopping_rounds=50, verbose=False)
    oof_cb[va] = cb_model.predict_proba(Xl[va])[:, 1]
    
    log(f"  Fold {f+1} complete.")

oof_ens = (oof_lgb + oof_cb) / 2.0
best = max((f1_score(yl, (oof_ens >= t).astype(int), average="macro"), t)
           for t in np.arange(0.3, 0.71, 0.02))
log(f"OOF macro-F1 (hakeme karsi) = {best[0]:.5f} @ thr={best[1]:.2f}")

log("Inference on all 3.36M pairs...")
pred_lgb = np.zeros(n, dtype=np.float32)
pred_cb = np.zeros(n, dtype=np.float32)

for f in range(5):
    tr = fl != f; va = fl == f
    
    lgb_model = lgb.LGBMClassifier(objective="binary", n_estimators=600, learning_rate=0.05,
                                   num_leaves=127, min_data_in_leaf=100, feature_fraction=0.9,
                                   bagging_fraction=0.9, bagging_freq=1, n_jobs=-1, verbose=-1)
    lgb_model.fit(Xl[tr], yl[tr])
    pred_lgb += lgb_model.predict_proba(X)[:, 1] / 5.0
    
    cb_model = CatBoostClassifier(iterations=600, learning_rate=0.05, depth=8,
                                  loss_function="Logloss", verbose=False)
    cb_model.fit(Xl[tr], yl[tr])
    pred_cb += cb_model.predict_proba(X)[:, 1] / 5.0

pred_ens = (pred_lgb + pred_cb) / 2.0

# Sadece acik kaynakli etiketlerle override uyguluyoruz
final = np.where(mask, y, (pred_ens >= best[1]).astype(np.int8)).astype(int)

log(f"Final pos_rate = {final.mean():.4f} (Target ~0.30)")

out_path = os.path.join(DATA, "submission_ensemble_pure_os.csv")
pd.DataFrame({"id": ids, "prediction": final}).to_csv(out_path, index=False)
log(f"Saved pure OS submission to: {out_path}")
