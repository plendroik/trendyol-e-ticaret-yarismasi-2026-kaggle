"""
LightGBM STACKER (the 0.94-recipe): meta-learner over all model scores +
embedding/lexical features, trained on the LLM-judge labels (~590k test pairs).

Features per (query, item) pair:
  - 7 model scores: ce, full, ai, qaug, llm, distill, tybert_distill
  - embedding cosine (TY-ecomm bi-encoder)
  - lexical: token overlap count, jaccard, query token count, title token count
Labels: test_judge_labels + rescue_labels (mini judge), gpt4o_labels override last.
CV: 5 folds grouped by term_id (hash) -> OOF macro-F1 vs judge labels.
Refit on all labels -> score ALL 3.36M pairs -> artifacts/stacker_test.npy

Then v10 submission: judge label where available, stacker elsewhere (threshold
from OOF-best, sanity-checked against overall pos_rate ~0.30).

Usage: python build_stacker.py
"""
import os, re, zlib, time
import numpy as np, pandas as pd
import lightgbm as lgb
from sklearn.metrics import f1_score

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
EMB = os.path.join(DATA, "emb")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


log("Scores...")
S = {}
for name in ["trendyol_ce", "trendyol_full", "trendyol_ai", "trendyol_qaug",
             "trendyol_llm", "distill", "tybert_distill", "tybert_distill2", "distill2",
             "xlmr_distill"]:
    S[name] = np.load(os.path.join(ART, f"ce_test_{name}.npy")).astype(np.float32)

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
# normalized embeddings assumed; compute row-wise dot in chunks
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
t2toks = {r.term_id: set(tok.findall(str(r.query).lower())) for r in terms.itertuples(index=False)}
i2toks = {r.item_id: set(tok.findall(str(r.title).lower())) for r in items.itertuples(index=False)}
ov = np.zeros(n, np.float32); jac = np.zeros(n, np.float32)
qn = np.zeros(n, np.float32); tn = np.zeros(n, np.float32)
for k in range(n):
    a = t2toks.get(tids[k], set()); b = i2toks.get(iids[k], set())
    inter = len(a & b); uni = len(a | b) or 1
    ov[k] = inter; jac[k] = inter / uni; qn[k] = len(a); tn[k] = len(b)
    if k % 1000000 == 0:
        log(f"  lex {k}/{n}")

log("Within-term rank features (LTR)...")
# her adayin kendi sorgusunun aday havuzundaki goreli konumu — mutlak skordan
# cok daha kararli bir sinyal (sorgu basina ~104 rakip)
df_r = pd.DataFrame({"term": tids, "d2": S["distill2"], "tb2": S["tybert_distill2"],
                     "cos": cos})
g = df_r.groupby("term")
rank_d2 = g["d2"].rank(pct=True).to_numpy(np.float32)
rank_tb2 = g["tb2"].rank(pct=True).to_numpy(np.float32)
rank_cos = g["cos"].rank(pct=True).to_numpy(np.float32)
z_d2 = ((df_r["d2"] - g["d2"].transform("mean")) /
        (g["d2"].transform("std") + 1e-6)).to_numpy(np.float32)
n_cand = g["d2"].transform("size").to_numpy(np.float32)

X = np.column_stack(list(S.values()) +
                    [cos, ov, jac, qn, tn,
                     rank_d2, rank_tb2, rank_cos, z_d2, n_cand]).astype(np.float32)
feat_names = list(S.keys()) + ["emb_cos", "tok_overlap", "jaccard", "q_ntok", "t_ntok",
                               "rank_d2", "rank_tb2", "rank_cos", "z_d2", "n_cand"]
log(f"X: {X.shape}")

log("Labels...")
lab = pd.read_csv(os.path.join(DATA, "test_judge_labels.csv"))
frames = [lab]
p = os.path.join(DATA, "rescue_labels.csv")
if os.path.exists(p):
    frames.append(pd.read_csv(p))
lab = pd.concat(frames, ignore_index=True).drop_duplicates("id", keep="first")
id2lab = dict(zip(lab["id"].astype(str), lab["label"].astype(int)))
g4 = pd.read_csv(os.path.join(DATA, "gpt4o_labels.csv"))
for i_, l_ in zip(g4["id"].astype(str), g4["label"].astype(int)):
    id2lab[i_] = l_

y = np.full(n, -1, dtype=np.int8)
for k in range(n):
    v = id2lab.get(ids[k])
    if v is not None:
        y[k] = v
mask = y >= 0
log(f"labeled rows: {mask.sum():,}  pos_rate={y[mask].mean():.3f}")

# term-grouped folds via hash
fold = np.array([zlib.crc32(str(t).encode()) % 5 for t in tids], dtype=np.int8)

np.savez_compressed(os.path.join(ART, "stacker_features.npz"),
                    X=X, y=y, fold=fold, feat_names=np.array(feat_names))
log("features dumped -> stacker_features.npz")

params = dict(objective="binary", learning_rate=0.05, num_leaves=127,
              min_data_in_leaf=100, feature_fraction=0.9, bagging_fraction=0.9,
              bagging_freq=1, num_threads=8, verbosity=-1)

log("5-fold OOF...")
oof = np.zeros(mask.sum(), dtype=np.float32)
Xl = X[mask]; yl = y[mask].astype(np.int32); fl = fold[mask]
models = []
for f in range(5):
    tr = fl != f; va = fl == f
    m = lgb.train(params, lgb.Dataset(Xl[tr], yl[tr], feature_name=feat_names),
                  num_boost_round=800,
                  valid_sets=[lgb.Dataset(Xl[va], yl[va])],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
    oof[va] = m.predict(Xl[va], num_iteration=m.best_iteration)
    models.append(m)
    log(f"  fold {f}: best_iter={m.best_iteration}")

best = max((f1_score(yl, (oof >= t).astype(int), average="macro"), t)
           for t in np.arange(0.3, 0.71, 0.02))
log(f"OOF macroF1 (hakeme karsi) = {best[0]:.4f} @ thr={best[1]:.2f}")
imp = sorted(zip(feat_names, models[0].feature_importance("gain")), key=lambda x: -x[1])
log("importance: " + ", ".join(f"{a}={b:.0f}" for a, b in imp[:6]))

log("Scoring all pairs (fold-model average)...")
pred = np.zeros(n, dtype=np.float32)
for m in models:
    pred += m.predict(X, num_iteration=m.best_iteration).astype(np.float32)
pred /= len(models)
np.save(os.path.join(ART, "stacker_test5.npy"), pred)
log(f"saved stacker_test.npy  mean={pred.mean():.3f}")

# v10: judge labels where available, stacker elsewhere at OOF-best threshold
final = np.where(mask, y, (pred >= best[1]).astype(np.int8)).astype(int)
log(f"v10 pos_rate={final.mean():.4f} (hedef ~0.30)")
pd.DataFrame({"id": sub["id"], "prediction": final}).to_csv(
    os.path.join(DATA, "submission_stacker_v10.csv"), index=False)
log("Saved -> submission_stacker_v10.csv")
