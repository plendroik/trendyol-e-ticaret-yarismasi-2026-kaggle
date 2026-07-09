"""
Build the LLM-judge training set (the "label war" lever, step 2).

Takes the proven 0.83 base (artifacts/train_pairs.parquet: 250k real positives +
700k random easy negatives) and APPENDS all LLM-judge labels from llm_labels.csv
(embedding-ANN candidates judged by gpt-4o-mini: ~75k extra positives + ~26k clean
hard negatives that mirror the test's candidate distribution). One lever changed.

Folds are inherited from the base per term (GroupKFold-consistent, no leakage).
On (term_id, item_id) collisions the LLM label wins (verified > assumed).

Output: artifacts/train_pairs_llm.parquet
"""
import os, argparse, pandas as pd

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")

ap = argparse.ArgumentParser()
ap.add_argument("--only", choices=["all", "neg", "pos"], default="all",
                help="which LLM labels to add: all / neg-only / pos-only")
ap.add_argument("--out", default="train_pairs_llm.parquet")
args = ap.parse_args()

base = pd.read_parquet(os.path.join(ART, "train_pairs.parquet"))
llm = pd.read_csv(os.path.join(DATA, "llm_labels.csv"))
if args.only == "neg":
    llm = llm[llm.label == 0].reset_index(drop=True)
elif args.only == "pos":
    llm = llm[llm.label == 1].reset_index(drop=True)
print(f"base={len(base)} pos_rate={base.label.mean():.3f}   llm={len(llm)} (only={args.only}) pos_rate={llm.label.mean():.3f}")

# fold per term from base (all llm terms are training terms -> mapping complete)
term_fold = base.drop_duplicates("term_id").set_index("term_id")["fold"]
llm["fold"] = llm["term_id"].map(term_fold)
missing = llm["fold"].isna().sum()
if missing:
    print(f"WARN: {missing} llm rows with unknown term fold -> dropped")
    llm = llm.dropna(subset=["fold"])
llm["fold"] = llm["fold"].astype(base["fold"].dtype)

# combine; LLM label wins on duplicate (term_id, item_id)
comb = pd.concat([llm[base.columns.tolist()], base], ignore_index=True)
before = len(comb)
comb = comb.drop_duplicates(["term_id", "item_id"], keep="first").reset_index(drop=True)
print(f"dedup: {before} -> {len(comb)}  (collisions={before - len(comb)})")

out = os.path.join(ART, args.out)
comb.to_parquet(out, index=False)
print(f"final: {len(comb)} rows  pos_rate={comb.label.mean():.3f}")
print(comb.groupby("fold").agg(n=("label", "size"), pos=("label", "mean")))
print(f"Saved -> {out}")
