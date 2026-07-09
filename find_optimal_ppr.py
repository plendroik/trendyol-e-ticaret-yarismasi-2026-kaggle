"""
Her model icin holdout uzerinde optimal PPR (Positive Prediction Rate) bulur.
Farkli threshold'larda macro-F1 hesaplar, en iyi PPR'i bulur ve o PPR ile submission uretir.

Kullanim:
  python find_optimal_ppr.py                    # tum modelleri analiz et
  python find_optimal_ppr.py --suffix trendyol_ai  # tek model
"""
import os, argparse, glob, numpy as np, pandas as pd
from sklearn.metrics import f1_score

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")


def find_best_ppr(scores, labels, name=""):
    """Holdout skorlari uzerinde PPR 0.20-0.45 arasinda sweep yap, optimal bul."""
    results = []
    for ppr in np.arange(0.20, 0.46, 0.01):
        thr = np.quantile(scores, 1 - ppr)
        pred = (scores >= thr).astype(int)
        f1 = f1_score(labels, pred, average="macro")
        results.append((ppr, thr, f1, pred.mean()))

    results.sort(key=lambda x: -x[2])  # en iyi F1'e gore sirala
    best_ppr, best_thr, best_f1, actual_ppr = results[0]

    print(f"\n{'='*60}")
    print(f"  MODEL: {name}")
    print(f"  OPTIMAL PPR: {best_ppr:.2f}  (threshold={best_thr:.4f})")
    print(f"  Holdout macro-F1: {best_f1:.5f}")
    print(f"  Gercek pos_rate: {actual_ppr:.4f}")
    print(f"{'='*60}")

    # Top 5 PPR'i goster
    print(f"  {'PPR':>6}  {'Threshold':>10}  {'macro-F1':>10}  {'pos_rate':>10}")
    print(f"  {'-'*6}  {'-'*10}  {'-'*10}  {'-'*10}")
    for ppr, thr, f1, pr in results[:8]:
        marker = " <-- BEST" if ppr == best_ppr else ""
        print(f"  {ppr:>6.2f}  {thr:>10.4f}  {f1:>10.5f}  {pr:>10.4f}{marker}")

    return best_ppr, best_thr, best_f1


def generate_submission(test_scores, best_ppr, suffix):
    """Optimal PPR ile submission uret."""
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    thr = np.quantile(test_scores, 1 - best_ppr)
    pred = (test_scores >= thr).astype(int)
    ppr_int = int(best_ppr * 100)
    out = os.path.join(DATA, f"submission_{suffix}_opt_p{ppr_int}.csv")
    pd.DataFrame({"id": sub["id"], "prediction": pred}).to_csv(out, index=False)
    print(f"  >> {os.path.basename(out)}  (pos_rate={pred.mean():.4f})")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suffix", default=None, help="tek model analiz et")
    args = ap.parse_args()

    # Holdout meta (labels)
    meta_path = os.path.join(ART, "ce_holdout_meta.npy")
    if not os.path.exists(meta_path):
        print("ce_holdout_meta.npy bulunamadi!"); return
    meta = np.load(meta_path)
    labels = meta[:, 0].astype(int)

    # Modelleri bul
    if args.suffix:
        suffixes = [args.suffix]
    else:
        files = glob.glob(os.path.join(ART, "ce_holdout_*.npy"))
        suffixes = []
        for f in files:
            name = os.path.basename(f).replace("ce_holdout_", "").replace(".npy", "")
            if name != "meta":
                suffixes.append(name)
        suffixes.sort()

    print(f"Toplam {len(suffixes)} model bulundu: {suffixes}")
    print(f"Holdout: {len(labels)} satir, prevalans={labels.mean():.3f}")

    all_results = []

    for suffix in suffixes:
        ho_path = os.path.join(ART, f"ce_holdout_{suffix}.npy")
        te_path = os.path.join(ART, f"ce_test_{suffix}.npy")

        if not os.path.exists(ho_path):
            print(f"\n  {suffix}: holdout skoru yok, atlaniyor"); continue

        ho_scores = np.load(ho_path)

        # Boyut uyumu kontrol
        if len(ho_scores) != len(labels):
            print(f"\n  {suffix}: boyut uyumsuz ({len(ho_scores)} vs {len(labels)}), atlaniyor")
            continue

        best_ppr, best_thr, best_f1 = find_best_ppr(ho_scores, labels, suffix)
        all_results.append((suffix, best_ppr, best_f1))

        # Test skoru varsa optimal submission uret
        if os.path.exists(te_path):
            te_scores = np.load(te_path)
            generate_submission(te_scores, best_ppr, suffix)
        else:
            print(f"  (test skoru henuz yok, submission uretilmedi)")

    # Ozet tablo
    if all_results:
        print(f"\n{'='*60}")
        print(f"  OZET: Tum modellerin optimal PPR degerleri")
        print(f"{'='*60}")
        print(f"  {'Model':<25}  {'Opt PPR':>8}  {'Holdout F1':>11}")
        print(f"  {'-'*25}  {'-'*8}  {'-'*11}")
        all_results.sort(key=lambda x: -x[2])
        for name, ppr, f1 in all_results:
            print(f"  {name:<25}  {ppr:>8.2f}  {f1:>11.5f}")


if __name__ == "__main__":
    main()
