"""
Standalone test scorer for a trained cross-encoder. Run in a FRESH process so no
training-time optimizer/grad memory lingers (that caused VRAM thrashing during the
in-script inference). Memory-safe: no_grad + modest batch.

Usage:
  python score_test.py --ce_model artifacts/ce_model_berturk_clean --suffix berturk_clean
  python score_test.py --ce_model artifacts/ce_model_berturk_hard --suffix berturk_hard --basic_docs

Output: artifacts/ce_test_<suffix>.npy   (aligned to submission_pairs.csv order)
"""
import os, re, time, argparse, numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

DATA_DIR = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA_DIR, "artifacts")
device = "cuda" if torch.cuda.is_available() else "cpu"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


_ATTR_KEYS = ["renk", "materyal", "desen", "kumaş tipi", "ortam", "stil",
              "kol tipi", "yaka tipi", "boy", "kalıp"]
_ATTR_RE = {k: re.compile(re.escape(k) + r":\s*([^,]+)") for k in _ATTR_KEYS}
_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")


def build_docs(items, rich=True):
    titles = items["title"].fillna("").astype(str).tolist()
    brands = items["brand"].fillna("").astype(str).tolist()
    cats = items["category"].fillna("").astype(str).tolist()
    genders = items["gender"].fillna("").astype(str).tolist()
    ages = items["age_group"].fillna("").astype(str).tolist()
    attrs = items["attributes"].fillna("").astype(str).tolist()
    out = []
    for t, b, c, g, ag, a in zip(titles, brands, cats, genders, ages, attrs):
        al = a.lower()
        if rich:
            cat_full = c.replace("/", " > ") if c else ""
            parts = [f"{t}", f"marka {b}", f"kategori {cat_full}", f"{g} {ag}"]
            for k in _ATTR_KEYS:
                m = _ATTR_RE[k].search(al)
                if m:
                    parts.append(f"{k} {m.group(1).strip()}")
            out.append(" . ".join(parts))
        else:
            leaf = c.split("/")[-1] if c else ""
            cm = _C.search(al); mm = _M.search(al)
            out.append(f"{t} . marka {b} kategori {leaf} renk {cm.group(1).strip() if cm else ''} "
                       f"materyal {mm.group(1).strip() if mm else ''}")
    return out


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ce_model", required=True)
    ap.add_argument("--suffix", required=True)
    ap.add_argument("--max_len", type=int, default=192)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--basic_docs", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    log(f"model={args.ce_model}  suffix={args.suffix}  batch={args.batch}  rich={not args.basic_docs}")

    terms = pd.read_csv(os.path.join(DATA_DIR, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA_DIR, "items.csv"))
    sub = pd.read_csv(os.path.join(DATA_DIR, "submission_pairs.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    item2doc = dict(zip(items.item_id, build_docs(items, rich=not args.basic_docs)))
    qs = sub["term_id"].map(term2q).fillna("").tolist()
    ds = sub["item_id"].map(item2doc).fillna("").tolist()
    log(f"test pairs={len(qs)}")

    tok = AutoTokenizer.from_pretrained(args.ce_model)
    model = AutoModelForSequenceClassification.from_pretrained(args.ce_model).to(device).eval()

    out = np.zeros(len(qs), np.float32); pos = 0
    for i in range(0, len(qs), args.batch):
        enc = tok(qs[i:i+args.batch], ds[i:i+args.batch], truncation=True,
                  max_length=args.max_len, padding=True, return_tensors="pt").to(device)
        with torch.autocast("cuda", dtype=torch.float16):
            lg = model(**enc).logits
        out[pos:pos+lg.shape[0]] = torch.softmax(lg.float(), 1)[:, 1].cpu().numpy()
        pos += lg.shape[0]
        if (i // args.batch) % 2000 == 0:
            log(f"  {pos}/{len(qs)} ({pos/len(qs)*100:.1f}%, {time.time()-t0:.0f}s)")
    np.save(os.path.join(ART, f"ce_test_{args.suffix}.npy"), out)
    log(f"saved ce_test_{args.suffix}.npy  DONE in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
