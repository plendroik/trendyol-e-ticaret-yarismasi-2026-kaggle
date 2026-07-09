"""
Local LLM judge via Ollama (rule-safe: data never leaves the machine).

--validate: measure judge quality on ground truth we ALREADY have:
  recall  = % of KNOWN positives (training_pairs) the judge calls 1
  FPR     = % of random (query, item) pairs (certain negatives) it calls 1
A usable judge needs recall >= ~0.93 and FPR <= ~0.05. We validate BEFORE
labeling anything at scale (lesson from the 0.79 API run).

Usage:
  python judge_local.py --validate --n 500                    # quick smoke
  python judge_local.py --validate --n 3000 --model qwen2.5:7b-instruct
"""
import os, re, json, time, argparse, random
import numpy as np, pandas as pd
import urllib.request

DATA = r"C:\Users\ASUS\Desktop\trendyol"
OLLAMA = "http://localhost:11434/api/generate"
BATCH = 10
_C = re.compile(r"renk:\s*([^,]+)"); _M = re.compile(r"materyal:\s*([^,]+)")

SYS = ("Sen Trendyol arama-alaka uzmanısın. Her (sorgu | ürün) çifti için ürünün "
       "aramanın MAKUL bir sonucu olup olmadığına karar ver. KURAL: ürün, sorgunun "
       "istediği ürün tipiyle/kategorisiyle AYNI ya da onu karşılıyorsa 1 (alakalı) — "
       "renk/model/marka farkı ÖNEMSİZ, aynı tip yeterli. Ürün TAMAMEN farklı bir "
       "tip/kategori ise 0. Kararsız kalırsan 1 ver. Her çift için TEK satırda "
       "SADECE 0 ya da 1 yaz, sırayla, başka hiçbir şey yazma.")

# single-pair mode: few-shot, one decision per call (easier for 7B models).
# Ground truth is INCLUSIVE (ESCI-style "reasonable result"): brand queries match
# the brand's products, category queries match anything in it, near types count.
SYS1 = ("Sen Trendyol arama-alaka hakemisin. Ürün, sorgunun MAKUL bir sonucu mu? "
        "Kurallar: Sorgu marka ise o markanın her ürünü 1. Sorgu kategori ise o "
        "kategorideki her ürün 1. Ürün tipi aynı veya yakın kullanım amaçlıysa 1 "
        "(renk/beden/model/marka farkı önemsiz). SADECE ürün bambaşka bir ihtiyaca "
        "yönelikse 0. SADECE tek karakter yaz: 0 ya da 1.")
FEWSHOT = ("Sorgu: kırmızı kadın elbise | Ürün: siyah uzun kollu elbise [elbise]\n1\n"
           "Sorgu: puma bayan ayakkabı | Ürün: zenit çift kişilik yatak örtüsü [yatak örtüsü]\n0\n"
           "Sorgu: avent | Ürün: natural yenidoğan biberon seti [biberon] marka:philips avent\n1\n"
           "Sorgu: yolluk | Ürün: kaymaz taban halı yolluk mutfak [halı]\n1\n"
           "Sorgu: laptop çantası | Ürün: paslanmaz çelik tencere seti [tencere]\n0\n")


def prod_text(title, cat, attrs, brand=None, gender=None):
    """Brand is CRITICAL: many queries are brand names absent from the title
    (avent, stanley, schneider...) — without it the judge can't see relevance."""
    leaf = cat.split("/")[-1] if isinstance(cat, str) and cat else ""
    al = attrs.lower() if isinstance(attrs, str) else ""
    cm = _C.search(al); mm = _M.search(al)
    t = (title or "")[:90]
    parts = [t, f"[{leaf}]"]
    if isinstance(brand, str) and brand and brand.lower() != "unknown":
        parts.append(f"marka:{brand}")
    if isinstance(gender, str) and gender and gender.lower() != "unknown":
        parts.append(gender)
    if cm:
        parts.append(f"renk:{cm.group(1).strip()}")
    if mm:
        parts.append(f"materyal:{mm.group(1).strip()}")
    return " ".join(parts).strip()


def ollama_call(model, prompt, retries=3, system=None, num_predict=None):
    body = json.dumps({"model": model, "prompt": prompt,
                       "system": system or SYS, "stream": False,
                       "options": {"temperature": 0.0,
                                   "num_predict": num_predict or BATCH * 8}}).encode()
    for a in range(retries):
        try:
            req = urllib.request.Request(OLLAMA, data=body,
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as r:
                return json.loads(r.read())["response"]
        except Exception as e:
            print(f"  ollama error ({a+1}/{retries}): {e}", flush=True); time.sleep(5)
    return ""


def judge(model, batch):
    """batch: list of (query, prodtext) -> (labels, n_parsed). Positional bare 0/1
    lines (Qwen's natural format); falls back to 'numara. 0/1'. Parse miss -> 1."""
    lines = "\n".join(f"{i+1}. {q} | {p}" for i, (q, p) in enumerate(batch))
    resp = ollama_call(model, "Çiftler:\n" + lines)
    bare = [l.strip() for l in resp.splitlines() if l.strip() in ("0", "1")]
    if len(bare) == len(batch):
        return [int(b) for b in bare], len(batch)
    labs = {}
    for l in resp.splitlines():
        m = re.match(r"\s*(\d+)\s*[.:\-]\s*([01])\b", l.strip())
        if m:
            labs[int(m.group(1))] = int(m.group(2))
    return [labs.get(i + 1, 1) for i in range(len(batch))], len(labs)


def judge_single(model, q, p):
    """One pair per call, few-shot. Returns 0/1 (parse miss -> 1)."""
    prompt = FEWSHOT + f"Sorgu: {q} | Ürün: {p}\n"
    resp = ollama_call(model, prompt, system=SYS1, num_predict=4).strip()
    m = re.search(r"[01]", resp)
    return int(m.group(0)) if m else 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true")
    ap.add_argument("--n", type=int, default=500, help="positives and negatives each")
    ap.add_argument("--model", default="qwen2.5:7b-instruct")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--single", action="store_true", help="one pair per call (few-shot)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print(f"model={args.model}  n={args.n}/side", flush=True)
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    train = pd.read_csv(os.path.join(DATA, "training_pairs.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
    all_items = items["item_id"].tolist()

    # known positives (ground truth label=1)
    pos = train.sample(args.n, random_state=args.seed)
    pos_pairs = [(term2q[t], itxt[i]) for t, i in zip(pos.term_id, pos.item_id)]
    # random items for random train terms -> certain negatives
    tlist = list(train.term_id.unique())
    neg_pairs = [(term2q[rng.choice(tlist)], itxt[rng.choice(all_items)])
                 for _ in range(args.n)]

    t0 = time.time()

    def run(pairs, name):
        out = []; parsed = 0
        if args.single:
            for k, (q, p) in enumerate(pairs):
                out.append(judge_single(args.model, q, p))
                if k and k % 100 == 0:
                    el = time.time() - t0
                    print(f"  {name} {k}/{len(pairs)}  ({el:.0f}s, {k/el:.1f} pair/s)", flush=True)
            return np.array(out)
        for b in range(0, len(pairs), BATCH):
            labs, np_ = judge(args.model, pairs[b:b + BATCH])
            out += labs; parsed += np_
            if b % (BATCH * 10) == 0 and b:
                el = time.time() - t0
                print(f"  {name} {b}/{len(pairs)}  ({el:.0f}s, {b/el:.1f} pair/s)", flush=True)
        print(f"  {name} parse orani: {parsed}/{len(pairs)}", flush=True)
        return np.array(out)

    p = run(pos_pairs, "pos")
    n = run(neg_pairs, "neg")
    el = time.time() - t0
    print(f"\n=== {args.model} ===")
    print(f"RECALL (bilinen pozitifte 1 deme): {p.mean():.3f}   hedef >= 0.93")
    print(f"FPR    (rastgele ciftte 1 deme)  : {n.mean():.3f}   hedef <= 0.05")
    print(f"hiz: {(len(p)+len(n))/el:.1f} cift/sn  toplam {el:.0f}s")


if __name__ == "__main__":
    main()
