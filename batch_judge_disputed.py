"""
Re-judge the DISPUTED band pairs: where the student trio (distill2/tybert2/xlmr
average) strongly contradicts the mini judge. The remaining ceiling is judge
error (~35k) and it concentrates exactly here.

Two tiers (run both, parallel queues):
  python batch_judge_disputed.py --model gpt-4.1-mini --minconf 0.5 --chunk 8000
  python batch_judge_disputed.py --model gpt-4o --topk 8000 --chunk 370
(gpt-4o tier-1 batch queue = 90k tokens -> tiny chunks.)

Output: disputed_<model>.csv (id,label). Resume-safe.
"""
import os, io, re, json, time, argparse
import numpy as np, pandas as pd
from openai import OpenAI
from judge_local import prod_text
from judge_test_band import SYS, get_key

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--minconf", type=float, default=0.5)
    ap.add_argument("--topk", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=8000)
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())
    OUT = os.path.join(DATA, f"disputed_{args.model.replace('.','').replace('-','_')}.csv")

    log("Loading...")
    ce = np.load(os.path.join(ART, "ce_test_trendyol_ce.npy"))
    savg = (np.load(os.path.join(ART, "ce_test_distill2.npy")) +
            np.load(os.path.join(ART, "ce_test_tybert_distill2.npy")) +
            np.load(os.path.join(ART, "ce_test_xlmr_distill.npy"))) / 3
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
    ids = sub["id"].astype(str).to_numpy()
    tids = sub["term_id"].to_numpy(); iids = sub["item_id"].to_numpy()
    lab = pd.read_csv(os.path.join(DATA, "test_judge_labels.csv"))
    id2 = dict(zip(lab["id"].astype(str), lab["label"].astype(int)))

    band = (ce >= 0.03) & (ce <= 0.97)
    idx = np.where(band)[0]
    jl = np.array([id2.get(ids[i], -1) for i in idx])
    m = jl >= 0
    conf = np.abs(jl[m] - savg[idx[m]])
    cand = idx[m][np.argsort(-conf)]
    confs = np.sort(conf)[::-1]
    if args.topk:
        cand = cand[:args.topk]
    else:
        cand = cand[confs >= args.minconf]

    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["id"].astype(str))
    todo = [i for i in cand if ids[i] not in done]
    log(f"disputed={len(cand)} done={len(done)} todo={len(todo)} -> {os.path.basename(OUT)}")

    header = not os.path.exists(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8")
    if header:
        fout.write("id,label\n"); fout.flush()

    cost = 0.0
    for c0 in range(0, len(todo), args.chunk):
        rows = todo[c0:c0 + args.chunk]
        buf = io.BytesIO()
        for i in rows:
            q = term2q.get(tids[i], ""); p = itxt.get(iids[i], "")
            line = {"custom_id": ids[i], "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": args.model, "temperature": 0.0, "max_tokens": 6,
                             "messages": [{"role": "system", "content": SYS},
                                          {"role": "user", "content": f"Çiftler:\n1. {q} | {p}"}]}}
            buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        job = None
        for attempt in range(10):
            buf.seek(0)
            try:
                f = client.files.create(file=("disp.jsonl", buf), purpose="batch")
                job = client.batches.create(input_file_id=f.id,
                                            endpoint="/v1/chat/completions",
                                            completion_window="24h")
            except Exception as e:
                log(f"  submit blocked ({e}); 120s"); time.sleep(120); continue
            log(f"  chunk {c0//args.chunk+1}/{(len(todo)-1)//args.chunk+1} ({len(rows)}) job={job.id}")
            while True:
                time.sleep(45)
                job = client.batches.retrieve(job.id)
                if job.status in ("completed", "failed", "expired", "cancelled"):
                    break
            if job.status == "completed":
                break
            log(f"    {job.status}; retry 120s"); job = None; time.sleep(120)
        if job is None:
            log("    chunk given up"); continue
        n_ok = 0
        for lr in client.files.content(job.output_file_id).text.splitlines():
            if not lr.strip():
                continue
            r = json.loads(lr)
            resp = r.get("response") or {}
            if resp.get("status_code") != 200:
                continue
            body = resp["body"]; u = body.get("usage", {})
            rate_in = 1.25 if args.model == "gpt-4o" else 0.20
            rate_out = 5.0 if args.model == "gpt-4o" else 0.80
            cost += (u.get("prompt_tokens", 0) * rate_in + u.get("completion_tokens", 0) * rate_out) / 1e6
            txt = body["choices"][0]["message"]["content"].strip()
            mm = re.search(r"([01])\b", txt[::-1])
            fout.write(f"{r['custom_id']},{int(mm.group(1)) if mm else 1}\n"); n_ok += 1
        fout.flush()
        log(f"    collected {n_ok}  cost=${cost:.2f}")
    fout.close()
    log(f"DONEDISPUTED. cost=${cost:.2f} -> {OUT}")


if __name__ == "__main__":
    main()
