"""
gpt-4o SECOND OPINION on the pairs where the mini judge and the CE model clash
hardest (mini judge errors concentrate there; gpt-4o benched at RECALL 0.99 /
FPR 0.02 vs mini's 0.94/0.06).

Selects the top-K conflict pairs by strength: judge=0 with high CE score, or
judge=1 with very low CE score. Labels them via the Batch API (gpt-4o batch:
$1.25/1M in, $5/1M out -> ~$0.31/1k pairs). Output goes to a SEPARATE file that
build steps apply LAST (overrides mini).

Usage:
  python batch_judge_4o.py --topk 25000 --chunk 4000
Output: gpt4o_labels.csv  (id,label)
"""
import os, io, re, json, time, argparse
import numpy as np, pandas as pd
from openai import OpenAI
from judge_local import prod_text
from judge_test_band import SYS, get_key, CE_TEST

DATA = r"C:\Users\ASUS\Desktop\trendyol"
OUT4 = os.path.join(DATA, "gpt4o_labels.csv")
# gpt-4o's tier-1 batch queue is only 90k tokens -> unusable. gpt-4.1-mini benched
# RECALL 0.96 / FPR 0.05 (better than 4o-mini) and is 6x cheaper than 4o.
MODEL = "gpt-4.1-mini"
IN_RATE, OUT_RATE = 0.20 / 1e6, 0.80 / 1e6      # batch prices


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=25000)
    ap.add_argument("--chunk", type=int, default=4000)
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())

    log("Loading data...")
    ce = np.load(CE_TEST)
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
    ids = sub["id"].astype(str).to_numpy()
    tids = sub["term_id"].to_numpy(); iids = sub["item_id"].to_numpy()

    mini = pd.read_csv(os.path.join(DATA, "test_judge_labels.csv"))
    id2lab = dict(zip(mini["id"].astype(str), mini["label"].astype(int)))

    # conflict strength: judge=0 -> ce score; judge=1 -> (1 - ce/0.2 scaled) for low ce
    band = (ce >= 0.03) & (ce <= 0.85)
    cand = []
    for i in np.where(band)[0]:
        v = id2lab.get(ids[i])
        if v == 0 and ce[i] >= 0.5:
            cand.append((ce[i], i))                 # model emin 1, hakem 0
        elif v == 1 and ce[i] <= 0.08:
            cand.append((1.0 - ce[i], i))           # model emin 0, hakem 1
    cand.sort(reverse=True)
    done = set()
    if os.path.exists(OUT4):
        done = set(pd.read_csv(OUT4)["id"].astype(str))
    pick = [i for _, i in cand if ids[i] not in done][:args.topk]
    log(f"conflicts={len(cand)}  picked={len(pick)}  (done={len(done)})")

    header = not os.path.exists(OUT4)
    fout = open(OUT4, "a", newline="", encoding="utf-8")
    if header:
        fout.write("id,label\n"); fout.flush()

    cost = 0.0
    for c0 in range(0, len(pick), args.chunk):
        rows = pick[c0:c0 + args.chunk]
        buf = io.BytesIO()
        for i in rows:
            q = term2q.get(tids[i], ""); p = itxt.get(iids[i], "")
            line = {"custom_id": ids[i], "method": "POST", "url": "/v1/chat/completions",
                    "body": {"model": MODEL, "temperature": 0.0, "max_tokens": 6,
                             "messages": [{"role": "system", "content": SYS},
                                          {"role": "user", "content": f"Çiftler:\n1. {q} | {p}"}]}}
            buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))

        job = None
        for attempt in range(8):     # queue-full jobs FAIL at job level -> retry chunk
            buf.seek(0)
            for a in range(60):
                try:
                    f = client.files.create(file=("c4o.jsonl", buf), purpose="batch")
                    job = client.batches.create(input_file_id=f.id,
                                                endpoint="/v1/chat/completions",
                                                completion_window="24h")
                    break
                except Exception as e:
                    log(f"  submit blocked ({e}); wait 120s"); time.sleep(120)
            if job is None:
                continue
            log(f"  chunk {c0//args.chunk+1}/{(len(pick)-1)//args.chunk+1} "
                f"({len(rows)}) try{attempt+1} job={job.id}")
            while True:
                time.sleep(60)
                job = client.batches.retrieve(job.id)
                if job.status in ("completed", "failed", "expired", "cancelled"):
                    break
                rc = job.request_counts
                log(f"    {job.status}  {getattr(rc,'completed',0)}/{getattr(rc,'total',0)}")
            if job.status == "completed":
                break
            log(f"    job {job.status}; retry in 180s"); job = None; time.sleep(180)
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
            cost += u.get("prompt_tokens", 0) * IN_RATE + u.get("completion_tokens", 0) * OUT_RATE
            txt = body["choices"][0]["message"]["content"].strip()
            m = re.search(r"([01])\b", txt[::-1])
            fout.write(f"{r['custom_id']},{int(m.group(1)) if m else 1}\n"); n_ok += 1
        fout.flush()
        log(f"    collected {n_ok}  cost=${cost:.2f}")
    fout.close()
    log(f"DONE4O. cost=${cost:.2f} -> {OUT4}")


if __name__ == "__main__":
    main()
