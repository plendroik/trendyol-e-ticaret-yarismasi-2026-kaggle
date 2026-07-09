"""
Label the CE uncertain band via the OpenAI BATCH API.

Why: the regular API's 10k requests/DAY cap (tier 1) kills the 278k single-pair
plan. The Batch API has separate (queue-token) limits and is 50% cheaper
(~$5 for the band). Same single-pair format = same validated quality
(recall 0.94 / FPR 0.06).

Chunks the TODO pairs into JSONL jobs (<= CHUNK lines, under the enqueued-token
cap), submits them one at a time, polls, appends results to test_judge_labels.csv
(same file as judge_test_band.py; fully resume-safe both ways). Failed/errored
lines are retried on the next outer sweep.

Usage:
  python batch_judge.py --lo 0.2 --hi 0.85 --chunk 8000
Output: test_judge_labels.csv  (id,label)
"""
import os, io, re, json, time, argparse
import numpy as np, pandas as pd
from openai import OpenAI
from judge_local import prod_text
from judge_test_band import SYS, get_key, CE_TEST, OUT

DATA = r"C:\Users\ASUS\Desktop\trendyol"
MODEL = "gpt-4o-mini"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def load_todo(args, sub, ce):
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["id"].astype(str))
    band = (ce >= args.lo) & (ce <= args.hi)
    ids = sub["id"].astype(str).to_numpy()
    todo = [i for i in np.where(band)[0] if ids[i] not in done]
    return todo, done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lo", type=float, default=0.20)
    ap.add_argument("--hi", type=float, default=0.85)
    ap.add_argument("--chunk", type=int, default=8000)
    ap.add_argument("--sweeps", type=int, default=3)
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

    header_needed = not os.path.exists(OUT)
    fout = open(OUT, "a", newline="", encoding="utf-8")
    if header_needed:
        fout.write("id,label\n"); fout.flush()

    total_cost = 0.0
    for sweep in range(args.sweeps):
        todo, done = load_todo(args, sub, ce)
        log(f"sweep {sweep+1}: done={len(done)} todo={len(todo)}")
        if not todo:
            break
        for c0 in range(0, len(todo), args.chunk):
            rows = todo[c0:c0 + args.chunk]
            buf = io.BytesIO()
            for i in rows:
                q = term2q.get(tids[i], ""); p = itxt.get(iids[i], "")
                line = {"custom_id": ids[i], "method": "POST", "url": "/v1/chat/completions",
                        "body": {"model": MODEL, "temperature": 0.0, "max_tokens": 6,
                                 "messages": [{"role": "system", "content": SYS},
                                              {"role": "user", "content": f"Çiftler:\n1. {q} | {p}"}]}}
                buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
            buf.seek(0)

            # submit (retry while the enqueued-token quota is full)
            for a in range(60):
                try:
                    f = client.files.create(file=("chunk.jsonl", buf), purpose="batch")
                    job = client.batches.create(input_file_id=f.id,
                                                endpoint="/v1/chat/completions",
                                                completion_window="24h")
                    break
                except Exception as e:
                    log(f"  submit blocked ({e}); wait 120s"); time.sleep(120); buf.seek(0)
            else:
                log("  submit failed 60x -> skip chunk"); continue
            log(f"  chunk {c0//args.chunk + 1}/{(len(todo)-1)//args.chunk + 1} "
                f"({len(rows)} pairs) job={job.id}")

            # poll
            while True:
                time.sleep(60)
                job = client.batches.retrieve(job.id)
                if job.status in ("completed", "failed", "expired", "cancelled"):
                    break
                rc = job.request_counts
                log(f"    {job.status}  {getattr(rc, 'completed', 0)}/{getattr(rc, 'total', 0)}")
            if job.status != "completed":
                log(f"    job {job.status} -> ids will retry next sweep"); continue

            # collect
            n_ok = n_err = 0
            content = client.files.content(job.output_file_id).text
            for lineraw in content.splitlines():
                if not lineraw.strip():
                    continue
                r = json.loads(lineraw)
                cid = r["custom_id"]
                resp = r.get("response") or {}
                if resp.get("status_code") != 200:
                    n_err += 1; continue
                body = resp["body"]
                u = body.get("usage", {})
                total_cost += (u.get("prompt_tokens", 0) * 0.075 +
                               u.get("completion_tokens", 0) * 0.30) / 1e6
                txt = body["choices"][0]["message"]["content"].strip()
                m = re.search(r"([01])\b", txt[::-1])
                lab = int(m.group(1)) if m else 1
                fout.write(f"{cid},{lab}\n"); n_ok += 1
            fout.flush()
            log(f"    collected ok={n_ok} err={n_err}  cost so far=${total_cost:.2f}")

    fout.close()
    todo, done = load_todo(args, sub, ce)
    log(f"DONE. labeled total={len(done)}  remaining={len(todo)}  batch cost=${total_cost:.2f}")


if __name__ == "__main__":
    main()
