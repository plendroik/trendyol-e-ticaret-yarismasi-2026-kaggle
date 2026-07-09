"""
Phase 1b: judge the RESCUE candidates — pairs the main CE buried (< 0.03) but at
least one other 0.83-class model scores > 0.5. Missed true positives concentrate
here (class-1 recall is our biggest loss). Writes to a SEPARATE file so it can run
concurrently with the band labeler (no same-file append races).

Usage: python batch_judge_rescue.py --chunk 8000
Output: rescue_labels.csv (id,label)
"""
import os, io, re, json, time, argparse
import numpy as np, pandas as pd
from openai import OpenAI
from judge_local import prod_text
from judge_test_band import SYS, get_key

DATA = r"C:\Users\ASUS\Desktop\trendyol"
ART = os.path.join(DATA, "artifacts")
OUT = os.path.join(DATA, "rescue_labels.csv")
MODEL = "gpt-4o-mini"


def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunk", type=int, default=8000)
    args = ap.parse_args()
    client = OpenAI(api_key=get_key())

    log("Loading...")
    ce = np.load(os.path.join(ART, "ce_test_trendyol_ce.npy"))
    fu = np.load(os.path.join(ART, "ce_test_trendyol_full.npy"))
    ai = np.load(os.path.join(ART, "ce_test_trendyol_ai.npy"))
    qa = np.load(os.path.join(ART, "ce_test_trendyol_qaug.npy"))
    sub = pd.read_csv(os.path.join(DATA, "submission_pairs.csv"))
    terms = pd.read_csv(os.path.join(DATA, "terms.csv"))
    items = pd.read_csv(os.path.join(DATA, "items.csv"))
    term2q = dict(zip(terms.term_id, terms["query"].fillna("").astype(str)))
    itxt = {r.item_id: prod_text(r.title, r.category, r.attributes, r.brand, r.gender)
            for r in items.itertuples(index=False)}
    ids = sub["id"].astype(str).to_numpy()
    tids = sub["term_id"].to_numpy(); iids = sub["item_id"].to_numpy()

    sel = np.where((ce < 0.03) & ((fu > 0.5) | (ai > 0.5) | (qa > 0.5)))[0]
    done = set()
    if os.path.exists(OUT):
        done = set(pd.read_csv(OUT)["id"].astype(str))
    todo = [i for i in sel if ids[i] not in done]
    log(f"rescue candidates={len(sel)} done={len(done)} todo={len(todo)}")

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
                    "body": {"model": MODEL, "temperature": 0.0, "max_tokens": 6,
                             "messages": [{"role": "system", "content": SYS},
                                          {"role": "user", "content": f"Çiftler:\n1. {q} | {p}"}]}}
            buf.write((json.dumps(line, ensure_ascii=False) + "\n").encode("utf-8"))
        job = None
        for attempt in range(8):
            buf.seek(0)
            try:
                f = client.files.create(file=("rescue.jsonl", buf), purpose="batch")
                job = client.batches.create(input_file_id=f.id,
                                            endpoint="/v1/chat/completions",
                                            completion_window="24h")
            except Exception as e:
                log(f"  submit blocked ({e}); wait 120s"); time.sleep(120); continue
            log(f"  chunk {c0//args.chunk+1}/{(len(todo)-1)//args.chunk+1} ({len(rows)}) job={job.id}")
            while True:
                time.sleep(60)
                job = client.batches.retrieve(job.id)
                if job.status in ("completed", "failed", "expired", "cancelled"):
                    break
            if job.status == "completed":
                break
            log(f"    job {job.status}; retry 180s"); job = None; time.sleep(180)
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
            cost += (u.get("prompt_tokens", 0) * 0.075 + u.get("completion_tokens", 0) * 0.30) / 1e6
            txt = body["choices"][0]["message"]["content"].strip()
            m = re.search(r"([01])\b", txt[::-1])
            fout.write(f"{r['custom_id']},{int(m.group(1)) if m else 1}\n"); n_ok += 1
        fout.flush()
        log(f"    collected {n_ok}  cost=${cost:.2f}")
    fout.close()
    log(f"DONERESCUE. cost=${cost:.2f} -> {OUT}")


if __name__ == "__main__":
    main()
