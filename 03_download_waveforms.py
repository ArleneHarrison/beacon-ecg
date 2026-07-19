#!/usr/bin/env python3
"""Download MIMIC-IV-ECG paired waveforms LOCALLY through the US proxy (cookie-session auth),
then they get pushed to the 5090. Threaded, resumable, re-logins on cookie expiry."""
import argparse, os, re, time, threading, sys
import requests, pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE = "https://physionet.org/files/mimic-iv-ecg/1.0"
LOGIN = "https://physionet.org/login/"
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
PROXY = None   # set from --proxy

_lock = threading.Lock()

def login(sess, user, pw):
    r = sess.get(LOGIN, timeout=30)
    m = re.search(r'name="csrfmiddlewaretoken" value="([^"]+)"', r.text)
    csrf = m.group(1) if m else sess.cookies.get("csrftoken")
    r = sess.post(LOGIN, data={"csrfmiddlewaretoken": csrf, "username": user,
                               "password": pw, "next": "/"},
                  headers={**UA, "Referer": LOGIN}, timeout=30)
    ok = "sessionid" in sess.cookies.get_dict()
    return ok

def make_session(user, pw):
    s = requests.Session()
    if PROXY: s.proxies = PROXY
    s.headers.update(UA)
    if not login(s, user, pw): raise SystemExit("LOGIN FAILED")
    return s

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", default=""); ap.add_argument("--out", required=True)
    ap.add_argument("--user", required=True); ap.add_argument("--pw", required=True)
    ap.add_argument("--subset", choices=["index","all"], default="index")
    ap.add_argument("--pathlist", default="", help="file with one record path per line (overrides cohort)")
    ap.add_argument("--workers", type=int, default=12); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--proxy", default="", help="e.g. http://127.0.0.1:7897 ; empty=direct")
    a = ap.parse_args()
    global PROXY
    PROXY = {"http": a.proxy, "https": a.proxy} if a.proxy else None
    if a.pathlist:
        paths = [ln.strip() for ln in open(a.pathlist) if ln.strip()]
    else:
        df = pd.read_parquet(a.cohort)
        if a.subset == "index": df = df[df["is_index_ecg"] == 1]
        paths = df["path"].tolist()
    if a.limit: paths = paths[:a.limit]
    print(f"records: {len(paths)}  workers: {a.workers}  out: {a.out}", flush=True)

    tl = threading.local()
    def sess():
        if not hasattr(tl, "s"): tl.s = make_session(a.user, a.pw)
        return tl.s
    done = [0]; failed = []; bytes_ = [0]; t0 = time.time()
    def fetch(p):
        dst = os.path.join(a.out, p)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        for ext in (".hea", ".dat"):
            fp = dst + ext
            if os.path.exists(fp) and os.path.getsize(fp) > 0: continue
            for attempt in range(3):
                try:
                    r = sess().get(f"{BASE}/{p}{ext}", timeout=60)
                    if r.status_code == 200 and r.content:
                        with open(fp, "wb") as f: f.write(r.content)
                        with _lock: bytes_[0] += len(r.content)
                        break
                    elif r.status_code in (401, 403):        # cookie expired -> relogin
                        login(sess(), a.user, a.pw)
                    else: time.sleep(1)
                except Exception:
                    time.sleep(2)
            else:
                return p
        return None
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(fetch, p): p for p in paths}
        for fut in as_completed(futs):
            r = fut.result()
            with _lock:
                done[0] += 1
                if r: failed.append(r)
                if done[0] % 500 == 0 or done[0] == len(paths):
                    el = time.time() - t0
                    print(f"  {done[0]}/{len(paths)}  ok={done[0]-len(failed)} fail={len(failed)}  "
                          f"{bytes_[0]/1e6:.0f}MB  {bytes_[0]/1e6/max(el,1):.2f}MB/s  {el:.0f}s", flush=True)
    print(f"DONE ok={len(paths)-len(failed)} fail={len(failed)}  {bytes_[0]/1e6:.0f}MB in {time.time()-t0:.0f}s", flush=True)
    if failed:
        open(os.path.join(a.out,"_failed.txt"),"w").write("\n".join(failed))

if __name__ == "__main__": main()
