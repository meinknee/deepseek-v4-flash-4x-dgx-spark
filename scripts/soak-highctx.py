#!/usr/bin/env python3
"""High-context wedge reproduction for DSv4F TP4 (graph path).

The earlier soak used SHORT prompts, so accumulated context never grew and the
context-driven wedge never triggered. This test:
  Phase A: one big request to establish ~500K+ tokens of KV context (paid once).
  Phase B: sustained 4-way concurrent DECODE on top of that same prefix
           (prefix-cache hits, cheap), for DURATION or until a wedge.

Wedge signal (client side): a request that times out (>TIMEOUT s) or the API
refusing/HTTP-000. On wedge it stops immediately and prints WEDGE + wall clock,
so the operator can gdb the stuck worker while it is frozen.

Run ON the head node (hits localhost:8888, --network host). No tokenizer needed;
Phase A reports the server's actual prompt_tokens.
"""
import json, sys, time, urllib.request, urllib.error
from concurrent.futures import ThreadPoolExecutor, as_completed

URL = "http://localhost:8888/v1/chat/completions"
MODEL = "deepseek-v4-flash-0731"
SEED_CHARS = 2_500_000     # ~600K tokens at ~4 chars/tok
CONC = 4                   # match max-num-seqs
DURATION = 1200            # Phase B seconds (20 min) unless wedge
TIMEOUT = 150              # per-request; exceed => wedge suspect

# coherent-ish technical filler; content is irrelevant to the top-k path (it
# operates over KV length), we just need many real tokens.
_PARA = (
    "In a tensor-parallel decode step the sparse lightning indexer scores every "
    "key-value block against the query projection, then selects the top-k blocks "
    "per row before the paged multi-query attention reads them; the radix top-k "
    "kernel stages partial histograms in shared memory and reduces across warps. "
)
SEED = (_PARA * (SEED_CHARS // len(_PARA) + 1))[:SEED_CHARS]


def call(suffix, max_tokens, timeout=TIMEOUT):
    body = json.dumps({
        "model": MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "messages": [
            {"role": "user", "content": SEED + "\n\nQuestion: " + suffix},
        ],
    }).encode()
    req = urllib.request.Request(URL, data=body, headers={"content-type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = time.time() - t0
    u = d.get("usage", {})
    return dt, u.get("prompt_tokens", 0), u.get("completion_tokens", 0)


def phase_a():
    print("PHASE A: establishing high context (~600K tok prefill, one request)...", flush=True)
    t0 = time.time()
    try:
        dt, pt, ct = call("Summarize the mechanism above in one sentence.", 200, timeout=900)
    except Exception as e:
        print(f"PHASE-A-FAILED: {type(e).__name__}: {e}", flush=True)
        return 0
    print(f"PHASE A done: prompt_tokens={pt} completion={ct} wall={dt:.1f}s "
          f"({ct/dt:.1f} tok/s decode)", flush=True)
    if pt < 400_000:
        print(f"WARNING: prompt_tokens {pt} < 400K — seed too small to test the threshold", flush=True)
    return pt


def phase_b():
    print(f"PHASE B: {CONC}-way sustained high-context decode for up to {DURATION}s "
          f"(or until wedge)...", flush=True)
    t0 = time.time()
    ok = fail = 0
    consec_fail = 0
    tok = 0.0
    n = 0
    with ThreadPoolExecutor(max_workers=CONC) as ex:
        while time.time() - t0 < DURATION:
            futs = [ex.submit(call, f"Explain point {n+i} in two sentences.", 400)
                    for i in range(CONC)]
            for f in as_completed(futs):
                try:
                    dt, pt, ct = f.result()
                    ok += 1; consec_fail = 0; tok += ct
                    last = f"{ct}tok/{dt:.1f}s"
                except Exception as e:
                    fail += 1; consec_fail += 1
                    last = f"FAIL {type(e).__name__}"
                    if consec_fail >= 3 or isinstance(e, urllib.error.URLError):
                        el = time.time() - t0
                        print(f"*** WEDGE SUSPECTED at +{el:.0f}s: {consec_fail} consecutive "
                              f"failures, last={type(e).__name__}: {e}", flush=True)
                        print(f"*** {time.strftime('%H:%M:%S')} — GDB THE WORKER NOW", flush=True)
                        print(f"=== phase B end: {ok} ok, {fail} fail, WEDGE=True ===", flush=True)
                        return True
            n += CONC
            el = time.time() - t0
            print(f"  [+{el:.0f}s] {ok} ok, {fail} fail, {tok/max(el,1):.1f} tok/s agg, last {last}", flush=True)
    print(f"=== phase B end: {ok} ok, {fail} fail, WEDGE=False, {tok/max(time.time()-t0,1):.1f} tok/s agg ===", flush=True)
    return False


if __name__ == "__main__":
    pt = phase_a()
    if pt == 0:
        sys.exit(1)
    wedged = phase_b()
    sys.exit(2 if wedged else 0)
