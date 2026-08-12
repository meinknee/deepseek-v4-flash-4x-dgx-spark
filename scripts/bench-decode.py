#!/usr/bin/env python3
import json, time, urllib.request
URL = "http://localhost:8888/v1/chat/completions"
P = "In a distributed system the consensus protocol must handle node failures gracefully. "


def call(prompt, mx):
    b = json.dumps({"model": "deepseek-v4-flash-0731", "max_tokens": mx, "temperature": 0.0,
                    "messages": [{"role": "user", "content": prompt}]}).encode()
    r = urllib.request.Request(URL, data=b, headers={"content-type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=300))
    return time.time() - t, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]


def decode_rate(prompt):
    call(prompt, 8)              # prime prefix cache
    wa, pt, _ = call(prompt, 8)
    wb, _, _ = call(prompt, 208)
    return pt, 200.0 / max(wb - wa, 1e-6)


tests = {
    "tiny (~200 tok)": "Explain how a hash table handles collisions.",
    "small (~2K tok)": P * 40 + "Summarize the ideas above and explain leader election.",
    "grown (~32K tok)": P * 2600 + "Summarize the ideas above and explain leader election.",
    "big (~128K tok)": P * 10500 + "Summarize the ideas above and explain leader election.",
}
print("%18s %9s %13s" % ("scenario", "ctx_tok", "decode_tok/s"))
for name, p in tests.items():
    pt, rate = decode_rate(p)
    print("%18s %9d %13.1f" % (name, pt, rate))
