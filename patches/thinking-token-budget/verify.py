#!/usr/bin/env python3
"""On-GPU execution gate for _thinking_budget_kernel. Launches the REAL Triton
kernel on synthetic tensors laid out like RequestState and asserts the masked
logits (force vs. no-mask, including the spec-decode multi-row layout). Run
inside the serving image on a FREE GPU.
Usage: python3 verify.py [path/to/thinking_budget_state.py]  (default: sibling module)"""
import importlib.util
import os
import sys

import torch

MOD = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "thinking_budget_state.py")
spec = importlib.util.spec_from_file_location("tbs", MOD)
tbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tbs)
kernel = tbs._thinking_budget_kernel

dev = torch.device("cuda")
VOCAB = 16
END = 9
FILL = 1
MAXLEN = 4096
MAXREQ = 8

# Per-request state (indexed by req_state_idx).
budget = torch.zeros(MAXREQ, dtype=torch.int32, device=dev)
prompt_len = torch.zeros(MAXREQ, dtype=torch.int32, device=dev)
total_len = torch.zeros(MAXREQ, dtype=torch.int32, device=dev)
all_token_ids = torch.zeros((MAXREQ, MAXLEN), dtype=torch.int32, device=dev)

def set_req(r, budget_v, prompt_n, output_tokens):
    budget[r] = budget_v
    prompt_len[r] = prompt_n
    total_len[r] = prompt_n + len(output_tokens)
    # put some junk in the prompt region (incl. an END, to prove it's excluded)
    all_token_ids[r, :prompt_n] = END
    for i, t in enumerate(output_tokens):
        all_token_ids[r, prompt_n + i] = t

# Scenario rows: (req, expanded_local_pos, in-flight input_id, expect)
# req0: budget5, 6 filler outputs, no end -> FORCE
set_req(0, 5, 3, [FILL] * 6)
# req1: budget5, end already at output pos2 -> NO-MASK
set_req(1, 5, 2, [FILL, FILL, END, FILL, FILL, FILL, FILL, FILL])
# req2: budget0 disabled -> NO-MASK
set_req(2, 0, 4, [FILL] * 100)
# req3 spec: budget5, committed output_len=3 (no end), 3 draft rows
set_req(3, 5, 5, [FILL, FILL, FILL])

rows = [
    # (req, local_pos, inflight_token, expect_force)
    (0, 0, FILL, True),    # eff=6>=5, open -> force
    (1, 0, FILL, False),   # end already emitted -> no-mask
    (2, 0, FILL, False),   # disabled -> no-mask
    (3, 0, FILL, False),   # spec eff=3<5 -> no-mask
    (3, 1, FILL, False),   # spec eff=4<5 -> no-mask
    (3, 2, FILL, True),    # spec eff=5>=5, open -> force
]
# req3's three rows must be contiguous with first_pos so token_idx-pos lands right.
# Layout: row order = [r0, r1, r2, r3p0, r3p1, r3p2]; the r3 rows are contiguous
# and start at index 3, so for r3p2 (token_idx=5,pos=2) cur_req_first_pos=3. Its
# in-flight draft window input_ids[3:5] are r3p0,r3p1 tokens = FILL (no end). Good.
num_tokens = len(rows)
expanded_idx_mapping = torch.tensor([r[0] for r in rows], dtype=torch.int32, device=dev)
expanded_local_pos = torch.tensor([r[1] for r in rows], dtype=torch.int32, device=dev)
input_ids = torch.tensor([r[2] for r in rows], dtype=torch.int32, device=dev)

torch.manual_seed(0)
logits = torch.randn((num_tokens, VOCAB), dtype=torch.float32, device=dev)
orig = logits.clone()

kernel[(num_tokens,)](
    logits, logits.stride(0),
    expanded_idx_mapping,
    budget,
    all_token_ids, all_token_ids.stride(0),
    prompt_len, total_len,
    input_ids,
    expanded_local_pos,
    END, VOCAB,
    SLACK=16, SCAN_BLOCK=256, LOGITS_BLOCK_SIZE=8192,
)
torch.cuda.synchronize()

fails = 0
for i, (req, pos, tok, expect_force) in enumerate(rows):
    row = logits[i]
    if expect_force:
        others = torch.cat([row[:END], row[END + 1:]])
        ok = bool(torch.isinf(others).all() and (others < 0).all()
                  and torch.isfinite(row[END]) and torch.equal(row[END], orig[i, END]))
        verdict = "FORCE ok" if ok else "FORCE FAIL"
    else:
        ok = bool(torch.equal(row, orig[i]))
        verdict = "unchanged ok" if ok else "CHANGED FAIL"
    fails += not ok
    print(f"[{'OK ' if ok else 'FAIL'}] row{i} req{req} pos{pos} expect={'force' if expect_force else 'no-mask':7s} -> {verdict}")

print()
print("ALL GPU KERNEL CASES PASS" if not fails else f"{fails} GPU CASES FAILED")
sys.exit(1 if fails else 0)
