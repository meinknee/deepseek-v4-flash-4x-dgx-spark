# Backport of open upstream vLLM PR #50843: clamp block-argmax token ids to
# vocab_size-1. Prevents out-of-range ids (~78/min on DSv4F under bursty
# concurrency per the PR's production A/B) that kill the engine in 90s-5.5min.
# Idempotent; logs and continues on any mismatch.
from pathlib import Path

BASE = Path("/usr/local/lib/python3.12/dist-packages/vllm/v1/worker/gpu")
SITES = [
    (BASE / "spec_decode/rejection_sampler_utils.py",
     "        token_id = block_idx * BLOCK_SIZE + idx",
     "        token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)"),
    (BASE / "spec_decode/rejection_sampler_utils.py",
     "    token_id = block_idx * BLOCK_SIZE + idx",
     "    token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)"),
    (BASE / "sample/gumbel.py",
     "    token_id = block_idx * BLOCK_SIZE + idx",
     "    token_id = tl.minimum(block_idx * BLOCK_SIZE + idx, vocab_size - 1)"),
]
for path, old, new in SITES:
    try:
        lines = path.read_text().split("\n")
        if any(l == new for l in lines):
            print(f"50843-clamp: {path.name}: already patched")
            continue
        hits = [i for i, l in enumerate(lines) if l == old]
        assert len(hits) == 1, f"anchor x{len(hits)} in {path.name}"
        lines[hits[0]] = new
        t = "\n".join(lines)
        compile(t, str(path), "exec")
        path.write_text(t)
        print(f"50843-clamp: {path.name}: installed")
    except Exception as e:
        print(f"50843-clamp: WARN {path.name}: {e} (continuing)")
