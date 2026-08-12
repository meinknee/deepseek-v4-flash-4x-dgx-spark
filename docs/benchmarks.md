# Benchmarks — honestly labeled

**Single-stream ≠ aggregate.** Every number below states its condition. Hardware: 4× NVIDIA DGX
Spark (GB10, sm_121), RoCE fabric, TP=4, `deepseek-ai/DeepSeek-V4-Flash-0731`, DSpark k=5 greedy,
temperature 0, 1M `max-model-len`, 4 `max-num-seqs`.

## Single stream (idle cluster, prime-then-marginal method — `scripts/bench-decode.py`)

| content | tok/s |
|---|---|
| natural prose | **50–56** |
| structured / predictable (counting, code-ish) | **~108–112** |

Speed is speculative-decode-dependent: mean acceptance ~2.9 cold → **4.5 warm** under real mixed
traffic. Per-position acceptance decays on content that needs attention lookups (the draft's
sliding window is 128 tokens — architecture, not a bug; chain-predictable content drafts at ~100%
through all 5 positions).

## High context / aggregate

| metric | value |
|---|---|
| 496,705-token prefill (one request) | **341 s** |
| 4-way sustained decode @ ~500K ctx | **31.3 tok/s aggregate** (~7.8/stream) |
| soak: 752 requests @ ~500K ctx, 4-way | **0 failures, no wedge** |
| KV pool @ gmu 0.75 | 44.2 GiB · **4.37× concurrency at 1M** |

For reference, the same soak on the community fork image (Anemll 0.25 lineage) measured 24.9 tok/s
aggregate and a 428 s prefill on identical prompts — this stack is ~+25% on both — while the fork
is ~15–20% faster on **single-stream prose** (its B12X MXFP4 MoE kernels). Pick your trade.

Methodology notes: decode rate isolates generation (prime the prefix cache, then measure the delta
between a max_tokens=8 and a max_tokens=208 call). Aggregate numbers state concurrency and context.
