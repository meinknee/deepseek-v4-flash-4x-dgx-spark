# DeepSeek-V4-Flash-0731 on 4× DGX Spark — upstream vLLM 0.27, TP=4, 1M context

Is it groundbreaking: no. Is it a fork: no — **upstream vLLM 0.27.0** plus one dependency pin and a
handful of small, documented patches (all with upstream linkage, most already fixed-but-unreleased
upstream). Big context: yes — **1M, actually usable: 4 concurrent 1M-capable streams**, soak-tested
with 752 requests at ~500K context and zero failures.

As far as we know this is the first public 4-node recipe on *upstream* vLLM (the excellent
existing 4× recipes run the community fork image), and the highest published concurrency at 1M
context on Spark hardware — 2-node KV pools fit ~2 streams at 1M; ours measures 4.37×. If your workload fits 2 nodes and you want maximum
single-stream speed, the fork-image recipes are ~15–20% faster there — see
[docs/benchmarks.md](docs/benchmarks.md) for the honest trade.

## What you get

- **Serving:** `deepseek-ai/DeepSeek-V4-Flash-0731` (284B MoE), TP=4 across 4× GB10, 1M
  `max-model-len`, 4 streams, DSpark speculative decoding k=5 (greedy — measured +12% over
  probabilistic at temperature 0), DeepGEMM MoE + dense on sm12x.
- **Numbers** (single stream, idle): ~50–56 tok/s prose, ~110 structured. Aggregate @ 4×500K ctx:
  31.3 tok/s. 496K-token prefill: 341 s. Details + methodology: [docs/benchmarks.md](docs/benchmarks.md).
- **Everything that broke:** the [10-blocker ledger](docs/blockers.md) — each failure between
  `docker run` and production, root-caused, with fixes. This is probably the most useful file here.
- **Native thinking-token budget** on the DSpark (V2) runner — a from-scratch port
  (`thinking_token_budget` is V1-only upstream, so DSpark installs 400 the field). Caps `<think>`
  so it can't eat the whole `max_tokens`. Root cause + design: [docs/thinking-token-budget.md](docs/thinking-token-budget.md).
- **Ops:** launch / teardown / health-watchdog scripts, a high-context soak test, rollback notes.

## Quickstart

```bash
# 0. prerequisites: 4 Sparks on a RoCE fabric (node i at ${FABRIC_PREFIX}.$((i+1))),
#    passwordless SSH between them, weights downloaded to $MODELS_DIR on every node.

# 1. build the image ON a Spark (~2-4 h; do NOT build on a node that is serving)
cd build && bash build-image.sh
# distribute: docker save vllm-gb10-027:dg0810 | ssh <node> docker load

# 2. launch — followers first, then the head
ssh node2 'FABRIC_PREFIX=10.0.0 bash tp4-launch.sh 1'
ssh node3 'FABRIC_PREFIX=10.0.0 bash tp4-launch.sh 2'
ssh node4 'FABRIC_PREFIX=10.0.0 bash tp4-launch.sh 3'
bash launch/tp4-launch.sh 0        # head; serves OpenAI API on :8888

# 3. first boot: ~15 min (JIT compiles the patched FlashInfer kernel once, cached after).
curl localhost:8888/health

# 4. validate like you mean it (short soaks hide the failure modes — see the ledger)
python3 scripts/soak-highctx.py
```

After any crash-loop: **tear down all four containers before relaunching** (`docker rm -f` +
`pkill -f "vllm serve"` on every node) — stale rendezvous state stalls the next boot (blocker 3).

## The two bugs you will hit on stock installs (and why patches ship here)

1. **vLLM 0.27.x pins a DeepGEMM with zero sm12x support** → `Unknown SF transformation` /
   `Unsupported architecture`. Known upstream (vllm#47436; fix PR #50796 pins the same rev this
   recipe builds). Our build script swaps the pin.
2. **FlashInfer's sm120 sparse-MLA decode dispatch lacks topk=256** → every DSpark spec-decode
   config crashes (`Check failed: num_tokens > 64`). Reported three times upstream (#3828, #3988,
   #4336) and **fixed on FlashInfer main 2026-08-08 (PR #4380)** — but the fix is in **no stable
   release** (≤ 0.6.17), and vLLM installs 0.6.16.post3. The entrypoint applies an equivalent
   backport (native 256 kernels — output verified bitwise-identical to the padded-512 workaround)
   and evicts the prebuilt `flashinfer_jit_cache` module that would otherwise shadow it.
   **Delete this patch when FlashInfer ≥ 0.6.18 ships.**

Plus two smaller ones: a backport of the open vocab-clamp PR (vllm#50843) that prevents an
engine-death class under bursty load, and a gate for an unguarded DeepGEMM call in the mHC path
(unreported upstream; issue draft in [docs/mhc-issue-draft.md](docs/mhc-issue-draft.md)).
Full inventory: [patches/README.md](patches/README.md).

And one feature, not a bug: [`patches/thinking-token-budget/`](patches/thinking-token-budget/) ports
`thinking_token_budget` into the V2 GPU runner. DSpark forces the V2 runner, and upstream wired the
budget only into V1 — so on this recipe a stock install returns *"not yet supported by the V2 model
runner"*, and an unbounded `<think>` can spend the whole `max_tokens` and return an empty turn. The
port re-implements it in the V2 sampler (one hook, covers k=5 spec-decode) so you keep DSpark **and**
get a real cap. Why the gate is genuine and how the port works:
[docs/thinking-token-budget.md](docs/thinking-token-budget.md).

## Known limitations, honestly

- Single-stream prose trails the fork-image stacks by ~15–20% (their B12X MXFP4 MoE kernels are
  faster; upstream's DeepGEMM path wins on prefill and multi-stream aggregate instead).
- `--block-size` must stay 256 (64 breaks the compress-128 layer page math) and DSpark k must be 5
  (the 0.27 validator floor for this checkpoint). Both explained in the ledger.
- The draft's 128-token sliding window means late speculative positions accept poorly on
  long-range-lookup content. This is the checkpoint's architecture, not a config problem.

## Credits

This stands on a lot of community work:

- **[MiaAI-Lab](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)** — the origin
  DSpark recipe; our tokenizer compat layer and NCCL fabric settings derive from it.
- **[joesinvestments](https://github.com/joesinvestments/DeepSeek-V4-Flash-0731-TP4-4x-DGX-Spark)** —
  the prior 4× tuning recipe (fork image) this deployment started from.
- **Anemll** (`dspark-vllm-gx10`) — the fork image that served us for weeks and whose overlay
  contains the original pad-to-512 workaround for the FlashInfer gap.
- **eous, aldakata, Lin-shuaibi** — filed the FlashInfer dispatch bug (#3828/#3988/#4336) long
  before we hit it. **lucifer1004** — the SM120 sparse-MLA kernels, the upstream enablement
  (vllm#43477), and the merged FlashInfer fix (#4380).
- **Odrec / ZacharyZcR** — the DeepGEMM pin issue (#47436) and fix PR (#50796). **Mirrdhyn** — an
  alternative DeepGEMM repoint (vllm PR #51959). **alexbi29** — the vocab-clamp fix (vllm#50843)
  backported here.
- **[ecohash-co](https://github.com/ecohash-co/deepseek-v4-flash-dgx-spark-vllm027)** — independent
  upstream-0.27 write-up (TP=2) of the same two headline bugs, published the same week.
- **jasl** — the sm12x research corpus in vllm PR #41834. **eugr**
  ([spark-vllm-docker](https://github.com/eugr/spark-vllm-docker)) — build patterns for this
  hardware. **lukealonso** — the b12x kernels behind the fork baseline we benchmarked against.

## Weights & license

Model weights are downloaded separately from Hugging Face and remain under DeepSeek's upstream
terms. This repo: Apache-2.0.
