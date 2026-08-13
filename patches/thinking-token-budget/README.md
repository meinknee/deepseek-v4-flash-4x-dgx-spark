# thinking-token-budget — native `thinking_token_budget` for the V2 GPU runner

Wires the per-request `thinking_token_budget` sampling param into vLLM's **V2 GPU
model runner** — the runner that DSpark speculative decoding forces on, and the one
place upstream never wired the feature. With this, a request that sets
`thinking_token_budget: N` has its reasoning block hard-closed (`</think>` forced)
once it has emitted `N` tokens, leaving the rest of `max_tokens` for the answer.

Full root-cause and design writeup: [`docs/thinking-token-budget.md`](../../docs/thinking-token-budget.md).

## Why it's needed

On stock vLLM 0.27.x, setting `thinking_token_budget` on this deployment returns
**HTTP 400** (`… not yet supported by the V2 model runner`). That refusal is
correct as shipped: DSpark is implemented **only** by the V2 runner
(`config/vllm.py`: *"DSpark is implemented only by the V2 GPU model runner … V1 …
can't run dspark"*), so `--speculative-config method=dspark` forces V2 — but the
thinking-budget machinery (`ThinkingBudgetStateHolder`) is wired **only in the V1
path** (`v1/sample/…`). The entire V2 GPU stack (`v1/worker/gpu/…`) has none of it.
So on this recipe you otherwise get **k=5 spec-decode OR a thinking budget, never
both**. Without a cap, a long `<think>` can spend the whole `max_tokens` and the
turn comes back with empty `content`.

## What it does

- Adds `v1/worker/gpu/sample/thinking_budget_state.py` — a `ThinkingBudgetState`
  modelled 1:1 on the existing `BadWordsState` / `LogitBiasState`: fixed-size
  per-request GPU buffers + **one eager Triton mask**. It hooks
  `Sampler.apply_sampling_params`, which is the single funnel for **both** normal
  decode and the DSpark spec-decode verify path (`rejection_sampler._verify`), so
  one insertion covers k=5 automatically.
- The mask forces the (single) `</think>` token via the same allowed-token
  primitive `LogitBiasState` uses (save the end-token logit, `-inf` the vocab,
  restore it). It fires only while the block is open — a bounded scan of the
  already-emitted tokens detects an existing close and no-ops thereafter, so it
  never loops. Pure read-only masking, **no `post_update` counter** → nothing
  enters a captured CUDA graph (the sampler runs eager, after graph replay).
- Threads `reasoning_config` into the `Sampler` constructor (for the `</think>`
  id) and **no-ops only** the V2-runner refusal in `input_processor.py` (the
  `reasoning_config` precondition is kept — you must still launch with
  `--reasoning-parser`).

**Inert unless used:** with no `thinking_token_budget` on a request the state is a
no-op (host gate short-circuits before any kernel launch), so existing traffic is
byte-for-byte unchanged.

## Apply

Applied idempotently at container start by `launch/tp4-entrypoint.sh`. Standalone:

```bash
python3 apply.py                 # patches the installed vllm package
python3 apply.py /path/to/vllm   # or an explicit vLLM root
```

Each edit is guarded by a marker and an exactly-one-occurrence anchor assertion,
and every touched file is byte-compiled before it is kept; a re-run is a no-op.

## Use

`thinking_token_budget` is a top-level field of the OpenAI chat/completions request
(it already flows end-to-end in vLLM; only the V2 consumer was missing):

```json
{ "model": "deepseek-v4-flash-0731",
  "messages": [ … ],
  "chat_template_kwargs": {"thinking": true},
  "max_tokens": 2000,
  "thinking_token_budget": 1024 }
```

A serving proxy can inject a sensible default when the client omits one. Scaling it
to the request (e.g. `min(ceiling, int(max_tokens * 0.6))`, floored) keeps room for
the answer at every request size — a flat ceiling only bites when `max_tokens`
exceeds it.

## Verify / revert

- `verify.py` — runs the Triton kernel on synthetic tensors (force vs. no-mask,
  including the spec-decode multi-row layout). Run it on a free GPU inside the
  serving image.
- Revert: this patch only adds a file and makes anchored edits; restore the three
  touched files (`sampler.py`, `model_runner.py`, `input_processor.py`) from a
  clean wheel and delete `thinking_budget_state.py`, or rebuild the image.

## Upstream status

The V2 GPU runner does not implement `thinking_token_budget` as of vLLM 0.27.0
(the feature exists only on the V1 runner). This is a from-scratch port into the V2
sampler idiom, not a cherry-pick. Remove it if/when upstream wires the budget into
the V2 runner.
