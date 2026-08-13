# Thinking-token budget on the V2 (DSpark) runner — the blank turn, and the real fix

**TL;DR** On this recipe, `thinking_token_budget` is refused with HTTP 400, and the
refusal is *correct* as shipped — DSpark forces vLLM's **V2** GPU model runner, and
the thinking-budget feature is wired only into the **V1** runner. So out of the box
you get **k=5 spec-decode _or_ a thinking budget, never both**, and a long `<think>`
can spend the entire `max_tokens` and return an empty `content` (a "blank turn").
The fix is a from-scratch port of the budget into the V2 sampler. It ships in
[`patches/thinking-token-budget/`](../patches/thinking-token-budget/).

## The symptom

DeepSeek-V4-Flash with reasoning enabled will, on some prompts, spend its whole
output budget inside the reasoning block: it hits `finish_reason=length` having
emitted only `reasoning_content` and an empty `content`. A client that renders only
`content` shows a blank message.

The obvious lever is vLLM's per-request `thinking_token_budget` (mask logits toward
the reasoning-end token once the budget is spent, so the block closes and the model
answers). On this deployment it returns:

```
thinking_token_budget is not yet supported by the V2 model runner.
Run vLLM with VLLM_USE_V2_MODEL_RUNNER=0 to use thinking_token_budget.
```

## Why the refusal is genuine (not a stale guard)

It is tempting to read that gate (`v1/engine/input_processor.py`) as over-broad and
delete it. Don't — it guards a feature the V2 runner does not implement:

- **DSpark forces the V2 runner.** `config/vllm.py` is explicit:
  *"DSpark is implemented only by the V2 GPU model runner … V1 … can't run dspark."*
  So `--speculative-config method=dspark` (this recipe's k=5) sets
  `use_v2_model_runner=True`.
- **The budget is wired only in V1.** `ThinkingBudgetStateHolder`,
  `maybe_create_thinking_budget_state_holder`, and its `apply_to_logits` live in
  `v1/sample/…` and are referenced by the **V1** input batch and the **V1**
  rejection sampler. The entire **V2** GPU stack (`v1/worker/gpu/…` — its input
  batch, its `spec_decode/rejection_sampler.py`, its `sample/sampler.py`) has **no**
  thinking-budget code at all.

So the two tempting shortcuts both fail:

- **Drop the gate.** The request is then accepted but nothing enforces the budget —
  under V2 the holder is never even constructed. The `<think>` runs unbounded; the
  blank turns return. It *looks* fixed and does nothing. Strictly worse.
- **Drop to the V1 runner** (`VLLM_USE_V2_MODEL_RUNNER=0`) to get the native budget.
  V1 can't run DSpark, so you lose k=5 spec-decode — the biggest single-stream speed
  lever on this checkpoint. Not acceptable for production.

To keep k=5 *and* get a real token cap, the budget has to be implemented **in the V2
sampler**.

## The port

The V2 sampler (`v1/worker/gpu/sample/sampler.py`) is not a `SamplingMetadata` +
logits-processor pipeline like V1; it is a set of persistent per-request GPU **state
objects** (`SamplingStates`, `PenaltiesState`, `LogitBiasState`, `BadWordsState`),
each with `add_request` / `apply_staged_writes` / an in-place `apply_*` Triton mask,
all invoked from `apply_sampling_params`. The V1 holder's API
(`sync_batch(BatchUpdate)`, `apply_to_logits(..., spec_token_ids=…)`) doesn't fit,
and its Python-per-step state mutation is the wrong shape for this design. So the
budget is re-implemented in the V2 idiom:

- **`ThinkingBudgetState`** ([`thinking_budget_state.py`](../patches/thinking-token-budget/thinking_budget_state.py)),
  cloned 1:1 from `BadWordsState`: a fixed-size `UvaBackedTensor` of per-request
  budgets + a host gate array, and one Triton kernel.
- **One hook.** It is called inside `Sampler.apply_sampling_params`, right after
  `bad_words` and before temperature/top-k/top-p. That single method is the funnel
  for **both** normal decode (`Sampler.sample`) and the DSpark verify path
  (`gpu/spec_decode/rejection_sampler.py::_verify`), so k=5 is covered with no
  spec-decode-specific code.
- **Correct token count under spec decode.** The kernel counts emitted tokens as
  `effective_len = output_len + expanded_local_pos` — committed output
  (`all_token_ids[prompt_len:total_len]`) fused with the in-flight draft offset —
  exactly as `bad_words` does, so the count is right across all draft positions of a
  request in one call. It never writes a per-request counter (that would double-count
  across the multi-row spec layout); the mask is pure read-only.
- **Forcing `</think>`.** Once `effective_len >= budget` and the block is still open,
  the kernel forces the (single) reasoning-end token via the same allowed-token
  primitive `LogitBiasState` uses: save the end-token logit, drive the whole vocab
  to `-inf`, restore it (both `tl.debug_barrier()`s are mandatory). A **bounded**
  backward scan (`[0, min(effective_len, budget+slack))`) of the emitted stream
  detects an already-present close and no-ops, so it never loops into an infinite
  `</think>`.
- **CUDA-graph-safe by construction.** The V2 sampler runs **eager**, after the
  captured forward replays — the same place `bad_words`/`logit_bias` already mask —
  so per-request logit forcing never enters a captured graph. (This is exactly why
  the V1 holder, which mutates Python dicts in the hot path, was deemed unsafe here.)
- **Inert unless used.** A host gate (`np.any(use_budget[idx_mapping_np])`) added to
  `_requires_logits_processing` short-circuits before any kernel launch, so requests
  without a budget are byte-for-byte unchanged.

Two more small edits complete it: thread `reasoning_config` into the `Sampler`
constructor (to resolve the `</think>` id at build time, no tokenizer needed), and
no-op **only** the V2-runner branch of the input-processor gate (the
`reasoning_config` precondition is kept).

## Validation

- **Kernel** ([`verify.py`](../patches/thinking-token-budget/verify.py)): the real
  Triton kernel on synthetic `RequestState`-shaped tensors — force vs. no-mask across
  disabled / under-budget / over-budget-open / already-closed / spec-decode-multi-row
  cases.
- **Live behaviour:** with a small budget the reasoning stops at ~budget tokens and a
  correct answer follows; a client-set budget is honoured; requests with no budget
  and non-reasoning requests are untouched.

## Serving note: pick the budget relative to `max_tokens`

If a proxy injects a default budget when the client sends none, make it **adaptive**:
`min(ceiling, max(floor, int(max_tokens * 0.6)))`. A flat ceiling only bites when
`max_tokens` exceeds it — for a smaller request the model still fills `max_tokens`
with thinking and there's no room for an answer. Scaling to the request guarantees
headroom for the content at every size. (A proxy can also, as an independent safety
net, surface a truncated `reasoning` as `content` when `content` is empty, so a turn
is never blank even below the budget.)

## Upstream

As of vLLM 0.27.0 the V2 GPU runner does not implement `thinking_token_budget`; it
exists only on the V1 runner. This is a port into the V2 sampler, not a cherry-pick.
Remove the patch if upstream wires the budget into the V2 runner.
