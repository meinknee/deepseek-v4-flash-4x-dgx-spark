# Patches

All patches are applied **idempotently at container start** by `launch/tp4-entrypoint.sh`
(exact-anchor matching, log-and-continue). They live here as standalone reference.

| Patch | What | Upstream status |
|---|---|---|
| `flashinfer-topk256/` | Adds the missing topk=256 column to FlashInfer's sm120 sparse-MLA decode dispatch (python gate + CU template instantiations) and **evicts the prebuilt `flashinfer_jit_cache` module** so the JIT compiles the patched source. Without it, every DSpark spec-decode config crashes (`Check failed: num_tokens > 64`). | Reported: flashinfer #3828 / #3988 / #4336. **Fixed on main 2026-08-08 (PR #4380, same approach)** — but in NO stable release ≤ 0.6.17, so this backport is required on every wheel vLLM 0.27.x installs. Delete when flashinfer ≥ 0.6.18 releases. |
| `vocab-clamp-50843.py` | Bounds block-argmax token ids to `vocab_size-1` in 3 Triton kernels. Prevents out-of-range token ids that kill the engine under bursty concurrency. | Open upstream: vLLM PR #50843 (alexbi29). Backport verbatim. |
| `mhc-deepgemm-gate/` | Gates the one UNgated `tf32_hc_prenorm_gemm` (DeepGEMM) call in `mhc/tilelang.py`'s broadcast variant, matching its sister functions' `is_deep_gemm_supported()` fallback. Inert when DeepGEMM works; prevents a crash class when it doesn't. | Unreported upstream as of 2026-08-12 — issue draft in `docs/`. |
| tokenizer/chat_utils tweaks (in entrypoint) | DeepSeek-V4 encoding compat + lenient historical tool-call JSON parsing (heals truncated tool args that otherwise 400 the thread forever). | Checkpoint/ops-specific; optional. |
