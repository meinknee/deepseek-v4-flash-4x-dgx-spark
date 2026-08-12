# DRAFT — vLLM issue: ungated DeepGEMM call in mhc/tilelang.py broadcast variant

**Status: for repo-owner review before filing at github.com/vllm-project/vllm/issues.**

---

**Title:** [Bug] DeepSeek-V4 mHC broadcast path calls `tf32_hc_prenorm_gemm` (DeepGEMM) without the
`is_deep_gemm_supported()` gate its sibling functions use

## Symptom

Serving DeepSeek-V4-Flash-0731 on hardware where DeepGEMM is unavailable or broken (e.g. GB10 /
sm_121 with the currently-pinned DeepGEMM `e21c821`, which has no sm12x dispatch — see #47436),
the profile-run forward crashes:

```
RuntimeError: Assertion error (deepgemm-src/csrc/apis/hyperconnection.hpp:56): Unsupported architecture
```

even with `VLLM_USE_DEEP_GEMM=0`, which correctly disables every *other* DeepGEMM call site.

## Root cause

In `vllm/model_executor/kernels/mhc/tilelang.py`, the two per-layer hot paths gate their DeepGEMM
usage and fall back to the TileLang kernels:

```python
use_deep_gemm = is_deep_gemm_supported()
...
if use_deep_gemm:
    tf32_hc_prenorm_gemm(...)
else:
    _tilelang_hc_prenorm_gemm(...)
```

but the **broadcast variant** (`mhc_pre_broadcast_tilelang`, the first-layer path, ~line 374 at
v0.27.0) imports and calls `tf32_hc_prenorm_gemm` **unconditionally**, with `n_splits` computed for
the DeepGEMM layout. On any platform where `is_deep_gemm_supported()` is False (env-disabled or
unsupported arch), this is the one remaining crash site.

## Fix (validated in production, TP=4 on 4× GB10)

Mirror the sibling functions' gate:

```python
from vllm.utils.deep_gemm import is_deep_gemm_supported
_use_dg = is_deep_gemm_supported()
n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64)) if _use_dg else 1
...
if _use_dg:
    tf32_hc_prenorm_gemm(residual_flat, fn_broadcast, gemm_out_mul, gemm_out_sqrsum, n_splits)
else:
    _tilelang_hc_prenorm_gemm(residual_flat, fn_broadcast, gemm_out_mul, gemm_out_sqrsum, hidden_size, 1)
```

(`hc_mult=1` satisfies `_tilelang_hc_prenorm_gemm`'s shape asserts for the broadcast shapes:
`x.shape[1] == 1 × hidden_size`; buffers sized with `n_splits=1`.)

Happy to send this as a PR.

Environment: vLLM 0.27.0, DeepSeek-V4-Flash-0731, 4× GB10 (sm_121), TP=4, CUDA 13.
