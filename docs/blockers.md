# The blocker ledger — every failure between `docker run` and production

Ten distinct defects stood between upstream vLLM 0.27.0 and a serving 4×-Spark cluster. Each cost
at least one ~12-minute boot to find. They are listed in the order you will hit them, with fixes.
(Upstream linkage per bug is in `patches/README.md`; several were independently found by others —
see Credits in the main README.)

## The 10 blockers (each cost ≥1 boot; in order encountered)

1. **`--moe-backend flashinfer_b12x` invalid for MXFP4** — stock b12x is NVFP4-only; Anemll's
   B12X_MXFP4 is fork-only. → `--moe-backend auto`.
2. **NCCL rendezvous hang** — `NCCL_IB_GID_INDEX=3` points at the link-local fe80 GID on this stack
   (fabric IPv4 GID is idx5); nccl 2.30.7 wedges silently (0% cpu/gpu after tp/ep group select).
   → `NCCL_IB_ADDR_RANGE=10.0.0.0/16`. **GID fix is stack-specific** (Anemll needs GID_INDEX=3!).
3. **Crash-loop leaves stale rendezvous state** (follower TCPStore "shut down too early") → full
   teardown (docker rm + pkill vllm on ALL nodes) before every relaunch.
4. **DeepGEMM "Unknown SF transformation" (dense FP8 post-processing)** — 0.27's DeepGEMM pin has
   zero sm12x support. Interim fix was cutlass; REAL fix = the dg0810 image (fixed DeepGEMM).
5. **Same assert in MoE weight conversion** (`_pack_deepgemm_mxfp4_scales`) — same root cause.
   (`VLLM_MOE_USE_DEEP_GEMM=0` does NOT affect the MXFP4 oracle — env trap; the oracle reads
   `VLLM_USE_DEEP_GEMM`.)
6. **triton MoE rejected** on sm_121 ("kernel does not support current device") — marlin was the
   only stock MXFP4 MoE for sm_121 (live-verified select+repack) until DeepGEMM was fixed.
7. **DSv4 mHC hyper-connection calls DeepGEMM ungated** (broadcast variant, mhc/tilelang.py:374) →
   "hyperconnection.hpp:56 Unsupported architecture" on the broken lib. → entrypoint patch gates it
   to `_tilelang_hc_prenorm_gemm(..., hidden_size, 1)` (inert now that DeepGEMM works).
8. **`--linear-backend cutlass` fails at RUNTIME dispatch on sm_121** (`scaled_mm_helper.hpp:17`) —
   the compiled c3x sm120 kernels don't cover DSv4's blockwise-FP8 shape. → dense on fixed DeepGEMM.
9. **`--block-size 64` is IMPOSSIBLE** — page math `block//compress_ratio×584` zeroes for the
   compress-128 layers (ZeroDivisionError). Block 256 stays. **k<5 is also impossible** (0.27
   validator: k ≥ dspark_block_size=5). So spec on 0.27 = k=5 or nothing.
10. **THE k=5 BLOCKER: missing topk=256 in FlashInfer's sm120 decode dispatch.** Debug print
    captured the tuple: `mt=DSV4 nt=5 nh=16 topk=256 d_qk=512 pbs=64` — all in-table EXCEPT
    topk=256 (allowlist had 128/512/1024; 0.27 adaptive-topk narrows 512→256 at short ctx) → 5-token
    spec batches fell to the >64-token big kernel → tvm assert. **Three-part fix (in the
    entrypoint):** (a) extend `_DECODE_DSV4_DISPATCH` frozenset (span-bounded edit — "(16, 128),"
    is NOT unique, dsv3_2's table has it too); (b) add `DSV4_DISPATCH(*, 256)` instantiations in
    `flashinfer/data/csrc/sparse_mla_sm120_decode_dsv4.cu` (kernel templated <MT,H,TOPK,PBS>);
    (c) **evict the prebuilt `flashinfer_jit_cache/jit_cache/sparse_mla_sm120/`** — `is_aot()`
    loads the shipped .so and IGNORES patched csrc; nvcc is in-image, JIT compiles the patched
    kernel (~3 min first boot; cached on the volume after). **Upstream this to FlashInfer.**

## Ops notes

- Boot ~15 min cold (NCCL init is slow-looking under INFO; weight load ~90 s warm-cache; JIT/autotune).
- The full teardown-relaunch cycle (`docker rm -f` everywhere → followers → head) is REQUIRED after
  any crash; `--restart` alone reboots into stale-rendezvous stalls.
- Debug workflow that cracked blocker 10: patch a tuple-print into the fallback branch, one boot,
  read stderr. Offline-validate any entrypoint patch in a scratch `docker run` FIRST (a broken
  anchor costs a 12-min boot).
- Follow-ups: acceptance-tail investigation (jasl 0804 draft fix), upstream the topk-256 fix to
  FlashInfer, consider PR #50796's DeepGEMM pin bump when it merges (removes the dg0810 delta),
  wedge-watchdog arming (`head-node:~/ds_watchdog.sh`), Sunny/Clamps precheck re-verify.
