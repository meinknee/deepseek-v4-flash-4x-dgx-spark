#!/bin/bash
# DSv4F TP=4 entrypoint — STOCK vLLM 0.27.0 image (vllm-gb10-027:local), 2026-08-12.
# Ported from the Anemll entrypoint: kv-cache-dtype nvfp4_ds_mla -> fp8_ds_mla (0.27
# has no nvfp4_ds_mla; fp8_ds_mla = SAME memory, verified). moe-backend flashinfer_b12x
# valid verbatim (config/kernel.py MoEBackend). All 3 in-container patches are made
# BOOT-SAFE (log + continue, never exit) — 0.27's native deepseek_v4 tokenizer may not
# need the Anemll compat tweaks, and a shifted line must not crash the first boot.
set -e
export PATH="/usr/local/cuda/bin:/usr/local/bin:${PATH:-}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

VP="$(python3 -c 'import vllm,os;print(os.path.dirname(vllm.__file__))')"

# ── (best-effort) MiaAI tokenizer compat: copy encoding + reasoning_effort default ──
ENCODING_SOURCE=""
for candidate in /cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/*/encoding/encoding_dsv4.py; do
  [ -f "${candidate}" ] && ENCODING_SOURCE="${candidate}" && break
done
if [ -n "${ENCODING_SOURCE}" ]; then
  cp "${ENCODING_SOURCE}" "${VP}/tokenizers/deepseek_v4_encoding.py" 2>/dev/null \
    && echo "entrypoint: copied encoding_dsv4.py" || echo "entrypoint: WARN encoding copy failed (continuing)"
else
  echo "entrypoint: WARN encoding_dsv4.py not found (0.27 tokenizer may not need it; continuing)"
fi
python3 - "$VP" <<'PYEOF' || echo "entrypoint: WARN reasoning_effort patch skipped (continuing)"
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "tokenizers" / "deepseek_v4.py"
if not p.exists():
    print("entrypoint: deepseek_v4.py absent, skip reasoning_effort patch"); sys.exit(0)
s = p.read_text()
new = ('elif reasoning_effort in ("max", "xhigh"):\n                reasoning_effort = "max"\n'
       '            elif reasoning_effort == "high":\n                reasoning_effort = "high"\n'
       '            else:\n                reasoning_effort = "low"')
old = ('elif reasoning_effort in ("max", "xhigh"):\n                reasoning_effort = "max"\n'
       '            else:\n                reasoning_effort = "high"')
if new in s:
    print("entrypoint: reasoning_effort already patched")
elif old in s:
    p.write_text(s.replace(old, new)); print("entrypoint: reasoning_effort patched -> default low")
else:
    print("entrypoint: reasoning_effort block not found (0.27 layout differs; leaving default)")
PYEOF

# ── (important) Lenient tool-history parsing — prevents tool-call 400 loops ──
python3 - "$VP" <<'PYEOF' || echo "entrypoint: WARN chat_utils patch skipped (continuing)"
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "entrypoints" / "chat_utils.py"
s = p.read_text()
new = """                if content := function.get("arguments"):
                    if not isinstance(content, (dict, list)):
                        try:
                            parsed = json.loads(content)
                        except json.JSONDecodeError:
                            parsed = None
                            for _suf in ('"', '"}', '"}}', '"]}', '}', '}}'):
                                try:
                                    parsed = json.loads(content + _suf)
                                    break
                                except json.JSONDecodeError:
                                    continue
                        function["arguments"] = parsed if parsed is not None else {}"""
old = """                if content := function.get("arguments"):
                    if not isinstance(content, (dict, list)):
                        parsed = json.loads(content)
                        function["arguments"] = parsed if parsed is not None else {}"""
if new in s:
    print("entrypoint: chat_utils lenient parse already present")
elif old in s:
    p.write_text(s.replace(old, new)); print("entrypoint: chat_utils lenient parse installed")
else:
    print("entrypoint: chat_utils strict block not found (0.27 layout differs; leaving as-is)")
PYEOF

# ── (0.27 fix #7) mhc broadcast fn: gate the ungated DeepGEMM call ───────────
# Upstream bug: mhc/tilelang.py broadcast variant calls tf32_hc_prenorm_gemm
# UNconditionally (sisters are gated). On sm_121 + VLLM_USE_DEEP_GEMM=0 this
# asserts "Unsupported architecture" (hyperconnection.hpp:56). Route to the
# sister functions tilelang fallback. Idempotent, boot-safe.
python3 - "$VP" <<'PYPATCH' || echo "entrypoint: WARN mhc patch skipped (continuing)"
import sys
from pathlib import Path
p = Path(sys.argv[1]) / "model_executor" / "kernels" / "mhc" / "tilelang.py"
s = p.read_text()
marker = "is_deep_gemm_supported()  # dsv4f-gb10 patch"
if marker in s:
    print("entrypoint: mhc patch already present"); sys.exit(0)
old_n = "    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64))"
new_n = ("    from vllm.utils.deep_gemm import is_deep_gemm_supported\n"
         "    _use_dg = is_deep_gemm_supported()  # dsv4f-gb10 patch\n"
         "    n_splits = compute_num_split(64, hidden_size, cdiv(num_tokens, 64)) if _use_dg else 1")
old_c = ("    from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm\n\n"
         "    tf32_hc_prenorm_gemm(\n"
         "        residual_flat,\n"
         "        fn_broadcast,\n"
         "        gemm_out_mul,\n"
         "        gemm_out_sqrsum,\n"
         "        n_splits,\n"
         "    )")
new_c = ("    if _use_dg:\n"
         "        from vllm.utils.deep_gemm import tf32_hc_prenorm_gemm\n"
         "        tf32_hc_prenorm_gemm(\n"
         "            residual_flat,\n"
         "            fn_broadcast,\n"
         "            gemm_out_mul,\n"
         "            gemm_out_sqrsum,\n"
         "            n_splits,\n"
         "        )\n"
         "    else:\n"
         "        _tilelang_hc_prenorm_gemm(\n"
         "            residual_flat,\n"
         "            fn_broadcast,\n"
         "            gemm_out_mul,\n"
         "            gemm_out_sqrsum,\n"
         "            hidden_size,\n"
         "            1,\n"
         "        )")
if s.count(old_n) != 1 or s.count(old_c) != 1:
    print(f"entrypoint: mhc patch anchors not unique (n={s.count(old_n)}, c={s.count(old_c)}) - SKIPPING")
    sys.exit(0)
s = s.replace(old_n, new_n).replace(old_c, new_c)
p.write_text(s)
print("entrypoint: mhc broadcast DeepGEMM-gate patch installed")
PYPATCH


# ── DEBUG: log the failing dispatch tuple in flashinfer sm120 sparse-MLA ──────
python3 - <<'PYDBG' || echo "entrypoint: WARN flashinfer debug patch skipped"
from pathlib import Path
import flashinfer, os
p = Path(os.path.dirname(flashinfer.__file__)) / "mla" / "_sparse_mla_sm120.py"
s = p.read_text()
marker = "[dsv4f-debug]"
if marker in s:
    print("entrypoint: flashinfer debug already present")
else:
    old = "        module.sparse_mla_sm120_paged_attention(\n            q,"
    new = ("        if num_tokens <= 64:\n"
           "            import sys as _sys\n"
           "            print(f\"[dsv4f-debug] big-kernel fallback: mt={model_type} nt={num_tokens} "
           "nh={num_heads} topk={topk} d_qk={d_qk} pbs={kv_pbs} extra_topk={extra_topk}\", "
           "file=_sys.stderr, flush=True)\n"
           "        module.sparse_mla_sm120_paged_attention(\n            q,")
    assert s.count(old) == 1, f"anchor count {s.count(old)}"
    p.write_text(s.replace(old, new))
    print("entrypoint: flashinfer debug patch installed")
PYDBG


# ── FIX: add topk=256 to flashinfer sm120 decode-dsv4 dispatch (gate + kernel) ─
python3 - <<'PYFIX' || echo "entrypoint: WARN topk-256 fix skipped"
from pathlib import Path
import flashinfer, os, re
fi = Path(os.path.dirname(flashinfer.__file__))
# 1) python gate: span-bounded edit inside _DECODE_DSV4_DISPATCH only
p = fi / "mla" / "_sparse_mla_sm120.py"
s = p.read_text()
if "(16, 256)," in s:
    print("entrypoint: topk-256 python gate already present")
else:
    start = s.index("_DECODE_DSV4_DISPATCH = frozenset(")
    end = s.index(")", s.index("}", start))
    span = s[start:end]
    assert "(16, 128)," in span and "(16, 256)," not in span
    new_span = span.replace("(16, 128),", "(16, 128),\n        (8, 256),\n        (16, 256),\n        (32, 256),\n        (64, 256),\n        (128, 256),", 1)
    s = s[:start] + new_span + s[end:]
    p.write_text(s)
    print("entrypoint: topk-256 python gate installed")
# 2) CU launcher: instantiate the 256 column (dsv4 file only)
c = fi / "data" / "csrc" / "sparse_mla_sm120_decode_dsv4.cu"
s2 = c.read_text()
if "DSV4_DISPATCH(16, 256)" in s2:
    print("entrypoint: topk-256 CU instantiation already present")
else:
    old = "  DSV4_DISPATCH(16, 128)"
    assert s2.count(old) == 1, f"cu anchor count {s2.count(old)}"
    s2 = s2.replace(old, "  DSV4_DISPATCH(8, 256)\n  DSV4_DISPATCH(16, 128)\n  DSV4_DISPATCH(16, 256)\n  DSV4_DISPATCH(32, 256)\n  DSV4_DISPATCH(64, 256)\n  DSV4_DISPATCH(128, 256)")
    c.write_text(s2)
    print("entrypoint: topk-256 CU instantiation installed")
PYFIX
# workspace JIT cache kept: holds the PATCHED compiled module after first boot
rm -rf /usr/local/lib/python3.12/dist-packages/flashinfer_jit_cache/jit_cache/sparse_mla_sm120 2>/dev/null
echo "entrypoint: prebuilt sparse_mla module evicted -> JIT will compile patched csrc"

# ── #50843 crash insurance: vocab-bound clamp in Triton argmax kernels ──
python3 -c "import base64;exec(base64.b64decode('IyBCYWNrcG9ydCBvZiBvcGVuIHVwc3RyZWFtIHZMTE0gUFIgIzUwODQzOiBjbGFtcCBibG9jay1hcmdtYXggdG9rZW4gaWRzIHRvCiMgdm9jYWJfc2l6ZS0xLiBQcmV2ZW50cyBvdXQtb2YtcmFuZ2UgaWRzICh+NzgvbWluIG9uIERTdjRGIHVuZGVyIGJ1cnN0eQojIGNvbmN1cnJlbmN5IHBlciB0aGUgUFIncyBwcm9kdWN0aW9uIEEvQikgdGhhdCBraWxsIHRoZSBlbmdpbmUgaW4gOTBzLTUuNW1pbi4KIyBJZGVtcG90ZW50OyBsb2dzIGFuZCBjb250aW51ZXMgb24gYW55IG1pc21hdGNoLgpmcm9tIHBhdGhsaWIgaW1wb3J0IFBhdGgKCkJBU0UgPSBQYXRoKCIvdXNyL2xvY2FsL2xpYi9weXRob24zLjEyL2Rpc3QtcGFja2FnZXMvdmxsbS92MS93b3JrZXIvZ3B1IikKU0lURVMgPSBbCiAgICAoQkFTRSAvICJzcGVjX2RlY29kZS9yZWplY3Rpb25fc2FtcGxlcl91dGlscy5weSIsCiAgICAgIiAgICAgICAgdG9rZW5faWQgPSBibG9ja19pZHggKiBCTE9DS19TSVpFICsgaWR4IiwKICAgICAiICAgICAgICB0b2tlbl9pZCA9IHRsLm1pbmltdW0oYmxvY2tfaWR4ICogQkxPQ0tfU0laRSArIGlkeCwgdm9jYWJfc2l6ZSAtIDEpIiksCiAgICAoQkFTRSAvICJzcGVjX2RlY29kZS9yZWplY3Rpb25fc2FtcGxlcl91dGlscy5weSIsCiAgICAgIiAgICB0b2tlbl9pZCA9IGJsb2NrX2lkeCAqIEJMT0NLX1NJWkUgKyBpZHgiLAogICAgICIgICAgdG9rZW5faWQgPSB0bC5taW5pbXVtKGJsb2NrX2lkeCAqIEJMT0NLX1NJWkUgKyBpZHgsIHZvY2FiX3NpemUgLSAxKSIpLAogICAgKEJBU0UgLyAic2FtcGxlL2d1bWJlbC5weSIsCiAgICAgIiAgICB0b2tlbl9pZCA9IGJsb2NrX2lkeCAqIEJMT0NLX1NJWkUgKyBpZHgiLAogICAgICIgICAgdG9rZW5faWQgPSB0bC5taW5pbXVtKGJsb2NrX2lkeCAqIEJMT0NLX1NJWkUgKyBpZHgsIHZvY2FiX3NpemUgLSAxKSIpLApdCmZvciBwYXRoLCBvbGQsIG5ldyBpbiBTSVRFUzoKICAgIHRyeToKICAgICAgICBsaW5lcyA9IHBhdGgucmVhZF90ZXh0KCkuc3BsaXQoIlxuIikKICAgICAgICBpZiBhbnkobCA9PSBuZXcgZm9yIGwgaW4gbGluZXMpOgogICAgICAgICAgICBwcmludChmIjUwODQzLWNsYW1wOiB7cGF0aC5uYW1lfTogYWxyZWFkeSBwYXRjaGVkIikKICAgICAgICAgICAgY29udGludWUKICAgICAgICBoaXRzID0gW2kgZm9yIGksIGwgaW4gZW51bWVyYXRlKGxpbmVzKSBpZiBsID09IG9sZF0KICAgICAgICBhc3NlcnQgbGVuKGhpdHMpID09IDEsIGYiYW5jaG9yIHh7bGVuKGhpdHMpfSBpbiB7cGF0aC5uYW1lfSIKICAgICAgICBsaW5lc1toaXRzWzBdXSA9IG5ldwogICAgICAgIHQgPSAiXG4iLmpvaW4obGluZXMpCiAgICAgICAgY29tcGlsZSh0LCBzdHIocGF0aCksICJleGVjIikKICAgICAgICBwYXRoLndyaXRlX3RleHQodCkKICAgICAgICBwcmludChmIjUwODQzLWNsYW1wOiB7cGF0aC5uYW1lfTogaW5zdGFsbGVkIikKICAgIGV4Y2VwdCBFeGNlcHRpb24gYXMgZToKICAgICAgICBwcmludChmIjUwODQzLWNsYW1wOiBXQVJOIHtwYXRoLm5hbWV9OiB7ZX0gKGNvbnRpbnVpbmcpIikK').decode())" || echo "entrypoint: WARN 50843 clamp skipped (continuing)"

HEADLESS_FLAG=""
[ "${RANK}" != "0" ] && HEADLESS_FLAG="--headless"

exec /usr/local/bin/vllm serve deepseek-ai/DeepSeek-V4-Flash-0731 \
  --served-model-name deepseek-v4-flash-0731 --host 0.0.0.0 --port 8888 --trust-remote-code \
  --tensor-parallel-size 4 --pipeline-parallel-size 1 \
  --kv-cache-dtype fp8_ds_mla --block-size 256 \
  --max-model-len 1048576 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8264 \
  --max-cudagraph-capture-size 32 \
  --gpu-memory-utilization 0.75 \
  --compilation-config '{"cudagraph_mode":"FULL_DECODE_ONLY"}' \
  --override-generation-config '{"temperature":0.0}' \
  --speculative-config '{"method":"dspark","num_speculative_tokens":5,"draft_sample_method":"greedy"}' \
  --enable-prefix-caching --async-scheduling --enable-chunked-prefill \
  --tokenizer-mode deepseek_v4 --distributed-executor-backend mp \
  --moe-backend auto --enable-flashinfer-autotune \
  --tool-call-parser deepseek_v4 --enable-auto-tool-choice --reasoning-parser deepseek_v4 \
  --default-chat-template-kwargs '{"thinking":false}' \
  --generation-config vllm \
  --nnodes 4 --node-rank "${RANK}" --master-addr "${MASTER_ADDR:-10.0.0.1}" --master-port 25310 \
  ${HEADLESS_FLAG}
