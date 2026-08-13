#!/usr/bin/env python3
"""Idempotently wire the native thinking_token_budget into the vLLM V2 GPU runner.

Writes `thinking_budget_state.py` (this dir) into the vLLM package and applies the
Sampler / model_runner / input_processor edits that register it. Safe to re-run;
each edit is guarded by a marker and an exactly-one-occurrence anchor assertion,
and every touched file is byte-compiled before it is kept.

    python3 apply.py [VLLM_ROOT]     # default: the installed vllm package

Verified against vLLM 0.27.0. See ./README.md for the what/why and revert steps.
"""
import py_compile
import sys
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
VLLM_ROOT = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else pathlib.Path(
    __import__("vllm", fromlist=["__file__"]).__file__).parent

MODULE_REL = "v1/worker/gpu/sample/thinking_budget_state.py"

# (relative path under vllm root, exact old, new, marker-means-already-applied)
PATCHES = [
    ("v1/worker/gpu/sample/sampler.py",
     "from vllm.v1.worker.gpu.sample.bad_words import BadWordsState\n",
     "from vllm.v1.worker.gpu.sample.bad_words import BadWordsState\n"
     "from vllm.v1.worker.gpu.sample.thinking_budget_state import ThinkingBudgetState\n",
     "from vllm.v1.worker.gpu.sample.thinking_budget_state import ThinkingBudgetState"),

    ("v1/worker/gpu/sample/sampler.py",
     "        num_speculative_tokens: int = 1,\n"
     "        use_fp64_gumbel: bool = False,\n"
     "    ):\n",
     "        num_speculative_tokens: int = 1,\n"
     "        use_fp64_gumbel: bool = False,\n"
     "        reasoning_config=None,\n"
     "    ):\n",
     "        reasoning_config=None,\n    ):"),

    ("v1/worker/gpu/sample/sampler.py",
     "        self.bad_words_state = BadWordsState(req_states)\n"
     "        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)\n",
     "        self.bad_words_state = BadWordsState(req_states)\n"
     "        self.thinking_budget_state = ThinkingBudgetState(\n"
     "            req_states, vocab_size, reasoning_config\n"
     "        )\n"
     "        self.logprob_token_ids_state = LogprobTokenIdsState(max_num_reqs, device)\n",
     "self.thinking_budget_state = ThinkingBudgetState("),

    ("v1/worker/gpu/sample/sampler.py",
     "        self.bad_words_state.add_request(req_idx, sampling_params)\n"
     "        self.logprob_token_ids_state.add_request(req_idx, sampling_params)\n",
     "        self.bad_words_state.add_request(req_idx, sampling_params)\n"
     "        self.thinking_budget_state.add_request(req_idx, sampling_params)\n"
     "        self.logprob_token_ids_state.add_request(req_idx, sampling_params)\n",
     "self.thinking_budget_state.add_request(req_idx, sampling_params)"),

    ("v1/worker/gpu/sample/sampler.py",
     "        self.bad_words_state.apply_staged_writes()\n"
     "        self.logprob_token_ids_state.apply_staged_writes()\n",
     "        self.bad_words_state.apply_staged_writes()\n"
     "        self.thinking_budget_state.apply_staged_writes()\n"
     "        self.logprob_token_ids_state.apply_staged_writes()\n",
     "self.thinking_budget_state.apply_staged_writes()"),

    ("v1/worker/gpu/sample/sampler.py",
     "        # Apply bad words masking in place.\n"
     "        self.bad_words_state.apply_bad_words(\n"
     "            logits,\n"
     "            expanded_idx_mapping,\n"
     "            idx_mapping_np,\n"
     "            input_ids,\n"
     "            expanded_local_pos,\n"
     "        )\n\n"
     "        # Apply temperature in place.\n",
     "        # Apply bad words masking in place.\n"
     "        self.bad_words_state.apply_bad_words(\n"
     "            logits,\n"
     "            expanded_idx_mapping,\n"
     "            idx_mapping_np,\n"
     "            input_ids,\n"
     "            expanded_local_pos,\n"
     "        )\n\n"
     "        # Apply thinking-token budget: force </think> once the budget is spent.\n"
     "        self.thinking_budget_state.apply_thinking_budget(\n"
     "            logits,\n"
     "            expanded_idx_mapping,\n"
     "            idx_mapping_np,\n"
     "            input_ids,\n"
     "            expanded_local_pos,\n"
     "        )\n\n"
     "        # Apply temperature in place.\n",
     "self.thinking_budget_state.apply_thinking_budget("),

    ("v1/worker/gpu/sample/sampler.py",
     "        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):\n"
     "            return True\n\n"
     "        states = self.sampling_states\n",
     "        if np.any(self.bad_words_state.num_bad_words.np[idx_mapping_np] > 0):\n"
     "            return True\n"
     "        if np.any(self.thinking_budget_state.use_budget[idx_mapping_np]):\n"
     "            return True\n\n"
     "        states = self.sampling_states\n",
     "if np.any(self.thinking_budget_state.use_budget[idx_mapping_np]):"),

    # thread reasoning_config into the Sampler construction
    ("v1/worker/gpu/model_runner.py",
     "                num_speculative_tokens=self.decode_query_len,\n"
     "                use_fp64_gumbel=self.model_config.use_fp64_gumbel,\n"
     "            )\n",
     "                num_speculative_tokens=self.decode_query_len,\n"
     "                use_fp64_gumbel=self.model_config.use_fp64_gumbel,\n"
     "                reasoning_config=self.vllm_config.reasoning_config,\n"
     "            )\n",
     "reasoning_config=self.vllm_config.reasoning_config,"),

    # no-op ONLY the V2-runner refusal (keep the reasoning_config precondition)
    ("v1/engine/input_processor.py",
     "                if self.use_v2_model_runner:\n"
     "                    raise VLLMValidationError(\n"
     "                        \"thinking_token_budget is not yet supported by the V2 \"\n",
     "                if False:  # patched: thinking_token_budget wired into the V2 runner\n"
     "                    raise VLLMValidationError(\n"
     "                        \"thinking_token_budget is not yet supported by the V2 \"\n",
     "if False:  # patched: thinking_token_budget wired into the V2 runner"),
]


def main():
    changed = 0
    mod_path = VLLM_ROOT / MODULE_REL
    want = (HERE / "thinking_budget_state.py").read_bytes()
    if not mod_path.exists() or mod_path.read_bytes() != want:
        mod_path.write_bytes(want)
        changed += 1
        print("WROTE  ", mod_path)
    else:
        print("ALREADY", MODULE_REL)
    py_compile.compile(str(mod_path), doraise=True)

    for rel, old, new, marker in PATCHES:
        p = VLLM_ROOT / rel
        s = p.read_text()
        if marker in s:
            print("ALREADY", rel, "::", marker[:44])
            continue
        n = s.count(old)
        if n != 1:
            raise SystemExit(
                f"ABORT {rel}: expected 1 anchor, found {n} "
                f"(anchor starts {old[:50]!r}) — vLLM version drift?")
        p.write_text(s.replace(old, new, 1))
        py_compile.compile(str(p), doraise=True)
        changed += 1
        print("PATCHED", rel, "::", marker[:44])

    print(f"\nthinking-token-budget: done (changed={changed})")


if __name__ == "__main__":
    main()
