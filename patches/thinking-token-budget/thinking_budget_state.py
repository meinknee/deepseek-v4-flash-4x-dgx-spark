# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# V2 GPU-runner native thinking-token budget.  Backport (2026-08-13) of the
# per-request `thinking_token_budget` feature into the V2 model runner, which
# DSpark speculative decoding forces on (the stock V1 ThinkingBudgetStateHolder
# is API-incompatible with, and unwired from, the V2 sampler stack).
#
# Modelled 1:1 on BadWordsState / LogitBiasState: a fixed-size per-request GPU
# state set + one eager Triton mask that runs AFTER the captured forward, inside
# Sampler.apply_sampling_params (which serves BOTH the normal-decode and the
# spec-decode/_verify paths).  Pure read-only masking -> no per-step counter,
# no post_update hook, CUDA-graph-safe by construction.
#
# Behaviour: once a request has emitted `thinking_token_budget` output tokens
# without closing its reasoning block, force the (single) reasoning-end token
# `</think>` by masking every other token to -inf (save/restore the end-token
# logit, exactly like LogitBiasState's allowed_token_ids primitive).  Fires only
# while the block is still open; a bounded scan of the already-emitted tokens
# detects an existing close and no-ops thereafter.
import numpy as np
import torch

from vllm.sampling_params import SamplingParams
from vllm.triton_utils import tl, triton
from vllm.v1.worker.gpu.buffer_utils import UvaBackedTensor
from vllm.v1.worker.gpu.states import RequestState

# How far past the budget the end-token may legitimately sit (natural or forced
# close lands within a couple tokens of the budget). Bounds the detect scan.
_END_SCAN_SLACK = 16
_SCAN_BLOCK = 256          # power-of-two, vectorised detect-scan block
_LOGITS_BLOCK_SIZE = 8192  # matches LogitBiasState's vocab sweep block


class ThinkingBudgetState:
    """Per-request thinking-token budget for the V2 GPU sampler."""

    def __init__(
        self,
        req_states: RequestState,
        vocab_size: int,
        reasoning_config,
    ):
        self.req_states = req_states
        self.max_num_reqs = req_states.max_num_reqs
        self.device = req_states.device
        self.vocab_size = vocab_size

        # On-switch: reasoning must be configured AND expose a SINGLE end token
        # (DeepSeek's </think> = one id). A multi-token end delimiter would need
        # sequential forcing; disable rather than mis-force.
        end_ids = None
        enabled = reasoning_config is not None and getattr(
            reasoning_config, "enabled", False
        )
        if enabled:
            end_ids = reasoning_config.reasoning_end_token_ids
        self.is_enabled = bool(enabled and end_ids is not None and len(end_ids) == 1)
        self.end_token_id = int(end_ids[0]) if self.is_enabled else -1

        # Per-request budget (int32); 0/absent => disabled for that row.
        self.thinking_token_budget = UvaBackedTensor(
            self.max_num_reqs, dtype=torch.int32
        )
        self.thinking_token_budget.np.fill(0)
        # Host gate array (mirrors PenaltiesState.use_penalty), indexed by
        # idx_mapping_np in _requires_logits_processing.
        self.use_budget = np.zeros(self.max_num_reqs, dtype=bool)

    def add_request(self, req_idx: int, sampling_params: SamplingParams) -> None:
        budget = 0
        if self.is_enabled:
            b = getattr(sampling_params, "thinking_token_budget", None)
            # -1 (unlimited) / None / <=0 => no budget.
            if b is not None and b > 0:
                budget = int(b)
        self.thinking_token_budget.np[req_idx] = budget
        self.use_budget[req_idx] = budget > 0

    def apply_staged_writes(self) -> None:
        self.thinking_token_budget.copy_to_uva()

    def apply_thinking_budget(
        self,
        logits: torch.Tensor,
        expanded_idx_mapping: torch.Tensor,
        idx_mapping_np: np.ndarray,
        input_ids: torch.Tensor,
        expanded_local_pos: torch.Tensor,
    ) -> None:
        if not self.is_enabled:
            return
        # Host-side per-launch skip (no budgeted request in this batch).
        if not np.any(self.use_budget[idx_mapping_np]):
            return
        num_tokens = logits.shape[0]
        _thinking_budget_kernel[(num_tokens,)](
            logits,
            logits.stride(0),
            expanded_idx_mapping,
            self.thinking_token_budget.gpu,
            self.req_states.all_token_ids.gpu,
            self.req_states.all_token_ids.gpu.stride(0),
            self.req_states.prompt_len.gpu,
            self.req_states.total_len.gpu,
            input_ids,
            expanded_local_pos,
            self.end_token_id,
            self.vocab_size,
            SLACK=_END_SCAN_SLACK,
            SCAN_BLOCK=_SCAN_BLOCK,
            LOGITS_BLOCK_SIZE=_LOGITS_BLOCK_SIZE,
        )


@triton.jit
def _thinking_budget_kernel(
    logits_ptr,
    logits_stride,
    expanded_idx_mapping_ptr,
    budget_ptr,
    all_token_ids_ptr,
    all_token_ids_stride,
    prompt_len_ptr,
    total_len_ptr,
    input_ids_ptr,
    expanded_local_pos_ptr,
    end_token_id,
    vocab_size,
    SLACK: tl.constexpr,
    SCAN_BLOCK: tl.constexpr,
    LOGITS_BLOCK_SIZE: tl.constexpr,
):
    token_idx = tl.program_id(0).to(tl.int64)
    req_state_idx = tl.load(expanded_idx_mapping_ptr + token_idx)
    budget = tl.load(budget_ptr + req_state_idx)
    if budget <= 0:
        return

    # Tokens emitted so far for THIS logit row: committed output + in-flight
    # draft offset (mirrors bad_words' effective_len = output_len + pos).
    pos = tl.load(expanded_local_pos_ptr + token_idx)
    cur_req_first_pos = token_idx - pos
    prompt_len = tl.load(prompt_len_ptr + req_state_idx)
    total_len = tl.load(total_len_ptr + req_state_idx)
    output_len = total_len - prompt_len
    effective_len = output_len + pos
    if effective_len < budget:
        return  # still within the thinking budget

    # Over budget: has </think> already been emitted? Scan the bounded window
    # [0, min(effective_len, budget+SLACK)) over committed output + in-flight
    # draft tokens. If present, the block is (being) closed -> do nothing.
    scan_n = tl.minimum(effective_len, budget + SLACK)
    output_base = all_token_ids_ptr + req_state_idx * all_token_ids_stride + prompt_len
    committed_n = tl.minimum(output_len, scan_n)
    already = 0
    for s in range(0, committed_n, SCAN_BLOCK):
        offs = s + tl.arange(0, SCAN_BLOCK)
        m = offs < committed_n
        toks = tl.load(output_base + offs, mask=m, other=-1)
        already = tl.maximum(already, tl.max(tl.where(toks == end_token_id, 1, 0)))
    for s in range(0, pos, SCAN_BLOCK):
        offs = s + tl.arange(0, SCAN_BLOCK)
        m = offs < pos
        toks = tl.load(input_ids_ptr + cur_req_first_pos + offs, mask=m, other=-1)
        already = tl.maximum(already, tl.max(tl.where(toks == end_token_id, 1, 0)))
    if already > 0:
        return

    # Force </think>: save its logit, drive the whole vocab to -inf, restore it
    # (LogitBiasState allowed_token_ids primitive; both barriers are mandatory).
    saved = tl.load(logits_ptr + token_idx * logits_stride + end_token_id)
    tl.debug_barrier()
    for i in range(0, vocab_size, LOGITS_BLOCK_SIZE):
        offset = i + tl.arange(0, LOGITS_BLOCK_SIZE)
        tl.store(
            logits_ptr + token_idx * logits_stride + offset,
            -float("inf"),
            mask=offset < vocab_size,
        )
    tl.debug_barrier()
    tl.store(logits_ptr + token_idx * logits_stride + end_token_id, saved)
