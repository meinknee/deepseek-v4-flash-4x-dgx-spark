#!/bin/bash
# tp4-launch.sh — launch the DeepSeek-V4-Flash TP=4 container on ONE node.
# Usage: tp4-launch.sh <rank 0-3>   (run on each node; rank 0 = head, serves the API)
# Configure via env (defaults follow the common 4x-Spark RoCE fabric layout):
#   FABRIC_PREFIX  node fabric IPs are ${FABRIC_PREFIX}.$((rank+1))   [default 10.0.0]
#   MASTER_ADDR    head node fabric IP                                 [default ${FABRIC_PREFIX}.1]
#   MODELS_DIR     host dir with the HF cache (weights)                [default $HOME/llm-models]
#   IMAGE          serving image                                       [default vllm-gb10-027:dg0810]
#   NCCL_IB_HCA / NCCL_SOCKET_IFNAME  set for YOUR rail (defaults below are ConnectX-7 port f1)
set -e
RANK="$1"
[ -n "$RANK" ] || { echo "usage: $0 <rank 0-3>"; exit 2; }
FABRIC_PREFIX="${FABRIC_PREFIX:-10.0.0}"
MASTER_ADDR="${MASTER_ADDR:-${FABRIC_PREFIX}.1}"
MODELS_DIR="${MODELS_DIR:-$HOME/llm-models}"
IMAGE="${IMAGE:-vllm-gb10-027:dg0810}"
IP="${FABRIC_PREFIX}.$((RANK + 1))"
mkdir -p "$HOME/.cache/dspark-tmp"

# Mandatory pre-boot cache ritual on this memory-marginal unified-memory box:
# skipping it once cost us a multi-hour livelock (see docs/blockers.md).
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches; echo 1 > /proc/sys/vm/compact_memory'   2>/dev/null || echo "warn: cache ritual skipped (needs sudo)"

docker rm -f dsv4f-tp4 >/dev/null 2>&1 || true
docker run -d --name dsv4f-tp4 \
  --restart unless-stopped \
  --network host --ipc host --gpus all \
  --shm-size 68719476736 --memory 118g --memory-swap 118g \
  --ulimit memlock=-1:-1 --ulimit stack=67108864:67108864 \
  --ulimit nofile=1048576:1048576 \
  --device /dev/infiniband:/dev/infiniband \
  -v "$MODELS_DIR:/cache/huggingface:rw" \
  -v "$HOME/.cache/dspark-tmp:/tmp:rw" \
  -v "$(dirname "$0")/tp4-entrypoint.sh:/tp4-entrypoint.sh:ro" \
  -e RANK="$RANK" \
  -e VLLM_HOST_IP="$IP" \
  -e MASTER_ADDR="$MASTER_ADDR" -e MASTER_PORT=25310 \
  -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e HF_HUB_DISABLE_XET=1 \
  -e VLLM_CACHE_ROOT=/cache/huggingface/vllm-cache \
  -e FLASHINFER_WORKSPACE_BASE=/cache/huggingface/flashinfer \
  -e FLASHINFER_CUDA_ARCH_LIST=12.1a -e TORCH_CUDA_ARCH_LIST=12.1a \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1800 \
  -e NCCL_NET=IB -e NCCL_IB_DISABLE=0 -e NCCL_IB_HCA="${NCCL_IB_HCA:-rocep1s0f1}" \
  -e NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}" \
  -e NCCL_IB_ADDR_RANGE="${FABRIC_PREFIX}.0/16" \
  -e NCCL_IB_ROCE_VERSION_NUM=2 -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_CROSS_NIC=1 -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_DEBUG=WARN \
  --entrypoint bash \
  "$IMAGE" \
  /tp4-entrypoint.sh
echo "rank $RANK launched on $IP"
