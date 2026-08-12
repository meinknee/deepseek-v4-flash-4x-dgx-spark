#!/bin/bash
# Build the serving image: upstream vLLM 0.27.0 for GB10 (sm_121), then swap the
# DeepGEMM pin to a rev with sm12x support (vLLM's pinned e21c821 has NONE —
# vllm issue #47436, open fix PR #50796 pins the same rev we use here).
#
# Run ON a Spark (native aarch64). Needs ~100 GB free disk, 2-4 h. The CUDA
# compile is memory-hungry: do NOT build while serving on the same node.
set -euo pipefail
DEEPGEMM_REV="${DEEPGEMM_REV:-2fd67329ec2942f65ba35d561256ab6ed3b903cb}"  # nv_dev+situ

git clone --branch v0.27.0 --depth 1 https://github.com/vllm-project/vllm.git vllm-src
cd vllm-src

# swap the DeepGEMM pin (both cmake fetch + install helper reference it)
sed -i "s/e21c821f39a2056d68067a466c64ddc942200106/${DEEPGEMM_REV}/g" \
  cmake/external_projects/deepgemm.cmake tools/install_deepgemm.sh
grep -rn "${DEEPGEMM_REV}" cmake/external_projects/deepgemm.cmake >/dev/null || {
  echo "pin swap failed"; exit 1; }

export DOCKER_BUILDKIT=1
docker build \
  -f docker/Dockerfile \
  --target vllm-openai \
  --build-arg torch_cuda_arch_list=12.1a \
  --build-arg max_jobs=8 \
  --build-arg nvcc_threads=4 \
  --build-arg RUN_WHEEL_CHECK=false \
  -t vllm-gb10-027:dg0810 \
  .
echo "built vllm-gb10-027:dg0810 — distribute to all nodes:"
echo "  docker save vllm-gb10-027:dg0810 | ssh <node> docker load"
