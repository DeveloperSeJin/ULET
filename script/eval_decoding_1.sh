#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <cuda_id> [subject]"
  echo "  Example: $0 0 unique_sent"
  exit 1
}

# 인자: 3개(필수) + 1개(선택)
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
fi

cuda=cuda:"$1"

if [ -n "${2:-}" ]; then
    subject=$2
else
    subject="unique_sent"
fi


python3 evaluation.py \
    --task_name task1_task2_taskNRv2 \
    -t task1_task2_taskNRv2 \
    --path save_model/checkpoints/EEGPT/best/text_emedding_DDPM-module.pt \
    -cuda cuda:1
