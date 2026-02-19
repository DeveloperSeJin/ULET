#!/bin/bash

set -euo pipefail

usage() {
  echo "Usage: $0 <cuda_id> [subject]"
  echo "  Example: $0  0 unique_sent"
  exit 1
}

# 인자: 3개(필수) + 1개(선택)
if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
fi

cuda=cuda:"$1"


# ['DELTA', 'wo_diffusion', 'DConv', 'EEGNet', 'wo_pretrained','main_diffusion', 'full_diffusion']
# [pretrain, DConv_pretrain, EEGNet_pretrain, wo_diffusion_pretrain]
# subject ['ZAB', 'ZDM', 'ZGW', 'ZJM', 'ZJN', 'ZJS', 'ZKB', 'ZKH', 'ZKW', 'ZMG', 'ZPH']

if [ -n "${2:-}" ]; then
    subject=$2
else
    subject="unique_sent"
fi

batch=32

save_name=text_embedding

python3 pretrain.py \
    --task_name task1_task2_taskNRv2 \
    --num_epoch 100 \
    -lr 0.00002 \
    -b $batch \
    --cuda "$cuda" \
    -setting "$subject" \
    -s ./save_model/checkpoints/pretrained_models \
    --save_name "$save_name"

python3 train.py  \
    --task_name task1_task2_taskNRv2 \
    --num_epoch_step1 100 \
    --num_epoch_step2 50 \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b $batch \
    --cuda "$cuda" \
    -path "./save_model/checkpoints/pretrained_models/best/${save_name}-module.pt" \
    --save_name "$save_name"

python3 train.py  \
    --task_name task1_task2_taskNRv2 \
    --num_epoch_step1 100 \
    --num_epoch_step2 50 \
    -lr1 0.00002 \
    -lr2 0.00002 \
    -b 32 \
    --cuda cuda:1 \
    -path "./save_model/checkpoints/pretrained_models/best/text_embedding-module.pt" \
    --save_name text_emedding_DDPM

# 활성화 옵션 메모:
# -con \
# -geo\
# -kl \


# #pretrain
# python3 train_eegpt.py \
#     --cuda cuda:2\
#     -s ./save_model/checkpoints/EEGPT

