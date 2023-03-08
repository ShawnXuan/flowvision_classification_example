#!/bin/bash
DEVICE_NUM_PER_NODE=${1:-1}
DATA_ROOT=${2:-/path/to/data_dir}

python3 -m oneflow.distributed.launch \
    --nproc_per_node $DEVICE_NUM_PER_NODE \
    --nnodes 1 \
    --node_rank 0 \
    --master_addr 127.0.0.1 \
    finetune.py \
        --batch_size 64 \
	--save_snapshot \
        --data_dir=$DATA_ROOT 
