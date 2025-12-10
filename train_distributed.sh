#!/bin/bash
# Set PYTHONPATH to include apex directory
export PYTHONPATH=$(pwd)/apex:$PYTHONPATH
CUDA_VISIBLE_DEVICES=0,1,2,3 python3 -m torch.distributed.launch --nproc_per_node=4 train_distributed.py