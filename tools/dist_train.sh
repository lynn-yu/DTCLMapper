#!/usr/bin/env bash

#CONFIG=$1
#GPUS=$2
PORT=${PORT:-28519}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python3 -m torch.distributed.launch --nproc_per_node=1 --master_port=$PORT \
    $(dirname "$0")/train_maptr.py  --launcher pytorch ${@:3} --deterministic #--resume-from '/data0/lsy/4.v2_MapTR-main/work_dirs/run_0/epoch_4.pth'
