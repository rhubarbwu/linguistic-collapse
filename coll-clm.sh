#!/bin/bash

#SBATCH --output=logs/collect/%x.%j.log
#SBATCH --err=logs/collect/%x.%j.log
##SBATCH --gres=gpu:a10s0:1
#SBATCH --mem=8G

. ./env-n

nvidia-smi

BATCH_SIZE=${BATCH_SIZE:-16}
CONFIG=${CONFIG:-TinyStories-01x0064_01n}
DECAY=${DECAY:-}
DATASET=roneneldan/TinyStories
DATA_DIR=$STORAGE/roneneldan___tiny_stories
DEVICE=${DEVICE:-cuda:0}
SAVE_EVERY=${SAVE_EVERY:-1024}
STAGE=${STAGE:-means}
CKPT_IDX=${CKPT_IDX:-0}
DTYPE=${DTYPE:-float32}

echo $DATASET
echo $DATA_DIR
echo $HF_HOME

python run_clm.py --model_name_or_path $SCRATCH/TS/$CONFIG --output_dir $SCRATCH/TS/$CONFIG --tokenizer_name EleutherAI/gpt-neo-125M --do_collect --cache_dir $SCRATCH --stats_dir $SCRATCH/stats/ --dataset_name $DATASET --data_dir $DATA_DIR --dataloader_num_workers 2 --preprocessing_num_workers 2 --device $DEVICE --batch_size $BATCH_SIZE --save_every $SAVE_EVERY --stage $STAGE --model_ckpt_idx $CKPT_IDX --torch_dtype $DTYPE $@
