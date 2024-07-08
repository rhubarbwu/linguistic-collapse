#!/bin/bash

#SBATCH --qos=normal
#SBATCH --output=logs/train/%x.%j.log
#SBATCH --err=logs/train/%x.%j.log
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G

. ~/.bashrc
. ./env-h

nvidia-smi

BATCH_SIZE=${BATCH_SIZE:-1}
CONFIG=${CONFIG:-config.json}
DATASET=roneneldan/TinyStories
DATA_DIR=$DATASET_DIR/roneneldan___tiny_stories
EPOCHS=${EPOCHS:-1}
DECAY=${DECAY:-0}

echo $DATASET
echo $DATA_DIR
echo $MODEL_NAME # manually set if not running with batch-train.sh

mkdir -p $SCRATCH/TS

HF_HOME=$SCRATCH python run_clm.py --config_name $CONFIG --output_dir $SCRATCH/TS/$MODEL_NAME --tokenizer_name EleutherAI/gpt-neo-125M --do_train --save_strategy epoch --num_train_epochs $EPOCHS --logging_steps 10000 --per_device_train_batch_size $BATCH_SIZE --cache_dir $SCRATCH/TS --dataset_name $DATASET --data_dir $DATA_DIR --dataloader_num_workers 2 --preprocessing_num_workers 2 --bf16 --weight_decay $DECAY
