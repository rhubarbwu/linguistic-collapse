#!/bin/sh

. ./env-h

EPOCHS=$1 # number of total epochs to train for; not the same as checkpoint index
EXPR=$2 # custom identifier for level of weight decay: n=0, d=5e-4, L=0.1
DECAY=${DECAY:-0}

launch() {
    L=$1
    d=$2
    HOURS_PER_EPOCH=$3
    BATCH_SIZE=$4
    RUN_ID=$5
    MODEL_NAME=TinyStories-$L"x"$d"_"$EPOCHS$EXPR$RUN_ID
    HOURS=$((EPOCHS * HOURS_PER_EPOCH))

    echo $MODEL_NAME $HOURS
    return # breakpoint 

    sbatch --time=$HOURS:00:00 --export=DECAY=$DECAY,EXPR=$EXPR,MODEL_NAME=$MODEL_NAME,BATCH_SIZE=$BATCH_SIZE,EPOCHS=$EPOCHS -J $MODEL_NAME train.sh
}

exit # breakpoint

case $3 in
01)
    launch 01 0064 2 16
    launch 01 0128 2 16
    launch 01 0256 2 16
    launch 01 0512 2 16
    launch 01 0768 2 16
    launch 01 1024 2 16
    ;;
02)
    launch 02 0064 2 16
    launch 02 0128 2 16
    launch 02 0256 2 16
    launch 02 0512 2 16
    launch 02 0768 2 16
    launch 02 1024 3 8
    ;;
04)
    launch 04 0064 3 8
    launch 04 0128 3 8
    launch 04 0256 3 8
    launch 04 0512 3 8
    launch 04 0768 3 8
    launch 04 1024 4 8
    ;;
08)
    launch 08 0064 4 8
    launch 08 0128 4 8
    launch 08 0256 4 8
    launch 08 0512 4 4
    launch 08 0768 5 4
    launch 08 1024 6 4
    ;;
12)
    launch 12 0064 5 4
    launch 12 0128 5 4
    launch 12 0256 6 4
    launch 12 0512 6 4
    launch 12 0768 7 4
    launch 12 1024 8 4
    ;;
esac
