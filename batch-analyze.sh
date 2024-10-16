#!/bin/bash

#SBATCH --account=
#SBATCH --qos=normal
#SBATCH --job-name=analyze
#SBATCH --output=logs/analyze/%x.%j.log
#SBATCH --err=logs/analyze/%x.%j.log
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=8G
#SBATCH --time=2:00:0
#SBATCH --export=ENV=GPU
##SBATCH --dependency=afterok:

. ./env-v

FILES=$(ls $SCRATCH/stats/*/*d*@* $SCRATCH/stats/*/*L@*)

OUTPUT=lc-2024_10_16 # useful for timestamping results
case $ENV in
ALL)
    analyze -loss -dual -etf -kern log -snr -decs -nor -o $OUTPUT -i $FILES -ts 512 -f
GPU)
    analyze -etf -kern log -snr -o $OUTPUT -i $FILES -ts 512 -f
    ;;
CKPT)
    analyze -dual -loss -o $OUTPUT -i $FILES -f
    ;;
CPU)
    analyze -decs -nor -o $OUTPUT -i $FILES -f
    ;;
esac
# done
