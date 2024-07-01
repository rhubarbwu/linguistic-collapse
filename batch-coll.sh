DECAY=${DECAY:-n}
DTYPE=${DTYPE:-float32}

IDX=$1
STG=$2

case $IDX in
0) EPOCHS="01" ;;
2) EPOCHS="03" ;;
5) EPOCHS="10" ;;
9) EPOCHS="10" ;;
*) exit ;;
esac

launch() {
    CONFIG=TinyStories-$1x$2"_"$EPOCHS$DECAY$SEED
    CONFIG=${CONFIG//EPOCHS/$EPOCHS}
    case $STG in
    means | vars)
        HOURS=$3
        ;;
    decs)
        HOURS=1
        ;;
    *)
        echo "invalid stage" $STD
        return
        ;;
    esac
    DEVICE=$4
    DEPEND=$5
    JOB_NAME=$1"${STG:0:1}"$2$DECAY$IDX

    if [ -n "$DEPEND" ]; then
        sbatch --gres=gpu:$DEVICE:1 --time=$HOURS:00:00 --export=CONFIG=$CONFIG,DTYPE=$DTYPE,CKPT_IDX=$IDX,STAGE=$STG -J $JOB_NAME --dependency=afterok:$DEPEND coll-clm.sh
    else
        sbatch --gres=gpu:$DEVICE:1 --time=$HOURS:00:00 --export=CONFIG=$CONFIG,DTYPE=$DTYPE,CKPT_IDX=$IDX,STAGE=$STG -J $JOB_NAME coll-clm.sh
    fi
}

exit # breakpoint

launch 01 0064 6 a100 # add means job ID as dependency for vars/decs
launch 01 0128 6 a100
launch 01 0256 6 a100
launch 01 0512 6 a100
launch 01 0768 6 a100
launch 01 1024 6 a100
launch 02 0064 6 a100
launch 02 0128 6 a100
launch 02 0256 6 a100
launch 02 0512 6 a100
launch 02 0768 6 a100
launch 02 1024 6 a100
launch 04 0064 6 a100
launch 04 0128 6 a100
launch 04 0256 6 a100
launch 04 0512 6 a100
launch 04 0768 6 a100
launch 04 1024 6 a100
launch 08 0064 6 a100
launch 08 0128 6 a100
launch 08 0256 6 a100
launch 08 0512 6 a100
launch 08 0768 7 a100
launch 08 1024 8 a100
launch 12 0064 6 a100
launch 12 0128 6 a100
launch 12 0256 6 a100
launch 12 0512 6 a100
launch 12 0768 7 a100
launch 12 1024 8 a100
