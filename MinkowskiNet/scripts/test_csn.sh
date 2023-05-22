#!/usr/bin/env bash

set -x
# Exit script when a command returns nonzero state
set -e

set -o pipefail

PYTHONUNBUFFERED="True"
export OMP_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=$1
CATEGORY=$2
EXPERIMENT=$3
INPUT_FEAT=$4
BATCH_SIZE=$5
NUM_NEIGHBORS=$6
MAX_EPOCH=$7
TIME=$(date +"%Y-%m-%d_%H-%M-%S")

DATASET=${DATASET:-PartnetVoxelization0_05Dataset}
MODEL=${MODEL:-HRNetSimCSN3S}
OPTIMIZER=${OPTIMIZER:-SGD}
LR=${LR:-0.5e-1}
SCHEDULER=${SCHEDULER:-ReduceLROnPlateau}

BASE_DIR=./outputs/$DATASET/$MODEL/$CATEGORY-$OPTIMIZER-l$LR-b$BATCH_SIZE-k$NUM_NEIGHBORS-$SCHEDULER-e$MAX_EPOCH-$EXPERIMENT-$INPUT_FEAT
OUTPATH=$BASE_DIR/${TIME}_evaluation
EXP_DIR=($(ls $BASE_DIR | sort))
WEIGHTS=$BASE_DIR/$EXP_DIR/checkpoint_${MODEL}.pth

VERSION=$(git rev-parse HEAD)

# Save the experiment detail and dir to the common log file
mkdir -p $OUTPATH

LOG="$OUTPATH/$TIME.txt"
SAVE_PRED_DIR="$OUTPATH/results"

# put the arguments on the first line for easy resume
echo -e "
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --partnet_category $CATEGORY \
    --model $MODEL \
    --k_neighbors $NUM_NEIGHBORS \
    --lr $LR \
    --optimizer $OPTIMIZER \
    --scheduler $SCHEDULER \
    --input_feat $INPUT_FEAT" >> $LOG
echo Logging output to "$LOG"
echo $(pwd) >> $LOG
echo "Version: " $VERSION >> $LOG
echo "Git diff" >> $LOG
echo "" >> $LOG
git diff | tee -a $LOG
echo "" >> $LOG
echo -e "-------------------------------System Information----------------------------" >> $LOG
echo -e "Hostname:\t\t"`hostname` >> $LOG
echo -e "GPU(s):\t\t$CUDA_VISIBLE_DEVICES" >> $LOG
nvidia-smi | tee -a $LOG

time python -W ignore tasks/main_csn.py \
    --log_dir $OUTPATH \
    --dataset $DATASET \
    --partnet_category $CATEGORY \
    --model $MODEL \
    --k_neighbors $NUM_NEIGHBORS \
    --save_pred_dir $SAVE_PRED_DIR \
    --input_feat $INPUT_FEAT \
    --weights $WEIGHTS \
    $8 2>&1 | tee -a "$LOG"
