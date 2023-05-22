#!/usr/bin/env bash

declare -a CATEGORIES=("Bed" "Bottle" "Chair" "Clock" "Dishwasher" "Display" "Door" "Earphone" "Faucet" "Knife"
                       "Lamp" "Microwave" "Refrigerator" "StorageFurniture" "Table" "TrashCan" "Vase")
declare -a TRAIN_NUM=(133 315 4489 406 111 633 149 147 435 221 1554 133 136 1588 5707 221 741)

SHOW_CATS="--show_categories"

if [ -z "$1" ]
then
    echo "You need to specify a PartNet category, e.g., './scripts/training_csn.sh Bed' \
(use the '$SHOW_CATS' argument to list all available categories)"
    exit
fi

if [ $1 = $SHOW_CATS ]
then
    echo "PartNet categories with L3 annotations:"
    echo "---------------------------------------"
    for i in "${!CATEGORIES[@]}"
    do
        echo -e "\t$(echo $i+1 | bc).\t"${CATEGORIES[$i]}
    done
    exit
fi

if [ -z "$2" ]
then
    echo "You need to specify the number of neighbors, e.g., './scripts/training_csn.sh Bed 0' (for SSA only or 1 2 3 for CSA)"
    exit
fi

PARTNET_PATH="./Dataset/PartNet/sem_seg_h5"
CAT=$1
LEVEL=3
BATCH_SIZE=8
ITER_SIZE=1
INPUT_FEAT="xyz"
LOSS="cross_entropy"
NUM_NEIGHBORS=$2
MAX_EPOCH=200
GPU=0

for i in "${!CATEGORIES[@]}"
do
  if [ $CAT = ${CATEGORIES[$i]} ]
      then
          NUMERATOR=$(echo ${TRAIN_NUM[$i]}+$BATCH_SIZE-1 | bc)
          N_BATCHES=$(echo $NUMERATOR/$BATCH_SIZE | bc)
          STAT_FREQ=$(echo $N_BATCHES/4 | bc)
        ./scripts/train_csn.sh $GPU ${CATEGORIES[$i]}-$LEVEL $LOSS $INPUT_FEAT $BATCH_SIZE $NUM_NEIGHBORS $MAX_EPOCH \
        "--partnet_path $PARTNET_PATH --load_h5 True --normalize_coords True --normalize_method sphere --distort_partnet True
        --avg_feat True --opt_speed True --return_neighbors True --stat_freq $STAT_FREQ --save_param_histogram True
        --iter_size $ITER_SIZE --num_workers 0"
        exit
    fi
done

echo "ERROR: '$CAT' is not a PartNet category with L3 annotations"

