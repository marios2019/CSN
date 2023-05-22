#!/usr/bin/env bash

declare -a CATEGORIES=("Bed" "Bottle" "Chair" "Clock" "Dishwasher" "Display" "Door" "Earphone" "Faucet" "Knife"
		       "Lamp" "Microwave" "Refrigerator" "StorageFurniture" "Table" "TrashCan" "Vase")

SHOW_CATS="--show_categories"

if [ -z "$1" ]
then
    echo "You need to specify a PartNet category, e.g., './scripts/testing_csn.sh Bed' \
(use the '$SHOW_CATS' argument to list all available categories). You call also use the 'all' option to evaluate all categories."
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
    echo "You need to specify the number of neighbors, e.g., './scripts/testing_csn.sh Bed 0' (for SSA only or 1 2 3 for CSA)"
    exit
fi

PARTNET_PATH="./Dataset/PartNet/sem_seg_h5"
CAT=$1
LEVEL=3
BATCH_SIZE=8
INPUT_FEAT="xyz"
LOSS="cross_entropy"
NUM_NEIGHBORS=$2
MAX_EPOCH=200
GPU=0

FOUND=false
for i in "${!CATEGORIES[@]}"
do
    if [ $CAT = ${CATEGORIES[$i]} ] || [ $CAT = "all" ]
    then
        ./scripts/test_csn.sh $GPU ${CATEGORIES[$i]}-$LEVEL $MODEL $LOSS $INPUT_FEAT $BATCH_SIZE $NUM_NEIGHBORS $MAX_EPOCH \
        "--partnet_path $PARTNET_PATH --is_train False --load_h5 True --normalize_coords True --normalize_method sphere
        --distort_partnet True --avg_feat True --opt_speed True --return_neighbors True"
        FOUND=true
    fi
done

if [ $FOUND = false ]
then
    echo "ERROR: '$CAT' is not a PartNet category with L3 annotations"
fi
