#!/usr/bin/env bash

declare -a CATEGORIES=("Bed" "Bottle" "Chair" "Clock" "Dishwasher" "Display" "Door" "Earphone" "Faucet" "Knife"
                       "Lamp" "Microwave" "Refrigerator" "StorageFurniture" "Table" "TrashCan" "Vase")

SHOW_CATS="--show_categories"

if [ -z "$1" ]
then
    echo "You need to specify a PartNet category, e.g., './scripts/testing_hrnet.sh Bed' \
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

PARTNET_PATH="./Dataset/PartNet/sem_seg_h5"
CAT=$1
BATCH_SIZE=8
LEVEL=3
INPUT_FEAT="xyz"
LOSS="cross_entropy"
MAX_EPOCH=200
GPU=0

FOUND=false
for i in "${!CATEGORIES[@]}"
do
    if [ $CAT = ${CATEGORIES[$i]} ] || [ $CAT = "all" ]
    then
        ./scripts/test_hrnet.sh $GPU ${CATEGORIES[$i]}-$LEVEL $LOSS $INPUT_FEAT $BATCH_SIZE $MAX_EPOCH "--partnet_path $PARTNET_PATH \
        --is_train False --load_h5 True --normalize_coords True --normalize_method sphere --avg_feat True --opt_speed True"
       FOUND=true
    fi
done

