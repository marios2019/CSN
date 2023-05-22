#!/usr/bin/env bash
function gdrive_download () {
  CONFIRM=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://docs.google.com/uc?export=download&id=$1" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
  wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$CONFIRM&id=$1" -O $2
  rm -rf /tmp/cookies.txt
}

declare -a MODELS_NAME=("HRNetSeg3S_models.zip" "HRNetSimCSN3S_SSA_models.zip" "HRNetSimCSN3S_K1_models.zip"
                        "HRNetSimCSN3S_K2_models.zip" "HRNetSimCSN3S_K3_models.zip")
declare -a MODELS_ID=("1WIOii5OzrzYfyg2mX40cQZjYOvaOdnWE" "1MxD-7Gra09CCcGo59b6ogmjEy3ML4Kt9"
                      "1TrlFsdUfqWcw-135hgLJMLbsoS1DULBQ" "1sTSGVlStY5Zx5iEyK8_NDA1hyzWxsFjW"
                      "1YHh_qFSFJCWZliLbcGoEwlPGzSwIPmqW")

for i in "${!MODELS_NAME[@]}"
do
    gdrive_download ${MODELS_ID[$i]} ${MODELS_NAME[$i]}
    unzip ${MODELS_NAME[$i]}
    rm ${MODELS_NAME[$i]}
done
