#!/bin/bash
#########################################################################
# Author: 
# Created Time: Mon 01 Apr 2019 09:15:32 PM CST
# File Name: load_data.sh
# Description: 
#########################################################################

CEPH_PATH="xx"
LOCAL_PATH="xx"

if (( $(find  $LOCAL_PATH -maxdepth 1 -mindepth 1 |wc -l)  >= 1 ));then
    echo "file exits"
    exit
fi

mkdir -p $LOCAL_PATH
cd $LOCAL_PATH
START_TIME=$(date +%s)
END_TIME=$(date +%s)
function GetUseTime()
{
        END_TIME=$(date +%s)
        (( USE_TIME=$END_TIME - $START_TIME ))
        START_TIME=$END_TIME
        echo $USE_TIME
        return $USE_TIME
}
cd $LOCAL_PATH
for part in $(find $CEPH_PATH -type f )
do
        echo $part
        cp $part $LOCAL_PATH &
done
echo "start to download file, waiting----------"
wait
USE_TIME=$(GetUseTime)
echo "down file time eslapse "$USE_TIME

cat finetune_det_data.tar.gz.* |  tar -Ipigz -x --no-same-owner

USE_TIME=$(GetUseTime)
echo "un tar file time eslapse "$USE_TIME
echo "file ready.."
rm finetune_det_data.gz.* -rf

cd -
