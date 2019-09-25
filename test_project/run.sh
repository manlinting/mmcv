if [[ $(pwd | grep /data/user/data1/) ]];then
    echo "run on local"
else
    echo "run on ecphfs"
    working_path=$(cd $(dirname $0); pwd)
    cd $working_path
fi

ulimit -c unlimited
export PYTHONUNBUFFERED="True"

if ((  $# < 1 )); then
    echo "Usage: sh run.sh instance [checkpoint_dir]"
    exit
fi

INSTANCE=$1
CHECKPOINT=""

if (( $# > 1 )) ;then
    CHECKPOINT=$2
    echo "resume in"$CHECKPOINT
    mkdir -p $CHECKPOINT
else
    postfix=$(date +%Y%m%d_%H%M)
    CHECKPOINT=stage_${INSTANCE}_$postfix
    mkdir -p $CHECKPOINT
fi

cp train_mmcv.py config/$INSTANCE.yaml $CHECKPOINT

#load data ceph
sh load_data.sh

# set gpu
gpus=$(cat config/$INSTANCE.yaml|  shyaml get-value gpus  |  awk 'BEGIN{ORS=","}{print $2}' | awk '{print $0}')
export CUDA_VISIBLE_DEVICES=$gpus

python train_mmcv.py --config config/$INSTANCE.yaml --checkpoint $CHECKPOINT
