# Dependence

1. pytorch 1.0
2. pip install tarfile
3. mmcv 

use docker : lincolnlin/train:2.0  要挂载cephfs/g_wxg_ob_dc 到/data/user下

nvidia-docker run --net=host --rm -it --ipc=host --name=mmcv_train -v /data1/:/data/user/data1/ -v /mnt/yardcephfs/mmyard/g_wxg_ob_dc/:/data/user/cephfs/ docker.yard.oa.com:14917/lincolnlin/train:2.0

# Usage

## 目录介绍

config 配置文件目录
data  转换后的训练数据目录

## 修改配置

## 运行

sh run.sh [config]   例如  sh run.sh resnet50
