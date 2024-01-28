#!/bin/bash

export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-64-pod-vm


TFDS_DATA_DIR="gs://jaxtpu-tfds-imagenet-eu-west4-a"
LAION_PATH="gs://jaxtpu-data-eu-west4/laion400m_blip_filtered"
# LAION_PATH="gs://jaxtpu-data-eu-west4/laion-400m-cv2resize-356m"
#WORK_DIR=/home/jyang347/checkpoints
WORK_DIR='gs://lxh_jaxtpu_eu_ckpt/jinruiyang_ckpt/clipa/tpu-v3-128-pod-vm-spot-flexivit-b16'
WANDB_log=b6b4e923d9e742d710ad470384368394b14a1df2 # only if you set wandb.log_wandb=True then you can revise the project name and experiment name


echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME

# clean the TPU cores
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "rm -r /tmp/tpu_logs/"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python3"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python3"

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "pkill -f python"


sleep 5


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "cd ~/CLIPA/clipa_jax/ &&  . bv_venv/bin/activate && wandb login $WANDB_log"



gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd ~/CLIPA/clipa_jax/ && \
. bv_venv/bin/activate && \
cd ~/CLIPA/clipa_jax && \
TFDS_DATA_DIR=${TFDS_DATA_DIR} bash scripts/tools/run_tpu.sh flexi_main \
--config=configs/model_flexi/flexi_pretraining.py  --config.wandb.log_wandb=False  --workdir=$WORK_DIR --config.input.data.data_dir=$LAION_PATH --config.evals.disclf.data_dir=$TFDS_DATA_DIR"
