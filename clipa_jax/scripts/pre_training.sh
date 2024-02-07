#!/bin/bash

export PROJECT_ID=focus-album-323718
export ZONE=us-central2-b
export TPU_NAME=tpu-pod-v4-64


# TFDS_DATA_DIR="gs://jaxtpu-tfds-imagenet-eu-west4-a"
# LAION_PATH="gs://jaxtpu-data-eu-west4/dfn2b_noblur_tfrecord"
TFDS_DATA_DIR="gs://jaxtpu-data-us-central2/imagenet2012"
LAION_PATH="gs://jaxtpu-data-us-central2/dfn2b_noblur_tfrecord"
# LAION_PATH="gs://jaxtpu-data-eu-west4/laion-400m-cv2resize-356m"
#WORK_DIR=/home/jyang347/checkpoints
WORK_DIR='gs://lxh_jaxtpu_us_central2/jinruiyang_ckpt/clipa/tpu-pod-v4-64-vit-b16-align-dfn-onsubset-res64-align_orig_So150m'
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

# entry_name=flexi_main
# config_name=configs/model_flexi/dfn_unmask_tuning.py
entry_name=main
config_name=configs/model_b/dfn_64_32_pre_training.py



gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd ~/CLIPA/clipa_jax/ && \
. bv_venv/bin/activate && \
cd ~/CLIPA/clipa_jax && \
TFDS_DATA_DIR=${TFDS_DATA_DIR} bash scripts/tools/run_tpu.sh $entry_name \
--config=$config_name  --config.wandb.log_wandb=False  --workdir=$WORK_DIR --config.input.data.data_dir=$LAION_PATH --config.evals.disclf.data_dir=$TFDS_DATA_DIR"
