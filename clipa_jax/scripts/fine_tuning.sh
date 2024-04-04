#!/bin/bash

export PROJECT_ID=priors-medical-ai
export ZONE=us-east1-d
export TPU_NAME=tpu-v3-256-pod-vm


TFDS_DATA_DIR="gs://tfds-imagenet-us-east1"
LAION_PATH="gs://pmi-data-us-east1/datacomp1b/shards"
MASK_INIT='gs://lxh-jax-us-east1/jinruiyang_ckpt/clipa/v3-256-sovit400m14-datacomp1b-bs65k_84_8_gap_sin2d_8e-6_align_H14_2.56b_FLOPs/checkpoint.npz'
WORK_DIR='gs://lxh-jax-us-east1/jinruiyang_ckpt/clipa/v3-256-flexi_sovit400m14_datacomp1b_32k_240_32_gap_sin2d_H14_2.56b_FLOPs_ft_224_512m_mask'
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

#entry_name=main
entry_name=flexi_main
#config_name='configs/sovit/so400m_84_8_pre_training.py'
#config_name='configs/sovit/so400m_unmask_tuning_224_scheduleX4.py'
config_name='configs/sovit/so400m_unmask_tuning_336_scheduleX1.py'
#config_name='configs/sovit/so400m_unmask_tuning.py'
#config_name='configs/model_h/unmask_tuning_224_scheduleX4.py'
config_name='configs/model_h/unmask_tuning_336_scheduleX1.py'
config_name=configs/sovit/so400m_flexi_unmask_tuning_240_scheduleX4.py
#config_name='configs/model_l/unmask_tuning.py'
#config_name='configs/model_l/unmask_tuning_336_scheduleX1.py'


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all \
--command "cd ~/CLIPA/clipa_jax/ && \
. bv_venv/bin/activate && \
cd ~/CLIPA/clipa_jax && \
TFDS_DATA_DIR=${TFDS_DATA_DIR} bash scripts/tools/run_tpu.sh $entry_name \
--config=$config_name  --config.wandb.log_wandb=True  --config.masked_init=$MASK_INIT  --workdir=$WORK_DIR --config.input.data.data_dir=$LAION_PATH --config.evals.disclf.data_dir=$TFDS_DATA_DIR"
