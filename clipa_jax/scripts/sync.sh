export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-128-pod-vm-spot

echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME


# sync_name=/home/jyang347/CLIPA/clipa_jax/configs/convnext/convnext_pre_training.py
sync_name=/home/jyang347/CLIPA/clipa_jax/models/convnext.py
sync_dir=$(dirname $sync_name)


# upload files to all your pods (make sure all files are synced)
#gcloud alpha compute tpus tpu-vm scp --recurse ~/CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
#gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "mkdir /home/jyang347/checkpoints"
# gcloud alpha compute tpus tpu-vm scp  ~/CLIPA/clipa_jax/main.py  $TPU_NAME:~/CLIPA/clipa_jax/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
# gcloud alpha compute tpus tpu-vm scp  ~/CLIPA/clipa_jax/scripts/pre_training.sh  $TPU_NAME:~/CLIPA/clipa_jax/scripts/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
#gcloud alpha compute tpus tpu-vm scp  /home/jyang347/CLIPA/clipa_jax/configs/model_b/64_32_pre_training.py  $TPU_NAME:/home/jyang347/CLIPA/clipa_jax/configs/model_b/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
gcloud alpha compute tpus tpu-vm scp  $sync_name  $TPU_NAME:$sync_dir/  --zone=$ZONE --worker=all --project ${PROJECT_ID}



