export PROJECT_ID=focus-album-323718
export ZONE=us-central2-b
export TPU_NAME=tpu-pod-v4-64

echo  $PROJECT_ID
echo $ZONE
echo $TPU_NAME

# gcloud compute config-ssh # need to configure ssh first time

# upload files to all your pods (make sure all files are synced)
gcloud alpha compute tpus tpu-vm scp --recurse ~/CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}
# gcloud alpha compute tpus tpu-vm scp --recurse ~/CLIPA/  $TPU_NAME:~/  --zone=$ZONE --worker=all --project ${PROJECT_ID}

# gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo rm -r ~/CLIPA/clipa_jax/bv_venv"


#  prepare env will create an env installed all required softwares
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "cd ~/CLIPA/clipa_jax &&  bash scripts/tools/prepare_env.sh"

