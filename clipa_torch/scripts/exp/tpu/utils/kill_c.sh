export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-32-pod-vm-spot-1

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "killall -9 python python3"