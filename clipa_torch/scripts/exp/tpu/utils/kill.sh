export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-32-pod-vm-spot-1

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f python"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo lsof -w /dev/accel0"
# find processes on each worker that are using tpu, then kill it, like
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo kill -9 pids"