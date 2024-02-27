export PROJECT_ID=focus-album-323718
export ZONE=us-central2-b
export TPU_NAME=tpu-pod-v4-64

gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "killall -9 python python3"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f python"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo pkill -f python3"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo lsof -t /dev/accel0 | xargs -r -I {} sudo kill -9 {}"
gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo rm -rf /tmp/libtpu_lockfile /tmp/tpu_logs"
#gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo lsof -w /dev/accel0"
#gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command "sudo kill -9 pids"
