export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-32-pod-vm-spot-1


# gcloud compute config-ssh # need to configure ssh first time
# gcloud alpha compute tpus tpu-vm scp --recurse ~/CLIPA/  $TPU_NAME:~/ --zone=$ZONE --worker=all
#gcloud alpha compute tpus tpu-vm scp ~/CLIPA/clipa_torch/training/main.py  $TPU_NAME:~/CLIPA/clipa_torch/training/ --zone=$ZONE --worker=all
#gcloud alpha compute tpus tpu-vm scp ~/CLIPA/clipa_torch/launch_xla.py  $TPU_NAME:~/CLIPA/clipa_torch/ --zone=$ZONE --worker=all
gcloud alpha compute tpus tpu-vm scp ~/CLIPA/clipa_torch/training/train.py  $TPU_NAME:~/CLIPA/clipa_torch/training/ --zone=$ZONE --worker=all
#gcloud alpha compute tpus tpu-vm scp ~/CLIPA/clipa_torch/training/main_2.py  $TPU_NAME:~/CLIPA/clipa_torch/training/ --zone=$ZONE --worker=all
#gcloud alpha compute tpus tpu-vm scp /home/jyang347/CLIPA/clipa_torch/training/reader_tfds.py $TPU_NAME:/home/jyang347/CLIPA/clipa_torch/training/ --zone=$ZONE --worker=all
#gcloud alpha compute tpus tpu-vm scp --recurse  ~/xla/ $TPU_NAME:~/ --zone=$ZONE --worker=all
gcloud alpha compute tpus tpu-vm scp ~/CLIPA/clipa_torch/open_clip/loss.py $TPU_NAME:~/CLIPA/clipa_torch/open_clip/ --zone=$ZONE --worker=all
#pwd
#~/CLIPA/clipa_torch/scripts/exp/tpu/vit_b16/new_i50_t16_pretrain.sh
