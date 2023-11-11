export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-vm-5
WANDB_log=b6b4e923d9e742d710ad470384368394b14a1df2 # only if you set wandb.log_wandb=True then you can revise the project name and experiment name

cd ~/CLIPA/clipa_torch && pip3 install -r requirements.txt
python3 -m wandb login $WANDB_log && python3 -m wandb online


#gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command="pip install --upgrade transformers==4.28"
## prepara env && login wandb

# gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command \
# " cd ~/CLIPA/clipa_torch && pip3 install -r requirements.txt"

# gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command="pip3 install -U click"

# gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command \
# "python3 -m wandb login $WANDB_log && python3 -m wandb online"
