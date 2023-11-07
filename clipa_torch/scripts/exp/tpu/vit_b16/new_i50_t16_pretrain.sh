export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-32-pod-vm-spot-1


gcloud alpha compute tpus tpu-vm ssh $TPU_NAME  --project=$PROJECT_ID --zone=$ZONE --worker=all --command=" \
PJRT_DEVICE=TPU NUMBA_NUM_THREADS=1 python3 -W ignore  ~/CLIPA/clipa_torch/launch_xla.py --num-devices 8 training.main \
    --save-frequency 1 \
    --save-most-recent \
    --zeroshot-frequency 1 \
    --train-data "gs://jaxtpu-data-eu-west4/laion-400m-cv2resize-356m" \
    --dataset-type tfrecord \
    --lr "2.048e-3" \
    --beta1 0.9 \
    --beta2 0.95 \
    --warmup 782 \
    --wd 0.2 \
    --batch-size 1024 \
    --aug-cfg scale='(0.4, 1.0)' \
    --pos-embed 'sin_cos_2d' \
    --epochs=6 \
    --workers=6 \
    --model ViT-B-16-CL16 \
    --precision 'fp32' \
    --local-loss \
    --gather-with-grad \
    --force-image-size 112 \
    --to-float-on-device \
    --grad-checkpointing \
    --log-every-n-steps 32 --zeroshot-steps 1526 --val-steps 1526 \
    --seed 0 \
    --logs ./logs/ \
    --name ./debug_xla_v3_32_11/ \
    --imagenet-val "gs://jaxtpu-tfds-imagenet-eu-west4-a/imagenet2012""
