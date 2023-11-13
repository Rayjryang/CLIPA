export PROJECT_ID=focus-album-323718
export ZONE=europe-west4-a
export TPU_NAME=tpu-v3-vm-5
# export NUM_TPU_CORES=1
# os.environ[''] = '8'
# export XLA_NO_SPECIAL_SCALARS=1
# export TPU_PROCESS_BOUNDS=1,1,1
# export  TPU_VISIBLE_CHIPS=0

    # --to-float-on-device \
PJRT_DEVICE=TPU  python3  ~/CLIPA/clipa_torch/launch_xla.py --num-devices 8 training.main \
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
    --batch-size 512 \
    --aug-cfg scale='(0.4, 1.0)' \
    --pos-embed 'sin_cos_2d' \
    --epochs=6 \
    --workers=6 \
    --model ViT-B-16-CL16 \
    --precision 'fp32' \
    --local-loss \
    --gather-with-grad \
    --to-float-on-device \
    --force-image-size 112 \
    --grad-checkpointing \
    --log-every-n-steps 1 --zeroshot-steps 1526 --val-steps 1526 \
    --seed 0 \
    --logs ./logs/ \
    --name ./debug_xla_v3_32_13/ \
    --imagenet-val "gs://jaxtpu-tfds-imagenet-eu-west4-a/imagenet2012"
