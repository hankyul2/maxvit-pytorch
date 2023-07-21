## Tutorial

1. clone this repo
   
   ```bash
   git clone https://github.com/hankyul2/maxvit-pytorch
   cd maxvit-pytorch
   ```

2. run following command to train maxvit in imagenet-1k dataset.

   4 gpus (RTX 3090) needs about 5 days for training `maxvit_tiny_tf_224` model.
   
    ```bash
    torchrun --nproc_per_node=4 --master_port=12345 train.py /path/to/imagenet --model maxvit_tiny_tf_224 --aa rand-m15-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --remode pixel --reprob 0.25 --drop-path .2 --opt adamw --weight-decay .05 --sched cosine --epochs 300 --lr 3e-3 --warmup-lr 1e-6 --warmup-epoch 30 --min-lr 1e-5 -b 64 -tb 4096 --smoothing 0.1 --clip-grad 1.0 -j 8 --amp --pin-mem --channels-last 
    ```
   
3. run following command to valid maxvit in imagenet-1k dataset.
 
    It should give you Acc@1 83.820 (16.180) Acc@5 96.528 (3.472)

    ```bash 
    python3 valid.py /path/to/imagenet --img-size 224 --crop-pct 0.95 --cuda 0 --model maxvit_tiny_tf_224 --pretrained
    ```
