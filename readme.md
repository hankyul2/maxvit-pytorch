# MaxViT (PyTorch version)

This repo contains the unofficial PyTorch-version MaxViT model, training, and validation codes. This repo is written to share the PyTorch-version training hyper-parameters of MaxViT. For this, we just copy-and-paste the training hyper-parameters shown in [table 12 of the original paper](https://arxiv.org/pdf/2204.01697.pdf) with the modification of the number GPUs (we use 4 GPUs). Since most codes including model, train, and valid are copy-pasted from [Timm github](https://github.com/huggingface/pytorch-image-models), the credits should be given to [@rwightman](https://github.com/rwightman) and the original authors. See also their repos:

- [tensorflow-version maxvit by authors](https://github.com/google-research/maxvit).
- [pytorch-version maxvit by timm](https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/maxxvit.py).



## Tutorial

*Test environments: `torch==1.11.0` & `timm==0.9.2`*

1. Clone this repo
   
   ```bash
   git clone https://github.com/hankyul2/maxvit-pytorch
   cd maxvit-pytorch
   ```

2. Run the following command to train MaxViT-T in imagenet-1k dataset. For model variants, just change the `--drop-path` to `0.3 (small)` and `0.4 (base)`. 

    Training time: about 5 days for the `maxvit_tiny_tf_224` model with 4 GPUs (RTX 3090, 24GB).
    
    ```bash
   torchrun --nproc_per_node=4 --master_port=12345 train.py /path/to/imagenet --model maxvit_tiny_tf_224 --aa rand-m15-mstd0.5-inc1 --mixup .8 --cutmix 1.0 --remode pixel --reprob 0.25 --drop-path .2 --opt adamw --weight-decay .05 --sched cosine --epochs 300 --lr 3e-3 --warmup-lr 1e-6 --warmup-epoch 30 --min-lr 1e-5 -b 64 -tb 4096 --smoothing 0.1 --clip-grad 1.0 -j 8 --amp --pin-mem --channels-last 
   ```
   
3. Run the following command to reproduce the validation results of MaxViT-T in the imagenet-1k dataset.

    Results: ** Acc@1 83.820 (16.180) Acc@5 96.528 (3.472)*
    
    ```bash 
    python3 valid.py /path/to/imagenet --img-size 224 --crop-pct 0.95 --cuda 0 --model maxvit_tiny_tf_224 --pretrained
    ```
    
    

## Experiment result

| Model                | Image size | #Param | FLOPs | Top1  | Artifacts                                                    |
| -------------------- | ---------- | ------ | ----- | ----- | ------------------------------------------------------------ |
| MaxViT-T (paper)     | 224        | 31M    | 5.6G  | 83.62 |                                                              |
| MaxViT-T (this repo) | 224        | 31M    | 5.6G  | 83.82 | [[args.yaml]](https://github.com/hankyul2/maxvit-pytorch/releases/download/v0.0.1/maxvit-tiny-tf-224.yaml), [[ckpt.pth.tar]](https://github.com/hankyul2/maxvit-pytorch/releases/download/v0.0.1/maxvit-tiny-tf-224.pth.tar), [[train.log]](https://github.com/hankyul2/maxvit-pytorch/releases/download/v0.0.1/maxvit-tiny-tf-224.log), [[metric.csv]](https://github.com/hankyul2/maxvit-pytorch/releases/download/v0.0.1/maxvit-tiny-tf-224.csv) |

<img src="https://github.com/hankyul2/maxvit-pytorch/assets/31476895/323d3ff9-b602-47ef-b1fb-75469335bba7" width="800" height="616">

## References

```
@inproceedings{tu2022maxvit,
  title={Maxvit: Multi-axis vision transformer},
  author={Tu, Zhengzhong and Talebi, Hossein and Zhang, Han and Yang, Feng and Milanfar, Peyman and Bovik, Alan and Li, Yinxiao},
  booktitle={European conference on computer vision},
  pages={459--479},
  year={2022},
  organization={Springer}
}
```

