# Caterpillar

This is a Pytorch implementation for the paper "Caterpillar: A Pure-MLP Architecture with Shifted-Pillars-Concatenation"

![image](https://github.com/sunjin19126/Caterpillar/blob/main/Img/Cpr%2BBlock%2BSPC.png)

Caterpillar on Small-Scale Image Classification
| Networks | MIN | C10 | C100 | Fashion | Params | FLOPs |
| :-- |:--:|:--:|:--:|:--:|:--:|:--:|
| Caterpillar-Mi | 74.14 | 95.54 | 79.41 | 95.14 | 5.9M  | 0.4G |
| Caterpillar-Tx | 77.27 | 96.54 | 82.69 | 95.38 | 16.0M | 1.1G |
| Caterpillar-T  | 78.16 | 97.10 | 83.86 | 95.72 | 28.4M | 1.9G |
| Caterpillar-S  | 78.94 | 97.22 | 84.40 | 95.80 | 58.0M | 4.1G |
| Caterpillar-B  | 79.06 | 97.35 | 84.77 | 95.85 | 78.8M | 5.5G |

Caterpillar on ImageNet-1k Classification
| Networks | Params | FLOPs | Top-1 Acc. | Log | Ckpt |
| :-- |:--:|:--:|:--:|:--:|:--:|
| Caterpillar-Mi | 6M  | 1.2G  | 76.3 | 95.14 | 5.9M  |
| Caterpillar-Tx | 16M | 3.4G  | 80.9 | 95.38 | 16.0M |
| Caterpillar-T  | 29M | 6.0G  | 82.4 | 95.72 | 28.4M |
| Caterpillar-S  | 60M | 12.5G | 83.5 | 95.80 | 58.0M |
| Caterpillar-B  | 80M | 17.0G | 83.7 | 95.85 | 78.8M |

ResNet(/SPC) on Small-Scale Image Classification
| Networks | Nc | MIN | C10 | C100 | Fashion | Params | FLOPs |
| :-- |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
| Res-18       | 64  | 70.95 | 95.54 | 77.66 | 95.11 | 11.2M | 0.7G |
| Res-18(SPC)  | 64  | 70.10 | 94.52 | 76.19 | 94.90 | 2.6M  | 0.2G |
| Res-18(SPC)  | 96  | 71.88 | 95.72 | 78.35 | 95.33 | 5.7M  | 0.4G |
| Res-18(SPC)  | 128 | 73.24 | 95.84 | 79.77 | 95.54 | 10.2M | 0.8G |

ResNet(/SPC) on ImageNet-1k Classification
| Networks | Nc | Params | FLOPs | Top-1 Acc. | Log | Ckpt |
| :-- |:--:|:--:|:--:|:--:|:--:|:--:|
| Res-18       | 64  | 12M | 1.8G | 70.6 | 5.9M  | 0.4G |
| Res-18(SPC)  | 64  | 3M  | 0.6G | 69.1 | 28.4M | 1.9G |
| Res-18(SPC)  | 96  | 7M  | 1.3G | 73.6 | 58.0M | 4.1G |
| Res-18(SPC)  | 128 | 11M | 2.2G | 75.3 | 78.8M | 5.5G |



### Code overview

The proposed model of Caterpillar is in `caterpillar.py`. 

The SPC-based model of ResNet(SPC) is in `resnet_spc.py`.

The comparison models are in `models4Comparison`.

The ablation models are in `models4Ablation`.

We trained all models using the `timm` framework, which we copied from [here](https://github.com/huggingface/pytorch-image-models). Inside `pytorch-image-models-main`, we have made the following modifications.
+ Added Caterpillars
  + added `timm/models/caterpillar.py`
  + modified `timm/models/__init__.py`

### Data Preparation
Download and extract datasets with train and val images from:

[Mini-ImageNet (84×84)](https://drive.google.com/file/d/1xDhH7WJzZBdjzxCfc0hT0p8cVkXLGK5l/view?usp=share_link)

[CIFAR-10](https://drive.google.com/file/d/1KVnDI3UUcMFFYBPISQU84T89s5W1SPLH/view?usp=share_link)

[CIFAR-100](https://drive.google.com/file/d/1ajh7cM7mZz8shLzy0PnkxLzO4Osv6m0S/view?usp=share_link)

[Fashion-MNIST](https://drive.google.com/file/d/1AXWFH6FYaFbrtovb2kM30n4dv3l17hU1/view?usp=share_link)

[ImageNet-1k](https://image-net.org/)

The directory structure should be: 
```
│path/to/dataset/
├──train/
│  ├── n01440764
│  │   ├── n01440764_10026.JPEG
│  │   ├── n01440764_10027.JPEG
│  │   ├── ......
│  ├── ......
├──val/
│  ├── n01440764
│  │   ├── ILSVRC2012_val_00000293.JPEG
│  │   ├── ILSVRC2012_val_00002138.JPEG
│  │   ├── ......
│  ├── ......
```   


### Training

To train Caterpillar-T on Mini-ImageNet on a single node with 2 gpus:
```
sh distributed_train.sh 2  /path/to/mini-imagenet  --train-split train  --val-split val  --model Caterpillar_T_MIN  --num-classes 100  --input-size 3 84 84  --epochs 300  -b 512  --opt adamw  --lr 1e-3  --sched cosine  --min-lr 1e-5  --weight-decay 0.05  --warmup-epochs 20  --warmup-lr 1e-6  --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0  --drop-path 0  --aug-repeats 3  --reprob 0.25  --smoothing 0.1  -j 8  --amp-impl native --amp  --seed 42
```

To train Caterpillar-T on CIFAR-10 on a single node with 2 gpus:
```
sh distributed_train.sh 2  /path/to/cifar-10  --train-split train  --val-split val  --model Caterpillar_T_C10  --num-classes 10  --input-size 3 32 32  --epochs 300  -b 512  --opt adamw  --lr 1e-3  --sched cosine  --min-lr 1e-5  --weight-decay 0.05  --warmup-epochs 20  --warmup-lr 1e-6  --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0  --drop-path 0  --aug-repeats 3  --reprob 0.25  --smoothing 0.1  -j 8  --amp-impl native --amp  --seed 42  
```

To train Caterpillar-T on CIFAR-100 on a single node with 2 gpus:
```
sh distributed_train.sh 2  /path/to/cifar-100  --train-split train  --val-split val  --model Caterpillar_T_C100  --num-classes 100  --input-size 3 32 32  --epochs 300  -b 512  --opt adamw  --lr 1e-3  --sched cosine  --min-lr 1e-5  --weight-decay 0.05  --warmup-epochs 20  --warmup-lr 1e-6  --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0  --drop-path 0  --aug-repeats 3  --reprob 0.25  --smoothing 0.1  -j 8  --amp-impl native --amp  --seed 42  
```

To train Caterpillar-T on Fashion-MNIST on a single node with 2 gpus:
```
sh distributed_train.sh 2  /path/to/fashion-mnist  --train-split train  --val-split val  --model Caterpillar_T_FM  --num-classes 10  --input-size 3 28 28  --epochs 300  -b 512  --opt adamw  --lr 1e-3  --sched cosine  --min-lr 1e-5  --weight-decay 0.05  --warmup-epochs 20  --warmup-lr 1e-6  --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0  --drop-path 0  --aug-repeats 3  --reprob 0.25  --smoothing 0.1  -j 8  --amp-impl native --amp  --seed 42  
```

To train Caterpillar-T on ImageNet-1k on a single node with 8 gpus:
```
sh distributed_train.sh 8  /path/to/imagenet  --train-split train  --val-split val  --model  Caterpillar_T_IN1k  --num-classes 1000  --input-size 3 224 224  --epochs 300  -b 128  --opt adamw  --lr 1e-3  --sched cosine  --min-lr 1e-5  --weight-decay 0.05  --warmup-epochs 20  --warmup-lr 1e-6  --aa rand-m9-mstd0.5-inc1  --mixup 0.8 --cutmix 1.0  --drop-path 0  --aug-repeats 3  --reprob 0.25  --smoothing 0.1  -j 8  --amp-impl native --amp  --seed 42  --model-ema  --model-ema-decay 0.99996
```

### License
Caterpillar is released under MIT License.


