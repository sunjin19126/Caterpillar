# Using Caterpillar to Nibble Small-Scale Images
This is a Pytorch implementation for the paper "Using Caterpillar to Nibble Small-Scale Images"

### Code overview

The proposed model of Caterpillar is in `caterpillar.py`. 

The comparison models are in `models4Comparison`.

We trained all models using the `timm` framework, which we copied from [here](https://github.com/huggingface/pytorch-image-models). Inside `pytorch-image-models`, we have made the following modifications.
+ Added Caterpillars
  + added `timm/models/caterpillar.py`
  + modified `timm/models/__init__.py`

### Data Preparation
Download and extract datasets with train and val images from:

[Mini-ImageNet](https://drive.google.com/file/d/1xDhH7WJzZBdjzxCfc0hT0p8cVkXLGK5l/view?usp=share_link)

[CIFAR-10](https://drive.google.com/file/d/1KVnDI3UUcMFFYBPISQU84T89s5W1SPLH/view?usp=share_link)

[CIFAR-100](https://drive.google.com/file/d/1ajh7cM7mZz8shLzy0PnkxLzO4Osv6m0S/view?usp=share_link)

[Fashion-MNIST](https://drive.google.com/file/d/1AXWFH6FYaFbrtovb2kM30n4dv3l17hU1/view?usp=share_link)

[ImageNet-1k](https://image-net.org/)

The directory structure is 
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


