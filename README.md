# DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification

Created by Yongming Rao, Wenliang Zhao, Benlin Liu, Jiwen Lu, Jie Zhou, Cho-Jui Hsieh

This repository contains PyTorch implementation for DynamicViT.

We introduce a dynamic token sparsification framework to prune redundant tokens in vision transformers progressively and dynamically based on the input:

![intro](figs/intro.gif)

Our code is based on [pytorch-image-models](https://github.com/rwightman/pytorch-image-models), [DeiT](https://github.com/facebookresearch/deit) and [LV-ViT](https://github.com/zihangJiang/TokenLabeling)

[[Project]](https://dynamicvit.ivg-research.xyz/)

## Usage

### Requirements

- torch>=1.7.0
- torchvision>=0.8.1
- timm==0.4.5

Data preparation: Download and extract ImageNet images from http://image-net.org/. The directory structure should be

```
│ILSVRC2012/
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

Model preparation: Download pre-trained DeiT and LV-ViT models:
```
sh download_pretrain.sh
```

### Training & evaluation

To train DynamicViT models on ImageNet, run:

DeiT-small
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_dynamic_vit.py  --output_dir logs/dynamic-vit_deit-small --arch deit_small --input-size 224 --batch-size 96 --data-path /path/to/ILSVRC2012/ --epochs 30 --dist-eval --distill  --base_rate 0.7
```

LV-ViT-S
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_dynamic_vit.py  --output_dir logs/dynamic-vit_lvvit-s --arch lvvit_s --input-size 224 --batch-size 64 --data-path /path/to/ILSVRC2012/ --epochs 30 --dist-eval --distill  --base_rate 0.7
```

LV-ViT-M
```
python -m torch.distributed.launch --nproc_per_node=8 --use_env main_dynamic_vit.py  --output_dir logs/dynamic-vit_lvvit-m --arch lvvit_m --input-size 224 --batch-size 48 --data-path /path/to/ILSVRC2012/ --epochs 30 --dist-eval --distill  --base_rate 0.7
```

## License
MIT License
