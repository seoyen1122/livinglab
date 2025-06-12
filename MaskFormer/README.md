# livinglab

# AR Segmentation with MaskFormer

This repository contains the implementation of Atmospheric River (AR) segmentation using MaskFormer with Swin Transformer backbone on the ClimateNet dataset.

The original MaskFormer repo was customized and simplified for this task.

## Setup

1. Clone the repo.
2. Install dependencies.
3. Download pretrained swin weights and place under pretrained/.

Pretrained Swin: https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_base_patch4_window7_224_22k.pth

Training:

python train_net.py --config-file configs/climatenet/semantic/maskformer_Swin_p16_bs8_w10.yaml --num-gpus 1
