# Transformer-Based Atmospheric River Segmentation Model from Climate Reanalysis Data 🌍

This repository contains projects related to Atmospheric River (AR) segmentation based on deep learning models.

## Overview 🔍
We apply semantic segmentation models to detect and analyze Atmospheric Rivers (ARs) in climate reanalysis data, Climatenet. The repository includes two deep learning model:

- **MaskFormer**: Transformer-based segmentation model using mask classification.
- **ChannelViT**: Vision Transformer variant designed to handle multi-channel climate variables.

## Dataset preparation 📂

- Multi-channel climate data processing (e.g. TMQ, U850, V850, PSL)
- Patch-based training pipelines
- Experiment tracking for AR segmentation performance
- Code organized for reproducibility and further model extension

## Settings ⚙
Set model_name and make dataset/model_name directory
e.g. dataset/channnelvit

'''
mkdir -p dataset/channelvit 
'''
1. Download Climatenet dataset from https://gmd.copernicus.org/articles/14/107/2021/. Store them as following structure:
   dataset
   -> channelvit
     -> train        //303 images (≈ 66%)
     -> variance     //95 images ( ≈ 21%) 
     -> test         //61 images ( ≈ 13%)
* From original dataset, split train into train if the year is 2004 or before, else variance.
  
2. 
