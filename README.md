# MPFC-DPC Experiment Code

This repository contains the experiment code for the MPFC-DPC method built on top of the UFC dataset distillation pipeline.

## Core Generation Files

- `ufc_generation/ufc_cda_cifar_sam_dire.py`
  - MPFC-DPC generation for CIFAR-10 and CIFAR-100.
- `ufc_generation/ufc_cda_tiny_sam_dire.py`
  - MPFC-DPC generation for Tiny-ImageNet.
- `ufc_generation/ufc_cda_imagenet100_sam_dire.py`
  - MPFC-DPC generation for ImageNet-100 using ImageNet-1K pretrained teachers.
- `ufc_generation/utils.py`
  - Shared generation utilities, including BN feature hooks, image clipping, denormalization, and saving.

## Supporting Code

- `models/`
  - Teacher network definitions used by the generation scripts.
- `ufc_validation/`
  - Static and dynamic validation scripts.
- `sh/`
  - Example commands for running the MPFC-DPC generation scripts.

## Not Included

Large or generated experiment artifacts are intentionally excluded:

- datasets
- initialization tensors/images
- pretrained checkpoints
- synthetic generated images
- W&B logs
- Python caches and local server files
