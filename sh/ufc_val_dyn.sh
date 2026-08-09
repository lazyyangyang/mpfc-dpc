#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

# CIFAR-10 dynamic validation
python ufc_validation/val_dyn.py \
    --epochs 80 \
    --batch-size 64 \
    --ipc 10 \
    --M 4 \
    --syn-data-path syn/cifar10-ipc10/generated_results \
    --output-dir syn/cifar10-ipc10/generated_results \
    --wandb-name cifar10-ipc10-dyn \
    --dataset cifar10 \
    --networks resnet18

# CIFAR-100 dynamic validation
python ufc_validation/val_dyn.py \
    --epochs 80 \
    --batch-size 64 \
    --ipc 10 \
    --M 4 \
    --syn-data-path syn/cifar100-ipc10/generated_results \
    --output-dir syn/cifar100-ipc10/generated_results \
    --wandb-name cifar100-ipc10-dyn \
    --dataset cifar100 \
    --networks resnet18

# Tiny-ImageNet dynamic validation
python ufc_validation/val_dyn.py \
    --epochs 80 \
    --batch-size 64 \
    --ipc 10 \
    --M 4 \
    --syn-data-path syn/tiny-ipc10/generated_results \
    --output-dir syn/tiny-ipc10/generated_results \
    --wandb-name tiny-ipc10-dyn \
    --dataset tiny \
    --networks resnet18

# ImageNet-100 dynamic validation
python ufc_validation/val_dyn.py \
    --epochs 80 \
    --batch-size 32 \
    --ipc 10 \
    --M 3 \
    --syn-data-path syn/imagenet100-ipc10/generated_results \
    --output-dir syn/imagenet100-ipc10/generated_results \
    --wandb-name imagenet100-ipc10-dyn \
    --dataset imagenet100 \
    --networks resnet18 \
    --imagenet100-class-list data/wnids.txt \
    --imagenet100-num-class 100 \
    --imagenet-class-index data/imagenet_class_index.json \
    --imagenet100-val-root data/val
