#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0

# CIFAR-10
python ufc_generation/ufc_cda_cifar_sam_dire.py \
    --iteration 1500 \
    --r-bn 1 \
    --batch-size 10 \
    --lr 0.25 \
    --ipc 10 \
    --exp-name generated_results \
    --wandb-name cifar10-ipc10 \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/c10/ \
    --dataset cifar10 \
    --easy2hard-mode step \
    --milestone 0.65 \
    --rcd 1.0 \
    --rcdm 1.0 \
    --redm 0.5

# CIFAR-100
python ufc_generation/ufc_cda_cifar_sam_dire.py \
    --iteration 1500 \
    --r-bn 1 \
    --batch-size 100 \
    --lr 0.25 \
    --ipc 10 \
    --exp-name generated_results \
    --wandb-name cifar100-ipc10 \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/c100/ \
    --dataset cifar100 \
    --easy2hard-mode step \
    --milestone 0.65 \
    --rcd 1.0 \
    --rcdm 1.0 \
    --redm 0.5

# Tiny-ImageNet
python ufc_generation/ufc_cda_tiny_sam_dire.py \
    --iteration 1500 \
    --r-bn 1 \
    --batch-size 100 \
    --lr 0.25 \
    --ipc 10 \
    --exp-name generated_results \
    --wandb-name tiny-ipc10 \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/tiny/ \
    --dataset tiny \
    --easy2hard-mode step \
    --milestone 0.65 \
    --rcd 1.0 \
    --rcdm 1.0 \
    --redm 0.5

# ImageNet-100
python ufc_generation/ufc_cda_imagenet100_sam_dire.py \
    --iteration 1500 \
    --r-bn 1 \
    --batch-size 20 \
    --lr 0.1 \
    --ipc 10 \
    --exp-name generated_results \
    --wandb-name imagenet100-ipc10 \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/tiny/ \
    --class-list data/wnids.txt \
    --dataset imagenet100 \
    --num-class 100 \
    --easy2hard-mode step \
    --milestone 0.65 \
    --rcd 1.0 \
    --rcdm 1.0 \
    --redm 0.5
