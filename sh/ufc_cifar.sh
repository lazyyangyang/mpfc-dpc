#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

# CIFAR-100 example
python ufc_generation/ufc_cda_cifar_sam_dire.py \
    --iteration 1000 \
    --r-bn 1 \
    --batch-size 100 \
    --lr 0.25 \
    --exp-name generated_results \
    --wandb-name cifar100-ipc10-mpfc-dpc \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/c100/ \
    --ipc 10 \
    --dataset cifar100

# CIFAR-10 example
python ufc_generation/ufc_cda_cifar_sam_dire.py \
    --iteration 1000 \
    --r-bn 1 \
    --batch-size 10 \
    --lr 0.25 \
    --exp-name generated_results \
    --wandb-name cifar10-ipc10-mpfc-dpc \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/c10/ \
    --ipc 10 \
    --dataset cifar10

# Tiny-ImageNet example
python ufc_generation/ufc_cda_tiny_sam_dire.py \
    --iteration 1000 \
    --r-bn 1 \
    --batch-size 100 \
    --lr 0.25 \
    --exp-name generated_results \
    --wandb-name tiny-ipc10-mpfc-dpc \
    --store-best-images \
    --syn-data-path syn/ \
    --init_path init_images/tiny/ \
    --ipc 10 \
    --dataset tiny
